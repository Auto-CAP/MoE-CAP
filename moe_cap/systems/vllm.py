#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Launch vLLM server using: python -m vllm.launch_server [options]

This module provides a simple way to start the vLLM OpenAI-compatible API server
without using the CLI command. It wraps the serve functionality from vllm.entrypoints.openai.
"""

import sys
import os
import uvloop
import json
import textwrap
from collections import defaultdict
from typing import Any, Optional, Union
import regex as re
import yaml
import torch
import time

from argparse import (
    Action,
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    ArgumentTypeError,
    Namespace,
    RawDescriptionHelpFormatter,
    _ArgumentGroup,
)

# ============================================================================
# CRITICAL: Import and patch GPUModelRunner BEFORE any other vLLM imports
# ============================================================================
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, AsyncGPUModelRunnerOutput
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, 
                             AsyncModelRunnerOutput,
                             ModelRunnerOutput)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.utils import record_function_or_nullcontext
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.sequence import IntermediateTensors
from vllm.forward_context import (BatchDescriptor, set_forward_context)
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.logger import init_logger

logger = init_logger(__name__)

# ============================================================================
# Global recording state - using file-based flags for multiprocessing safety
# ============================================================================
import tempfile
import threading

RECORDING_FLAG_FILE = os.path.join(tempfile.gettempdir(), "vllm_batch_recording.flag")
RECORDING_DATA_FILE = os.path.join(tempfile.gettempdir(), "vllm_batch_records.jsonl")
_record_lock = threading.Lock()

class RecordingState:
    """Global state for recording batch statistics - multiprocessing safe."""
    
    def __init__(self):
        # Clean up any stale files on init
        self._cleanup_files()
    
    def _cleanup_files(self):
        """Remove recording flag and data files."""
        try:
            if os.path.exists(RECORDING_FLAG_FILE):
                os.remove(RECORDING_FLAG_FILE)
            if os.path.exists(RECORDING_DATA_FILE):
                os.remove(RECORDING_DATA_FILE)
        except Exception:
            pass
    
    def is_recording(self):
        """Check if recording is active (file-based flag)."""
        return os.path.exists(RECORDING_FLAG_FILE)
    
    def start_recording(self, output_file: str = None):
        """Start recording batch statistics."""
        # Create flag file
        with open(RECORDING_FLAG_FILE, 'w') as f:
            f.write('1')
        # Clear data file
        if os.path.exists(RECORDING_DATA_FILE):
            os.remove(RECORDING_DATA_FILE)
        logger.info("Started recording batch statistics (file-based)")
    
    def stop_recording(self):
        """Stop recording batch statistics."""
        if os.path.exists(RECORDING_FLAG_FILE):
            os.remove(RECORDING_FLAG_FILE)
        count = self.get_record_count()
        logger.info(f"Stopped recording. Total records: {count}")
    
    def add_record(self, record: dict):
        """Add a record to the data file (thread-safe, process-safe)."""
        with _record_lock:
            with open(RECORDING_DATA_FILE, 'a') as f:
                f.write(json.dumps(record) + '\n')
    
    def get_records(self):
        """Get all recorded statistics."""
        if not os.path.exists(RECORDING_DATA_FILE):
            return []
        records = []
        with open(RECORDING_DATA_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
    
    def get_record_count(self):
        """Get count of records without loading all."""
        if not os.path.exists(RECORDING_DATA_FILE):
            return 0
        count = 0
        with open(RECORDING_DATA_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    
    def clear_records(self):
        """Clear all recorded statistics."""
        count = self.get_record_count()
        if os.path.exists(RECORDING_DATA_FILE):
            os.remove(RECORDING_DATA_FILE)
        logger.info(f"Cleared {count} records")
        return count

recording_state = RecordingState()

# ============================================================================
# Custom execute_model implementation
# ============================================================================
@torch.inference_mode()
def execute_model_custom(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
    """Custom execute_model with latency tracking."""
    
    with record_function_or_nullcontext("Preprocess"):
        with self.synchronize_input_prep():
            # Update persistent batch states.
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(
                    scheduler_output, self.vllm_config)
            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.input_batch.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs")
            # Prepare the decoder inputs.
            (attn_metadata, logits_indices, spec_decode_metadata,
             num_scheduled_tokens_np, spec_decode_common_attn_metadata,
             max_query_len, ubatch_slices, num_tokens_after_padding
             ) = self._prepare_inputs(scheduler_output)
        (
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_across_dp,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
        ) = self._preprocess(scheduler_output, intermediate_tensors,
                             ubatch_slices, num_tokens_after_padding)
        uniform_decode = (max_query_len
                          == self.uniform_decode_query_len) and (
                              num_scheduled_tokens
                              == self.input_batch.num_reqs * max_query_len)
        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                           uniform_decode=uniform_decode)
        cudagraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(batch_descriptor)
    
    if ubatch_slices is not None:
        num_input_tokens = ubatch_slices[0].num_tokens
    
    # ======== START TIMING ========
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    # Run the model
    with (set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
    ), record_function_or_nullcontext("Forward"),
          self.maybe_get_kv_connector_output(scheduler_output) as
          kv_connector_output):
        model_output = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
    
    with record_function_or_nullcontext("Postprocess"):
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None
        if not self.broadcast_pp_output:
            if not get_pp_group().is_last_rank:
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                return hidden_states
            if self.is_pooling_model:
                output = self._pool(hidden_states, num_scheduled_tokens,
                                    num_scheduled_tokens_np)
                output.kv_connector_output = kv_connector_output
                return output
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
        else:
            assert not self.is_pooling_model
            if not get_pp_group().is_last_rank:
                all_gather_tensors = {
                    "residual":
                    not is_residual_scattered_for_sp(
                        self.vllm_config, num_input_tokens)
                }
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors,
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors)
                logits = None
            else:
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            model_output_broadcast_data = {}
            if logits is not None:
                model_output_broadcast_data["logits"] = logits.contiguous()
            model_output_broadcast_data = get_pp_group(
            ).broadcast_tensor_dict(model_output_broadcast_data,
                                    src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]
        
        if scheduler_output.grammar_bitmask is not None:
            apply_grammar_bitmask(scheduler_output, self.input_batch,
                                  logits, self.device)
    
    with record_function_or_nullcontext("Sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)
    
    def propose_draft_token_ids(sampled_token_ids):
        assert spec_decode_common_attn_metadata is not None
        with record_function_or_nullcontext("Draft"):
            self._draft_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )
    
    use_padded_batch_for_eagle = self.speculative_config and \
        self.speculative_config.use_eagle() and \
        not self.speculative_config.disable_padded_drafter_batch
    effective_drafter_max_model_len = self.max_model_len
    if effective_drafter_max_model_len is None:
        effective_drafter_max_model_len = self.model_config.max_model_len
    if (self.speculative_config
            and self.speculative_config.draft_model_config is not None
            and self.speculative_config.draft_model_config.max_model_len
            is not None):
        effective_drafter_max_model_len = (
            self.speculative_config.draft_model_config.max_model_len)
    input_fits_in_drafter = spec_decode_common_attn_metadata and (
        spec_decode_common_attn_metadata.seq_lens.max() +
        self.speculative_config.num_speculative_tokens
        <= effective_drafter_max_model_len)
    if use_padded_batch_for_eagle and input_fits_in_drafter:
        propose_draft_token_ids(sampler_output.sampled_token_ids)
    
    with record_function_or_nullcontext("Bookkeep"):
        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(scheduler_output, sampler_output,
                                   logits, hidden_states,
                                   num_scheduled_tokens)
    
    if (self.speculative_config and not use_padded_batch_for_eagle
            and input_fits_in_drafter):
        propose_draft_token_ids(valid_sampled_token_ids)
    
    with record_function_or_nullcontext("EPLB"):
        self.eplb_step()
    
    output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=[],
        kv_connector_output=kv_connector_output,
        num_nans_in_logits=num_nans_in_logits,
    )
    
    # ======== END TIMING ========
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    latency = end_time - start_time
    batch_size = input_ids.size(0)
    forward_mode = "decode" if uniform_decode else "prefill"
    sum_seq_len = num_input_tokens
    
    # Record batch statistics if recording is enabled (file-based check for multiprocessing)
    if recording_state.is_recording():
        rec_dict = {
            "batch_size": batch_size,
            "latency": latency,
            "seq_lens_sum": sum_seq_len,
            "forward_mode": forward_mode,
            "expert_activation": 0  # Will be populated later with actual expert data
        }
        recording_state.add_record(rec_dict)
    
    if not self.use_async_scheduling:
        return output
    return AsyncGPUModelRunnerOutput(
        model_runner_output=output,
        sampled_token_ids=sampler_output.sampled_token_ids,
        invalid_req_indices=invalid_req_indices,
        async_output_copy_stream=self.async_output_copy_stream,
    )


# ============================================================================
# Apply the patch immediately
# ============================================================================
print(f"[PID {os.getpid()}] Applying custom execute_model patch...", flush=True)
GPUModelRunner.execute_model = execute_model_custom
print(f"[PID {os.getpid()}] Patch applied! Method name: {GPUModelRunner.execute_model.__name__}", flush=True)

# Verify the patch
assert GPUModelRunner.execute_model.__name__ == "execute_model_custom", \
    f"Patch verification failed! Got: {GPUModelRunner.execute_model.__name__}"


# ============================================================================
# Now import the rest of vLLM components
# ============================================================================
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args


# ============================================================================
# Argument parser classes (unchanged from original)
# ============================================================================
class SortedHelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    """SortedHelpFormatter that sorts arguments by their option strings."""

    def _split_lines(self, text, width):
        single_newline = re.compile(r"(?<!\n)\n(?!\n)\s*")
        multiple_newlines = re.compile(r"\n{2,}\s*")
        text = single_newline.sub(" ", text)
        lines = re.split(multiple_newlines, text)
        return sum([textwrap.wrap(line, width) for line in lines], [])

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)


class FlexibleArgumentParser(ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    _deprecated: set[Action] = set()
    _json_tip: str = (
        "When passing JSON CLI arguments, the following sets of arguments "
        "are equivalent:\n"
        '   --json-arg \'{"key1": "value1", "key2": {"key3": "value2"}}\'\n'
        "   --json-arg.key1 value1 --json-arg.key2.key3 value2\n\n"
        "Additionally, list elements can be passed individually using +:\n"
        '   --json-arg \'{"key4": ["value3", "value4", "value5"]}\'\n'
        "   --json-arg.key4+ value3 --json-arg.key4+='value4,value5'\n\n"
    )
    _search_keyword: str | None = None

    def __init__(self, *args, **kwargs):
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = SortedHelpFormatter
        self.add_json_tip = kwargs.pop("add_json_tip", True)
        super().__init__(*args, **kwargs)

    if sys.version_info < (3, 13):
        def parse_known_args(self, args=None, namespace=None):
            if args is not None and "--disable-log-requests" in args:
                logger.warning_once(
                    "argument '--disable-log-requests' is deprecated and "
                    "replaced with '--enable-log-requests'. This will be "
                    "removed in v0.12.0."
                )
            namespace, args = super().parse_known_args(args, namespace)
            for action in FlexibleArgumentParser._deprecated:
                if (
                    hasattr(namespace, dest := action.dest)
                    and getattr(namespace, dest) != action.default
                ):
                    logger.warning_once("argument '%s' is deprecated", dest)
            return namespace, args

        def add_argument(self, *args, **kwargs):
            deprecated = kwargs.pop("deprecated", False)
            action = super().add_argument(*args, **kwargs)
            if deprecated:
                FlexibleArgumentParser._deprecated.add(action)
            return action

        class _FlexibleArgumentGroup(_ArgumentGroup):
            def add_argument(self, *args, **kwargs):
                deprecated = kwargs.pop("deprecated", False)
                action = super().add_argument(*args, **kwargs)
                if deprecated:
                    FlexibleArgumentParser._deprecated.add(action)
                return action

        def add_argument_group(self, *args, **kwargs):
            group = self._FlexibleArgumentGroup(self, *args, **kwargs)
            self._action_groups.append(group)
            return group

    def format_help(self):
        if self._subparsers is not None:
            return super().format_help()

        formatter = self._get_formatter()

        if (search_keyword := self._search_keyword) is not None:
            search_keyword = search_keyword.lower().replace("_", "-")
            if search_keyword == "all":
                self.epilog = self._json_tip
                return super().format_help()

            for group in self._action_groups:
                if group.title and group.title.lower() == search_keyword:
                    formatter.start_section(group.title)
                    formatter.add_text(group.description)
                    formatter.add_arguments(group._group_actions)
                    formatter.end_section()
                    formatter.add_text(self._json_tip)
                    return formatter.format_help()

            matched_actions = []
            for group in self._action_groups:
                for action in group._group_actions:
                    if any(
                        search_keyword in opt.lower() for opt in action.option_strings
                    ):
                        matched_actions.append(action)
            if matched_actions:
                formatter.start_section(f"Arguments matching '{search_keyword}'")
                formatter.add_arguments(matched_actions)
                formatter.end_section()
                formatter.add_text(self._json_tip)
                return formatter.format_help()

            formatter.add_text(
                f"No group or arguments matching '{search_keyword}'.\n"
                "Use '--help' to see available groups or "
                "'--help=all' to see all available parameters."
            )
            return formatter.format_help()

        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)

        formatter.start_section("Config Groups")
        config_groups = ""
        for group in self._action_groups:
            if not group._group_actions:
                continue
            title = group.title
            description = group.description or ""
            config_groups += f"{title: <24}{description}\n"
        formatter.add_text(config_groups)
        formatter.end_section()

        formatter.add_text(self.epilog)
        return formatter.format_help()

    def parse_args(self, args: list[str] | None = None, namespace: Namespace | None = None):
        if args is None:
            args = sys.argv[1:]

        if args and args[0] == "serve":
            try:
                model_idx = next(
                    i
                    for i, arg in enumerate(args)
                    if arg == "--model" or arg.startswith("--model=")
                )
                logger.warning(
                    "With `vllm serve`, you should provide the model as a "
                    "positional argument or in a config file instead of via "
                    "the `--model` option. "
                    "The `--model` option will be removed in v0.13."
                )

                if args[model_idx] == "--model":
                    model_tag = args[model_idx + 1]
                    rest_start_idx = model_idx + 2
                else:
                    model_tag = args[model_idx].removeprefix("--model=")
                    rest_start_idx = model_idx + 1

                args = [
                    "serve",
                    model_tag,
                    *args[1:model_idx],
                    *args[rest_start_idx:],
                ]
            except StopIteration:
                pass

        if "--config" in args:
            args = self._pull_args_from_config(args)

        def repl(match: re.Match) -> str:
            return match.group(0).replace("_", "-")

        pattern = re.compile(r"(?<=--)[^\.]*")

        processed_args = list[str]()
        for i, arg in enumerate(args):
            if arg.startswith("--help="):
                FlexibleArgumentParser._search_keyword = arg.split("=", 1)[-1].lower()
                processed_args.append("--help")
            elif arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = pattern.sub(repl, key, count=1)
                    processed_args.append(f"{key}={value}")
                else:
                    key = pattern.sub(repl, arg, count=1)
                    processed_args.append(key)
            elif arg.startswith("-O") and arg != "-O" and arg[2] != ".":
                mode = arg[3:] if arg[2] == "=" else arg[2:]
                processed_args.append(f"-O.mode={mode}")
            elif (
                arg == "-O"
                and i + 1 < len(args)
                and args[i + 1] in {"0", "1", "2", "3"}
            ):
                processed_args.append("-O.mode")
            else:
                processed_args.append(arg)

        def create_nested_dict(keys: list[str], value: str) -> dict[str, Any]:
            nested_dict: Any = value
            for key in reversed(keys):
                nested_dict = {key: nested_dict}
            return nested_dict

        def recursive_dict_update(
            original: dict[str, Any],
            update: dict[str, Any],
        ) -> set[str]:
            duplicates = set[str]()
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(original.get(k), dict):
                    nested_duplicates = recursive_dict_update(original[k], v)
                    duplicates |= {f"{k}.{d}" for d in nested_duplicates}
                elif isinstance(v, list) and isinstance(original.get(k), list):
                    original[k] += v
                else:
                    if k in original:
                        duplicates.add(k)
                    original[k] = v
            return duplicates

        delete = set[int]()
        dict_args = defaultdict[str, dict[str, Any]](dict)
        duplicates = set[str]()
        for i, processed_arg in enumerate(processed_args):
            if i in delete:
                continue

            if processed_arg.startswith("-") and "." in processed_arg:
                if "=" in processed_arg:
                    processed_arg, value_str = processed_arg.split("=", 1)
                    if "." not in processed_arg:
                        continue
                else:
                    value_str = processed_args[i + 1]
                    delete.add(i + 1)

                if processed_arg.endswith("+"):
                    processed_arg = processed_arg[:-1]
                    value_str = json.dumps(list(value_str.split(",")))

                key, *keys = processed_arg.split(".")
                try:
                    value = json.loads(value_str)
                except json.decoder.JSONDecodeError:
                    value = value_str

                arg_dict = create_nested_dict(keys, value)
                arg_duplicates = recursive_dict_update(dict_args[key], arg_dict)
                duplicates |= {f"{key}.{d}" for d in arg_duplicates}
                delete.add(i)
        
        processed_args = [a for i, a in enumerate(processed_args) if i not in delete]
        if duplicates:
            logger.warning("Found duplicate keys %s", ", ".join(duplicates))

        for dict_arg, dict_value in dict_args.items():
            processed_args.append(dict_arg)
            processed_args.append(json.dumps(dict_value))

        return super().parse_args(processed_args, namespace)

    def check_port(self, value):
        try:
            value = int(value)
        except ValueError:
            msg = "Port must be an integer"
            raise ArgumentTypeError(msg) from None

        if not (1024 <= value <= 65535):
            raise ArgumentTypeError("Port must be between 1024 and 65535")

        return value

    def _pull_args_from_config(self, args: list[str]) -> list[str]:
        assert args.count("--config") <= 1, "More than one config file specified!"

        index = args.index("--config")
        if index == len(args) - 1:
            raise ValueError(
                "No config file specified! "
                "Please check your command-line arguments."
            )

        file_path = args[index + 1]
        config_args = self.load_config_file(file_path)

        if args[0].startswith("-"):
            args = config_args + args[0:index] + args[index + 2 :]
        elif args[0] == "serve":
            model_in_cli = len(args) > 1 and not args[1].startswith("-")
            model_in_config = any(arg == "--model" for arg in config_args)

            if not model_in_cli and not model_in_config:
                raise ValueError(
                    "No model specified! Please specify model either "
                    "as a positional argument or in a config file."
                )

            if model_in_cli:
                args = (
                    [args[0]]
                    + [args[1]]
                    + config_args
                    + args[2:index]
                    + args[index + 2 :]
                )
            else:
                args = [args[0]] + config_args + args[1:index] + args[index + 2 :]
        else:
            args = [args[0]] + config_args + args[1:index] + args[index + 2 :]

        return args

    def load_config_file(self, file_path: str) -> list[str]:
        extension: str = file_path.split(".")[-1]
        if extension not in ("yaml", "yml"):
            raise ValueError(
                f"Config file must be of a yaml/yml type. {extension} supplied"
            )

        processed_args: list[str] = []
        config: dict[str, int | str] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. Check path correctness",
                file_path,
            )
            raise ex

        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    processed_args.append("--" + key)
            elif isinstance(value, list):
                if value:
                    processed_args.append("--" + key)
                    for item in value:
                        processed_args.append(str(item))
            else:
                processed_args.append("--" + key)
                processed_args.append(str(value))

        return processed_args


# ============================================================================
# Custom API endpoints for recording
# ============================================================================
def add_custom_endpoints(app):
    """Add custom endpoints to the FastAPI app for batch statistics recording."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    @app.post("/start_batch_recording")
    async def start_batch_recording():
        """Start recording batch statistics."""
        recording_state.start_recording()
        return JSONResponse(content={
            "status": "success",
            "message": "Started recording batch statistics"
        })
    
    @app.post("/stop_batch_recording")
    async def stop_batch_recording():
        """Stop recording batch statistics."""
        recording_state.stop_recording()
        return JSONResponse(content={
            "status": "success",
            "message": "Stopped recording batch statistics",
            "total_records": recording_state.get_record_count()
        })
    
    @app.post("/dump_batch_recording")
    async def dump_batch_recording():
        """Dump batch statistics to file and return as JSON."""
        records = recording_state.get_records()
        return JSONResponse(content={
            "status": "success",
            "records": records,
            "total_records": len(records)
        })
    
    @app.get("/batch_recording_status")
    async def batch_recording_status():
        """Get current recording status."""
        return JSONResponse(content={
            "is_recording": recording_state.is_recording(),
            "total_records": recording_state.get_record_count()
        })
    
    @app.post("/clear_batch_recording")
    async def clear_batch_recording():
        """Clear all recorded batch statistics."""
        count = recording_state.clear_records()
        return JSONResponse(content={
            "status": "success",
            "message": f"Cleared {count} records"
        })


def main():
    """Main entry point for python -m vllm.launch_server"""
    
    parser = FlexibleArgumentParser(
        description="Launch a local OpenAI-compatible API server to serve LLM completions via HTTP. "
                    "Defaults to Qwen/Qwen3-0.6B if no model is specified.",
        usage="python -m vllm.launch_server [model_tag] [options]"
    )
    
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    
    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag
    
    validate_parsed_serve_args(args)
    
    # Monkey-patch build_app to add custom endpoints
    from vllm.entrypoints.openai import api_server
    original_build_app = api_server.build_app
    
    def patched_build_app(args):
        """Patched build_app that adds custom endpoints."""
        app = original_build_app(args)
        add_custom_endpoints(app)
        return app
    
    api_server.build_app = patched_build_app
    
    if args.headless or args.api_server_count < 1:
        from vllm.entrypoints.openai.serve import run_headless
        run_headless(args)
    elif args.api_server_count > 1:
        from vllm.entrypoints.openai.serve import run_multi_api_server
        run_multi_api_server(args)
    else:
        uvloop.run(run_server(args))


if __name__ == "__main__":
    main()