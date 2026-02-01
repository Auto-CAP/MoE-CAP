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

from moe_cap.utils.hardware_utils import get_gpu_details

logger = init_logger(__name__)

# ============================================================================
# CRITICAL: Apply expert distribution monkey patching BEFORE any other vLLM imports
# This must be done early so it applies to all worker processes
# ============================================================================
try:
    # Add path to extracted_expert_dist for imports
    _current_file_dir = os.path.dirname(os.path.abspath(__file__))
    _expert_dist_path = os.path.join(_current_file_dir, "..", "extracted_expert_dist")
    _expert_dist_path = os.path.abspath(_expert_dist_path)
    if _expert_dist_path not in sys.path:
        sys.path.insert(0, _expert_dist_path)
    
    from vllm_integration import apply_vllm_monkey_patching
    # print(f"[PID {os.getpid()}] Applying expert distribution monkey patching...", flush=True)
    apply_vllm_monkey_patching()
    # print(f"[PID {os.getpid()}] Expert distribution monkey patching applied successfully!", flush=True)
except ImportError as e:
    print(f"[PID {os.getpid()}] Warning: Could not import expert distribution patching: {e}", flush=True)
    print(f"[PID {os.getpid()}] Expert distribution recording will not be available.", flush=True)
except Exception as e:
    print(f"[PID {os.getpid()}] Warning: Failed to apply expert distribution patching: {e}", flush=True)
    import traceback
    traceback.print_exc()

# ============================================================================
# Global recording state - using file-based flags for multiprocessing safety
# ============================================================================
import tempfile
import threading
from pathlib import Path

RECORDING_FLAG_FILE = os.path.join(tempfile.gettempdir(), "vllm_batch_recording.flag")
RECORDING_DATA_FILE = os.path.join(tempfile.gettempdir(), "vllm_batch_records.jsonl")
_record_lock = threading.Lock()

# Expert distribution recording state
EXPERT_DISTRIBUTION_RECORDING_FLAG_FILE = os.path.join(tempfile.gettempdir(), "vllm_expert_distribution_recording.flag")
EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE = os.path.join(tempfile.gettempdir(), "vllm_expert_distribution_auto_start.flag")
EXPERT_DISTRIBUTION_OUTPUT_DIR = os.path.join(os.getcwd(), "logs/expert_distribution")
_expert_record_lock = threading.Lock()
_forward_pass_id_counter = 0
_forward_pass_id_lock = threading.Lock()

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
GLOBAL_GPU_TYPE = get_gpu_details()
# ============================================================================
# Expert Distribution Recording State 
# ============================================================================
class ExpertDistributionRecordingState:
    """State for expert distribution recording with automatic JSONL output."""
    
    def __init__(self):
        self.expert_record_list = []
        self.output_dir = EXPERT_DISTRIBUTION_OUTPUT_DIR
        self.model_path = None
        self.enabled = False
        self.checked_auto_start = False
    
    def set_model_path(self, model_path: str):
        """Set the model path for output file naming."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_name = model_path.replace("/", "_")
        self.model_path = f"{sanitized_name}_{timestamp}"
    
    def enable(self):
        """Enable automatic expert distribution recording."""
        self.enabled = True
        # self.expert_record_list = [] # Don't clear list on re-enable to preserve history
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        # logger.info("Expert distribution automatic recording enabled") # Reduce log spam on workers

    def disable(self):
        """Disable automatic expert distribution recording."""
        self.enabled = False
    
    def add_record(self, record: dict):
        """Add a record and write to JSONL file."""
        if not self.enabled:
            return
        
        with _expert_record_lock:
            self.expert_record_list.append(record)
            
            # Write to JSONL file immediately (like sglang.py)
            if self.model_path:
                output_file = os.path.join(self.output_dir, f"{self.model_path}/expert_distribution_record.jsonl")
                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)
                
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
    
    def get_records(self):
        """Get all recorded records."""
        return self.expert_record_list.copy()

expert_distribution_recording_state = ExpertDistributionRecordingState()

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
    
    # Lazy initialization of recording state on workers
    if not expert_distribution_recording_state.checked_auto_start:
        if os.path.exists(EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE):
            expert_distribution_recording_state.enable()
        expert_distribution_recording_state.checked_auto_start = True
    
    # Ensure model path is set if recording is enabled
    if expert_distribution_recording_state.enabled and not expert_distribution_recording_state.model_path:
        if hasattr(self, 'model_config') and hasattr(self.model_config, 'model'):
            expert_distribution_recording_state.set_model_path(self.model_config.model)

    world_size = self.vllm_config.parallel_config.world_size
    gpu_raw_type = GLOBAL_GPU_TYPE
    
    # Lazy initialization of recording state on workers
    if not expert_distribution_recording_state.checked_auto_start:
        if os.path.exists(EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE):
            expert_distribution_recording_state.enable()
        expert_distribution_recording_state.checked_auto_start = True
    
    # Ensure model path is set if recording is enabled
    if expert_distribution_recording_state.enabled and not expert_distribution_recording_state.model_path:
        if hasattr(self, 'model_config') and hasattr(self.model_config, 'model'):
            expert_distribution_recording_state.set_model_path(self.model_config.model)
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
    
    # Track forward pass ID 
    global _forward_pass_id_counter
    with _forward_pass_id_lock:
        _forward_pass_id_counter += 1
        forward_pass_id = _forward_pass_id_counter
    
    # Collect expert distribution data (per_pass mode)
    expert_activation = 0
    expert_utilization = 0
    try:
        # Try to get expert distribution data from the model runner
        if hasattr(self, 'expert_distribution_recorder') and self.expert_distribution_recorder is not None:
            recorder = self.expert_distribution_recorder
            # Check if recording is active
            if hasattr(recorder, '_recording') and recorder._recording:
                # CRITICAL: Use per_pass collection logic (collect -> reset -> append)
                # This is what allows it to work with CUDAGraph (where forward hooks don't run CPU code)
                
                # 1. Collect data from gatherer (syncs if needed)
                if hasattr(recorder, '_gatherer'):
                    collected_data = recorder._gatherer.collect()
                    # Inject forward_mode into collected data so it's available for dump/accumulation
                    collected_data['forward_mode'] = forward_mode
                    recorder._gatherer.reset()
                    
                    # 2. Append to accumulator with current pass ID
                    if hasattr(recorder, '_accumulator'):
                        recorder._accumulator.append(forward_pass_id, collected_data)
                    
                    # 3. Calculate metric for logging from the collected data
                    # collected_data['expert_count'] is [num_layers, num_experts]
                    # Note: The key is 'expert_count' (singular) in the recorder implementation
                    if 'expert_count' in collected_data:
                        counts = collected_data['expert_count']
                        if counts is not None:
                            # Average activated experts per layer
                            # (counts > 0).float() -> [num_layers, num_experts]
                            # .sum(dim=1) -> [num_layers]
                            # .mean() -> scalar
                            active = (counts > 0).float().sum(dim=1).mean().item()
                            expert_activation = active
                    
                    # Calculate utilization
                    if expert_activation > 0 and hasattr(recorder, '_expert_location_metadata'):
                         num_experts = recorder._expert_location_metadata.num_logical_experts
                         if num_experts > 0:
                              expert_utilization = expert_activation / num_experts

    except Exception as e:
        # Don't fail if expert recording is not available or fails
        print(f"[ExpertDist-Error] Failed to calculate/record metrics: {e}", flush=True)
        import traceback
        traceback.print_exc()
        logger.debug(f"Could not collect expert distribution data: {e}")
    
    # Record batch statistics if recording is enabled (file-based check for multiprocessing)
    if recording_state.is_recording():
        # Use attn_metadata for seq_lens_sum if available (gives full context length)
        if hasattr(attn_metadata, "seq_lens"):
             if isinstance(attn_metadata.seq_lens, torch.Tensor):
                  sum_seq_len = attn_metadata.seq_lens.sum().item()
             elif isinstance(attn_metadata.seq_lens, list):
                  sum_seq_len = sum(attn_metadata.seq_lens)
        
        rec_dict = {
            "batch_size": batch_size,
            "latency": latency,
            "seq_lens_sum": sum_seq_len,
            "forward_mode": forward_mode,
            "expert_activation": expert_activation,
            # "expert_utilization": round(expert_utilization, 4),
            "gpu_num": world_size,
            "gpu_raw_type": gpu_raw_type
        }
        recording_state.add_record(rec_dict)
    
    # Automatic expert distribution recording 
    # Only record on rank 0 to avoid duplicates 
    if expert_distribution_recording_state.enabled:
        try:
            from vllm.distributed.parallel_state import get_tp_group
            tp_group = get_tp_group()
            tp_rank = tp_group.rank if tp_group is not None else 0
            
            if tp_rank == 0:  # Only record on rank 0 (like sglang.py)
                # Recalculate seq_lens_sum for this record as well
                if hasattr(attn_metadata, "seq_lens"):
                     if isinstance(attn_metadata.seq_lens, torch.Tensor):
                          sum_seq_len = attn_metadata.seq_lens.sum().item()
                     elif isinstance(attn_metadata.seq_lens, list):
                          sum_seq_len = sum(attn_metadata.seq_lens)
                
                record_dict = {
                    "forward_pass_id": forward_pass_id,
                    "batch_size": batch_size,
                    "latency": latency,
                    "seq_lens_sum": sum_seq_len,
                    "forward_mode": forward_mode,
                    "expert_activation": expert_activation,
                    # "expert_utilization": round(expert_utilization, 4)
                    "gpu_num": world_size,
                    "gpu_raw_type": gpu_raw_type
                }
                expert_distribution_recording_state.add_record(record_dict)
                logger.debug(f"Forward pass {forward_pass_id} completed with latency {latency:.4f}s, expert activation {expert_activation:.2f}")
        except Exception as e:
            logger.debug(f"Could not record expert distribution automatically: {e}")
    
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
# print(f"[PID {os.getpid()}] Applying custom execute_model patch...", flush=True)
GPUModelRunner.execute_model = execute_model_custom
# print(f"[PID {os.getpid()}] Patch applied! Method name: {GPUModelRunner.execute_model.__name__}", flush=True)

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
    from fastapi import FastAPI, Request
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
    
    # Expert distribution endpoints
    # Helper function to get engine client from request (vLLM v1 pattern)
    async def get_engine_client_from_request(request):
        """Get engine client from FastAPI request using vLLM's dependency injection."""
        # Primary method: Use vLLM's engine_client dependency (vLLM v1 pattern)
        try:
            from vllm.entrypoints.openai.api_server import engine_client
            engine_client_obj = await engine_client(request)
            if engine_client_obj is not None and hasattr(engine_client_obj, 'collective_rpc'):
                return engine_client_obj
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"Could not use vLLM's engine_client: {e}")
        
        # Fallback: Try to find engine in app.state
        try:
            # Check common locations
            for attr_name in ['engine', 'llm_engine', 'engine_core']:
                engine = getattr(request.app.state, attr_name, None)
                if engine is not None and hasattr(engine, 'collective_rpc'):
                    return engine
            
            # Check app.state._state dictionary
            if hasattr(request.app.state, '_state') and isinstance(request.app.state._state, dict):
                for value in request.app.state._state.values():
                    if hasattr(value, 'collective_rpc'):
                        return value
                    if hasattr(value, 'engine'):
                        engine = getattr(value, 'engine')
                        if hasattr(engine, 'collective_rpc'):
                            return engine
        except Exception as e:
            logger.debug(f"Error checking app.state: {e}")
        
        return None
    
    @app.post("/configure_expert_distribution")
    async def configure_expert_distribution(request: Request, mode: str = "stat", verbose: bool = False):
        """Configure expert distribution recording mode.
        
        Supported modes:
        - "stat": Aggregate statistics across tokens (default, fastest)
        - "per_token": Records per-token expert selections in detail (most detailed)
        - "per_pass": Records per-forward-pass expert activation metrics (balanced)
        
        Example:
            POST /configure_expert_distribution?mode=per_token
            POST /configure_expert_distribution?mode=per_pass
            POST /configure_expert_distribution?mode=stat
        """
        try:
            # Validate and normalize mode
            valid_modes = {"stat", "per_token", "per_pass"}
            mode_lower = mode.lower().strip()
            
            # Normalize common typos
            mode_normalizations = {
                "per_path": "per_pass",  # Common typo
                "stats": "stat",  # Plural form
                "per-token": "per_token",  # With dash
                "per-pass": "per_pass",  # With dash
            }
            
            if mode_lower in mode_normalizations:
                mode_lower = mode_normalizations[mode_lower]
                logger.warning(f"Normalized mode '{mode}' to '{mode_lower}'")
            
            # Validate mode
            if mode_lower not in valid_modes:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"Invalid recording mode: '{mode}'. Valid modes are: {', '.join(sorted(valid_modes))}",
                        "valid_modes": sorted(valid_modes)
                    }
                )
            
            engine_client_obj = await get_engine_client_from_request(request)
            if engine_client_obj is None or not hasattr(engine_client_obj, 'collective_rpc'):
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": "Engine not available. Server may not be fully initialized."}
                )
            
            # Use normalized mode
            await engine_client_obj.collective_rpc("configure_expert_distribution_recording", args=(mode_lower, verbose))
            return JSONResponse(content={
                "status": "success",
                "message": f"Expert distribution recording configured with mode={mode_lower}",
                "mode": mode_lower,
                "original_mode": mode if mode != mode_lower else None
            })
        except Exception as e:
            import traceback
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e), "traceback": traceback.format_exc()}
            )
    
    @app.post("/start_expert_distribution")
    async def start_expert_distribution(request: Request):
        """Start recording expert distributions."""
        try:
            engine_client_obj = await get_engine_client_from_request(request)
            if engine_client_obj is None or not hasattr(engine_client_obj, 'collective_rpc'):
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": "Engine not available. Server may not be fully initialized."}
                )
            
            await engine_client_obj.collective_rpc("start_expert_distribution_recording")
            return JSONResponse(content={
                "status": "success",
                "message": "Started recording expert distributions"
            })
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)}
            )
    
    @app.post("/stop_expert_distribution")
    async def stop_expert_distribution(request: Request):
        """Stop recording expert distributions."""
        try:
            engine_client_obj = await get_engine_client_from_request(request)
            if engine_client_obj is None or not hasattr(engine_client_obj, 'collective_rpc'):
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": "Engine not available. Server may not be fully initialized."}
                )
            
            await engine_client_obj.collective_rpc("stop_expert_distribution_recording")
            return JSONResponse(content={
                "status": "success",
                "message": "Stopped recording expert distributions"
            })
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)}
            )
    
    @app.post("/dump_expert_distribution")
    async def dump_expert_distribution(request: Request, summary_only: bool = True, pretty: bool = True):
        """Dump recorded expert distribution data.
        
        Args:
            summary_only: If True (default), return summary statistics instead of raw arrays.
                         If False, return full detailed data (can be very large).
            pretty: If True (default), return indented JSON for readability.
        
        Returns:
            Summary statistics similar to sglang.py format, or full data if summary_only=False
        """
        try:
            engine_client_obj = await get_engine_client_from_request(request)
            if engine_client_obj is None or not hasattr(engine_client_obj, 'collective_rpc'):
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": "Engine not available. Server may not be fully initialized."}
                )
            
            all_data = await engine_client_obj.collective_rpc("dump_expert_distribution_record")
            
            # Verify data format - should be a list of dicts (one per worker)
            if not isinstance(all_data, list):
                all_data = [all_data] if all_data else []
            
            response_data = None
            
            if summary_only:
                # Return clean summary similar to sglang.py
                summary = {
                    "status": "success",
                    "num_workers": len(all_data),
                    "summary": {}
                }
                
                # Get per-forward-pass records from worker data (Rank 0)
                # We use worker data because expert_distribution_recording_state is process-local and empty on API server
                records = []
                for worker_data in all_data:
                    if isinstance(worker_data, dict) and worker_data.get("rank") == 0 and "records" in worker_data:
                         records = worker_data["records"]
                         break
                
                if records:
                    # Calculate prefill/decode averages
                    prefill_activations = []
                    decode_activations = []
                    
                    for r in records:
                        # expert_activation calculation repeats logic from execute_model_custom
                        # but here we might have pre-calculated values if the recorder stored them?
                        # The recorder stores 'expert_count' tensor/array.
                        # We need to re-calculate 'active' from the array if it's not stored.
                        # Wait, the recorder stores RAW counts.
                        
                        # Helper to calculate activation from record
                        activation = 0
                        if "expert_count" in r:
                             counts = r["expert_count"]
                             # counts is a list (JSON) or numpy array
                             import numpy as np
                             c = np.array(counts)
                             if c.size > 0:
                                 # (counts > 0).sum(axis=1).mean()
                                 activation = (c > 0).sum(axis=1).mean()
                        elif "expert_counts" in r:
                             counts = r["expert_counts"]
                             import numpy as np
                             c = np.array(counts)
                             if c.size > 0:
                                 activation = (c > 0).sum(axis=1).mean()
                        
                        mode = r.get("forward_mode", "unknown")
                        if mode == "prefill":
                             prefill_activations.append(activation)
                        elif mode == "decode":
                             decode_activations.append(activation)

                    if prefill_activations:
                        summary["summary"]["average_expert_activation_prefill"] = sum(prefill_activations) / len(prefill_activations)
                    
                    if decode_activations:
                        summary["summary"]["average_expert_activation_decode"] = sum(decode_activations) / len(decode_activations)


                    # Clean sample records - remove large arrays to keep output compact
                    sample_records = []
                    for record in (records[:5] if len(records) > 5 else records):
                        clean_record = record.copy()
                        # Remove large arrays that cause formatting issues
                        clean_record.pop("expert_counts", None)  # per_pass mode: 2D array
                        clean_record.pop("activated_per_layer", None)  # per_pass mode: 1D array
                        clean_record.pop("topk_ids", None)  # per_token mode: 3D array [num_layers, num_tokens, topk]
                        sample_records.append(clean_record)
                    
                    # Simplify output: don't show sample records here, just count and file info
                    summary["summary"]["forward_pass_records"] = {
                        "count": len(records),
                        # "sample": sample_records, # Removed to reduce verbosity
                        "note": "Detailed records are in the JSONL file."
                    }
                
                # Extract summary from worker data
                worker_summaries = []
                for worker_data in all_data:
                    if not isinstance(worker_data, dict):
                        continue
                    
                    worker_summary = {
                        "rank": worker_data.get("rank", "unknown"),
                        "recording_mode": worker_data.get("recording_mode", "unknown"),
                        "num_layers": worker_data.get("num_layers"),
                        "num_experts": worker_data.get("num_physical_experts") or worker_data.get("num_experts"),
                    }
                    
                    # Add mode-specific summaries
                    if "aggregated_expert_counts" in worker_data:
                        counts = worker_data["aggregated_expert_counts"]
                        if isinstance(counts, list) and len(counts) > 0:
                            # Calculate summary statistics similar to sglang.py
                            try:
                                # Convert to torch tensor for calculations
                                counts_tensor = torch.tensor(counts, dtype=torch.float32)
                                
                                if counts_tensor.ndim == 2:  # [num_layers, num_experts]
                                    activated_experts = (counts_tensor > 0).float()
                                    activated_per_layer = activated_experts.sum(dim=1)
                                    avg_activated = activated_per_layer.mean()
                                    worker_summary["expert_activation_avg"] = float(avg_activated.item())
                                    worker_summary["total_expert_selections"] = int(counts_tensor.sum().item())
                                elif counts_tensor.ndim == 3:  # [num_forwards, num_layers, num_experts]
                                    latest_counts = counts_tensor[-1] if len(counts_tensor) > 0 else counts_tensor[0]
                                    activated_experts = (latest_counts > 0).float()
                                    activated_per_layer = activated_experts.sum(dim=1)
                                    avg_activated = activated_per_layer.mean()
                                    worker_summary["expert_activation_avg"] = float(avg_activated.item())
                                    worker_summary["total_forward_passes"] = len(counts_tensor)
                            except Exception as e:
                                # If conversion fails, just note that counts are available
                                worker_summary["expert_counts_available"] = True
                                worker_summary["expert_counts_shape"] = str(len(counts)) if isinstance(counts, list) else "unknown"
                    
                    if "records" in worker_data:
                        records = worker_data["records"]
                        worker_summary["num_records"] = len(records) if isinstance(records, list) else 0
                        if isinstance(records, list) and len(records) > 0:
                            # For sample record, only show summary stats, not full arrays
                            sample_record = records[0].copy()
                            # Remove large arrays to keep output compact
                            sample_record.pop("expert_counts", None)  # Remove 2D array (per_pass mode)
                            sample_record.pop("activated_per_layer", None)  # Remove 1D array (per_pass mode)
                            sample_record.pop("topk_ids", None)  # Remove 3D array (per_token mode) - shape: [num_layers, num_tokens, topk]
                            worker_summary["sample_record"] = sample_record
                    
                    worker_summaries.append(worker_summary)
                
                summary["summary"]["workers"] = worker_summaries
                
                # Add file location info if available
                if expert_distribution_recording_state.enabled and expert_distribution_recording_state.model_path:
                    output_file = os.path.join(
                        EXPERT_DISTRIBUTION_OUTPUT_DIR,
                        f"{expert_distribution_recording_state.model_path}/expert_distribution_record.jsonl"
                    )
                    summary["summary"]["jsonl_file"] = output_file
                    summary["summary"]["note"] = "Detailed per-forward-pass records are written to JSONL file automatically."
                
                response_data = summary
            else:
                # Return full data (original behavior)
                def make_serializable(obj):
                    """Recursively make an object JSON-serializable."""
                    import inspect
                    if inspect.iscoroutine(obj):
                        raise ValueError("Found coroutine in data - this should not happen")
                    elif isinstance(obj, (str, int, float, bool, type(None))):
                        return obj
                    elif isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_serializable(item) for item in obj]
                    elif hasattr(obj, '__dict__'):
                        return make_serializable(obj.__dict__)
                    else:
                        return str(obj)
                
                try:
                    response_data = {
                        "status": "success",
                        "data": make_serializable(all_data),
                        "num_workers": len(all_data),
                        "note": "Full detailed data returned. Use summary_only=true for cleaner output."
                    }
                except Exception as e:
                    logger.warning(f"Could not fully serialize data: {e}, converting to string")
                    response_data = {
                        "status": "success",
                        "data": str(all_data),
                        "num_workers": len(all_data),
                        "error": str(e)
                    }

            if pretty:
                from fastapi.responses import Response
                return Response(
                    content=json.dumps(response_data, indent=2, default=str),
                    media_type="application/json"
                )
            else:
                return JSONResponse(content=response_data)

        except Exception as e:
            import traceback
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e), "traceback": traceback.format_exc()}
            )
    
    @app.get("/expert_distribution_status")
    async def expert_distribution_status(request: Request):
        """Get expert distribution recording status."""
        try:
            engine_client_obj = await get_engine_client_from_request(request)
            if engine_client_obj is None:
                return JSONResponse(content={
                    "status": "warning",
                    "message": "Engine not available. Expert distribution recording may not be initialized yet."
                })
            
            return JSONResponse(content={
                "status": "success",
                "message": "Expert distribution recording is available. Use /configure_expert_distribution to configure."
            })
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)}
            )
    
    @app.get("/debug_expert_recording")
    async def debug_expert_recording(request: Request):
        """Debug endpoint to check expert recording setup."""
        try:
            engine_client_obj = await get_engine_client_from_request(request)
            if engine_client_obj is None or not hasattr(engine_client_obj, 'collective_rpc'):
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": "Engine not available. Server may not be fully initialized."}
                )
            
            try:
                all_data = await engine_client_obj.collective_rpc("dump_expert_distribution_record")
                return JSONResponse(content={
                    "status": "success",
                    "message": "Expert distribution recording is working",
                    "num_workers": len(all_data) if isinstance(all_data, list) else 1,
                    "sample_data": all_data[0] if isinstance(all_data, list) and len(all_data) > 0 else {}
                })
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": f"Could not get expert distribution data: {str(e)}"}
                )
        except Exception as e:
            import traceback
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e), "traceback": traceback.format_exc()}
            )
    
    @app.get("/debug_expert_recorder_state")
    async def debug_expert_recorder_state(request: Request):
        """Debug endpoint to check expert recorder state in workers."""
        try:
            engine_client_obj = await get_engine_client_from_request(request)
            if engine_client_obj is None:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": "Engine not available"}
                )
            
            # Use dump_expert_distribution_record to get recorder info
            results = await engine_client_obj.collective_rpc("dump_expert_distribution_record")
            if isinstance(results, list) and len(results) > 0:
                sample = results[0]
                return JSONResponse(content={
                    "status": "success",
                    "num_workers": len(results),
                    "sample_worker_data": {
                        "rank": sample.get("rank"),
                        "num_layers": sample.get("num_layers"),
                        "num_physical_experts": sample.get("num_physical_experts"),
                        "has_counts": bool(sample.get("aggregated_expert_counts")),
                        "has_records": bool(sample.get("records"))
                    }
                })
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "No worker data available"}
            )
        except Exception as e:
            import traceback
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e), "traceback": traceback.format_exc()}
            )


def main():
    """Main entry point for python -m vllm.launch_server"""
    
    parser = FlexibleArgumentParser(
        description="Launch a local OpenAI-compatible API server to serve LLM completions via HTTP. "
                    "Defaults to Qwen/Qwen3-0.6B if no model is specified.",
        usage="python -m vllm.launch_server [model_tag] [options]"
    )
    
    parser = make_arg_parser(parser)
    
    # Add custom argument for expert distribution auto-recording (similar to sglang.py)
    parser.add_argument(
        "--enable-expert-distribution-metrics",
        action="store_true",
        help="Automatically start expert distribution recording and write to JSONL file (similar to sglang.py)"
    )
    
    args = parser.parse_args()
    
    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag
    
    validate_parsed_serve_args(args)
    
    # Enable automatic expert distribution recording if flag is set (similar to sglang.py)
    if hasattr(args, "enable_expert_distribution_metrics") and args.enable_expert_distribution_metrics:
        expert_distribution_recording_state.enable()
        # Set model path for output file naming
        model_path = getattr(args, "model", None) or getattr(args, "model_tag", None) or "unknown_model"
        expert_distribution_recording_state.set_model_path(model_path)
        
        # Create file-based flag for worker processes to auto-start recording
        # This allows worker processes to check the flag and auto-start when creating the recorder
        with open(EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE, 'w') as f:
            f.write('1')
        logger.info("Expert distribution metrics enabled - recording will auto-start in worker processes")
    
    # Monkey-patch build_app to add custom endpoints
    from vllm.entrypoints.openai import api_server
    original_build_app = api_server.build_app
    
    def patched_build_app(args):
        """Patched build_app that adds custom endpoints."""
        app = original_build_app(args)
        
        # vLLM stores the engine in app.state, but we need to access it correctly
        # The engine is typically set by vLLM's build_app function
        # We'll access it through the endpoints using dependency injection pattern
        
        add_custom_endpoints(app)
        
        # Note: Auto-start is handled via file-based flag checked in worker processes
        # The recorder will auto-start when created if EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE exists
        
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