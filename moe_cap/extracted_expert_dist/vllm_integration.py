#!/usr/bin/env python3
"""
vLLM Integration for Expert Distribution Recording

This module provides the monkey patching logic to integrate expert distribution
recording functionality into vLLM. It patches vLLM's Worker, GPUModelRunner,
and MoE model classes to enable automatic expert distribution recording.

MONKEY PATCHING COVERAGE:
✅ Module replacement: Replaces vLLM's expert_distribution_recorder with our custom one
✅ Worker RPC methods: Adds configure/start/stop/dump expert distribution methods
✅ Recorder initialization: Automatic MoE model detection and recorder setup
✅ Model layer context: Patches Qwen2MoeDecoderLayer and DeepseekV2DecoderLayer
✅ FusedMoE integration: Uses built-in recording logic in vLLM's FusedMoE.select_experts

REQUIREMENTS:
- vLLM v0.11.0 or compatible version
- MoE model (Qwen/Qwen1.5-MoE-A2.7B, DeepSeek-V2, etc.)
- GPU with sufficient memory

Usage:
    from vllm_integration import apply_vllm_monkey_patching
    apply_vllm_monkey_patching()
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))


def apply_vllm_monkey_patching():
    """Apply vLLM monkey patching for expert distribution recording."""
    try:
        # Try to import vLLM and apply monkey patching
        import vllm
        print("Using real vLLM")

        # Import Worker early and store original __init__
        from vllm.v1.worker.gpu_worker import Worker
        _original_worker_init = Worker.__init__

        # Monkey patch vLLM to use our custom expert_distribution_recorder
        import vllm.distributed.eplb as eplb_module

        # Replace the module in sys.modules BEFORE any imports happen
        import expert_distribution_recorder as custom_recorder
        sys.modules['vllm.distributed.eplb.expert_distribution_recorder'] = custom_recorder
        eplb_module.expert_distribution_recorder = custom_recorder

        # Also ensure our moe_hooks is used
        import moe_hooks as custom_hooks
        sys.modules['vllm.distributed.eplb.moe_hooks'] = custom_hooks
        eplb_module.moe_hooks = custom_hooks

        print("Successfully monkey-patched vLLM with custom expert_distribution_recorder")

        # Monkey patch Worker to add expert distribution methods
        try:
            from vllm.v1.worker.gpu_worker import Worker

            # Add expert distribution recorder attribute if it doesn't exist
            if not hasattr(Worker, 'expert_distribution_recorder'):
                Worker.expert_distribution_recorder = None

            # Define the worker monkey patching method
            def _apply_worker_monkey_patching(self):
                """Apply monkey patching in the worker process for MoE model classes."""
                try:
                    import sys

                    # Apply module replacements in worker process
                    import expert_distribution_recorder as custom_recorder
                    sys.modules['vllm.distributed.eplb.expert_distribution_recorder'] = custom_recorder

                    import moe_hooks as custom_hooks
                    sys.modules['vllm.distributed.eplb.moe_hooks'] = custom_hooks

                    # Also monkey patch GPUModelRunner in worker process
                    try:
                        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

                        # Add expert_distribution_recorder attribute
                        if not hasattr(GPUModelRunner, 'expert_distribution_recorder'):
                            GPUModelRunner.expert_distribution_recorder = None

                        # Always ensure all methods are available in worker process
                        def _get_expert_location_metadata(self):
                            try:
                                model = self.get_model()
                                hf_config = self.model_config.hf_config

                                def is_mixture_of_experts(model):
                                    return hasattr(model, "num_logical_experts") and hasattr(model, "num_expert_layers")

                                if is_mixture_of_experts(model):
                                    num_logical_experts = model.num_logical_experts
                                    num_layers = model.num_expert_layers
                                    ep_size = getattr(model, 'ep_size', 1)
                                    num_physical_experts = num_logical_experts
                                    num_local_physical_experts = num_logical_experts // ep_size if ep_size > 1 else num_logical_experts
                                else:
                                    num_experts = getattr(hf_config, 'num_experts', 60)
                                    num_layers = 24
                                    num_physical_experts = num_experts
                                    num_local_physical_experts = num_experts
                                    ep_size = 1

                                from expert_distribution_recorder import ExpertLocationMetadata
                                return ExpertLocationMetadata(
                                    num_layers=num_layers,
                                    num_logical_experts=num_experts,
                                    num_physical_experts=num_physical_experts,
                                    num_local_physical_experts=num_local_physical_experts,
                                    ep_size=ep_size,
                                )
                            except Exception:
                                return None

                        def configure_expert_distribution_recording(self, recording_mode=None, enable_metrics=False, buffer_size=-1):
                            from expert_distribution_recorder import ExpertDistributionRecorder, set_global_expert_distribution_recorder
                            expert_location_metadata = self._get_expert_location_metadata()
                            if expert_location_metadata is None:
                                return

                            import torch
                            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

                            self.expert_distribution_recorder = ExpertDistributionRecorder.init_new(
                                recording_mode=recording_mode,
                                expert_location_metadata=expert_location_metadata,
                                rank=rank,
                                device=str(self.device),
                                buffer_size=buffer_size,
                                enable_metrics=enable_metrics,
                            )
                            set_global_expert_distribution_recorder(self.expert_distribution_recorder)

                        def start_expert_distribution_recording(self):
                            if self.expert_distribution_recorder:
                                self.expert_distribution_recorder.start_record()

                        def stop_expert_distribution_recording(self):
                            if self.expert_distribution_recorder:
                                self.expert_distribution_recorder.stop_record()

                        def dump_expert_distribution_record(self, output_path=None):
                            if self.expert_distribution_recorder:
                                return self.expert_distribution_recorder.dump_record(output_path)
                            return {}

                        def check_expert_buffer(self):
                            if self.expert_distribution_recorder:
                                buffer = self.expert_distribution_recorder.get_expert_counts_buffer()
                                return {
                                    "buffer_available": buffer is not None,
                                    "buffer_shape": buffer.shape if buffer is not None else None,
                                }
                            return {"buffer_available": False}

                        # Always apply the methods in worker process (overwrite any existing)
                        GPUModelRunner._get_expert_location_metadata = _get_expert_location_metadata
                        GPUModelRunner.configure_expert_distribution_recording = configure_expert_distribution_recording
                        GPUModelRunner.start_expert_distribution_recording = start_expert_distribution_recording
                        GPUModelRunner.stop_expert_distribution_recording = stop_expert_distribution_recording
                        GPUModelRunner.dump_expert_distribution_record = dump_expert_distribution_record
                        GPUModelRunner.check_expert_buffer = check_expert_buffer

                    except ImportError:
                        pass

                    # Import required modules
                    import inspect
                    from vllm.model_executor.models.utils import extract_layer_index

                    # Try to patch Qwen models
                    try:
                        from vllm.model_executor.models.qwen2_moe import Qwen2MoeDecoderLayer, Qwen2MoeSparseMoeBlock
                    except ImportError:
                        Qwen2MoeDecoderLayer = None
                        Qwen2MoeSparseMoeBlock = None

                    # Try to patch DeepSeek models
                    try:
                        from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
                    except ImportError:
                        DeepseekV2DecoderLayer = None

                    # Patch Qwen models if available
                    if Qwen2MoeDecoderLayer is not None:
                        # Store original __init__
                        original_init = Qwen2MoeDecoderLayer.__init__

                        def patched_qwen_init(self, *args, **kwargs):
                            # Extract layer_idx from prefix if prefix is provided
                            layer_idx = None
                            if len(args) >= 4:  # config, layer_id, quant_config, prefix
                                prefix = args[3]
                                layer_idx = extract_layer_index(prefix)
                            elif 'prefix' in kwargs:
                                prefix = kwargs['prefix']
                                layer_idx = extract_layer_index(prefix)

                            # Don't filter kwargs - let the original __init__ handle what it accepts
                            # The signature is: (self, config, layer_id, quant_config=None, prefix="", alt_stream=None)
                            # Just pass everything through and let the method handle unknown parameters
                            original_init(self, *args, **kwargs)

                            # Set layer_idx as instance attribute if we found it
                            if layer_idx is not None:
                                self.layer_idx = layer_idx

                        # Patch the Qwen2MoeDecoderLayer forward method
                        original_forward = Qwen2MoeDecoderLayer.forward

                        def patched_qwen_forward(self, *args, **kwargs):
                            # Get layer index
                            layer_idx = getattr(self, 'layer_idx', None)

                            if layer_idx is not None:
                                from expert_distribution_recorder import get_global_expert_distribution_recorder
                                recorder = get_global_expert_distribution_recorder()
                                if recorder is not None and hasattr(recorder, '_recording') and recorder._recording:
                                    # For per_pass and per_token modes, we need layer context for data collection
                                    # For stat mode, select_experts patching handles everything atomically
                                    if recorder._recording_mode in ["per_pass", "per_token"]:
                                        with recorder.with_current_layer(layer_idx):
                                            result = original_forward(self, *args, **kwargs)

                                        # For per_pass and per_token modes, collect data only on the last layer (23)
                                        # This simulates collecting once per complete forward pass (token through all layers)
                                        if recorder._recording_mode in ["per_pass", "per_token"] and layer_idx == 23:
                                            try:
                                                collected_data = recorder._gatherer.collect()
                                                recorder._gatherer.reset()

                                                # Use a simple incrementing counter for forward pass ID
                                                if not hasattr(recorder, '_forward_pass_counter'):
                                                    recorder._forward_pass_counter = 0
                                                forward_pass_id = recorder._forward_pass_counter
                                                recorder._forward_pass_counter += 1

                                                recorder._accumulator.append(forward_pass_id, collected_data)
                                            except Exception:
                                                pass

                                        return result

                            return original_forward(self, *args, **kwargs)

                        Qwen2MoeDecoderLayer.__init__ = patched_qwen_init
                        Qwen2MoeDecoderLayer.forward = patched_qwen_forward

                    # Patch DeepSeek models if available
                    if DeepseekV2DecoderLayer is not None:
                        # Patch the DeepseekV2DecoderLayer forward method
                        original_deepseek_forward = DeepseekV2DecoderLayer.forward

                        def patched_deepseek_forward(self, *args, **kwargs):
                            # Get layer index from self.mlp if it exists
                            layer_idx = getattr(self, 'layer_idx', None)

                            if layer_idx is not None:
                                from expert_distribution_recorder import get_global_expert_distribution_recorder
                                recorder = get_global_expert_distribution_recorder()
                                if recorder is not None and hasattr(recorder, '_recording') and recorder._recording:
                                    # For per_pass and per_token modes, we need layer context for data collection
                                    # For stat mode, select_experts patching handles everything atomically
                                    if recorder._recording_mode in ["per_pass", "per_token"]:
                                        with recorder.with_current_layer(layer_idx):
                                            result = original_deepseek_forward(self, *args, **kwargs)

                                        # For per_pass and per_token modes, collect data only on the last layer
                                        # Use a reasonable default for DeepSeek (assuming similar layer count)
                                        last_layer_idx = getattr(recorder, '_expert_location_metadata', None)
                                        if last_layer_idx:
                                            last_layer_idx = recorder._expert_location_metadata.num_layers - 1
                                        else:
                                            last_layer_idx = 59  # Conservative default for DeepSeek

                                        if recorder._recording_mode in ["per_pass", "per_token"] and layer_idx == last_layer_idx:
                                            try:
                                                collected_data = recorder._gatherer.collect()
                                                recorder._gatherer.reset()

                                                # Use a simple incrementing counter for forward pass ID
                                                if not hasattr(recorder, '_forward_pass_counter'):
                                                    recorder._forward_pass_counter = 0
                                                forward_pass_id = recorder._forward_pass_counter
                                                recorder._forward_pass_counter += 1

                                                recorder._accumulator.append(forward_pass_id, collected_data)
                                            except Exception:
                                                pass

                                        return result

                            return original_deepseek_forward(self, *args, **kwargs)

                        DeepseekV2DecoderLayer.forward = patched_deepseek_forward

                except Exception as e:
                    # Don't crash if monkey patching fails in worker
                    pass

            # Add the method to the Worker class
            Worker._apply_worker_monkey_patching = _apply_worker_monkey_patching

            def patched_init(self, *args, **kwargs):
                # Apply monkey patching in the worker process before model loading
                self._apply_worker_monkey_patching()

                # Only pass Worker.__init__ parameters to avoid passing them to model constructors
                # Worker.__init__ accepts: vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
                worker_kwargs = {k: v for k, v in kwargs.items()
                               if k in ['vllm_config', 'local_rank', 'rank', 'distributed_init_method', 'is_driver_worker']}

                # Call the original Worker.__init__ using super() to bypass our patched method
                super(Worker, self).__init__(**worker_kwargs)

                # Initialize expert distribution recorder (will be done lazily when needed)
                self.expert_distribution_recorder = None

            # Add expert distribution recorder methods to Worker
            def configure_expert_distribution_recorder(self, recording_mode: str | None = None, enable_metrics: bool = False, buffer_size: int = -1):
                """Configure the expert distribution recorder on the worker."""
                # Ensure model_runner exists (it should be initialized by now)
                if not hasattr(self, 'model_runner') or self.model_runner is None:
                    # If model_runner not ready yet, return error
                    return {"success": False, "error": "model_runner not initialized"}

                # Call the GPUModelRunner method directly to avoid delegation issues
                from expert_distribution_recorder import ExpertDistributionRecorder, set_global_expert_distribution_recorder

                # Get expert location metadata inline (avoid method call issues)
                try:
                    model = self.model_runner.get_model()
                    hf_config = self.model_runner.model_config.hf_config

                    def is_mixture_of_experts(model):
                        return hasattr(model, "num_logical_experts") and hasattr(model, "num_expert_layers")

                    if is_mixture_of_experts(model):
                        num_logical_experts = model.num_logical_experts
                        num_layers = model.num_expert_layers
                        ep_size = getattr(model, 'ep_size', 1)
                        num_physical_experts = num_logical_experts
                        num_local_physical_experts = num_logical_experts // ep_size if ep_size > 1 else num_logical_experts
                    else:
                        num_experts = getattr(hf_config, 'num_experts', 60)
                        num_layers = 24
                        num_physical_experts = num_experts
                        num_local_physical_experts = num_experts
                        ep_size = 1

                    from expert_distribution_recorder import ExpertLocationMetadata
                    expert_location_metadata = ExpertLocationMetadata(
                        num_layers=num_layers,
                        num_logical_experts=num_experts,
                        num_physical_experts=num_physical_experts,
                        num_local_physical_experts=num_local_physical_experts,
                        ep_size=ep_size,
                    )
                except Exception as e:
                    return {"success": False, "error": f"Failed to get expert metadata: {e}"}

                import torch
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

                self.model_runner.expert_distribution_recorder = ExpertDistributionRecorder.init_new(
                    recording_mode=recording_mode,
                    expert_location_metadata=expert_location_metadata,
                    rank=rank,
                    device=str(self.model_runner.device),
                    buffer_size=buffer_size,
                    enable_metrics=enable_metrics,
                )
                set_global_expert_distribution_recorder(self.model_runner.expert_distribution_recorder)

                recorder = self.model_runner.expert_distribution_recorder
                return {
                    "recording_mode": getattr(recorder, "_recording_mode", None),
                    "recording": recorder.recording,
                }

            # Backward compatibility alias for older RPC name
            def configure_expert_distribution_recording(self, mode: str, verbose: bool = False):
                """Configure expert distribution recording mode."""
                return self.configure_expert_distribution_recorder(recording_mode=mode, enable_metrics=verbose)

            def start_expert_distribution_recording(self):
                """Start recording expert distributions."""
                if hasattr(self, 'model_runner') and self.model_runner is not None:
                    self.model_runner.start_expert_distribution_recording()

            def dump_expert_distribution_record(self, output_path=None):
                """Dump recorded expert distribution data."""
                if hasattr(self, 'model_runner') and self.model_runner is not None:
                    return self.model_runner.dump_expert_distribution_record(output_path)
                return {}

            def stop_expert_distribution_recording(self):
                """Stop recording expert distributions."""
                if hasattr(self, 'model_runner') and self.model_runner is not None:
                    self.model_runner.stop_expert_distribution_recording()

            # Monkey patch the methods
            Worker.__init__ = patched_init
            Worker.configure_expert_distribution_recorder = configure_expert_distribution_recorder
            Worker.configure_expert_distribution_recording = configure_expert_distribution_recording
            Worker.start_expert_distribution_recording = start_expert_distribution_recording
            Worker.dump_expert_distribution_record = dump_expert_distribution_record
            Worker.stop_expert_distribution_recording = stop_expert_distribution_recording

            print("Successfully monkey-patched Worker with expert distribution methods")

            # Now monkey patch GPUModelRunner with expert distribution methods
            try:
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner

                # Add expert_distribution_recorder attribute
                if not hasattr(GPUModelRunner, 'expert_distribution_recorder'):
                    GPUModelRunner.expert_distribution_recorder = None

                def _get_expert_location_metadata(self):
                    """Extract expert location metadata from model config."""
                    try:
                        model = self.get_model()
                        hf_config = self.model_config.hf_config

                        # Check if it's a MoE model
                        def is_mixture_of_experts(model):
                            return hasattr(model, "num_logical_experts") and hasattr(model, "num_expert_layers")

                        if is_mixture_of_experts(model):
                            num_logical_experts = model.num_logical_experts
                            num_layers = model.num_expert_layers
                            ep_size = getattr(model, 'ep_size', 1)

                            num_physical_experts = num_logical_experts
                            num_local_physical_experts = num_logical_experts // ep_size if ep_size > 1 else num_logical_experts
                        else:
                            # Fallback to config-based extraction
                            num_experts = getattr(hf_config, 'num_experts', 60)  # Qwen default
                            num_layers = 24  # Qwen 1.5 MoE layers

                            num_physical_experts = num_experts
                            num_local_physical_experts = num_experts
                            ep_size = 1

                        from expert_distribution_recorder import ExpertLocationMetadata
                        return ExpertLocationMetadata(
                            num_layers=num_layers,
                            num_logical_experts=num_experts,
                            num_physical_experts=num_physical_experts,
                            num_local_physical_experts=num_local_physical_experts,
                            ep_size=ep_size,
                        )
                    except Exception as e:
                        print(f"Failed to extract expert location metadata: {e}")
                        return None

                def configure_expert_distribution_recording(self, recording_mode=None, enable_metrics=False, buffer_size=-1):
                    """Configure expert distribution recording."""
                    from expert_distribution_recorder import ExpertDistributionRecorder, set_global_expert_distribution_recorder

                    expert_location_metadata = self._get_expert_location_metadata()
                    if expert_location_metadata is None:
                        print("[ExpertRecorder] Could not extract expert location metadata from model")
                        return

                    import torch
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

                    self.expert_distribution_recorder = ExpertDistributionRecorder.init_new(
                        recording_mode=recording_mode,
                        expert_location_metadata=expert_location_metadata,
                        rank=rank,
                        device=str(self.device),
                        buffer_size=buffer_size,
                        enable_metrics=enable_metrics,
                    )
                    set_global_expert_distribution_recorder(self.expert_distribution_recorder)
                    print(f"[ExpertRecorder] Expert distribution recording configured with mode={recording_mode}")

                def start_expert_distribution_recording(self):
                    """Start recording expert distributions."""
                    if self.expert_distribution_recorder:
                        self.expert_distribution_recorder.start_record()

                def stop_expert_distribution_recording(self):
                    """Stop recording expert distributions."""
                    if self.expert_distribution_recorder:
                        self.expert_distribution_recorder.stop_record()

                def dump_expert_distribution_record(self, output_path=None):
                    """Dump recorded expert distribution data."""
                    if self.expert_distribution_recorder:
                        return self.expert_distribution_recorder.dump_record(output_path)
                    return {}

                def check_expert_buffer(self):
                    """Check if expert buffer is available."""
                    if self.expert_distribution_recorder:
                        buffer = self.expert_distribution_recorder.get_expert_counts_buffer()
                        return {
                            "buffer_available": buffer is not None,
                            "buffer_shape": buffer.shape if buffer is not None else None,
                        }
                    return {"buffer_available": False}

                # Monkey patch the methods
                GPUModelRunner._get_expert_location_metadata = _get_expert_location_metadata
                GPUModelRunner.configure_expert_distribution_recording = configure_expert_distribution_recording
                GPUModelRunner.start_expert_distribution_recording = start_expert_distribution_recording
                GPUModelRunner.stop_expert_distribution_recording = stop_expert_distribution_recording
                GPUModelRunner.dump_expert_distribution_record = dump_expert_distribution_record
                GPUModelRunner.check_expert_buffer = check_expert_buffer

                print("Successfully monkey-patched GPUModelRunner with expert distribution methods")

            except ImportError as e:
                print(f"Could not monkey-patch GPUModelRunner: {e}")

        except ImportError as e:
            print(f"vLLM not available: {e}")

    except ImportError as e:
        print(f"vLLM not available: {e}")

    # Patch FusedMoE.select_experts and SharedFusedMoE.select_experts to add our recording logic
    try:
        from vllm.model_executor.layers.fused_moe import FusedMoE
        original_select_experts = FusedMoE.select_experts

        def patched_select_experts(*args, **kwargs):
            # Call original
            result = original_select_experts(*args, **kwargs)

            # Add our recording logic
            try:
                from expert_distribution_recorder import get_global_expert_distribution_recorder
                recorder = get_global_expert_distribution_recorder()

                if recorder is not None and hasattr(recorder, '_recording') and recorder._recording:
                    # Get topk_ids from result (select_experts returns topk_weights, topk_ids, zero_expert_result)
                    topk_weights, topk_ids, _ = result

                    # For STAT mode, try to get layer_idx from context manager first, then fallback to inspection
                    effective_layer_idx = getattr(recorder, '_current_layer_idx', None)

                    # If context manager didn't set it, try to inspect the call stack to find layer_idx
                    if effective_layer_idx is None:
                        try:
                            import inspect
                            frame = inspect.currentframe()
                            while frame:
                                frame_locals = frame.f_locals
                                # Check if we're in a decoder layer forward method
                                self_obj = frame_locals.get('self', None)
                                if self_obj and hasattr(self_obj, 'layer_idx'):
                                    effective_layer_idx = self_obj.layer_idx
                                    break
                                frame = frame.f_back
                        except:
                            pass

                    if effective_layer_idx is not None:
                        if recorder._recording_mode == "stat":
                            # Use torch.compile-compatible atomic recording for stat mode
                            from moe_hooks import record_expert_selection_atomic
                            record_expert_selection_atomic(effective_layer_idx, topk_ids)
                        else:
                            # Use traditional callback for other modes (per_token, per_pass)
                            # Layer context is already set by decoder layer forward patching
                            recorder.on_select_experts(effective_layer_idx, topk_ids)
            except Exception as e:
                # Don't crash if recording fails
                pass

            return result

        FusedMoE.select_experts = staticmethod(patched_select_experts)

    except Exception:
        pass

    # Also patch SharedFusedMoE.select_experts
    try:
        from vllm.model_executor.layers.shared_fused_moe.shared_fused_moe import SharedFusedMoE
        original_shared_select_experts = SharedFusedMoE.select_experts

        def patched_shared_select_experts(*args, **kwargs):
            # Call original
            result = original_shared_select_experts(*args, **kwargs)

            # Add our recording logic
            try:
                from expert_distribution_recorder import get_global_expert_distribution_recorder
                recorder = get_global_expert_distribution_recorder()

                if recorder is not None and hasattr(recorder, '_recording') and recorder._recording:
                    # Get topk_ids from result (select_experts returns topk_weights, topk_ids, zero_expert_result)
                    topk_weights, topk_ids, _ = result

                    # For STAT mode, try to get layer_idx from context manager first, then fallback to inspection
                    effective_layer_idx = getattr(recorder, '_current_layer_idx', None)

                    # If context manager didn't set it, try to inspect the call stack to find layer_idx
                    if effective_layer_idx is None:
                        try:
                            import inspect
                            frame = inspect.currentframe()
                            while frame:
                                frame_locals = frame.f_locals
                                # Check if we're in a decoder layer forward method
                                self_obj = frame_locals.get('self', None)
                                if self_obj and hasattr(self_obj, 'layer_idx'):
                                    effective_layer_idx = self_obj.layer_idx
                                    break
                                frame = frame.f_back
                        except:
                            pass

                    if effective_layer_idx is not None:
                        if recorder._recording_mode == "stat":
                            # Use torch.compile-compatible atomic recording for stat mode
                            from moe_hooks import record_expert_selection_atomic
                            record_expert_selection_atomic(effective_layer_idx, topk_ids)
                        else:
                            # Use traditional callback for other modes (per_token, per_pass)
                            # Layer context is already set by decoder layer forward patching
                            recorder.on_select_experts(effective_layer_idx, topk_ids)
            except Exception as e:
                # Don't crash if recording fails
                pass

            return result

        SharedFusedMoE.select_experts = staticmethod(patched_shared_select_experts)

    except Exception:
        pass
