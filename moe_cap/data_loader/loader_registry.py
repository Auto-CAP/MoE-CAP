"""Registry mapping dataset names to loader classes and defaults.

This module centralizes the mapping from a short dataset name (e.g. 'gsm8k')
to the loader class and a suggested default `max_new_tokens` for generation.

When fixed_length_mode is enabled in config, the loader will wrap the base
dataset and pad/truncate inputs to fixed token lengths for benchmarking.
"""
from typing import Tuple
from . import GSM8KLoader, LongBenchV2Loader, NuminaMathLoader, MMLUProLoader, RulerLoader, FixedLengthLoader, ArenaHardLoader


_REGISTRY = {
    "gsm8k": (GSM8KLoader, 16384), #(Dataloader Class, default_max_new_tokens)
    "longbench_v2": (LongBenchV2Loader, 16384),
    "numinamath": (NuminaMathLoader, 16384),
    "mmlu-pro": (MMLUProLoader, 16384),
    "ruler": (RulerLoader, 16384),
    "arena-hard": (ArenaHardLoader, 16384),
}


def get_loader_for_task(task_name: str, config) -> Tuple[object, int]:
    """Return a tuple (loader_instance, default_max_new_tokens).

    If config.fixed_length_mode is True, returns FixedLengthLoader wrapping
    the specified dataset for performance benchmarking without accuracy eval.
    
    Raises KeyError if the task is unsupported.
    """
    key = task_name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"No loader registered for task '{task_name}'")
    
    # Check if fixed_length_mode is enabled
    if getattr(config, 'fixed_length_mode', False):
        # Use FixedLengthLoader with the base dataset name
        max_tokens = config.target_output_tokens or 1024
        return FixedLengthLoader(config, base_dataset_name=key), max_tokens
    
    LoaderCls, default_max = _REGISTRY[key]
    return LoaderCls(config), default_max
