"""Registry mapping dataset names to loader classes and defaults.

This module centralizes the mapping from a short dataset name (e.g. 'gsm8k')
to the loader class and a suggested default `max_new_tokens` for generation.
"""
from typing import Tuple
from . import GSM8KLoader, LongBenchV2Loader, NuminaMathLoader, MMLUProLoader, RulerLoader


_REGISTRY = {
    "gsm8k": (GSM8KLoader, 16384), #(Dataloader Class, default_max_new_tokens)
    "longbench_v2": (LongBenchV2Loader, 16384),
    "numinamath": (NuminaMathLoader, 16384),
    "mmlu-pro": (MMLUProLoader, 16384),
    "ruler": (RulerLoader, 16384),
}


def get_loader_for_task(task_name: str, config) -> Tuple[object, int]:
    """Return a tuple (loader_instance, default_max_new_tokens).

    Raises KeyError if the task is unsupported.
    """
    key = task_name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"No loader registered for task '{task_name}'")
    LoaderCls, default_max = _REGISTRY[key]
    return LoaderCls(config), default_max
