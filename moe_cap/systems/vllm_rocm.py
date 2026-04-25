"""ROCm-specific entry point for moe_cap vLLM server.

Use this for AMD GPUs (MI300X, MI355X, etc.) instead of `moe_cap.systems.vllm`.
GPU detection uses rocm-smi instead of nvidia-smi/GPUtil.
"""

import os
import subprocess


def _get_amd_gpu_type():
    """Detect AMD GPU. Returns formatted name like 'AMD-Instinct-MI355X-288GB'."""
    try:
        import math
        import torch

        raw = torch.cuda.get_device_name(0)
        gb = math.ceil(torch.cuda.get_device_properties(0).total_memory / 1024**3)
        # torch.cuda on ROCm reports names like 'AMD Instinct MI355X' or
        # 'AMD Radeon Graphics' or just 'gfx950'. Normalize to AMD-Instinct-<series>.
        if "MI355" in raw:
            return f"AMD-Instinct-MI355X-{gb}GB"
        if "MI325" in raw:
            return f"AMD-Instinct-MI325X-{gb}GB"
        if "MI300" in raw:
            return f"AMD-Instinct-MI300X-{gb}GB"
        if "MI250" in raw:
            return f"AMD-Instinct-MI250X-{gb}GB"
        name = raw.replace(" ", "-")
        if not name.startswith("AMD"):
            name = f"AMD-{name}"
        return f"{name}-{gb}GB"
    except Exception:
        return None


# Override get_gpu_details BEFORE importing vllm.py (which calls it at module level)
import moe_cap.utils.hardware_utils as _hw

_hw.get_gpu_details = _get_amd_gpu_type

# Now import vllm.py — it will use our patched get_gpu_details
from moe_cap.systems.vllm import *  # noqa: F401, F403
from moe_cap.systems.vllm import main as _main  # explicit re-export for entry


if __name__ == "__main__":
    _main()
