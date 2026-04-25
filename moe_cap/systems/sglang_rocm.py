"""ROCm-specific entry point for moe_cap SGLang server.

Use this for AMD GPUs instead of `moe_cap.systems.sglang`.
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
        # Normalize various torch.cuda names to MEM_BW_DICT key format
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


import moe_cap.utils.hardware_utils as _hw

_hw.get_gpu_details = _get_amd_gpu_type

from moe_cap.systems.sglang import *  # noqa: F401, F403
from moe_cap.systems.sglang import __name__ as _sglang_module_name  # keep explicit import side-effect visible


def main():
    """Run SGLang server entrypoint with ROCm GPU detection patched."""
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    import sys

    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
