"""ROCm-specific entry point for moe_cap SGLang server.

Use this for AMD GPUs instead of `moe_cap.systems.sglang`.
"""

import os
import subprocess


def _get_amd_gpu_type():
    """Detect AMD GPU using rocm-smi. Returns formatted name like 'AMD-Instinct-MI355X-288GB'."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Parse rocm-smi output, e.g.:
        # device,Card series,Card model,Card vendor,Card SKU
        # card0,Instinct MI355X,...,AMD,...
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[1]:
                    series = parts[1].replace(" ", "-")
                    name = f"AMD-{series}"

                    # Get memory size
                    mem_result = subprocess.run(
                        ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    mem_lines = mem_result.stdout.strip().split("\n")
                    if len(mem_lines) >= 2:
                        mem_parts = [p.strip() for p in mem_lines[1].split(",")]
                        # VRAM Total in bytes is usually in column 1
                        for p in mem_parts[1:]:
                            try:
                                bytes_val = int(p)
                                gb = round(bytes_val / 1024**3)
                                return f"{name}-{gb}GB"
                            except ValueError:
                                continue
                    return name
    except Exception:
        pass

    # Fall back to torch
    try:
        import math
        import torch

        name = torch.cuda.get_device_name(0).replace(" ", "-")
        gb = math.ceil(torch.cuda.get_device_properties(0).total_memory / 1024**3)
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
