from types import SimpleNamespace

import moe_cap.utils.hardware_utils as hardware_utils
from moe_cap.utils.cost_utils import GPU_COST_DICT, GPU_TDP_DICT


def test_get_gpu_details_handles_gb10_nan_memory(monkeypatch):
    monkeypatch.setattr(
        hardware_utils.GPUtil,
        "getGPUs",
        lambda: [SimpleNamespace(name="NVIDIA GB10", memoryTotal=float("nan"))],
    )

    assert hardware_utils.get_gpu_details() == "NVIDIA-GB10-UnknownGB"


def test_get_gpu_details_preserves_finite_memory(monkeypatch):
    monkeypatch.setattr(
        hardware_utils.GPUtil,
        "getGPUs",
        lambda: [SimpleNamespace(name="NVIDIA A100", memoryTotal=81920)],
    )

    assert hardware_utils.get_gpu_details() == "NVIDIA-A100-80GB"


def test_gb10_specs_exist_for_all_supported_precisions():
    keys = ("NVIDIA-GB10-UnknownGB", "NVIDIA-GB10-128GB")
    for key in keys:
        assert hardware_utils.MEM_BW_DICT[key] > 0
        for precision in ("float32", "float16", "bfloat16", "int8", "fp8", "fp4", "int4"):
            assert hardware_utils.PEAK_FLOPS_DICT[precision][key] > 0
        assert GPU_COST_DICT[key] > 0
        assert GPU_TDP_DICT[key] > 0
