"""Unit tests for precision resolution from HF quantization_config.

Regression guard for the bug where quantized checkpoints (MXFP4/INT4/FP8) were
recorded as bfloat16 because only torch_dtype was consulted.
"""
from moe_cap.model_loader.precision_utils import resolve_precision, normalize_model_id


def test_mxfp4_quant_method():
    # gpt-oss-120b (unsloth/openai): torch_dtype is bf16 but quant is mxfp4
    cfg = {"torch_dtype": "bfloat16",
           "quantization_config": {"quant_method": "mxfp4"}}
    assert resolve_precision(cfg) == "mxfp4"


def test_compressed_tensors_int4():
    # kimi-k2.5: compressed-tensors, 4-bit int weights -> int4
    cfg = {"torch_dtype": "bfloat16", "quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int"}}},
    }}
    assert resolve_precision(cfg) == "int4"


def test_compressed_tensors_fp8():
    # deepseek-r1 style: compressed-tensors, 8-bit float weights -> fp8
    cfg = {"torch_dtype": "bfloat16", "quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float"}}},
    }}
    assert resolve_precision(cfg) == "fp8"


def test_fp8_quant_method():
    cfg = {"torch_dtype": "bfloat16", "quantization_config": {"quant_method": "fp8"}}
    assert resolve_precision(cfg) == "fp8"


def test_nvfp4_kept_as_method():
    cfg = {"torch_dtype": "bfloat16", "quantization_config": {"quant_method": "nvfp4"}}
    assert resolve_precision(cfg) == "nvfp4"


def test_dense_falls_back_to_dtype():
    assert resolve_precision({"torch_dtype": "bfloat16"}) == "bfloat16"
    assert resolve_precision({"torch_dtype": "float16"}) == "float16"


def test_empty_or_bad_input():
    assert resolve_precision({}) is None
    assert resolve_precision(None) is None


def test_normalize_model_id():
    assert normalize_model_id("/workspace/models/gpt-oss-120b") == "models/gpt-oss-120b"
    assert normalize_model_id("/llm-cache-pvc/models/unsloth/gpt-oss-120b") == "unsloth/gpt-oss-120b"
    assert normalize_model_id("unsloth/gpt-oss-120b") == "unsloth/gpt-oss-120b"
    assert normalize_model_id("gpt-oss-120b") == "gpt-oss-120b"
    assert normalize_model_id("") == ""
