"""Resolve served quantization precision and normalize checkpoint ids.

HuggingFace writes ``torch_dtype: bfloat16`` into ``config.json`` even for
quantized checkpoints, so the dtype alone mislabels MXFP4/INT4/FP8 models as
bf16 (and, via the wrong bytes-per-param, implies impossible memory bandwidth).
The checkpoint's ``quantization_config`` is the source of truth; these helpers
read it. Kept free of heavy imports (``transformers`` etc.) so they can be
unit-tested against synthetic config dicts.
"""
from typing import Any, Dict, Optional


def _ct_weight_bits(qc: Dict[str, Any]):
    """(num_bits, type) of the weight quant in a compressed-tensors config."""
    groups = qc.get("config_groups") or {}
    for g in groups.values():
        w = (g or {}).get("weights") or {}
        if w.get("num_bits") is not None:
            return w.get("num_bits"), w.get("type")
    return qc.get("num_bits"), qc.get("type")


def _find_quant_config(cfg: Dict[str, Any]):
    """quantization_config, incl. multimodal nesting (text_config/thinker_config)."""
    if not isinstance(cfg, dict):
        return None
    qc = cfg.get("quantization_config")
    if qc:
        return qc
    for sub in ("text_config", "thinker_config", "language_config"):
        s = cfg.get(sub)
        if isinstance(s, dict):
            found = _find_quant_config(s)
            if found:
                return found
    return None


def _find_dtype(cfg: Dict[str, Any]):
    """Model dtype — HF uses ``torch_dtype`` or (newer) ``dtype``; may be nested."""
    if not isinstance(cfg, dict):
        return None
    for key in ("torch_dtype", "dtype"):
        v = cfg.get(key)
        if v:
            return str(v)
    for sub in ("text_config", "thinker_config", "language_config"):
        s = cfg.get(sub)
        if isinstance(s, dict):
            found = _find_dtype(s)
            if found:
                return found
    return None


def resolve_precision(cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve the *served* precision from an HF config dict.

    Prefer ``quantization_config`` (the checkpoint's real quantization, possibly
    nested in ``text_config`` for multimodal checkpoints); fall back to the
    model dtype only for genuinely dense models. Returns a precision string
    (``mxfp4``/``int4``/``fp8``/``bfloat16``/...) or None.
    """
    if not isinstance(cfg, dict):
        return None
    qc = _find_quant_config(cfg)
    if not qc:
        return _find_dtype(cfg)
    qm = str(qc.get("quant_method", "")).lower()
    # compressed-tensors is identified by quant_method OR by config_groups /
    # a "*-quantized" format (kimi-k2.5 omits quant_method entirely).
    is_ct = (qm == "compressed-tensors" or qc.get("config_groups")
             or "quantized" in str(qc.get("format", "")).lower())
    if is_ct:
        nbits, wtype = _ct_weight_bits(qc)
        if nbits == 4:
            return "int4" if str(wtype or "").lower() == "int" else "mxfp4"
        if nbits == 8:
            return "fp8"
    if qm in ("mxfp4", "nvfp4"):
        return qm  # keep the method name; don't collapse to a generic "fp4"
    if qm:
        return qm  # fp8, awq, gptq, ...
    return _find_dtype(cfg)


def normalize_model_id(name) -> str:
    """Reduce a checkpoint reference to an HF id (``org/model``).

    A bare local path like ``/workspace/models/gpt-oss-120b`` loses its org, so
    keep the last two path segments; leave a proper ``org/model`` id intact.
    """
    if not name:
        return name
    s = str(name).rstrip("/")
    if s.startswith("/") or s.count("/") >= 2:
        parts = [p for p in s.split("/") if p]
        return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else s)
    return s
