from .base_model_info_loader import BaseModelInfoRetriever
from typing import Any, Dict
from transformers import AutoConfig


def _pick(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _deep_get(d: Dict[str, Any], *keys):
    if not isinstance(d, dict):
        return None

    if "thinker_config" in d and isinstance(d["thinker_config"], dict):
        text_cfg = d["thinker_config"].get("text_config")
        if isinstance(text_cfg, dict):
            for k in keys:
                if k in text_cfg and text_cfg[k] is not None:
                    return text_cfg[k]
            for v in text_cfg.values():
                if isinstance(v, dict):
                    found = _deep_get(v, *keys)
                    if found is not None:
                        return found

    for k in keys:
        if k in d and d[k] is not None:
            return d[k]

    for v in d.values():
        if isinstance(v, dict):
            found = _deep_get(v, *keys)
            if found is not None:
                return found

    return None

def _has_kw_recursive(d: Dict[str, Any], keywords) -> bool:
    if not isinstance(d, dict):
        return False
    for k, v in d.items():
        lk = str(k).lower()
        if any(kw in lk for kw in keywords):
            return True
        if isinstance(v, dict) and _has_kw_recursive(v, keywords):
            return True
    return False


class HFModelInfoRetriever(BaseModelInfoRetriever):

    def __init__(self, config):
        super().__init__(config)

        valid_precisions = [
            "float32", "float16", "bfloat16",
            "int8", "int4", "awq", "gptq",
            "fp8", "fp4"
        ]
        if self.config.precision and self.config.precision not in valid_precisions:
            raise ValueError(
                f"Unsupported precision: {self.config.precision}. "
                f"Valid options are: {valid_precisions}"
            )

        self.hf_config = AutoConfig.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            trust_remote_code=True
        )
        self.cfg = self.hf_config.to_dict()

        self.model_name = self.config.model_id

    def get_model_precision_bits(self) -> float:
        p = (self.config.precision or "").lower()
        if p in ("float32", "fp32"):
            return 4.0
        if p in ("float16", "fp16"):
            return 2.0
        if p in ("bfloat16", "bf16"):
            return 2.0
        if p in ("int8", "fp8"):
            return 1.0
        if p in ("int4", "fp4"):
            return 0.5
        return 2.0  


    
    def get_attention_info(self) -> Dict[str, Any]:
        heads = _deep_get(self.cfg, "num_attention_heads", "encoder_attention_heads", "num_heads")
        kv_heads = _deep_get(self.cfg, "num_key_value_heads", "num_kv_heads", "num_key_value_groups")
        head_dim = _deep_get(self.cfg, "head_dim")  
        max_pos = _deep_get(self.cfg, "max_position_embeddings", "max_seq_len", "seq_length")

        if self.cfg.get("model_type") == "dbrx" or str(self.model_name).startswith("databricks/dbrx"):
            heads = heads or _deep_get(self.cfg, "n_heads")
            kv_heads = kv_heads or _deep_get(self.cfg, "attn_config", "kv_n_heads")

        return {
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "max_position_embeddings": max_pos,
        }


    def get_rope_info(self) -> Dict[str, Any]:
        if _deep_get(self.cfg, "use_alibi", "alibi"):
            return {"type": "alibi", "enabled": True}

        rope_theta = _deep_get(self.cfg, "rope_theta", "rotary_emb_base")
        rope_scaling = _deep_get(self.cfg, "rope_scaling")

        if self.cfg.get("model_type") == "dbrx" or str(self.model_name).startswith("databricks/dbrx"):
            attn_cfg = self.cfg.get("attn_config", {})
            if isinstance(attn_cfg, dict) and "rope_theta" in attn_cfg:
                rope_theta = attn_cfg["rope_theta"]  

        if rope_theta is not None or rope_scaling is not None:
            return {
                "type": "rope",
                "rope_theta": rope_theta,
                "rope_scaling": rope_scaling,
            }

        return {"type": "none"}

    

    def get_moe_info(self) -> Dict[str, Any]:
        n_experts = _deep_get(self.cfg, "num_experts_per_layer", "n_experts", "num_local_experts", "expert_number", "num_experts")
        top_k = _deep_get(self.cfg, "moe_top_k", "router_topk", "num_experts_per_tok", "topk", "n_routed_experts")
        n_shared = _deep_get(self.cfg, "n_shared_experts", "num_shared_experts", "shared_experts")
        d_ff = _deep_get(self.cfg, "moe_intermediate_size", "intermediate_size", "ffn_hidden_size")

        if self.cfg.get("model_type") == "dbrx" or str(self.model_name).startswith("databricks/dbrx"):
            ffn_cfg = self.cfg.get("ffn_config", {})
            if isinstance(ffn_cfg, dict):
                n_experts = n_experts or ffn_cfg.get("moe_num_experts")
                top_k = top_k or ffn_cfg.get("moe_top_k")
                d_ff = d_ff or ffn_cfg.get("ffn_hidden_size")

        moe_keywords = ["moe", "expert", "router", "gate"]
        found_moe_pattern = _has_kw_recursive(self.cfg, moe_keywords)

        has_expert_values = any([
            n_experts not in (None, 0),
            top_k not in (None, 0),
            n_shared not in (None, 0)
        ])

        is_moe = found_moe_pattern or has_expert_values

        return {
            "is_moe": is_moe,
            "num_experts_per_layer": n_experts,
            "moe_top_k": top_k,
            "num_shared_experts": n_shared,
            "ffn_dim": d_ff
        }



    def get_architecture_info(self) -> Dict[str, Any]:
        info = {
            "model_type": self.cfg.get("model_type"),
            "hidden_size": _deep_get(self.cfg, "hidden_size"),
            "intermediate_size": _deep_get(self.cfg, "intermediate_size", "ffn_hidden_size"),
            "num_hidden_layers": _deep_get(self.cfg, "num_hidden_layers"),
            "vocab_size": _deep_get(self.cfg, "vocab_size"),
            "max_position_embeddings": _deep_get(self.cfg, "max_position_embeddings", "max_seq_len", "seq_length"),
            "architectures": self.cfg.get("architectures"),
        }

        if info["model_type"] == "dbrx" or str(self.model_name).startswith("databricks/dbrx"):
            info["hidden_size"] = info["hidden_size"] or _deep_get(self.cfg, "d_model")
            info["intermediate_size"] = info["intermediate_size"] or _deep_get(self.cfg, "ffn_config", "ffn_hidden_size")
            info["num_hidden_layers"] = info["num_hidden_layers"] or _deep_get(self.cfg, "n_layers")
            info["max_position_embeddings"] = info["max_position_embeddings"] or _deep_get(self.cfg, "max_seq_len")
        return info