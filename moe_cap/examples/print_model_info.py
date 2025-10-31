import json
import sys
from moe_cap.configs.cap_config import CAPConfig
from moe_cap.model_loader.hf_model_info_loader import HFModelInfoRetriever



def main(cfg_path: str = "examples/model_config.json"):


    with open(cfg_path, "r") as f:
        cfg_json = json.load(f)

    cap_cfg = CAPConfig(
        model_id=cfg_json["model"],
        precision=cfg_json.get("precision"),
        revision=cfg_json.get("revision"),
        quantization=cfg_json.get("quantization")
    )

    retriever = HFModelInfoRetriever(cap_cfg)

    info = retriever.summarize()
    print(json.dumps(info, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "examples/model_config.json"
    main(cfg_path)
