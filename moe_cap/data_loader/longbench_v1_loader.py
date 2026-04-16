import os
import json
import random
import zipfile
from .base_data_loader import DataLoader
from moe_cap.configs.cap_config import CAPConfig
from typing import List

# English-only subsets (exclude _e variants and Chinese subsets)
_EN_SUBSETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
    "narrativeqa",
    "qmsum",
]

SEED = 42


def _ensure_data_dir() -> str:
    """Download and extract LongBench v1 data if not present."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "longbench_v1", "data")
    if os.path.exists(cache_dir) and any(
        f.endswith(".jsonl") for f in os.listdir(cache_dir)
    ):
        return cache_dir
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.dirname(cache_dir))
    extracted = os.path.join(os.path.dirname(cache_dir), "data")
    if extracted != cache_dir:
        os.rename(extracted, cache_dir)
    return cache_dir


class LongBenchV1Loader(DataLoader):
    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)
        data_dir = os.environ.get("LONGBENCH_V1_DATA_DIR") or _ensure_data_dir()

        self.prompts = []
        self.targets = []

        if config.dataset_subset:
            subsets = [config.dataset_subset]
        else:
            subsets = _EN_SUBSETS

        MAX_WORDS = 33000  # ~50K tokens, skip very long samples

        all_data = []
        for subset in subsets:
            path = os.path.join(data_dir, f"{subset}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("length", 0) > MAX_WORDS:
                        continue
                    all_data.append(obj)

        # Fixed seed shuffle for deterministic sampling
        rng = random.Random(SEED)
        rng.shuffle(all_data)

        for obj in all_data:
            context = obj["context"]
            question = obj["input"]
            prompt = (
                "Please read the following context and answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            )
            self.prompts.append(prompt)
            self.targets.append(
                {
                    "answers": obj.get("answers", []),
                    "dataset": obj.get("dataset", ""),
                    "all_classes": obj.get("all_classes"),
                }
            )

    def get_input(self) -> List[str]:
        return self.prompts

    def get_target(self) -> List[dict]:
        return self.targets
