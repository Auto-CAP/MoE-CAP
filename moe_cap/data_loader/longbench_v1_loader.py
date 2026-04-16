import os
import json
import random
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


class LongBenchV1Loader(DataLoader):
    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)
        data_dir = os.environ.get("LONGBENCH_V1_DATA_DIR", "/tmp/longbench_v1/data")

        self.prompts = []
        self.targets = []

        if config.dataset_subset:
            subsets = [config.dataset_subset]
        else:
            subsets = _EN_SUBSETS

        all_data = []
        for subset in subsets:
            path = os.path.join(data_dir, f"{subset}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
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
            answers = obj.get("answers", [])
            self.targets.append(answers[0] if answers else "")

    def get_input(self) -> List[str]:
        return self.prompts

    def get_target(self) -> List[str]:
        return self.targets
