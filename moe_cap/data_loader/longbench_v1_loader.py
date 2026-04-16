import os
import json
from .base_data_loader import DataLoader
from moe_cap.configs.cap_config import CAPConfig
from typing import List


class LongBenchV1Loader(DataLoader):
    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)
        subset = config.dataset_subset or "qasper"
        data_dir = os.environ.get("LONGBENCH_V1_DATA_DIR", "/tmp/longbench_v1/data")
        path = os.path.join(data_dir, f"{subset}.jsonl")

        self.prompts = []
        self.targets = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
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
