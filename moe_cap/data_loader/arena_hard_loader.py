from .base_data_loader import DataLoader
from typing import Any, Dict, List, Optional
from moe_cap.configs.cap_config import CAPConfig
from datasets import load_dataset


class ArenaHardLoader(DataLoader):
    """
    Loads and processes the Arena-Hard-Auto v0.1 benchmark.
    
    Dataset: lmarena-ai/arena-hard-auto-v0.1
    Contains 500 challenging single-turn prompts across diverse categories.
    Each sample has a 'turns' field with a list of dicts containing 'content'.
    
    This loader extracts the first turn's content as the prompt.
    No ground-truth answers are available (open-ended generation benchmark).
    """

    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)
        # Arena-Hard only has a 'train' split
        split = config.dataset_split
        if split != "train":
            print(f"Warning: Arena-Hard only has 'train' split. Ignoring split='{split}'.")
            split = "train"
        try:
            dataset = load_dataset(
                "lmarena-ai/arena-hard-auto-v0.1",
                split=split,
            )
        except Exception as e:
            print(f"Failed to load 'lmarena-ai/arena-hard-auto-v0.1'.")
            print("Please ensure you have an internet connection.")
            raise e

        # Optional subset filtering by cluster
        if config.dataset_subset:
            print(f"Filtering Arena-Hard for cluster: {config.dataset_subset}")
            dataset = dataset.filter(
                lambda x: x["cluster"] == config.dataset_subset
            )
            if len(dataset) == 0:
                print(f"Warning: No data found for cluster '{config.dataset_subset}'.")

        self.dataset = dataset
        self._process_data()

    def _process_data(self) -> None:
        """Extract the first turn content as the formatted prompt."""

        def format_prompt(example: Dict[str, Any]) -> Dict[str, str]:
            turns = example["turns"]
            content = turns[0]["content"] if turns else ""
            return {"formatted_prompt": content}

        self.dataset = self.dataset.map(format_prompt)

    def get_input(self) -> List:
        """Return the list of prompts."""
        return self.dataset["formatted_prompt"]

    def get_target(self) -> List:
        """No ground-truth answers for Arena-Hard (open-ended benchmark)."""
        return [""] * len(self.dataset)
