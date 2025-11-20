from .base_data_loader import DataLoader
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from moe_cap.configs.cap_config import CAPConfig


class RulerLoader(DataLoader):
    """Loads and processes the RULER dataset for long-context evaluation"""

    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)

        # RULER dataset uses 'default' configuration
        # The dataset_subset can filter by task type if provided
        self.dataset = load_dataset(
            "Tongyi-Zhiwen/ruler-128k-subset",
            split=config.dataset_split
        )
        self._process_data()

    def _process_data(self) -> None:
        """
        Process the dataset to format inputs and targets.

        The 'outputs' field is an array of possible answers.
        We extract the first output as the ground truth answer.
        We format the input as a prompt combining context and query.
        """
        def process_example(example: Dict[str, Any]) -> Dict[str, str]:
            # Extract the first output from the outputs array
            outputs = example.get('outputs', [])
            target_answer = outputs[0] if outputs else ""

            # Format the input prompt with context and query
            context = example.get('context', '')
            query = example.get('query', '')

            formatted_input = (
                f"Please read the following context and answer the question.\n\n"
                f"--- Context ---\n{context}\n\n"
                f"--- Question ---\n{query}\n\n"
                f"Answer:"
            )

            return {
                "formatted_input": formatted_input,
                "processed_answer": target_answer
            }

        self.dataset = self.dataset.map(process_example)

    def get_input(self) -> List[str]:
        """Returns the list of formatted input prompts"""
        return self.dataset["formatted_input"]

    def get_target(self) -> List[str]:
        """Returns the list of processed answers"""
        return self.dataset["processed_answer"]

