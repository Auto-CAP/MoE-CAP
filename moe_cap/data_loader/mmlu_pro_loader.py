from .base_data_loader import DataLoader
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from moe_cap.configs.cap_config import CAPConfig


class MMLUProLoader(DataLoader):
    """Loads and processes the MMLU-Pro dataset for multi-choice reasoning evaluation"""

    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)

        self.dataset = load_dataset(
            "TIGER-Lab/MMLU-Pro",
            split=config.dataset_split
        )
        self._process_data()

    def _process_data(self) -> None:
        """
        Process the dataset to format inputs and targets.

        The 'options' field contains a list of 10 multiple-choice options (A-J).
        We format the input as a structured prompt with question and options.
        The 'answer' field already contains the correct answer (A-J).
        """
        def process_example(example: Dict[str, Any]) -> Dict[str, str]:
            question = example.get('question', '')
            options = example.get('options', [])
            category = example.get('category', '')

            # Format options as A, B, C, ... J
            option_letters = [chr(65 + i) for i in range(len(options))]  # A, B, C, ..., J
            formatted_options = "\n".join(
                [f"{letter}. {opt}" for letter, opt in zip(option_letters, options)]
            )

            # Format the input prompt
            formatted_input = (
                f"Question (Category: {category}):\n{question}\n\n"
                f"Options:\n{formatted_options}\n\n"
                f"Please select the correct answer (A-J):\n"
            )

            return {
                "formatted_input": formatted_input,
                "processed_answer": example.get('answer', '')
            }

        self.dataset = self.dataset.map(process_example)

    def get_input(self) -> List[str]:
        """Returns the list of formatted input prompts"""
        return self.dataset["formatted_input"]

    def get_target(self) -> List[str]:
        """Returns the list of correct answers (A-J)"""
        return self.dataset["processed_answer"]
