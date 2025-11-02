from .base_data_loader import DataLoader
from moe_cap.configs.cap_config import CAPConfig
from datasets import load_dataset
from typing import List, Dict, Any

class LongBenchV2Loader(DataLoader):
    """
    Loads and processes tasks from the 'THUDM/LongBench-v2' benchmark.
    
    It loads the entire dataset and optionally filters it
    if 'config.dataset_subset' is provided, using it to
    filter the 'sub_domain' column.
    
    It formats the multiple-choice prompt from
    'choice_A', 'choice_B', 'choice_C', and 'choice_D'.
    """

    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)
        try:
            dataset = load_dataset(
                "THUDM/LongBench-v2", 
                split=config.dataset_split
            )
        except Exception as e:
            print(f"Failed to load 'THUDM/LongBench-v2'.")
            print("Please ensure you have an internet connection.")
            raise e
        
        #  use 'dataset_subset' to FILTER the loaded data
        if config.dataset_subset:
            print(f"Filtering LongBench v2 for subset (sub_domain): {config.dataset_subset}")
            self.dataset = dataset.filter(
                lambda x: x['sub_domain'] == config.dataset_subset
            )
            if len(self.dataset) == 0:
                print(f"Warning: No data found for sub_domain '{config.dataset_subset}'.")
        else:
            self.dataset = dataset

        self._process_data()

    def _process_data(self) -> None:
        """
        Formats the input into a prompt for the LLM.
        This function is updated to use the correct column names.
        """
        
        def format_prompt(example: Dict[str, Any]) -> Dict[str, str]:
            context = example['context']
            question = example['question']
            

            # pull choice columns
            options = (
                f"A. {example['choice_A']}\n"
                f"B. {example['choice_B']}\n"
                f"C. {example['choice_C']}\n"
                f"D. {example['choice_D']}"
            )
            
            # prompt template for multiple-choice QA
            prompt = (
                "Please read the following document and answer the multiple-choice question. "
                "Your answer should be the letter of the correct option only.\n\n"
                f"--- Document ---\n{context}\n\n"
                f"--- Question ---\n{question}\n\n"
                f"--- Options ---\n{options}\n\n"
                "Answer:"
            )
            return {"formatted_prompt": prompt}

        self.dataset = self.dataset.map(format_prompt)

    def get_input(self) -> List[str]:
        """Returns the list of formatted prompts."""
        return self.dataset["formatted_prompt"]

    def get_target(self) -> List[str]:
        """
        Returns the list of target answers (e.g "A", "B", "C").
        """
        return self.dataset["answer"]