from .base_data_loader import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from datasets import Dataset
from moe_cap.configs.cap_config import CAPConfig
from datasets import load_dataset
import re

class GSM8KLoader(DataLoader):
    """Loads and processes the GSM8K dataset"""

    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)
        subset = config.dataset_subset or "main"
        self.dataset = load_dataset("openai/gsm8k", subset, split=config.dataset_split)
        self._process_targets()

    def get_input(self) -> List:
        '''returns the list of questions'''
        return self.dataset["question"]

    def get_target(self) -> List:
        '''returns the list of processed final answers'''
        return self.dataset["processed_answer"]
    
    def _process_targets(self) -> None:
        '''
        Extracts the final numerical ansewr from the reasoning chain
        The answer is always after '####'
        (e.g 'answer': 'Natalia sold 48+24 = <<48+24=72>>72 clips altogether. #### 72'
        ->    'processed_answer': '72')
        '''
        def extract_answer(ans_str: str) -> str:
            match = re.search(r"####\s*([0-9,.-]+)", ans_str)
            if match:
                return match.group(1).replace(',', '')
            return ""
        
        self.dataset = self.dataset.map(
            lambda ex: {"processed_answer": extract_answer(ex["answer"])}
        )