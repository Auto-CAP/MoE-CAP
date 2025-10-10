from .base_data_loader import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from datasets import Dataset

class SimpleDataLoader(DataLoader):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

    def get_input(self) -> List:
        pass

    def get_target(self) -> List:
        pass