from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datasets import Dataset
from moe_cap.configs.cap_config import CAPConfig


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def __init__(self, config: CAPConfig) -> None:
        """Initialize the data loader with a configuration dictionary."""
        self.config = config
    
    def get_input(self) -> List:
        """Get input data to the generation models."""
        raise NotImplementedError
    
    def get_target(self) -> List:
        """Get answers / labels"""
        raise NotImplementedError

    def get_eval_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics to compute during evaluation."""
        curr_supported_metrics = ["em", "f1"]
        metric = self.config.get("metrics", None)
        if not metric:
            return curr_supported_metrics
        if isinstance(metric, str):
            metric = [metric]
        elif isinstance(metric, list):
            for m in metric:
                if m not in curr_supported_metrics:
                    raise ValueError(f"Metric {m} not supported. Supported metrics: {curr_supported_metrics}")
        else:
            raise ValueError("Metrics should be a string or a list of strings.")
        return metric