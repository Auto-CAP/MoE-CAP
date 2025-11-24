from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..configs.cap_config import CAPConfig

# To run the example, you need to install the transformers library:
# pip install transformers
# pip install torch

class BaseModelInfoRetriever(ABC):
    """
    Abstract base class for retrieving and summarizing model architecture information.

    This class defines a standard interface for extracting key details about a
    transformer model, such as its architecture, attention mechanism, and quantization
    precision. Subclasses must implement the abstract methods to provide a
    concrete way of fetching this information from a specific source (e.g.,
    Hugging Face Hub, local files).
    """

    def __init__(self, config: CAPConfig) -> None:
        """
        Initializes the retriever with the model name and precision.

        Args:
            model_name: The identifier for the model.
            precision: The data type or quantization format used by the model.
        """
        VALID_PRECISIONS = ['float32', 'float16', 'bfloat16', 'int8', 'int4', 'awq', 'gptq', 'fp8', 'fp4']
        self.config = config
        # TODO do some precision type checking. If not in the valid list, raise error.

    @abstractmethod
    def get_model_precision_bytes(self) -> float:
        """Returns the effective number of bytes per parameter for the given precision."""
        pass

    @abstractmethod
    def get_attention_info(self) -> Dict[str, Any]:
        """Returns attention-related information."""
        pass

    @abstractmethod
    def get_rope_info(self) -> Dict[str, Any]:
        """Returns RoPE (rotary embedding) information if available."""
        pass

    @abstractmethod
    def get_moe_info(self) -> Dict[str, Any]:
        """Returns Mixture of Experts (MoE) configuration."""
        pass

    @abstractmethod
    def get_architecture_info(self) -> Dict[str, Any]:
        """Returns model-wide architecture information."""
        pass

    def summarize(self) -> Dict[str, Any]:
        """
        Aggregates all extracted information into a single dictionary.

        This method provides a consistent output format for any class that
        implements this interface.
        """
        arch_info = self.get_architecture_info()
        return {
            "model_name": self.model_name,
            "model_type": arch_info.get("model_type"),
            "precision_bytes_per_param": self.get_model_precision_bits(),
            "architecture": arch_info,
            "attention": self.get_attention_info(),
            "rope": self.get_rope_info(),
            "moe": self.get_moe_info()
        }