from typing import Any, Dict, List
from abc import ABC, abstractmethod
from serializable import Serializable

class BaseMemory(Serializable, ABC):
    """Base interface for memory in chains."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return key-value pairs given the text input to the chain.

        If None, return all memories.
        
        :param inputs: The text inputs to the chain.
        :type inputs: Dict[str, Any]

        :return: The key-value pairs representing the memories.
        :rtype: Dict[str, Any]
        """

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save the context of this model run to memory.

        :param inputs: The input values.
        :type inputs: Dict[str, Any]
        :param outputs: The output values.
        :type outputs: Dict[str, str]
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""