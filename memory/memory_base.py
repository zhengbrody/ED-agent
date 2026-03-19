from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel


class MemoryItem(BaseModel):
    memory_id: str
    memory_content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    storage_path: str = "./memory_storage"

    # Base configuration for display/statistics purposes
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120


class BaseMemory(ABC):
    """Base class for all memory types.

    Defines the common interface and behavior for all memory implementations.
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        pass

    @abstractmethod
    def update(self, memory_id: str, content: str = None,
               importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """Calculate the importance score of a memory item based on its content."""
        importance = base_importance

        # Boost importance for longer content
        if len(content) > 100:
            importance += 0.1

        # Boost importance if content contains high-priority keywords
        important_keywords = ["important", "critical", "must", "warning", "error", "attention"]
        if any(keyword in content.lower() for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()