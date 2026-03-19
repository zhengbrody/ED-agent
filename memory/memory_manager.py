"""Memory Manager - Unified interface for the memory core layer"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

from .memory_base import MemoryItem, MemoryConfig
from .types.working import WorkingMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory Manager - Unified interface for all memory operations.

    Responsibilities:
    - Memory lifecycle management
    - Memory priority and importance evaluation
    - Memory forgetting and cleanup
    """

    def __init__(self, config: Optional[MemoryConfig] = None, user_id: str = "default_user",
                 enable_working: bool = True):
        self.config = config or MemoryConfig()
        self.user_id = user_id
        self.memory_types = {}

        if enable_working:
            self.memory_types['working'] = WorkingMemory(self.config)

        logger.info(f"MemoryManager initialized. Enabled types: {list(self.memory_types.keys())}")

    def add_memory(self, memory_content: str, memory_type: str = "working",
                   importance: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None,
                   auto_classify: bool = True) -> str:
        """Add a memory item.

        Args:
            memory_content: Content of the memory
            memory_type: Type of memory (only 'working' is supported)
            importance: Importance score (0-1)
            metadata: Additional metadata
            auto_classify: Whether to auto-classify memory type

        Returns:
            Memory ID
        """
        if auto_classify:
            memory_type = self._classify_memory_type(memory_content, metadata)

        if importance is None:
            importance = self._calculate_importance(memory_content, metadata)

        memory_item = MemoryItem(
            memory_id=str(uuid.uuid4()),
            memory_content=memory_content,
            memory_type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )

        if memory_type in self.memory_types:
            memory_id = self.memory_types[memory_type].add(memory_item)
            logger.debug(f"Added memory to '{memory_type}': {memory_id}")
            return memory_id
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

    def retrieve_memories(self, query: str, memory_types: Optional[List[str]] = None,
                          limit: int = 10, min_importance: float = 0.0,
                          time_range: Optional[tuple] = None) -> List[MemoryItem]:
        """Retrieve memories matching a query.

        Args:
            query: Search query
            memory_types: List of memory types to search (defaults to all)
            limit: Maximum number of results
            min_importance: Minimum importance threshold
            time_range: Optional (start_time, end_time) filter

        Returns:
            List of matching MemoryItem objects
        """
        if memory_types is None:
            memory_types = list(self.memory_types.keys())

        all_results = []
        per_type_limit = max(1, limit // len(memory_types))

        for memory_type in memory_types:
            if memory_type in self.memory_types:
                memory_instance = self.memory_types[memory_type]
                try:
                    type_results = memory_instance.retrieve(
                        query=query,
                        limit=per_type_limit,
                        min_importance=min_importance,
                        user_id=self.user_id
                    )
                    all_results.extend(type_results)
                except Exception as e:
                    logger.warning(f"Error retrieving from '{memory_type}': {e}")

        all_results.sort(key=lambda x: x.importance, reverse=True)
        return all_results[:limit]

    def update_memory(self, memory_id: str, content: Optional[str] = None,
                      importance: Optional[float] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory item.

        Returns:
            True if updated successfully, False if not found
        """
        for memory_instance in self.memory_types.values():
            if memory_instance.has_memory(memory_id):
                return memory_instance.update(memory_id, content, importance, metadata)

        logger.warning(f"Memory not found: {memory_id}")
        return False

    def remove_memory(self, memory_id: str) -> bool:
        """Delete a memory item by ID.

        Returns:
            True if deleted successfully, False if not found
        """
        for memory_instance in self.memory_types.values():
            if memory_instance.has_memory(memory_id):
                return memory_instance.remove(memory_id)

        logger.warning(f"Memory not found: {memory_id}")
        return False

    def forget_memories(self, strategy: str = "importance_based", threshold: float = 0.1,
                        max_age_days: int = 30) -> int:
        """Forget low-value memories using the specified strategy.

        Args:
            strategy: 'importance_based', 'time_based', or 'capacity_based'
            threshold: Importance threshold below which memories are forgotten
            max_age_days: Maximum age in days for time-based forgetting

        Returns:
            Number of memories forgotten
        """
        total_forgotten = 0
        for memory_instance in self.memory_types.values():
            if hasattr(memory_instance, 'forget'):
                total_forgotten += memory_instance.forget(strategy, threshold, max_age_days)

        logger.info(f"Forgot {total_forgotten} memories.")
        return total_forgotten

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics for all memory types."""
        stats = {
            "user_id": self.user_id,
            "enabled_types": list(self.memory_types.keys()),
            "total_memories": 0,
            "memories_by_type": {},
            "config": {
                "max_capacity": self.config.max_capacity,
                "importance_threshold": self.config.importance_threshold,
                "decay_factor": self.config.decay_factor
            }
        }

        for memory_type, memory_instance in self.memory_types.items():
            type_stats = memory_instance.get_stats()
            stats["memories_by_type"][memory_type] = type_stats
            stats["total_memories"] += type_stats.get("count", 0)

        return stats

    def clear_all_memories(self):
        """Clear all memories across all memory types."""
        for memory_instance in self.memory_types.values():
            memory_instance.clear()
        logger.info("All memories cleared.")

    def _classify_memory_type(self, memory_content: str,
                               metadata: Optional[Dict[str, Any]]) -> str:
        """Auto-classify memory type (always returns 'working')."""
        if metadata and metadata.get("type"):
            return metadata["type"]
        return "working"

    def _calculate_importance(self, memory_content: str,
                               metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate importance score based on content and metadata."""
        importance = 0.5

        if len(memory_content) > 100:
            importance += 0.1

        important_keywords = ["important", "critical", "must", "warning", "error", "attention"]
        if any(kw in memory_content.lower() for kw in important_keywords):
            importance += 0.2

        if metadata:
            if metadata.get("priority") == "high":
                importance += 0.3
            elif metadata.get("priority") == "low":
                importance -= 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_memory_stats()
        return f"MemoryManager(user={self.user_id}, total={stats['total_memories']})"