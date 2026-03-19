
from typing import List, Dict, Any
from datetime import datetime, timedelta
import heapq

from ..memory_base import BaseMemory, MemoryItem, MemoryConfig


class WorkingMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens
        self.max_age_minutes = getattr(self.config, 'working_memory_ttl_minutes', 120)
        self.current_tokens = 0
        self.session_start = datetime.now()

        self.memories: List[MemoryItem] = []
        self.memory_heap = []

    def add(self, memory_item: MemoryItem) -> str:
        self._expire_old_memories()
        priority = self._calculate_priority(memory_item)
        heapq.heappush(self.memory_heap, (-priority, memory_item.timestamp, memory_item))
        self.memories.append(memory_item)
        self.current_tokens += len(memory_item.memory_content.split())
        self._enforce_capacity_limits()
        return memory_item.memory_id  # ✅ memory_id

    def retrieve(self, query: str, limit: int = 5, user_id: str = None, **kwargs) -> List[MemoryItem]:
        self._expire_old_memories()
        if not self.memories:
            return []

        active_memories = [m for m in self.memories if not m.metadata.get("forgotten", False)]
        if user_id:
            active_memories = [m for m in active_memories if m.user_id == user_id]
        if not active_memories:
            return []

        vector_scores = {}
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            documents = [query] + [m.memory_content for m in active_memories]  # ✅
            vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            for i, memory in enumerate(active_memories):
                vector_scores[memory.memory_id] = similarities[i]  # ✅
        except Exception:
            vector_scores = {}

        query_lower = query.lower()
        scored_memories = []

        for memory in active_memories:
            content_lower = memory.memory_content.lower()  # ✅

            vector_score = vector_scores.get(memory.memory_id, 0.0)  # ✅

            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / len(content_lower)
            else:
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = len(intersection) / len(query_words.union(content_words)) * 0.8

            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_relevance = keyword_score

            time_decay = self._calculate_time_decay(memory.timestamp)
            base_relevance *= time_decay
            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * importance_weight

            if final_score > 0:
                scored_memories.append((final_score, memory))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def update(self, memory_id: str, content: str = None, importance: float = None,
               metadata: Dict[str, Any] = None) -> bool:
        for memory in self.memories:
            if memory.memory_id == memory_id:  # ✅
                old_tokens = len(memory.memory_content.split())  # ✅

                if content is not None:
                    memory.memory_content = content  # ✅
                    new_tokens = len(content.split())
                    self.current_tokens = self.current_tokens - old_tokens + new_tokens

                if importance is not None:
                    memory.importance = importance

                if metadata is not None:
                    memory.metadata.update(metadata)

                self._update_heap_priority(memory)
                return True
        return False

    def remove(self, memory_id: str) -> bool:
        for i, memory in enumerate(self.memories):
            if memory.memory_id == memory_id:  # ✅
                removed_memory = self.memories.pop(i)
                self._mark_deleted_in_heap(memory_id)
                self.current_tokens -= len(removed_memory.memory_content.split())  # ✅
                self.current_tokens = max(0, self.current_tokens)
                return True
        return False

    def has_memory(self, memory_id: str) -> bool:
        return any(memory.memory_id == memory_id for memory in self.memories)  # ✅

    def clear(self):
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens = 0

    def get_stats(self) -> Dict[str, Any]:
        self._expire_old_memories()
        active_memories = self.memories
        return {
            "count": len(active_memories),
            "forgotten_count": 0,
            "total_count": len(self.memories),
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "avg_importance": sum(m.importance for m in active_memories) / len(
                active_memories) if active_memories else 0.0,
            "capacity_usage": len(active_memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            "token_usage": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            "memory_type": "working"
        }

    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        return sorted(self.memories, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_important(self, limit: int = 10) -> List[MemoryItem]:
        return sorted(self.memories, key=lambda x: x.importance, reverse=True)[:limit]

    def get_all(self) -> List[MemoryItem]:
        return self.memories.copy()

    def get_context_summary(self, max_length: int = 500) -> str:
        if not self.memories:
            return "No working memories available."

        sorted_memories = sorted(self.memories, key=lambda m: (m.importance, m.timestamp), reverse=True)
        summary_parts = []
        current_length = 0

        for memory in sorted_memories:
            content = memory.memory_content
            if current_length + len(content) <= max_length:
                summary_parts.append(content)
                current_length += len(content)
            else:
                remaining = max_length - current_length
                if remaining > 50:
                    summary_parts.append(content[:remaining] + "...")
                break

        return "Working Memory Context:\n" + "\n".join(summary_parts)

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 1) -> int:
        forgotten_count = 0
        current_time = datetime.now()
        to_remove = []

        cutoff_ttl = current_time - timedelta(minutes=self.max_age_minutes)
        for memory in self.memories:
            if memory.timestamp < cutoff_ttl:
                to_remove.append(memory.memory_id)

        if strategy == "importance_based":
            for memory in self.memories:
                if memory.importance < threshold:
                    to_remove.append(memory.memory_id)

        elif strategy == "time_based":
            cutoff_time = current_time - timedelta(hours=max_age_days * 24)
            for memory in self.memories:
                if memory.timestamp < cutoff_time:
                    to_remove.append(memory.memory_id)

        elif strategy == "capacity_based":
            if len(self.memories) > self.max_capacity:
                sorted_memories = sorted(self.memories, key=lambda m: self._calculate_priority(m))
                excess_count = len(self.memories) - self.max_capacity
                for memory in sorted_memories[:excess_count]:
                    to_remove.append(memory.memory_id)

        for memory_id in set(to_remove):
            if self.remove(memory_id):
                forgotten_count += 1

        return forgotten_count

    def _calculate_priority(self, memory: MemoryItem) -> float:
        priority = memory.importance
        time_decay = self._calculate_time_decay(memory.timestamp)
        priority *= time_decay
        return priority

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600
        decay_factor = self.config.decay_factor ** (hours_passed / 6)
        return max(0.1, decay_factor)

    def _enforce_capacity_limits(self):
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()
        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory()

    def _expire_old_memories(self):
        if not self.memories:
            return
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)
        kept = []
        removed_token_sum = 0
        for m in self.memories:
            if m.timestamp >= cutoff_time:
                kept.append(m)
            else:
                removed_token_sum += len(m.memory_content.split())
        if len(kept) == len(self.memories):
            return
        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_token_sum)
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))

    def _remove_lowest_priority_memory(self):
        if not self.memories:
            return
        lowest_priority = float('inf')
        lowest_memory = None
        for memory in self.memories:
            priority = self._calculate_priority(memory)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_memory = memory
        if lowest_memory:
            self.remove(lowest_memory.memory_id)

    def _update_heap_priority(self, memory: MemoryItem):
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))

    def _mark_deleted_in_heap(self, memory_id: str):
        pass