"""Memory tool - simplified version, registered directly to ToolExecutor"""

from datetime import datetime
from memory import MemoryManager, MemoryConfig

# Global memory manager singleton
_manager: MemoryManager = None
_session_id: str = None
_conv_count: int = 0


def _get_manager(user_id: str = "default_user") -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager(
            config=MemoryConfig(),
            user_id=user_id,
            enable_working=True,
        )
    return _manager


def memory(action: str, **kwargs) -> str:
    """Memory tool entry point.

    Args:
        action: Operation type
            - add       : Add a memory. Requires memory_content, optional memory_type(default working), importance(default 0.5)
            - search    : Search memories. Requires query, optional limit(default 5), memory_type
            - summary   : Get memory summary. Optional limit(default 10)
            - stats     : Get memory statistics
            - update    : Update a memory. Requires memory_id, optional memory_content, importance
            - remove    : Delete a memory. Requires memory_id
            - forget    : Bulk-forget low-value memories. Optional strategy, threshold
            - clear_all : Clear all memories (destructive)
    """
    global _session_id, _conv_count
    mgr = _get_manager()

    if _session_id is None:
        _session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Normalize field name: LLM sometimes sends 'content' instead of 'memory_content'
    if "content" in kwargs and "memory_content" not in kwargs:
        kwargs["memory_content"] = kwargs.pop("content")

    try:
        # ---------- add ----------
        if action == "add":
            memory_id = mgr.add_memory(
                memory_content=kwargs.get("memory_content", ""),
                memory_type=kwargs.get("memory_type", "working"),
                importance=kwargs.get("importance", 0.5),
                metadata={"session_id": _session_id, "timestamp": datetime.now().isoformat()},
                auto_classify=False,
            )
            return f"✅ Memory added (ID: {memory_id[:8]}...)"

        # ---------- search ----------
        elif action == "search":
            query = kwargs.get("query", "")
            memory_type = kwargs.get("memory_type")
            results = mgr.retrieve_memories(
                query=query,
                limit=kwargs.get("limit", 5),
                memory_types=[memory_type] if memory_type else None,
                min_importance=kwargs.get("min_importance", 0.1),
            )
            if not results:
                return f"🔍 No memories found for '{query}'"

            lines = [f"🔍 Found {len(results)} relevant memories:"]
            for i, m in enumerate(results, 1):
                preview = m.memory_content[:80] + "..." if len(m.memory_content) > 80 else m.memory_content
                lines.append(f"{i}. [{m.memory_type}] {preview} (importance: {m.importance:.2f})")
            return "\n".join(lines)

        # ---------- summary ----------
        elif action == "summary":
            stats = mgr.get_memory_stats()
            lines = [
                "📊 Memory System Summary",
                f"Total memories : {stats['total_memories']}",
                f"Current session: {_session_id}",
                f"Conversation   : {_conv_count} turns",
            ]
            for t, s in stats.get("memories_by_type", {}).items():
                lines.append(f"  • {t}: {s.get('count', 0)} items (avg importance: {s.get('avg_importance', 0):.2f})")
            return "\n".join(lines)

        # ---------- stats ----------
        elif action == "stats":
            stats = mgr.get_memory_stats()
            return (
                f"📈 Stats: total={stats['total_memories']}, "
                f"types={', '.join(stats['enabled_types'])}, "
                f"session={_session_id}, turns={_conv_count}"
            )

        # ---------- update ----------
        elif action == "update":
            ok = mgr.update_memory(
                memory_id=kwargs["memory_id"],
                content=kwargs.get("memory_content"),
                importance=kwargs.get("importance"),
            )
            return "✅ Memory updated" if ok else "⚠️ Memory not found"

        # ---------- remove ----------
        elif action == "remove":
            ok = mgr.remove_memory(kwargs["memory_id"])
            return "✅ Memory deleted" if ok else "⚠️ Memory not found"

        # ---------- forget ----------
        elif action == "forget":
            count = mgr.forget_memories(
                strategy=kwargs.get("strategy", "importance_based"),
                threshold=kwargs.get("threshold", 0.1),
                max_age_days=kwargs.get("max_age_days", 30),
            )
            return f"🧹 Forgot {count} memories"

        # ---------- clear_all ----------
        elif action == "clear_all":
            mgr.clear_all_memories()
            return "🧽 All memories cleared"

        else:
            return f"❌ Unsupported action: {action}"

    except Exception as e:
        return f"❌ Action failed ({action}): {e}"
