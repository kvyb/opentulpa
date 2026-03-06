"""mem0-backed memory service for the agent."""

from typing import Any

from mem0 import Memory


class MemoryService:
    """Dedicated memory layer using mem0 (local by default). Requires OPENAI_API_KEY for default embedder."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        user_id: str = "default",
    ) -> None:
        self._config = config
        self._memory: Memory | None = None
        self._user_id = user_id

    def _get_memory(self) -> Memory:
        if self._memory is None:
            if self._config:
                self._memory = Memory.from_config(self._config)
            else:
                self._memory = Memory()
        return self._memory

    def add(
        self,
        messages: list[dict[str, str]],
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        retries: int = 1,
    ) -> Any:
        """Add conversation or messages to memory."""
        uid = user_id or self._user_id
        mem = self._get_memory()
        attempts = max(0, int(retries)) + 1
        last_result: Any = None
        for _ in range(attempts):
            try:
                result = mem.add(
                    messages,
                    user_id=uid,
                    metadata=metadata or {},
                    infer=bool(infer),
                )
            except TypeError:
                # Compatibility path for mem0 versions that don't expose infer kwarg.
                result = mem.add(
                    messages,
                    user_id=uid,
                    metadata=metadata or {},
                )

            last_result = result
            if not bool(infer):
                return result

            # mem0 may swallow malformed JSON from LLM and return empty results.
            # Retry once to recover transient malformed-output failures.
            if isinstance(result, dict):
                results = result.get("results")
                if isinstance(results, list) and results:
                    return result
            elif isinstance(result, list):
                if result:
                    return result
            else:
                return result
        return last_result

    def add_text(
        self,
        text: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        retries: int = 1,
    ) -> Any:
        """Add a single text as a user message (mem0 infer/update flow)."""
        return self.add(
            [{"role": "user", "content": text}],
            user_id=user_id,
            metadata=metadata,
            infer=infer,
            retries=retries,
        )

    def search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories for the user."""
        uid = user_id or self._user_id
        mem = self._get_memory()
        extra_filters = metadata or {}

        # mem0 signatures changed across versions; try common variants.
        # 1) Newer style: explicit user_id argument.
        try:
            return mem.search(
                query,
                user_id=uid,
                filters=extra_filters,
                limit=limit,
            )
        except TypeError:
            pass
        except Exception:
            # fall through to compatibility paths
            pass

        # 2) Older style: user_id included in filters.
        filters: dict[str, Any] = {"user_id": uid}
        filters.update(extra_filters)
        try:
            return mem.search(
                query,
                filters=filters,
                limit=limit,
            )
        except TypeError:
            # 3) Minimal fallback.
            return mem.search(query, limit=limit)

    def get_all(
        self,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get recent memories for the user (search with broad query)."""
        return self.search(
            "all memories and context about the user",
            user_id=user_id,
            limit=limit,
        )

    @property
    def user_id(self) -> str:
        return self._user_id
