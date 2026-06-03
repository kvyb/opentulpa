"""mem0-backed memory service for the agent."""

import logging
import re
from typing import Any

from mem0 import Memory  # type: ignore[import-untyped]

_MEM0_NOOP_MESSAGES = frozenset(
    {
        "NOOP for Memory.",
        "NOOP for Memory (async).",
    }
)


class _Mem0NoopFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in _MEM0_NOOP_MESSAGES


_MEM0_NOOP_FILTER = _Mem0NoopFilter()

MEMORY_KIND_ALIASES: dict[str, str] = {
    "directive_profile": "directive_fact",
    "user_skill": "skill_fact",
    "uploaded_file": "file_fact",
    "uploaded_voice_message": "media_fact",
}

MEMORY_KIND_PRIORITY: dict[str, int] = {
    "directive_fact": 0,
    "preference_fact": 1,
    "user_profile_fact": 2,
    "life_fact": 3,
    "relationship_fact": 4,
    "contact_fact": 5,
    "project_fact": 6,
    "aspirations_fact": 7,
    "workflow_fact": 8,
    "skill_fact": 9,
    "code_fact": 10,
    "credential_fact": 11,
    "file_fact": 12,
    "media_fact": 13,
    "thread_context_rollup": 99,
}

_CREDENTIAL_FACT_PATTERNS = (
    r"\bapi[_ -]?key\b",
    r"\bclient[_ -]?secret\b",
    r"\baccess[_ -]?token\b",
    r"\brefresh[_ -]?token\b",
    r"\bstringsession\b",
    r"\bpassword\b",
)
_CODE_FACT_PATTERNS = (
    r"\bapi\b",
    r"\bwebhook\b",
    r"\boauth\b",
    r"\bintegration\b",
    r"\bspreadsheet\b",
    r"\btoolkit\b",
    r"\btool slug\b",
    r"\bconnected account\b",
    r"\bservice\b",
    r"\brepo(?:sitory)?\b",
    r"\bcodebase\b",
)
_ASPIRATION_FACT_PATTERNS = (
    r"\bgoal\b",
    r"\baspir\w+\b",
    r"\bplan(?:ning)?\b",
    r"\bwant to\b",
    r"\bintend to\b",
    r"\btrying to\b",
    r"\bbuild\b",
    r"\blaunch\b",
)
_LIFE_FACT_PATTERNS = (
    r"\btimezone\b",
    r"\butc[+-]?\d{1,2}\b",
    r"\bbirthday\b",
    r"\blives?\b",
    r"\bfrom\b",
    r"\bfamily\b",
    r"\bage\b",
)
_PREFERENCE_FACT_PATTERNS = (
    r"\bprefers?\b",
    r"\blikes?\b",
    r"\bdislikes?\b",
    r"\bfavorite\b",
    r"\bavoid\b",
)


def _install_mem0_noop_filter() -> None:
    logger = logging.getLogger("mem0.memory.main")
    if any(existing is _MEM0_NOOP_FILTER for existing in logger.filters):
        return
    logger.addFilter(_MEM0_NOOP_FILTER)


class MemoryService:
    """Dedicated memory layer using mem0 (local by default). Requires OPENAI_API_KEY for default embedder."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        user_id: str = "default",
    ) -> None:
        _install_mem0_noop_filter()
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

    @staticmethod
    def _extract_text_from_messages(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", "") or "").strip()
            if content:
                parts.append(content)
        return "\n".join(parts).strip()

    @staticmethod
    def _infer_memory_kind(*, text: str, metadata: dict[str, Any]) -> str | None:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return None
        explicit_kind = str(metadata.get("kind", "") or "").strip().lower()
        if explicit_kind:
            return MEMORY_KIND_ALIASES.get(explicit_kind, explicit_kind)
        if any(re.search(pattern, lowered) for pattern in _CREDENTIAL_FACT_PATTERNS):
            return "credential_fact"
        if any(re.search(pattern, lowered) for pattern in _CODE_FACT_PATTERNS):
            return "code_fact"
        if any(re.search(pattern, lowered) for pattern in _ASPIRATION_FACT_PATTERNS):
            return "aspirations_fact"
        if any(re.search(pattern, lowered) for pattern in _PREFERENCE_FACT_PATTERNS):
            return "preference_fact"
        if any(re.search(pattern, lowered) for pattern in _LIFE_FACT_PATTERNS):
            return "life_fact"
        return None

    @classmethod
    def _normalize_metadata_for_write(
        cls,
        metadata: dict[str, Any] | None,
        *,
        text: str,
    ) -> dict[str, Any]:
        normalized = dict(metadata or {})
        original_kind = str(normalized.get("kind", "") or "").strip().lower()
        kind = cls._infer_memory_kind(text=text, metadata=normalized)
        if kind:
            normalized["kind"] = kind
        if original_kind and kind and original_kind != kind:
            normalized.setdefault("legacy_kind", original_kind)
        return normalized

    @classmethod
    def _normalize_record(cls, item: Any) -> dict[str, Any] | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            string_metadata: dict[str, Any] = {}
            kind = (
                cls._infer_memory_kind(text=text, metadata=string_metadata)
                or "thread_context_rollup"
            )
            return {
                "id": "",
                "text": text,
                "memory": text,
                "score": None,
                "kind": kind,
                "metadata": {"kind": kind},
                "created_at": None,
                "updated_at": None,
                "thread_id": "",
                "skill_name": "",
                "source": "",
            }
        if not isinstance(item, dict):
            return None
        raw_metadata = item.get("metadata")
        metadata: dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        text = str(
            item.get("memory")
            or item.get("text")
            or item.get("content")
            or item.get("summary")
            or ""
        ).strip()
        if not text:
            return None
        kind = cls._infer_memory_kind(text=text, metadata=metadata) or "thread_context_rollup"
        metadata["kind"] = kind
        return {
            "id": str(item.get("id", "") or "").strip(),
            "text": text,
            "memory": text,
            "score": item.get("score"),
            "kind": kind,
            "metadata": metadata,
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "thread_id": str(
                metadata.get("thread_id", "") or item.get("thread_id", "") or ""
            ).strip(),
            "skill_name": str(
                metadata.get("skill_name", "") or item.get("skill_name", "") or ""
            ).strip(),
            "source": str(metadata.get("source", "") or item.get("source", "") or "").strip(),
        }

    @classmethod
    def normalize_records(cls, raw: Any) -> list[dict[str, Any]]:
        payload = raw
        if isinstance(raw, dict):
            if isinstance(raw.get("results"), list):
                payload = raw.get("results")
            elif isinstance(raw.get("memories"), list):
                payload = raw.get("memories")
            elif isinstance(raw.get("result"), list):
                payload = raw.get("result")
            elif isinstance(raw.get("memory"), (str, dict)):
                payload = [raw]
        if not isinstance(payload, list):
            payload = [payload] if payload not in (None, "") else []
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in payload:
            normalized = cls._normalize_record(item)
            if normalized is None:
                continue
            dedupe_key = (
                str(normalized.get("kind", "")).strip().lower(),
                str(normalized.get("text", "")).strip().lower(),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append(normalized)
        return records

    @staticmethod
    def grounding_priority(record: dict[str, Any]) -> tuple[int, float]:
        kind = str(record.get("kind", "") or "").strip().lower()
        priority = MEMORY_KIND_PRIORITY.get(kind, 50)
        raw_score = record.get("score")
        try:
            score = float(raw_score) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        return priority, -score

    @staticmethod
    def _mem0_add_has_results(result: Any) -> bool:
        if isinstance(result, dict):
            results = result.get("results")
            if isinstance(results, list):
                return bool(results)
            return bool(result)
        if isinstance(result, list):
            return bool(result)
        return result is not None

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
        prepared_metadata = self._normalize_metadata_for_write(
            metadata,
            text=self._extract_text_from_messages(messages),
        )
        attempts = max(0, int(retries)) + 1
        last_result: Any = None
        for _ in range(attempts):
            try:
                result = mem.add(
                    messages,
                    user_id=uid,
                    metadata=prepared_metadata,
                    infer=bool(infer),
                )
            except TypeError:
                # Compatibility path for mem0 versions that don't expose infer kwarg.
                result = mem.add(
                    messages,
                    user_id=uid,
                    metadata=prepared_metadata,
                )

            last_result = result
            if not bool(infer):
                return result

            # mem0 may swallow malformed JSON from LLM and return empty results.
            # Retry once to recover transient malformed-output failures.
            if self._mem0_add_has_results(result):
                return result

        if bool(infer):
            fallback_metadata = dict(prepared_metadata)
            fallback_metadata["inference_fallback"] = "mem0_empty_result"
            try:
                return mem.add(
                    messages,
                    user_id=uid,
                    metadata=fallback_metadata,
                    infer=False,
                )
            except TypeError:
                return mem.add(
                    messages,
                    user_id=uid,
                    metadata=fallback_metadata,
                )
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
        extra_filters = {
            str(key): value
            for key, value in dict(metadata or {}).items()
            if str(key) not in {"user_id", "agent_id", "run_id"}
        }
        raw_results: Any = []

        # mem0 signatures changed across versions; try common variants.
        # 1) Newer style: explicit user_id argument.
        try:
            raw_results = mem.search(
                query,
                user_id=uid,
                filters=extra_filters,
                limit=limit,
            )
            return self.normalize_records(raw_results)
        except TypeError:
            pass
        except Exception:
            # fall through to compatibility paths
            pass

        # 2) Older style: user_id included in filters.
        filters: dict[str, Any] = dict(extra_filters)
        filters["user_id"] = uid
        try:
            raw_results = mem.search(
                query,
                filters=filters,
                limit=limit,
            )
            return self.normalize_records(raw_results)
        except TypeError:
            # 3) Minimal fallback.
            raw_results = mem.search(query, limit=limit)
            return self.normalize_records(raw_results)

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
