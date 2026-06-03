from __future__ import annotations

import logging
from typing import Any

from opentulpa.memory.service import MemoryService


def test_memory_service_filters_only_mem0_noop_logs(caplog) -> None:
    MemoryService()
    logger = logging.getLogger("mem0.memory.main")

    with caplog.at_level(logging.INFO, logger="mem0.memory.main"):
        logger.info("NOOP for Memory.")
        logger.info("useful mem0 info")

    assert "NOOP for Memory." not in caplog.text
    assert "useful mem0 info" in caplog.text


class _FakeMem0:
    def __init__(self, *, search_result=None) -> None:
        self.search_result = search_result or []
        self.add_calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: object):
        del query, kwargs
        return self.search_result

    def add(self, messages, *, user_id: str, metadata: dict[str, object], infer: bool = True):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "metadata": metadata,
                "infer": infer,
            }
        )
        return {"results": [{"ok": True}]}


class _LegacySearchFakeMem0:
    def __init__(self) -> None:
        self.filters_calls: list[dict[str, object]] = []

    def search(self, query: str, **kwargs: object):
        if "user_id" in kwargs:
            raise TypeError("legacy signature")
        raw_filters = kwargs.get("filters")
        assert isinstance(raw_filters, dict)
        self.filters_calls.append(
            {
                "query": query,
                "filters": dict(raw_filters),
                "limit": kwargs.get("limit"),
            }
        )
        return []


class _EmptyInferenceFakeMem0:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []

    def add(self, messages, *, user_id: str, metadata: dict[str, object], infer: bool = True):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "metadata": metadata,
                "infer": infer,
            }
        )
        if infer:
            return {"results": []}
        return {"results": [{"memory": "raw fallback", "event": "ADD"}]}


def test_memory_service_normalizes_dict_style_search_results() -> None:
    memory = MemoryService()
    memory._memory = _FakeMem0(
        search_result={
            "results": [
                {
                    "id": "mem_1",
                    "memory": "Timezone is UTC+8.",
                    "score": 0.91,
                    "metadata": {"kind": "life_fact"},
                },
                {
                    "id": "mem_2",
                    "memory": "Saved skill for browser automation.",
                    "score": 0.72,
                    "metadata": {"kind": "user_skill", "skill_name": "browser-use-operator"},
                },
                {
                    "id": "mem_3",
                    "memory": "Saved skill for browser automation.",
                    "score": 0.55,
                    "metadata": {"kind": "user_skill", "skill_name": "browser-use-operator"},
                },
            ]
        }
    )

    results = memory.search("what do you know?", user_id="telegram_test", limit=5)

    assert len(results) == 2
    assert results[0]["kind"] == "life_fact"
    assert results[1]["kind"] == "skill_fact"
    assert results[1]["skill_name"] == "browser-use-operator"


def test_memory_service_infers_typed_kinds_on_write() -> None:
    memory = MemoryService()
    fake = _FakeMem0()
    memory._memory = fake

    memory.add_text(
        "Telegram bot API key is stored for the sandbox service.", user_id="telegram_test"
    )
    memory.add_text("User wants to launch a paid community this year.", user_id="telegram_test")
    memory.add_text("User timezone is UTC+8.", user_id="telegram_test")

    kinds = [str(call["metadata"].get("kind")) for call in fake.add_calls]
    assert kinds == ["credential_fact", "aspirations_fact", "life_fact"]


def test_memory_service_falls_back_to_raw_add_after_empty_inference_results() -> None:
    memory = MemoryService()
    fake = _EmptyInferenceFakeMem0()
    memory._memory = fake

    result = memory.add_text(
        "User prefers concise status updates.",
        user_id="telegram_test",
        retries=1,
    )

    assert result == {"results": [{"memory": "raw fallback", "event": "ADD"}]}
    assert [call["infer"] for call in fake.add_calls] == [True, True, False]
    assert fake.add_calls[-1]["metadata"] == {
        "kind": "preference_fact",
        "inference_fallback": "mem0_empty_result",
    }


def test_memory_service_preserves_explicit_user_scope_in_legacy_search() -> None:
    memory = MemoryService()
    fake = _LegacySearchFakeMem0()
    memory._memory = fake

    memory.search(
        "durable context",
        user_id="customer-1",
        limit=7,
        metadata={
            "user_id": "customer-2",
            "agent_id": "agent-2",
            "run_id": "run-2",
            "kind": ["directive_fact", "life_fact"],
        },
    )

    assert fake.filters_calls == [
        {
            "query": "durable context",
            "filters": {
                "user_id": "customer-1",
                "kind": ["directive_fact", "life_fact"],
            },
            "limit": 7,
        }
    ]
