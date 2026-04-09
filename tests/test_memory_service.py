from __future__ import annotations

import logging

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
        self.add_calls: list[dict[str, object]] = []

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

    memory.add_text("Telegram bot API key is stored for the sandbox service.", user_id="telegram_test")
    memory.add_text("User wants to launch a paid community this year.", user_id="telegram_test")
    memory.add_text("User timezone is UTC+8.", user_id="telegram_test")

    kinds = [str(call["metadata"].get("kind")) for call in fake.add_calls]
    assert kinds == ["credential_fact", "aspirations_fact", "life_fact"]
