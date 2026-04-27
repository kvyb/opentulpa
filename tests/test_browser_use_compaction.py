from __future__ import annotations

from opentulpa.agent.tools.browser_tools import _compact_browser_use_task_view


def test_compact_browser_use_task_view_default_hides_steps_and_truncates_output() -> None:
    payload = {
        "id": "task_uuid",
        "sessionId": "session_uuid",
        "status": "finished",
        "isSuccess": True,
        "task": "scrape something",
        "llm": "browser-use-llm",
        "output": "x" * 13000,
        "outputFiles": [{"id": "f1", "fileName": "report.csv"}],
        "steps": [
            {
                "number": 1,
                "url": "https://example.com",
                "nextGoal": "click next",
                "actions": ["click(button)"],
                "screenshotUrl": "https://img/1.png",
            }
        ],
    }

    compact = _compact_browser_use_task_view(payload)
    assert compact["steps_count"] == 1
    assert "steps_preview" not in compact
    assert compact["output_truncated"] is True
    assert str(compact["output"]).endswith("...")


def test_compact_browser_use_task_view_with_steps_preview() -> None:
    payload = {
        "steps": [
            {"number": 1, "url": "https://a.com", "nextGoal": "A", "actions": ["a1", "a2"]},
            {"number": 2, "url": "https://b.com", "nextGoal": "B", "actions": ["b1"]},
            {"number": 3, "url": "https://c.com", "nextGoal": "C", "actions": ["c1"]},
        ]
    }
    compact = _compact_browser_use_task_view(payload, include_steps=True, max_steps_preview=2)
    assert compact["steps_count"] == 3
    assert len(compact["steps_preview"]) == 2
    assert compact["steps_preview_truncated"] is True
