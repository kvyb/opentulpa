from __future__ import annotations

import pytest

from opentulpa.agent.tools_registry import _normalize_allowed_domains, register_runtime_tools


class _DummyRuntime:
    def __init__(self) -> None:
        self._active_customer_id = "u_1"

    async def _request_with_backoff(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise RuntimeError("unexpected internal API call")


def test_normalize_allowed_domains_filters_invalid_values() -> None:
    values = _normalize_allowed_domains(
        [
            "https://example.com/path",
            "docs.python.org",
            "localhost",
            "bad domain",
            "https://example.com/other",
            "",
        ]
    )
    assert values == ["example.com", "docs.python.org"]


@pytest.mark.asyncio
async def test_browser_use_run_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    tools = register_runtime_tools(_DummyRuntime())

    result = await tools["browser_use_run"].ainvoke({"task": "open docs"})
    assert "error" in result
    assert "BROWSER_USE_API_KEY missing" in str(result["error"])


@pytest.mark.asyncio
async def test_browser_use_task_control_validates_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_USE_API_KEY", "dummy")
    tools = register_runtime_tools(_DummyRuntime())

    result = await tools["browser_use_task_control"].ainvoke(
        {"task_id": "task_123", "action": "explode"}
    )
    assert "error" in result
    assert "invalid action" in str(result["error"])
