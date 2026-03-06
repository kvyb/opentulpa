from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


class _Schema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = False
    reason: str = ""


class _StructuredRunner:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    async def ainvoke(self, _messages: object) -> object:
        return self._payload


class _StructuredModel:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def with_structured_output(self, _schema: type[BaseModel]) -> _StructuredRunner:
        return _StructuredRunner(self._payload)


class _FallbackResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FallbackModel:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _messages: object) -> _FallbackResponse:
        return _FallbackResponse(self._content)


class _BrokenStructuredThenFallbackModel(_FallbackModel):
    def with_structured_output(self, _schema: type[BaseModel]) -> _StructuredRunner:
        raise RuntimeError("structured_unavailable")


@pytest.mark.asyncio
async def test_invoke_structured_model_prefers_native_structured_output() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _StructuredModel(_Schema(ok=True, reason="native"))

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "native"
    assert error is None


@pytest.mark.asyncio
async def test_invoke_structured_model_uses_strict_json_fallback() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _BrokenStructuredThenFallbackModel('{"ok": true, "reason": "fallback"}')

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "fallback"
    assert error is None


@pytest.mark.asyncio
async def test_invoke_structured_model_rejects_wrapped_non_json_text() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _BrokenStructuredThenFallbackModel('prefix {"ok": true, "reason": "x"} suffix')

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert parsed is None
    assert isinstance(error, str)
    assert "ValidationError" in error
