"""Shared guardrails for model invocation and streaming calls."""

from __future__ import annotations

import asyncio
from typing import Any

from opentulpa.agent.utils import content_to_text


class EmptyModelResponseError(RuntimeError):
    """Raised when a provider returns no assistant content and no tool calls."""


def _tool_calls_for_response(response: Any) -> list[Any]:
    direct_tool_calls = getattr(response, "tool_calls", None) or []
    if direct_tool_calls:
        return list(direct_tool_calls)
    additional_kwargs = getattr(response, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        raw_tool_calls = additional_kwargs.get("tool_calls") or []
        if raw_tool_calls:
            return list(raw_tool_calls)
    return []


def raise_if_empty_model_response(response: Any, *, model_name: str, phase: str) -> None:
    if content_to_text(getattr(response, "content", "")).strip():
        return
    if _tool_calls_for_response(response):
        return
    raise EmptyModelResponseError(
        f"empty model response from {model_name} during {phase}: no content or tool calls"
    )


async def next_stream_chunk_with_timeout(stream_iter: Any, *, timeout_seconds: float) -> Any:
    try:
        return await asyncio.wait_for(anext(stream_iter), timeout=timeout_seconds)
    except StopAsyncIteration:
        raise
    except TimeoutError as exc:
        raise TimeoutError(
            f"model stream chunk timeout after {timeout_seconds:.2f}s"
        ) from exc
