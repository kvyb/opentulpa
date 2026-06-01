"""Transport retry and OpenRouter routing policy for runtime model calls."""

from __future__ import annotations

import os
from typing import Any

RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
RETRYABLE_MODEL_ERROR_MARKERS = (
    "connecttimeout",
    "connection reset",
    "empty model response",
    "incomplete chunked read",
    "peer closed connection",
    "rate limit",
    "ratelimit",
    "readtimeout",
    "remoteprotocolerror",
    "server disconnected",
    "timeout",
    "too many requests",
    "toomanyrequests",
    "temporarily unavailable",
    "provider returned error",
)


def _status_code(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _exception_status_codes(exc: Exception) -> tuple[int, ...]:
    codes: list[int] = []
    for value in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(getattr(exc, "raw_response", None), "status_code", None),
    ):
        if code := _status_code(value):
            codes.append(code)
    return tuple(codes)


def model_transient_retry_limit() -> int:
    raw = str(os.getenv("OPENTULPA_MODEL_TRANSIENT_RETRIES", "2") or "2").strip()
    try:
        return max(0, min(5, int(raw)))
    except ValueError:
        return 2


def is_retryable_model_error(error_text: str) -> bool:
    normalized = str(error_text or "").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in RETRYABLE_MODEL_ERROR_MARKERS)


def is_retryable_model_exception(exc: Exception) -> bool:
    if any(code in RETRYABLE_HTTP_STATUS_CODES for code in _exception_status_codes(exc)):
        return True
    return is_retryable_model_error(f"{type(exc).__name__}: {exc}")


def model_transient_retry_delay_seconds(retry_index: int) -> float:
    return float(min(6.0, 0.75 * (2 ** max(0, int(retry_index)))))


def model_invoke_timeout_seconds() -> float:
    raw = str(os.getenv("OPENTULPA_MODEL_INVOKE_TIMEOUT_SECONDS", "60") or "60").strip()
    try:
        return max(0.05, min(300.0, float(raw)))
    except ValueError:
        return 60.0


def openrouter_max_retries() -> int:
    raw = str(os.getenv("OPENTULPA_OPENROUTER_MAX_RETRIES", "4") or "4").strip()
    try:
        return max(0, min(10, int(raw)))
    except ValueError:
        return 4


def openrouter_timeout_seconds() -> int:
    raw = str(os.getenv("OPENTULPA_OPENROUTER_TIMEOUT_SECONDS", "180") or "180").strip()
    try:
        return max(10, min(600, int(float(raw))))
    except ValueError:
        return 180


def openrouter_provider_routing_for_model(model_name: str | None) -> dict[str, Any]:
    slug = str(model_name or "").strip().lower()
    if slug == "deepseek/deepseek-v4-pro":
        return {"order": ["DeepSeek"], "allow_fallbacks": False}
    return {}
