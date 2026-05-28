"""Provider-neutral chat model initialization policies."""

from __future__ import annotations

from typing import Any


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(existing, value)
            continue
        merged[key] = value
    return merged


def disable_deepseek_v4_pro_thinking_extra(
    *, model_name: str, reasoning_effort: str | None
) -> dict[str, Any]:
    if reasoning_effort:
        return {}
    slug = str(model_name or "").strip().lower()
    if slug != "deepseek/deepseek-v4-pro":
        return {}
    return {
        "extra_body": {
            "reasoning": {"effort": "none"},
            "thinking": {"type": "disabled"},
        },
    }


def cap_max_completion_tokens_for_model(
    model_kwargs: dict[str, Any], *, model_name: str
) -> dict[str, Any]:
    if str(model_name or "").strip().casefold() != "google/gemini-3.1-flash-lite-preview":
        return model_kwargs
    capped = dict(model_kwargs)
    try:
        current = int(capped.get("max_completion_tokens", 1000) or 1000)
    except (TypeError, ValueError):
        current = 1000
    capped["max_completion_tokens"] = min(max(1, current), 1000)
    return capped


def chat_model_init_kwargs_for_model(
    base_kwargs: dict[str, Any],
    *,
    model_name: str,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    model_kwargs = cap_max_completion_tokens_for_model(dict(base_kwargs), model_name=model_name)
    model_kwargs.setdefault("streaming", True)
    extra = disable_deepseek_v4_pro_thinking_extra(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )
    if extra:
        model_kwargs = deep_merge_dicts(model_kwargs, extra)
    return model_kwargs
