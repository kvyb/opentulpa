"""OpenRouter-specific chat model construction."""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from opentulpa.agent import model_transport_policy as transport_policy
from opentulpa.agent.deepseek_chat_model import OpenRouterDeepSeekChatModel

DEFAULT_OPENROUTER_APP_REFERER = "https://github.com/kvyb/opentulpa"
DEFAULT_OPENROUTER_APP_TITLE = "OpenTulpa"


def looks_like_openrouter_base_url(base_url: str | None) -> bool:
    normalized = str(base_url or "").strip().lower()
    return "openrouter.ai" in normalized


def uses_openrouter_reasoning_adapter(*, model_name: str | None, base_url: str | None) -> bool:
    slug = str(model_name or "").strip().lower()
    return "deepseek" in slug and looks_like_openrouter_base_url(base_url)


def _uses_openrouter_chat_adapter(*, model_name: str | None, base_url: str | None) -> bool:
    slug = str(model_name or "").strip().lower()
    return looks_like_openrouter_base_url(base_url) and (
        "deepseek" in slug or slug.startswith("qwen/") or "qwen" in slug
    )


def _openrouter_reasoning_config(reasoning_effort: str | None) -> dict[str, Any]:
    effort = str(reasoning_effort or "").strip() or "none"
    return {"effort": effort, "exclude": False}


def openrouter_app_headers(
    *,
    base_url: str | None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    if not looks_like_openrouter_base_url(base_url):
        return {}
    source = env if env is not None else os.environ
    title = str(source.get("OPENROUTER_APP_TITLE", "")).strip() or DEFAULT_OPENROUTER_APP_TITLE
    headers: dict[str, str] = {"HTTP-Referer": DEFAULT_OPENROUTER_APP_REFERER}
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def build_openrouter_chat_model(
    *,
    model_name: str,
    base_kwargs: dict[str, Any],
    openrouter_base_url: str | None,
    reasoning_effort: str | None,
    chat_openai_cls: Any = ChatOpenAI,
) -> Any | None:
    if not _uses_openrouter_chat_adapter(
        model_name=model_name,
        base_url=openrouter_base_url,
    ):
        return None

    app_headers = openrouter_app_headers(base_url=openrouter_base_url)
    if uses_openrouter_reasoning_adapter(model_name=model_name, base_url=openrouter_base_url):
        return OpenRouterDeepSeekChatModel(
            model=model_name,
            api_key=SecretStr(str(base_kwargs.get("api_key") or "")),
            api_base=str(openrouter_base_url or base_kwargs.get("base_url") or ""),
            temperature=base_kwargs.get("temperature"),
            max_tokens=base_kwargs.get("max_completion_tokens"),
            timeout=transport_policy.openrouter_timeout_seconds(),
            extra_body={
                "provider": dict(transport_policy.openrouter_provider_routing_for_model(model_name)),
                "reasoning": _openrouter_reasoning_config(reasoning_effort),
            },
            default_headers=app_headers,
            streaming=bool(base_kwargs.get("streaming", True)),
            max_retries=transport_policy.openrouter_max_retries(),
            use_responses_api=False,
        )

    adapter_kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": base_kwargs.get("api_key"),
        "base_url": openrouter_base_url or base_kwargs.get("base_url"),
        "temperature": base_kwargs.get("temperature"),
        "max_completion_tokens": base_kwargs.get("max_completion_tokens"),
        "streaming": bool(base_kwargs.get("streaming", True)),
        "max_retries": transport_policy.openrouter_max_retries(),
        "timeout": transport_policy.openrouter_timeout_seconds(),
    }
    if "qwen" in str(model_name or "").strip().lower():
        adapter_kwargs["stream_usage"] = True
    if provider_routing := transport_policy.openrouter_provider_routing_for_model(model_name):
        adapter_kwargs["extra_body"] = {"provider": provider_routing}
    if app_headers:
        adapter_kwargs["default_headers"] = app_headers
    return chat_openai_cls(
        **{key: value for key, value in adapter_kwargs.items() if value is not None}
    )
