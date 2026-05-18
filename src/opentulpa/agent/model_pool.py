"""Model initialization and prompt-cache helpers for the runtime."""

from __future__ import annotations

import inspect
import logging
import os
import time
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel

from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage
from opentulpa.agent.utils import content_to_text as _content_to_text

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_APP_REFERER = "https://github.com/kvyb/opentulpa"
DEFAULT_OPENROUTER_APP_TITLE = "OpenTulpa"


def prompt_cache_control_payload(*, ttl_1h: bool) -> dict[str, Any]:
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl_1h:
        cache_control["ttl"] = "1h"
    return cache_control


def provider_prompt_cache_profile(
    *,
    enabled: bool,
    model_name: str,
    ttl_1h: bool,
) -> dict[str, Any]:
    slug = (model_name or "").strip().lower()
    if not enabled:
        return {
            "enabled": False,
            "strategy": "disabled",
            "supports_top_level": False,
            "supports_breakpoints": False,
            "cache_control": {},
            "model_name": model_name,
        }
    if "anthropic/" in slug or "claude" in slug:
        return {
            "enabled": True,
            "strategy": "top_level",
            "supports_top_level": True,
            "supports_breakpoints": True,
            "cache_control": prompt_cache_control_payload(ttl_1h=ttl_1h),
            "model_name": model_name,
        }
    if "gemini" in slug or slug.startswith("google/"):
        return {
            "enabled": True,
            "strategy": "breakpoint",
            "supports_top_level": False,
            "supports_breakpoints": True,
            "cache_control": prompt_cache_control_payload(ttl_1h=ttl_1h),
            "model_name": model_name,
        }
    if any(
        marker in slug
        for marker in (
            "openai/",
            "gpt-",
            "o1",
            "o3",
            "o4",
            "deepseek",
            "grok",
            "x-ai/",
            "z-ai/",
            "zai/",
            "glm",
            "moonshot",
            "kimi",
            "groq/",
        )
    ):
        return {
            "enabled": True,
            "strategy": "automatic",
            "supports_top_level": False,
            "supports_breakpoints": False,
            "cache_control": {},
            "model_name": model_name,
        }
    return {
        "enabled": True,
        "strategy": "unknown",
        "supports_top_level": False,
        "supports_breakpoints": False,
        "cache_control": {},
        "model_name": model_name,
    }


def provider_prompt_cache_invoke_extras(
    *,
    enabled: bool,
    model_name: str,
    ttl_1h: bool,
) -> dict[str, Any]:
    profile = provider_prompt_cache_profile(
        enabled=enabled,
        model_name=model_name,
        ttl_1h=ttl_1h,
    )
    if profile.get("strategy") != "top_level":
        return {}
    return {"extra_body": {"cache_control": dict(profile.get("cache_control") or {})}}


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


def looks_like_openrouter_base_url(base_url: str | None) -> bool:
    normalized = str(base_url or "").strip().lower()
    return "openrouter.ai" in normalized


def uses_openrouter_reasoning_adapter(*, model_name: str | None, base_url: str | None) -> bool:
    return "deepseek" in str(model_name or "").strip().lower() and looks_like_openrouter_base_url(
        base_url
    )


def openrouter_reasoning_config(reasoning_effort: str | None) -> dict[str, Any]:
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


def chat_model_init_kwargs_for_model(
    base_kwargs: dict[str, Any],
    *,
    model_name: str,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    model_kwargs = cap_max_completion_tokens_for_model(dict(base_kwargs), model_name=model_name)
    extra = disable_deepseek_v4_pro_thinking_extra(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )
    if extra:
        model_kwargs = deep_merge_dicts(model_kwargs, extra)
    return model_kwargs


def init_runtime_chat_model(
    model_name: str,
    *,
    base_kwargs: dict[str, Any],
    openrouter_base_url: str | None,
    reasoning_effort: str | None,
    init_chat_model_func: Any = init_chat_model,
    chat_openrouter_cls: Any = ChatOpenRouter,
) -> Any:
    if uses_openrouter_reasoning_adapter(model_name=model_name, base_url=openrouter_base_url):
        app_headers = openrouter_app_headers(base_url=openrouter_base_url)
        adapter_kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": base_kwargs.get("api_key"),
            "base_url": openrouter_base_url or base_kwargs.get("base_url"),
            "temperature": base_kwargs.get("temperature"),
            "max_completion_tokens": base_kwargs.get("max_completion_tokens"),
            "reasoning": openrouter_reasoning_config(reasoning_effort),
        }
        if referer := app_headers.get("HTTP-Referer"):
            adapter_kwargs["app_url"] = referer
        if title := app_headers.get("X-OpenRouter-Title"):
            adapter_kwargs["app_title"] = title
        return chat_openrouter_cls(
            **{key: value for key, value in adapter_kwargs.items() if value is not None}
        )

    return init_chat_model_func(
        model_name,
        **chat_model_init_kwargs_for_model(
            base_kwargs,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        ),
    )


def model_invoke_extras(runtime: Any, *, model_name: str | None = None) -> dict[str, Any]:
    target_model_name = str(model_name or getattr(runtime, "model_name", "") or "").strip()
    invoke_extras = dict(
        provider_prompt_cache_invoke_extras(
            enabled=bool(getattr(runtime, "_prompt_caching_enabled", False)),
            model_name=target_model_name,
            ttl_1h=bool(getattr(runtime, "_prompt_cache_ttl_1h", False)),
        )
    )
    if uses_openrouter_reasoning_adapter(
        model_name=target_model_name,
        base_url=getattr(runtime, "openrouter_base_url", None),
    ):
        return invoke_extras
    return deep_merge_dicts(
        invoke_extras,
        disable_deepseek_v4_pro_thinking_extra(
            model_name=target_model_name,
            reasoning_effort=getattr(runtime, "_reasoning_effort", None),
        ),
    )


def message_content_with_cache_breakpoint(
    content: Any,
    *,
    cache_control: dict[str, Any],
) -> Any:
    if isinstance(content, str):
        text = str(content)
        if not text.strip():
            return content
        return [{"type": "text", "text": text, "cache_control": dict(cache_control)}]
    if not isinstance(content, list):
        return content
    updated = list(content)
    for idx in range(len(updated) - 1, -1, -1):
        item = updated[idx]
        if isinstance(item, str):
            text = str(item)
            if not text.strip():
                continue
            updated[idx] = {"type": "text", "text": text, "cache_control": dict(cache_control)}
            return updated
        if isinstance(item, dict):
            item_type = str(item.get("type", "")).strip().lower()
            if item_type != "text" or "cache_control" in item:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            patched = dict(item)
            patched["cache_control"] = dict(cache_control)
            updated[idx] = patched
            return updated
    return content


def message_with_cache_breakpoint(message: Any, *, cache_control: dict[str, Any]) -> Any:
    content = message_content_with_cache_breakpoint(
        getattr(message, "content", None),
        cache_control=cache_control,
    )
    if content == getattr(message, "content", None):
        return message
    model_copy = getattr(message, "model_copy", None)
    copied = model_copy(deep=True) if callable(model_copy) else message.copy(deep=True)
    copied.content = content
    return copied


def infer_stable_system_prefix_count(messages: list[Any]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, SystemMessage):
            break
        if not _content_to_text(getattr(message, "content", "")).strip():
            break
        count += 1
    return count


def prepare_messages_for_prompt_cache(
    runtime: Any,
    messages: list[Any],
    *,
    model_name: str | None = None,
    stable_prefix_count: int = 0,
) -> list[Any]:
    profile = runtime.prompt_cache_profile(model_name=model_name)
    if profile.get("strategy") != "breakpoint":
        return messages
    cache_control = dict(profile.get("cache_control") or {})
    if not cache_control:
        return messages
    effective_stable_prefix_count = (
        int(stable_prefix_count)
        if int(stable_prefix_count) > 0
        else infer_stable_system_prefix_count(messages)
    )
    if effective_stable_prefix_count <= 0:
        return messages
    patched: list[Any] = list(messages)
    target_index: int | None = None
    for idx in range(min(effective_stable_prefix_count, len(patched)) - 1, -1, -1):
        if getattr(patched[idx], "content", None):
            target_index = idx
            break
    if target_index is None:
        return messages
    patched[target_index] = message_with_cache_breakpoint(
        patched[target_index],
        cache_control=cache_control,
    )
    return patched


def supports_ainvoke_kwargs(target: Any, kwargs: dict[str, Any]) -> bool:
    if not kwargs:
        return False
    ainvoke = getattr(target, "ainvoke", None)
    if not callable(ainvoke):
        return False
    try:
        sig = inspect.signature(ainvoke)
    except (TypeError, ValueError):
        return False
    params = sig.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return True
    return all(key in sig.parameters for key in kwargs)


def supports_astream_kwargs(target: Any, kwargs: dict[str, Any]) -> bool:
    if not kwargs:
        return False
    astream = getattr(target, "astream", None)
    if not callable(astream):
        return False
    try:
        sig = inspect.signature(astream)
    except (TypeError, ValueError):
        return False
    params = sig.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return True
    return all(key in sig.parameters for key in kwargs)


def _ai_message_from_stream_chunk(chunk: Any) -> AIMessage:
    if isinstance(chunk, AIMessage):
        return chunk
    content = getattr(chunk, "content", "")
    return AIMessage(
        content=content,
        additional_kwargs=dict(getattr(chunk, "additional_kwargs", {}) or {}),
        response_metadata=dict(getattr(chunk, "response_metadata", {}) or {}),
        id=getattr(chunk, "id", None),
        tool_calls=list(getattr(chunk, "tool_calls", []) or []),
        invalid_tool_calls=list(getattr(chunk, "invalid_tool_calls", []) or []),
        usage_metadata=getattr(chunk, "usage_metadata", None),
    )


async def ainvoke_model(
    runtime: Any,
    model: Any,
    messages: list[Any],
    *,
    model_name: str | None = None,
    stable_prefix_count: int = 0,
    call_context: dict[str, Any] | None = None,
) -> Any:
    resolved_model_name = runtime._resolve_model_name_for_runtime_call(
        model, explicit_name=model_name
    )
    prepared_messages = runtime.prepare_messages_for_prompt_cache(
        list(messages),
        model_name=resolved_model_name,
        stable_prefix_count=stable_prefix_count,
    )
    base_invoke_extras = runtime.model_invoke_extras(model_name=resolved_model_name)
    attempts = runtime._model_request_attempts(model_name=resolved_model_name)
    last_exc: Exception | None = None
    for attempt_index, attempt in enumerate(attempts):
        invoke_extras = deep_merge_dicts(
            dict(base_invoke_extras),
            dict(attempt.get("invoke_extras") or {}),
        )
        attempt_context = dict(call_context or {})
        attempt_context.update(dict(attempt.get("call_context") or {}))
        attempt_context["provider_attempt_name"] = (
            str(attempt.get("name") or "").strip() or "default"
        )
        attempt_context["provider_attempt_index"] = attempt_index + 1
        attempt_context["provider_attempt_count"] = len(attempts)
        callback_target = runtime._model_with_callbacks(model, call_context=attempt_context)
        response: Any | None = None
        error_text: str | None = None
        try:
            if supports_ainvoke_kwargs(callback_target, invoke_extras):
                response = await callback_target.ainvoke(prepared_messages, **invoke_extras)
            else:
                response = await callback_target.ainvoke(prepared_messages)
            return response
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            last_exc = exc
            if attempt_index + 1 >= len(attempts):
                raise
            logger.warning(
                "Model invocation via %s failed for %s; retrying with next provider route: %s",
                attempt_context["provider_attempt_name"],
                resolved_model_name,
                error_text,
            )
            runtime.log_behavior_event(
                event="llm.provider_fallback",
                model_name=resolved_model_name,
                failed_provider_attempt=attempt_context["provider_attempt_name"],
                next_provider_attempt=str(attempts[attempt_index + 1].get("name") or "").strip()
                or "default",
                error=error_text,
            )
        finally:
            runtime._record_llm_call_trace(
                model_name=resolved_model_name,
                prepared_messages=prepared_messages,
                stable_prefix_count=stable_prefix_count,
                response=response,
                error=error_text,
                call_context=attempt_context,
            )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Model invocation failed without attempts.")


async def astream_model(
    runtime: Any,
    model: Any,
    messages: list[Any],
    *,
    model_name: str | None = None,
    stable_prefix_count: int = 0,
    call_context: dict[str, Any] | None = None,
) -> Any:
    resolved_model_name = runtime._resolve_model_name_for_runtime_call(
        model, explicit_name=model_name
    )
    prepared_messages = runtime.prepare_messages_for_prompt_cache(
        list(messages),
        model_name=resolved_model_name,
        stable_prefix_count=stable_prefix_count,
    )
    base_invoke_extras = runtime.model_invoke_extras(model_name=resolved_model_name)
    attempts = runtime._model_request_attempts(model_name=resolved_model_name)
    last_exc: Exception | None = None
    for attempt_index, attempt in enumerate(attempts):
        invoke_extras = deep_merge_dicts(
            dict(base_invoke_extras),
            dict(attempt.get("invoke_extras") or {}),
        )
        attempt_context = dict(call_context or {})
        attempt_context.update(dict(attempt.get("call_context") or {}))
        attempt_context["provider_attempt_name"] = (
            str(attempt.get("name") or "").strip() or "default"
        )
        attempt_context["provider_attempt_index"] = attempt_index + 1
        attempt_context["provider_attempt_count"] = len(attempts)
        callback_target = runtime._model_with_callbacks(model, call_context=attempt_context)
        astream = getattr(callback_target, "astream", None)
        if not callable(astream):
            return await ainvoke_model(
                runtime,
                model,
                messages,
                model_name=resolved_model_name,
                stable_prefix_count=stable_prefix_count,
                call_context=call_context,
            )
        response: Any | None = None
        error_text: str | None = None
        try:
            accumulated: Any | None = None
            if supports_astream_kwargs(callback_target, invoke_extras):
                stream = astream(prepared_messages, **invoke_extras)
            else:
                stream = astream(prepared_messages)
            async for chunk in stream:
                accumulated = chunk if accumulated is None else accumulated + chunk
            if accumulated is None:
                response = AIMessage(content="")
            else:
                response = _ai_message_from_stream_chunk(accumulated)
            return response
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            last_exc = exc
            if attempt_index + 1 >= len(attempts):
                raise
            logger.warning(
                "Streaming model invocation via %s failed for %s; retrying with next provider route: %s",
                attempt_context["provider_attempt_name"],
                resolved_model_name,
                error_text,
            )
            runtime.log_behavior_event(
                event="llm.provider_fallback",
                model_name=resolved_model_name,
                failed_provider_attempt=attempt_context["provider_attempt_name"],
                next_provider_attempt=str(attempts[attempt_index + 1].get("name") or "").strip()
                or "default",
                error=error_text,
            )
        finally:
            runtime._record_llm_call_trace(
                model_name=resolved_model_name,
                prepared_messages=prepared_messages,
                stable_prefix_count=stable_prefix_count,
                response=response,
                error=error_text,
                call_context=attempt_context,
            )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Streaming model invocation failed without attempts.")


async def invoke_structured_model[StructuredModelT: BaseModel](
    runtime: Any,
    *,
    model: Any,
    messages: list[Any],
    schema: type[StructuredModelT],
    model_name: str | None = None,
    stable_prefix_count: int = 0,
    call_context: dict[str, Any] | None = None,
    clean_json_text_block: Any,
) -> tuple[StructuredModelT | None, str | None]:
    last_error: str | None = None
    resolved_model_name = runtime._resolve_model_name_for_runtime_call(
        model, explicit_name=model_name
    )
    prepared_messages = runtime.prepare_messages_for_prompt_cache(
        list(messages),
        model_name=resolved_model_name,
        stable_prefix_count=stable_prefix_count,
    )
    base_invoke_extras = runtime.model_invoke_extras(model_name=resolved_model_name)
    attempts = runtime._model_request_attempts(model_name=resolved_model_name)
    for attempt_index, attempt in enumerate(attempts):
        invoke_extras = deep_merge_dicts(
            dict(base_invoke_extras),
            dict(attempt.get("invoke_extras") or {}),
        )
        attempt_context = dict(call_context or {})
        attempt_context.update(dict(attempt.get("call_context") or {}))
        attempt_context["provider_attempt_name"] = (
            str(attempt.get("name") or "").strip() or "default"
        )
        attempt_context["provider_attempt_index"] = attempt_index + 1
        attempt_context["provider_attempt_count"] = len(attempts)
        callback_target = runtime._model_with_callbacks(model, call_context=attempt_context)
        structured = getattr(callback_target, "with_structured_output", None)
        payload: Any | None = None
        error_text: str | None = None
        trace_recorded = False
        invoke_started = time.monotonic()
        runtime.log_behavior_event(
            event="llm.invoke.start",
            model_name=resolved_model_name,
            call_site=str(attempt_context.get("call_site") or "runtime_model_invoke"),
            trace_id=str(attempt_context.get("trace_id") or ""),
            thread_id=str(attempt_context.get("thread_id") or ""),
            customer_id=str(attempt_context.get("customer_id") or ""),
            turn_mode=str(attempt_context.get("turn_mode") or ""),
            prompt_mode=str(attempt_context.get("prompt_mode") or ""),
            provider_attempt_name=str(attempt_context.get("provider_attempt_name") or "default"),
            provider_attempt_index=int(attempt_context.get("provider_attempt_index") or 1),
            provider_attempt_count=int(attempt_context.get("provider_attempt_count") or 1),
            prompt_message_count=len(prepared_messages),
            stable_prefix_count=int(stable_prefix_count),
            structured_output_supported=bool(callable(structured)),
        )
        if callable(structured):
            phase = "structured_output"
            try:
                structured_started = time.monotonic()
                runner = structured(schema)
                runtime.log_behavior_event(
                    event="llm.invoke.runner_ready",
                    model_name=resolved_model_name,
                    call_site=str(attempt_context.get("call_site") or "runtime_model_invoke"),
                    trace_id=str(attempt_context.get("trace_id") or ""),
                    thread_id=str(attempt_context.get("thread_id") or ""),
                    customer_id=str(attempt_context.get("customer_id") or ""),
                    provider_attempt_name=str(
                        attempt_context.get("provider_attempt_name") or "default"
                    ),
                    elapsed_ms=int((time.monotonic() - structured_started) * 1000),
                )
                phase = "provider_await"
                provider_started = time.monotonic()
                runtime.log_behavior_event(
                    event="llm.invoke.await_provider",
                    model_name=resolved_model_name,
                    call_site=str(attempt_context.get("call_site") or "runtime_model_invoke"),
                    trace_id=str(attempt_context.get("trace_id") or ""),
                    thread_id=str(attempt_context.get("thread_id") or ""),
                    customer_id=str(attempt_context.get("customer_id") or ""),
                    provider_attempt_name=str(
                        attempt_context.get("provider_attempt_name") or "default"
                    ),
                )
                if supports_ainvoke_kwargs(runner, invoke_extras):
                    payload = await runner.ainvoke(prepared_messages, **invoke_extras)
                else:
                    payload = await runner.ainvoke(prepared_messages)
                provider_elapsed_ms = int((time.monotonic() - provider_started) * 1000)
                if isinstance(payload, schema):
                    runtime.log_behavior_event(
                        event="llm.invoke.finish",
                        model_name=resolved_model_name,
                        call_site=str(attempt_context.get("call_site") or "runtime_model_invoke"),
                        trace_id=str(attempt_context.get("trace_id") or ""),
                        thread_id=str(attempt_context.get("thread_id") or ""),
                        customer_id=str(attempt_context.get("customer_id") or ""),
                        provider_attempt_name=str(
                            attempt_context.get("provider_attempt_name") or "default"
                        ),
                        provider_elapsed_ms=provider_elapsed_ms,
                        elapsed_ms=int((time.monotonic() - invoke_started) * 1000),
                        result_type=type(payload).__name__,
                    )
                    runtime._record_llm_call_trace(
                        model_name=resolved_model_name,
                        prepared_messages=prepared_messages,
                        stable_prefix_count=stable_prefix_count,
                        response=payload,
                        error=None,
                        call_context=attempt_context,
                    )
                    trace_recorded = True
                    return payload, None
                if isinstance(payload, dict):
                    parsed = schema.model_validate(payload)
                    runtime.log_behavior_event(
                        event="llm.invoke.finish",
                        model_name=resolved_model_name,
                        call_site=str(attempt_context.get("call_site") or "runtime_model_invoke"),
                        trace_id=str(attempt_context.get("trace_id") or ""),
                        thread_id=str(attempt_context.get("thread_id") or ""),
                        customer_id=str(attempt_context.get("customer_id") or ""),
                        provider_attempt_name=str(
                            attempt_context.get("provider_attempt_name") or "default"
                        ),
                        provider_elapsed_ms=provider_elapsed_ms,
                        elapsed_ms=int((time.monotonic() - invoke_started) * 1000),
                        result_type=type(payload).__name__,
                    )
                    runtime._record_llm_call_trace(
                        model_name=resolved_model_name,
                        prepared_messages=prepared_messages,
                        stable_prefix_count=stable_prefix_count,
                        response=parsed,
                        error=None,
                        call_context=attempt_context,
                    )
                    trace_recorded = True
                    return parsed, None
                error_text = (
                    f"TypeError: structured output returned unsupported type "
                    f"{type(payload).__name__}"
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                runtime.log_behavior_event(
                    event="llm.invoke.error",
                    model_name=resolved_model_name,
                    call_site=str(attempt_context.get("call_site") or "runtime_model_invoke"),
                    trace_id=str(attempt_context.get("trace_id") or ""),
                    thread_id=str(attempt_context.get("thread_id") or ""),
                    customer_id=str(attempt_context.get("customer_id") or ""),
                    provider_attempt_name=str(
                        attempt_context.get("provider_attempt_name") or "default"
                    ),
                    phase=phase,
                    elapsed_ms=int((time.monotonic() - invoke_started) * 1000),
                    error=error_text,
                )
            finally:
                if not trace_recorded and (payload is not None or error_text):
                    runtime._record_llm_call_trace(
                        model_name=resolved_model_name,
                        prepared_messages=prepared_messages,
                        stable_prefix_count=stable_prefix_count,
                        response=payload,
                        error=error_text,
                        call_context=attempt_context,
                    )
        if error_text:
            last_error = error_text
            if attempt_index + 1 >= len(attempts):
                break
            logger.warning(
                "Structured model invocation via %s failed for %s; retrying with next provider route: %s",
                attempt_context["provider_attempt_name"],
                resolved_model_name,
                error_text,
            )
            runtime.log_behavior_event(
                event="llm.provider_fallback",
                model_name=resolved_model_name,
                failed_provider_attempt=attempt_context["provider_attempt_name"],
                next_provider_attempt=str(attempts[attempt_index + 1].get("name") or "").strip()
                or "default",
                error=error_text,
            )
            continue
    try:
        response = await runtime.ainvoke_model(
            model,
            list(messages),
            model_name=resolved_model_name,
            stable_prefix_count=stable_prefix_count,
            call_context={
                **dict(call_context or {}),
                "call_site": str(
                    (call_context or {}).get("call_site") or "structured_model_fallback"
                ),
            },
        )
        raw = _content_to_text(getattr(response, "content", response)).strip()
        if raw:
            try:
                return schema.model_validate_json(clean_json_text_block(raw)), None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                repair_messages = list(messages) + [
                    SystemMessage(
                        content=(
                            "Your previous structured output could not be parsed against the required schema. "
                            "Return only one valid JSON object for that schema. Do not include markdown, prose, "
                            "tool calls, or explanatory text."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Schema name: {schema.__name__}\n"
                            f"Parse error: {last_error}\n"
                            f"Previous output:\n{raw[:4000]}"
                        )
                    ),
                ]
                runtime.log_behavior_event(
                    event="llm.invoke.structured_repair_retry",
                    model_name=resolved_model_name,
                    call_site=str((call_context or {}).get("call_site") or "structured_model_fallback"),
                    trace_id=str((call_context or {}).get("trace_id") or ""),
                    thread_id=str((call_context or {}).get("thread_id") or ""),
                    customer_id=str((call_context or {}).get("customer_id") or ""),
                    error=last_error,
                )
                repair_response = await runtime.ainvoke_model(
                    model,
                    repair_messages,
                    model_name=resolved_model_name,
                    stable_prefix_count=stable_prefix_count,
                    call_context={
                        **dict(call_context or {}),
                        "call_site": str(
                            (call_context or {}).get("call_site")
                            or "structured_model_fallback"
                        )
                        + "_repair",
                    },
                )
                repair_raw = _content_to_text(
                    getattr(repair_response, "content", repair_response)
                ).strip()
                if repair_raw:
                    return schema.model_validate_json(clean_json_text_block(repair_raw)), None
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    return None, last_error
