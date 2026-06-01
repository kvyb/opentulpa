"""Model initialization and prompt-cache helpers for the runtime."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import time
from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from opentulpa.agent import model_error_trace
from opentulpa.agent import model_transport_policy as transport_policy
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage
from opentulpa.agent.model_call_guards import (
    next_stream_chunk_with_timeout,
    raise_if_empty_model_response,
)
from opentulpa.agent.model_init_policy import (
    chat_model_init_kwargs_for_model,
    deep_merge_dicts,
    disable_deepseek_v4_pro_thinking_extra,
)
from opentulpa.agent.model_provider_profile import (
    model_provider_profile,
    provider_prompt_cache_invoke_extras,
)
from opentulpa.agent.openrouter_chat_factory import (
    build_openrouter_chat_model,
    looks_like_openrouter_base_url,
    uses_openrouter_reasoning_adapter,
)
from opentulpa.agent.utils import content_to_text as _content_to_text

logger = logging.getLogger(__name__)


async def _run_with_transient_model_retries(
    runtime: Any,
    *,
    model_name: str,
    attempt_context: dict[str, Any],
    operation: Any,
) -> Any:
    transient_retries = transport_policy.model_transient_retry_limit()
    provider_retry_index = 0
    while True:
        try:
            return await operation()
        except Exception as exc:
            retry_error_text = model_error_trace.exception_trace_text(exc)
            retry_error_fields = model_error_trace.exception_trace_fields(exc)
            if (
                provider_retry_index >= transient_retries
                or not transport_policy.is_retryable_model_exception(exc)
            ):
                raise
            delay_seconds = transport_policy.model_transient_retry_delay_seconds(
                provider_retry_index
            )
            provider_retry_index += 1
            runtime.log_behavior_event(
                event="llm.invoke.transient_retry",
                model_name=model_name,
                call_site=str(attempt_context.get("call_site") or "runtime_model_invoke"),
                trace_id=str(attempt_context.get("trace_id") or ""),
                thread_id=str(attempt_context.get("thread_id") or ""),
                customer_id=str(attempt_context.get("customer_id") or ""),
                provider_attempt_name=str(
                    attempt_context.get("provider_attempt_name") or "default"
                ),
                retry_index=provider_retry_index,
                retry_limit=transient_retries,
                delay_seconds=delay_seconds,
                error=retry_error_text,
                **retry_error_fields,
            )
            await asyncio.sleep(delay_seconds)


def init_runtime_chat_model(
    model_name: str,
    *,
    base_kwargs: dict[str, Any],
    openrouter_base_url: str | None,
    reasoning_effort: str | None,
    init_chat_model_func: Any = init_chat_model,
    chat_openai_cls: Any = None,
) -> Any:
    openrouter_model = build_openrouter_chat_model(
        model_name=model_name,
        base_kwargs=base_kwargs,
        openrouter_base_url=openrouter_base_url,
        reasoning_effort=reasoning_effort,
        **({"chat_openai_cls": chat_openai_cls} if chat_openai_cls else {}),
    )
    if openrouter_model is not None:
        return openrouter_model

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


def _openrouter_session_id_for_call(
    runtime: Any,
    *,
    model_name: str,
    call_context: dict[str, Any],
) -> str:
    if not looks_like_openrouter_base_url(getattr(runtime, "openrouter_base_url", None)):
        return ""
    if not model_provider_profile(model_name).openrouter_session_sticky:
        return ""
    thread_id = str(call_context.get("thread_id") or "").strip()
    customer_id = str(call_context.get("customer_id") or "").strip()
    if not thread_id and not customer_id:
        return ""
    raw = f"{customer_id}:{thread_id}"
    return f"opentulpa-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"


def _with_openrouter_session_id(
    runtime: Any,
    invoke_extras: dict[str, Any],
    *,
    model_name: str,
    call_context: dict[str, Any],
) -> dict[str, Any]:
    session_id = _openrouter_session_id_for_call(
        runtime,
        model_name=model_name,
        call_context=call_context,
    )
    if not session_id:
        return invoke_extras
    call_context["openrouter_session_id"] = session_id
    return deep_merge_dicts(invoke_extras, {"extra_body": {"session_id": session_id}})


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


def prompt_cache_breakpoint_message_index(
    messages: list[Any],
    *,
    effective_prefix_count: int,
) -> int | None:
    if effective_prefix_count <= 0:
        return None
    target_roles = (SystemMessage, HumanMessage, AIMessage)
    for idx in range(min(effective_prefix_count, len(messages)) - 1, -1, -1):
        message = messages[idx]
        if not isinstance(message, target_roles):
            continue
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            continue
        if getattr(message, "content", None):
            return idx
    return None


def prepare_messages_for_prompt_cache(
    runtime: Any,
    messages: list[Any],
    *,
    model_name: str | None = None,
    stable_prefix_count: int = 0,
    cacheable_prefix_count: int | None = None,
) -> list[Any]:
    profile = runtime.prompt_cache_profile(model_name=model_name)
    strategy = str(profile.get("strategy") or "")
    if strategy not in {"breakpoint", "explicit_stable_prefix"}:
        return messages
    cache_control = dict(profile.get("cache_control") or {})
    if not cache_control:
        return messages
    if strategy == "explicit_stable_prefix":
        effective_prefix_count = (
            int(cacheable_prefix_count)
            if cacheable_prefix_count is not None and int(cacheable_prefix_count) > 0
            else int(stable_prefix_count)
            if int(stable_prefix_count) > 0
            else infer_stable_system_prefix_count(messages)
        )
    else:
        effective_prefix_count = (
            int(stable_prefix_count)
            if int(stable_prefix_count) > 0
            else infer_stable_system_prefix_count(messages)
        )
    if effective_prefix_count <= 0:
        return messages
    patched: list[Any] = list(messages)
    target_index = prompt_cache_breakpoint_message_index(
        patched,
        effective_prefix_count=effective_prefix_count,
    )
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
    cacheable_prefix_count: int | None = None,
    call_context: dict[str, Any] | None = None,
) -> Any:
    resolved_model_name = runtime._resolve_model_name_for_runtime_call(
        model, explicit_name=model_name
    )
    prepared_messages = runtime.prepare_messages_for_prompt_cache(
        list(messages),
        model_name=resolved_model_name,
        stable_prefix_count=stable_prefix_count,
        cacheable_prefix_count=cacheable_prefix_count,
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
        invoke_extras = _with_openrouter_session_id(
            runtime,
            invoke_extras,
            model_name=resolved_model_name,
            call_context=attempt_context,
        )
        callback_target = runtime._model_with_callbacks(model, call_context=attempt_context)
        response: Any | None = None
        error_text: str | None = None
        error_fields: dict[str, str] = {}
        try:
            async def invoke_once(
                callback_target: Any = callback_target,
                invoke_extras: dict[str, Any] = invoke_extras,
            ) -> Any:
                timeout_seconds = transport_policy.model_invoke_timeout_seconds()
                if supports_ainvoke_kwargs(callback_target, invoke_extras):
                    invoke_awaitable = callback_target.ainvoke(
                        prepared_messages,
                        **invoke_extras,
                    )
                else:
                    invoke_awaitable = callback_target.ainvoke(prepared_messages)
                result = await asyncio.wait_for(invoke_awaitable, timeout=timeout_seconds)
                raise_if_empty_model_response(
                    result,
                    model_name=resolved_model_name,
                    phase="ainvoke",
                )
                return result

            response = await _run_with_transient_model_retries(
                runtime,
                model_name=resolved_model_name,
                attempt_context=attempt_context,
                operation=invoke_once,
            )
            return response
        except Exception as exc:
            error_text, error_fields = model_error_trace.log_invoke_error(runtime, exc=exc, model_name=resolved_model_name, attempt_context=attempt_context, phase="ainvoke")
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
                call_context={
                    **attempt_context,
                    **error_fields,
                    "cacheable_prefix_count": cacheable_prefix_count,
                },
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
    cacheable_prefix_count: int | None = None,
    call_context: dict[str, Any] | None = None,
    stream_config: Any | None = None,
) -> Any:
    resolved_model_name = runtime._resolve_model_name_for_runtime_call(
        model, explicit_name=model_name
    )
    prepared_messages = runtime.prepare_messages_for_prompt_cache(
        list(messages),
        model_name=resolved_model_name,
        stable_prefix_count=stable_prefix_count,
        cacheable_prefix_count=cacheable_prefix_count,
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
        invoke_extras = _with_openrouter_session_id(
            runtime,
            invoke_extras,
            model_name=resolved_model_name,
            call_context=attempt_context,
        )
        callback_target = runtime._model_with_callbacks(model, call_context=attempt_context)
        astream = getattr(callback_target, "astream", None)
        if not callable(astream):
            return await ainvoke_model(
                runtime,
                model,
                messages,
                model_name=resolved_model_name,
                stable_prefix_count=stable_prefix_count,
                cacheable_prefix_count=cacheable_prefix_count,
                call_context=call_context,
            )
        response: Any | None = None
        error_text: str | None = None
        error_fields: dict[str, str] = {}
        try:
            async def stream_once(
                callback_target: Any = callback_target,
                invoke_extras: dict[str, Any] = invoke_extras,
                astream: Any = astream,
            ) -> Any:
                accumulated: Any | None = None
                stream_kwargs = dict(invoke_extras)
                if stream_config is not None:
                    stream_kwargs["config"] = stream_config
                if supports_astream_kwargs(callback_target, invoke_extras):
                    if supports_astream_kwargs(callback_target, stream_kwargs):
                        stream = astream(prepared_messages, **stream_kwargs)
                    else:
                        stream = astream(prepared_messages, **invoke_extras)
                elif stream_config is not None and supports_astream_kwargs(
                    callback_target, {"config": stream_config}
                ):
                    stream = astream(prepared_messages, config=stream_config)
                else:
                    stream = astream(prepared_messages)
                stream_iter = stream.__aiter__()
                timeout_seconds = model_provider_profile(
                    resolved_model_name
                ).stream_chunk_timeout_seconds()
                try:
                    while True:
                        try:
                            chunk = await next_stream_chunk_with_timeout(
                                stream_iter,
                                timeout_seconds=timeout_seconds,
                            )
                        except StopAsyncIteration:
                            break
                        accumulated = chunk if accumulated is None else accumulated + chunk
                finally:
                    aclose = getattr(stream_iter, "aclose", None)
                    if callable(aclose):
                        await aclose()
                if accumulated is None:
                    result = AIMessage(content="")
                else:
                    result = _ai_message_from_stream_chunk(accumulated)
                raise_if_empty_model_response(
                    result,
                    model_name=resolved_model_name,
                    phase="astream",
                )
                return result

            response = await _run_with_transient_model_retries(
                runtime,
                model_name=resolved_model_name,
                attempt_context=attempt_context,
                operation=stream_once,
            )
            return response
        except Exception as exc:
            error_text, error_fields = model_error_trace.log_invoke_error(runtime, exc=exc, model_name=resolved_model_name, attempt_context=attempt_context, phase="astream")
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
                call_context={
                    **attempt_context,
                    **error_fields,
                    "cacheable_prefix_count": cacheable_prefix_count,
                },
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
    cacheable_prefix_count: int | None = None,
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
        cacheable_prefix_count=cacheable_prefix_count,
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
        invoke_extras = _with_openrouter_session_id(
            runtime,
            invoke_extras,
            model_name=resolved_model_name,
            call_context=attempt_context,
        )
        callback_target = runtime._model_with_callbacks(model, call_context=attempt_context)
        structured = getattr(callback_target, "with_structured_output", None)
        skip_native_structured = model_error_trace.skip_native_structured_output(resolved_model_name)
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
            structured_output_supported=bool(callable(structured) and not skip_native_structured),
            native_structured_output_skipped=skip_native_structured,
        )
        if callable(structured) and not skip_native_structured:
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
                transient_retries = transport_policy.model_transient_retry_limit()
                provider_retry_index = 0
                while True:
                    try:
                        if supports_ainvoke_kwargs(runner, invoke_extras):
                            payload = await runner.ainvoke(prepared_messages, **invoke_extras)
                        else:
                            payload = await runner.ainvoke(prepared_messages)
                        break
                    except Exception as exc:
                        retry_error_text = model_error_trace.exception_trace_text(exc)
                        retry_error_fields = model_error_trace.exception_trace_fields(exc)
                        if (
                            provider_retry_index >= transient_retries
                            or not transport_policy.is_retryable_model_exception(exc)
                        ):
                            raise
                        delay_seconds = transport_policy.model_transient_retry_delay_seconds(
                            provider_retry_index
                        )
                        provider_retry_index += 1
                        runtime.log_behavior_event(
                            event="llm.invoke.transient_retry",
                            model_name=resolved_model_name,
                            call_site=str(
                                attempt_context.get("call_site") or "runtime_model_invoke"
                            ),
                            trace_id=str(attempt_context.get("trace_id") or ""),
                            thread_id=str(attempt_context.get("thread_id") or ""),
                            customer_id=str(attempt_context.get("customer_id") or ""),
                            provider_attempt_name=str(
                                attempt_context.get("provider_attempt_name") or "default"
                            ),
                            retry_index=provider_retry_index,
                            retry_limit=transient_retries,
                            delay_seconds=delay_seconds,
                            error=retry_error_text,
                            **retry_error_fields,
                        )
                        await asyncio.sleep(delay_seconds)
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
                        call_context={
                            **attempt_context,
                            "cacheable_prefix_count": cacheable_prefix_count,
                        },
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
                        call_context={
                            **attempt_context,
                            "cacheable_prefix_count": cacheable_prefix_count,
                        },
                    )
                    trace_recorded = True
                    return parsed, None
                error_text = (
                    f"TypeError: structured output returned unsupported type "
                    f"{type(payload).__name__}"
                )
            except Exception as exc:
                error_text = model_error_trace.exception_trace_text(exc)
                error_fields = model_error_trace.exception_trace_fields(exc)
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
                    **error_fields,
                )
            finally:
                if not trace_recorded and (payload is not None or error_text):
                    runtime._record_llm_call_trace(
                        model_name=resolved_model_name,
                        prepared_messages=prepared_messages,
                        stable_prefix_count=stable_prefix_count,
                        response=payload,
                        error=error_text,
                        call_context={
                            **attempt_context,
                            "cacheable_prefix_count": cacheable_prefix_count,
                        },
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
                last_error = model_error_trace.exception_trace_text(exc)
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
        last_error = model_error_trace.exception_trace_text(exc)
    return None, last_error
