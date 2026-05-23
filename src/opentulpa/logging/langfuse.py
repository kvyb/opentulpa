"""Optional Langfuse observability wiring for OpenTulpa."""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)

_SECRET_KEY_RE = re.compile(
    r"(authorization|api[_-]?key|secret|token|password|passwd|cookie|set-cookie|bearer)",
    re.IGNORECASE,
)
_LANGFUSE_ENV_RE = re.compile(r"[^a-z0-9-_]+")
_MAX_STRING_CHARS = 12_000
_MAX_SEQUENCE_ITEMS = 50
_ACTIVE_TOOL_SPAN: contextvars.ContextVar[_LangfuseToolSpan | None] = contextvars.ContextVar(
    "opentulpa_langfuse_active_tool_span",
    default=None,
)
_ACTIVE_OBSERVATION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "opentulpa_langfuse_active_observation_id",
    default=None,
)
_ACTIVE_TRACE_USAGE: contextvars.ContextVar[tuple[_TraceUsageAccumulator, ...]] = (
    contextvars.ContextVar("opentulpa_langfuse_active_trace_usage", default=())
)


class _NoopContext:
    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        _ = kwargs


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_langfuse_environment(value: Any) -> str:
    text = _clean_text(value).lower()
    if not text:
        return ""
    text = _LANGFUSE_ENV_RE.sub("-", text).strip("-_")
    if not text:
        return ""
    if text.startswith("langfuse"):
        text = f"env-{text}"
    return text[:40].strip("-_")


def _default_environment_candidate(*, deployment_tag: str | None) -> str:
    return (
        _clean_text(deployment_tag)
        or _clean_text(os.environ.get("LANGFUSE_DEPLOYMENT_TAG"))
        or _clean_text(os.environ.get("RAILWAY_SERVICE_NAME"))
        or _clean_text(os.environ.get("RAILWAY_ENVIRONMENT_NAME"))
        or _clean_text(os.environ.get("RAILWAY_ENVIRONMENT"))
        or _clean_text(os.environ.get("RAILWAY_ENVIRONMENT_ID"))
        or "local"
    )


def _redact_string(value: str) -> str:
    text = str(value)
    if text.lower().startswith("data:"):
        if ";base64," in text:
            prefix, _, _ = text.partition(";base64,")
            return f"{prefix};base64,[redacted]"
        prefix, _, _ = text.partition(",")
        return f"{prefix},[redacted]"
    if len(text) > _MAX_STRING_CHARS:
        return f"{text[:_MAX_STRING_CHARS]}...[truncated {len(text) - _MAX_STRING_CHARS} chars]"
    return text


def redact_for_langfuse(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    """Redact secrets and inline media while preserving debuggable structure."""
    if key and _SECRET_KEY_RE.search(str(key)):
        return "[redacted]"
    if depth > 8:
        return "[max_depth]"
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return f"[bytes:{len(value)}]"
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, dict):
        if str(value.get("type", "")).strip() == "input_audio" and "data" in value:
            value = {**value, "data": "[redacted-inline-media]"}
        out: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            safe_key = str(raw_key)
            out[safe_key] = redact_for_langfuse(raw_value, key=safe_key, depth=depth + 1)
        return out
    if isinstance(value, list | tuple | set):
        items = list(value)
        redacted = [
            redact_for_langfuse(item, depth=depth + 1) for item in items[:_MAX_SEQUENCE_ITEMS]
        ]
        if len(items) > _MAX_SEQUENCE_ITEMS:
            redacted.append(f"[truncated {len(items) - _MAX_SEQUENCE_ITEMS} items]")
        return redacted
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        with suppress(Exception):
            return redact_for_langfuse(model_dump(), depth=depth + 1)
    return _redact_string(str(value))


def _json_safe(value: Any) -> Any:
    redacted = redact_for_langfuse(value)
    with suppress(Exception):
        json.dumps(redacted, ensure_ascii=False, default=str)
        return redacted
    return str(redacted)


def _int_value(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _float_value(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_value(mapping.get(key))
        if value is not None:
            return value
    return None


def _usage_details(record: dict[str, Any]) -> dict[str, int]:
    details: dict[str, int] = {}
    mapping = {
        "native_tokens_prompt": "input",
        "native_tokens_completion": "output",
        "native_tokens_total": "total",
        "native_tokens_cached": "cache_read_input_tokens",
        "native_tokens_cache_write": "cache_write_input_tokens",
        "native_tokens_reasoning": "reasoning_output_tokens",
    }
    for source_key, langfuse_key in mapping.items():
        value = _int_value(record.get(source_key))
        if value is not None:
            details[langfuse_key] = value
    usage = record.get("usage")
    if isinstance(usage, dict):
        openai_mapping = {
            "prompt_tokens": "input",
            "completion_tokens": "output",
            "total_tokens": "total",
            "input_tokens": "input",
            "output_tokens": "output",
        }
        for source_key, langfuse_key in openai_mapping.items():
            value = _int_value(usage.get(source_key))
            if value is not None:
                details.setdefault(langfuse_key, value)
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached = _int_value(prompt_details.get("cached_tokens"))
            cache_write = _int_value(prompt_details.get("cache_write_tokens"))
            if cached is not None:
                details.setdefault("cache_read_input_tokens", cached)
            if cache_write is not None:
                details.setdefault("cache_write_input_tokens", cache_write)
        native_cached = _int_value(usage.get("prompt_cache_hit_tokens"))
        native_miss = _int_value(usage.get("prompt_cache_miss_tokens"))
        if native_cached is not None:
            details.setdefault("cache_read_input_tokens", native_cached)
        if native_miss is not None:
            details.setdefault("cache_write_input_tokens", native_miss)
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning = _int_value(completion_details.get("reasoning_tokens"))
            if reasoning is not None:
                details.setdefault("reasoning_output_tokens", reasoning)
    return details


def _cost_details(record: dict[str, Any]) -> dict[str, float]:
    details: dict[str, float] = {}
    mapping = {
        "native_cost_prompt_usd": "input",
        "native_cost_completion_usd": "output",
        "native_cost_usd": "total",
    }
    for source_key, langfuse_key in mapping.items():
        value = _float_value(record.get(source_key))
        if value is not None:
            details[langfuse_key] = value
    native_cost_details = record.get("native_cost_details")
    if isinstance(native_cost_details, dict):
        prompt = _first_float(
            native_cost_details,
            "prompt",
            "input",
            "prompt_cost",
            "input_cost",
            "upstream_inference_prompt_cost",
        )
        completion = _first_float(
            native_cost_details,
            "completion",
            "completions",
            "output",
            "completion_cost",
            "completions_cost",
            "output_cost",
            "upstream_inference_completion_cost",
            "upstream_inference_completions_cost",
        )
        total = _first_float(
            native_cost_details,
            "total",
            "cost",
            "total_cost",
            "upstream_inference_cost",
        )
        if prompt is not None:
            details.setdefault("input", prompt)
        if completion is not None:
            details.setdefault("output", completion)
        if total is not None:
            details.setdefault("total", total)
    usage = record.get("usage")
    if isinstance(usage, dict):
        cost = _float_value(usage.get("cost"))
        if cost is not None:
            details.setdefault("total", cost)
        usage_cost_details = usage.get("cost_details")
        if isinstance(usage_cost_details, dict):
            prompt = _first_float(
                usage_cost_details,
                "prompt",
                "input",
                "prompt_cost",
                "input_cost",
                "upstream_inference_prompt_cost",
            )
            completion = _first_float(
                usage_cost_details,
                "completion",
                "completions",
                "output",
                "completion_cost",
                "completions_cost",
                "output_cost",
                "upstream_inference_completion_cost",
                "upstream_inference_completions_cost",
            )
            total = _first_float(
                usage_cost_details,
                "total",
                "cost",
                "total_cost",
                "upstream_inference_cost",
            )
            if prompt is not None:
                details.setdefault("input", prompt)
            if completion is not None:
                details.setdefault("output", completion)
            if total is not None:
                details.setdefault("total", total)
    return details


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "prompt_messages",
        "response_message",
        "response_text",
        "response_content",
        "response_tool_calls",
        "usage",
    }
    return {key: _json_safe(value) for key, value in record.items() if key not in excluded}


@dataclass
class _TraceUsageAccumulator:
    usage: dict[str, int] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)

    def add(self, *, usage: dict[str, int], cost: dict[str, float]) -> None:
        for key, usage_value in usage.items():
            self.usage[key] = int(self.usage.get(key, 0)) + int(usage_value)
        for key, cost_value in cost.items():
            self.cost[key] = float(self.cost.get(key, 0.0)) + float(cost_value)


@dataclass
class _LangfuseToolSpan:
    tracer: LangfuseTracer
    trace_id: str | None
    tool_name: str
    tool_call_id: str | None = None
    args: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _ctx: Any = None
    _observation: Any = None
    _token: contextvars.Token[Any] | None = None
    _result: Any = None
    _status: str = "ok"
    _side_effects: list[dict[str, Any]] = field(default_factory=list)
    _started: float = field(default_factory=time.monotonic)

    def __enter__(self) -> _LangfuseToolSpan:
        self._token = _ACTIVE_TOOL_SPAN.set(self)
        if not self.tracer.enabled:
            return self
        client = self.tracer._client_or_none()
        if client is None:
            return self
        kwargs: dict[str, Any] = {
            "as_type": "tool",
            "name": f"tool.{self.tool_name}",
            "input": _json_safe(self.args),
            "metadata": self.tracer.base_metadata(
                {
                    **self.metadata,
                    "tool_name": self.tool_name,
                    "tool_call_id": self.tool_call_id,
                    "opentulpa_trace_id": self.trace_id,
                }
            ),
        }
        trace_context = self.tracer.trace_context_payload(self.trace_id)
        current_trace_id = None
        current_trace = getattr(client, "get_current_trace_id", None)
        if callable(current_trace):
            with suppress(Exception):
                current_trace_id = current_trace()
        if trace_context and not current_trace_id:
            kwargs["trace_context"] = trace_context
        with suppress(Exception):
            self._ctx = client.start_as_current_observation(**kwargs)
            self._observation = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        if exc is not None:
            self._status = "error"
            self._result = {"error": f"{type(exc).__name__}: {exc}"}
        elapsed_ms = int((time.monotonic() - self._started) * 1000)
        metadata = self.tracer.base_metadata(
            {
                **self.metadata,
                "tool_name": self.tool_name,
                "tool_call_id": self.tool_call_id,
                "status": self._status,
                "duration_ms": elapsed_ms,
                "side_effect_count": len(self._side_effects),
                "side_effects": self._side_effects[:20],
            }
        )
        if self._observation is not None:
            update = getattr(self._observation, "update", None)
            if callable(update):
                with suppress(Exception):
                    update(
                        output=_json_safe(self._result),
                        metadata=metadata,
                        level="ERROR" if self._status == "error" else "DEFAULT",
                        status_message=self._status,
                    )
        if self._ctx is not None:
            with suppress(Exception):
                self._ctx.__exit__(exc_type, exc, tb)
        if self._token is not None:
            with suppress(Exception):
                _ACTIVE_TOOL_SPAN.reset(self._token)
        return False

    def set_result(self, result: Any, *, status: str = "ok") -> None:
        self._result = result
        self._status = _clean_text(status) or "ok"

    def add_side_effect(self, payload: dict[str, Any]) -> None:
        event = _clean_text(payload.get("event"))
        summary = {
            "event": event,
            "status": _clean_text(payload.get("status")) or None,
            "customer_id": _clean_text(payload.get("customer_id")) or None,
            "thread_id": _clean_text(payload.get("thread_id")) or None,
            "trace_id": _clean_text(payload.get("trace_id")) or None,
            "payload": _json_safe(payload),
        }
        self._side_effects.append(summary)


class LangfuseTracer:
    """Lazy Langfuse client wrapper.

    The tracer is inert unless explicitly enabled and all required credentials are set.
    """

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        base_url: str,
        deployment_tag: str | None = None,
        environment: str | None = None,
        content_level: str = "full_debug",
        client: Any | None = None,
        callback_handler_cls: type[Any] | None = None,
    ) -> None:
        self.public_key = _clean_text(public_key)
        self.secret_key = _clean_text(secret_key)
        self.base_url = _clean_text(base_url).rstrip("/")
        self._enabled = bool(self.public_key and self.secret_key and self.base_url)
        self.deployment_tag = _clean_text(deployment_tag) or _clean_text(
            os.environ.get("RAILWAY_SERVICE_NAME")
        )
        self.environment = _normalize_langfuse_environment(
            environment or _default_environment_candidate(deployment_tag=self.deployment_tag)
        )
        self.content_level = _clean_text(content_level) or "full_debug"
        self._client = client
        self._callback_handler_cls = callback_handler_cls
        self._client_failed = False

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._client_failed

    def _install_env(self) -> None:
        os.environ["LANGFUSE_PUBLIC_KEY"] = self.public_key
        os.environ["LANGFUSE_SECRET_KEY"] = self.secret_key
        os.environ["LANGFUSE_BASE_URL"] = self.base_url
        os.environ["LANGFUSE_HOST"] = self.base_url
        if self.environment:
            os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = self.environment

    def _client_or_none(self) -> Any | None:
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        try:
            self._install_env()
            from langfuse import get_client

            self._client = get_client()
            return self._client
        except Exception:
            self._client_failed = True
            logger.exception("Failed to initialize Langfuse client; disabling Langfuse tracing.")
            return None

    def deterministic_trace_id(self, seed: str | None) -> str | None:
        text = _clean_text(seed)
        if not text:
            return None
        client = self._client_or_none()
        create_trace_id = getattr(client, "create_trace_id", None)
        if callable(create_trace_id):
            with suppress(Exception):
                return str(create_trace_id(seed=text))
        with suppress(Exception):
            from langfuse import Langfuse

            create = getattr(Langfuse, "create_trace_id", None)
            if callable(create):
                return str(create(seed=text))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def trace_context_payload(self, trace_id: str | None) -> dict[str, str] | None:
        resolved = self.deterministic_trace_id(trace_id)
        return {"trace_id": resolved} if resolved else None

    def base_metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        base = {
            "deployment_tag": self.deployment_tag or None,
            "environment": self.environment or None,
            "content_level": self.content_level,
            "railway_project_id": _clean_text(os.environ.get("RAILWAY_PROJECT_ID")) or None,
            "railway_environment_id": _clean_text(os.environ.get("RAILWAY_ENVIRONMENT_ID")) or None,
            "railway_service_id": _clean_text(os.environ.get("RAILWAY_SERVICE_ID")) or None,
            "railway_deployment_id": _clean_text(os.environ.get("RAILWAY_DEPLOYMENT_ID")) or None,
            "git_commit_sha": _clean_text(
                os.environ.get("RAILWAY_GIT_COMMIT_SHA") or os.environ.get("GIT_COMMIT_SHA")
            )
            or None,
            "git_branch": _clean_text(
                os.environ.get("RAILWAY_GIT_BRANCH") or os.environ.get("GIT_BRANCH")
            )
            or None,
        }
        base.update(dict(metadata or {}))
        return cast("dict[str, Any]", _json_safe(base))

    def tags(self, tags: list[str] | tuple[str, ...] | None = None) -> list[str]:
        resolved = ["opentulpa"]
        if self.deployment_tag:
            resolved.append(self.deployment_tag)
        if self.environment:
            resolved.append(f"env:{self.environment}")
        for item in tags or []:
            text = _clean_text(item)
            if text and text not in resolved:
                resolved.append(text)
        return resolved

    @contextmanager
    def _observation_context(self, client: Any, kwargs: dict[str, Any]) -> Any:
        observation_context = client.start_as_current_observation(**kwargs)
        observation = observation_context.__enter__()
        observation_id = _clean_text(
            getattr(observation, "id", None) or getattr(observation, "observation_id", None)
        )
        token = _ACTIVE_OBSERVATION_ID.set(observation_id) if observation_id else None
        try:
            yield observation
        finally:
            if token is not None:
                with suppress(Exception):
                    _ACTIVE_OBSERVATION_ID.reset(token)
            with suppress(Exception):
                observation_context.__exit__(None, None, None)

    def _propagate_attributes(
        self,
        *,
        trace_name: str,
        user_id: str | None,
        session_id: str | None,
        tags: list[str] | None,
    ) -> Any:
        try:
            from langfuse import propagate_attributes

            return propagate_attributes(
                trace_name=trace_name,
                user_id=_clean_text(user_id) or None,
                session_id=_clean_text(session_id) or None,
                tags=self.tags(tags),
            )
        except Exception:
            return _NoopContext()

    @contextmanager
    def trace_context(
        self,
        *,
        name: str,
        trace_id: str | None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        if not self.enabled:
            yield _NoopObservation()
            return
        client = self._client_or_none()
        if client is None:
            yield _NoopObservation()
            return
        kwargs: dict[str, Any] = {
            "as_type": "span",
            "name": _clean_text(name) or "opentulpa.trace",
            "input": _json_safe(input),
            "metadata": self.base_metadata(
                {**dict(metadata or {}), "opentulpa_trace_id": _clean_text(trace_id) or None}
            ),
        }
        try:
            observation_context = self._observation_context(client, kwargs)
            observation = observation_context.__enter__()
        except Exception:
            logger.exception("Failed to create Langfuse trace context.")
            yield _NoopObservation()
            return
        attributes_context = self._propagate_attributes(
            trace_name=kwargs["name"],
            user_id=user_id,
            session_id=session_id,
            tags=tags,
        )
        with suppress(Exception):
            attributes_context.__enter__()
        usage_accumulator = _TraceUsageAccumulator()
        usage_token = _ACTIVE_TRACE_USAGE.set(
            (*_ACTIVE_TRACE_USAGE.get(), usage_accumulator)
        )
        try:
            yield observation
        finally:
            if usage_accumulator.usage or usage_accumulator.cost:
                update = getattr(observation, "update", None)
                if callable(update):
                    with suppress(Exception):
                        update(
                            usage_details=dict(usage_accumulator.usage) or None,
                            cost_details=dict(usage_accumulator.cost) or None,
                        )
            with suppress(Exception):
                _ACTIVE_TRACE_USAGE.reset(usage_token)
            with suppress(Exception):
                attributes_context.__exit__(None, None, None)
            with suppress(Exception):
                observation_context.__exit__(None, None, None)

    def build_callbacks(
        self,
        *,
        user_id: str | None,
        trace_id: str | None,
        session_id: str | None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> list[Any]:
        _ = user_id, session_id, metadata, tags
        if not self.enabled:
            return []
        self._install_env()
        callback_cls = self._callback_handler_cls
        if callback_cls is None:
            try:
                from langfuse.langchain import CallbackHandler

                callback_cls = CallbackHandler
                self._callback_handler_cls = callback_cls
            except Exception:
                logger.exception("Failed to import Langfuse LangChain callback handler.")
                return []
        try:
            active_observation_id = _clean_text(_ACTIVE_OBSERVATION_ID.get())
            if not active_observation_id:
                logger.debug("Skipping Langfuse callback handler without active root observation.")
                return []
            return [callback_cls()]
        except Exception:
            logger.exception("Failed to build Langfuse callback handler.")
            return []

    def record_generation(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        client = self._client_or_none()
        if client is None:
            return
        call_site = _clean_text(record.get("call_site")) or "runtime_model_invoke"
        kwargs: dict[str, Any] = {
            "as_type": "generation",
            "name": f"llm.{call_site}",
            "model": _clean_text(record.get("model_name")) or None,
            "input": _json_safe(record.get("prompt_messages")),
            "output": _json_safe(
                {
                    "text": record.get("response_text"),
                    "content": record.get("response_content"),
                    "tool_calls": record.get("response_tool_calls"),
                    "error": record.get("error"),
                }
            ),
            "metadata": self.base_metadata(_record_metadata(record)),
        }
        usage = _usage_details(record)
        cost = _cost_details(record)
        if usage:
            kwargs["usage_details"] = usage
        if cost:
            kwargs["cost_details"] = cost
        for accumulator in _ACTIVE_TRACE_USAGE.get():
            accumulator.add(usage=usage, cost=cost)
        trace_context = self.trace_context_payload(record.get("trace_id"))
        current_trace_id = None
        current_trace = getattr(client, "get_current_trace_id", None)
        if callable(current_trace):
            with suppress(Exception):
                current_trace_id = current_trace()
        if not trace_context and not current_trace_id:
            logger.debug(
                "Skipping Langfuse generation without trace context for call_site=%s.",
                call_site,
            )
            return
        if trace_context and not current_trace_id:
            kwargs["trace_context"] = trace_context
        try:
            with self._observation_context(client, kwargs):
                return
        except Exception:
            logger.exception("Failed to record Langfuse generation.")

    def record_span(
        self,
        *,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
        status: str = "ok",
    ) -> None:
        if not self.enabled:
            return
        client = self._client_or_none()
        if client is None:
            return
        kwargs: dict[str, Any] = {
            "as_type": "span",
            "name": _clean_text(name) or "opentulpa.span",
            "input": _json_safe(input),
            "output": _json_safe(output),
            "metadata": self.base_metadata({**dict(metadata or {}), "status": status}),
            "level": "ERROR" if status == "error" else "DEFAULT",
            "status_message": status,
        }
        trace_context = self.trace_context_payload(trace_id)
        current_trace_id = None
        current_trace = getattr(client, "get_current_trace_id", None)
        if callable(current_trace):
            with suppress(Exception):
                current_trace_id = current_trace()
        if trace_context and not current_trace_id:
            kwargs["trace_context"] = trace_context
        try:
            with self._observation_context(client, kwargs):
                return
        except Exception:
            logger.exception("Failed to record Langfuse span '%s'.", name)

    def tool_span(
        self,
        *,
        trace_id: str | None,
        tool_name: str,
        tool_call_id: str | None = None,
        args: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> _LangfuseToolSpan:
        return _LangfuseToolSpan(
            tracer=self,
            trace_id=trace_id,
            tool_name=_clean_text(tool_name) or "unknown",
            tool_call_id=_clean_text(tool_call_id) or None,
            args=args,
            metadata=dict(metadata or {}),
        )

    def record_behavior_event(self, payload: dict[str, Any]) -> None:
        active_span = _ACTIVE_TOOL_SPAN.get()
        if active_span is None:
            return
        active_span.add_side_effect(payload)

    def flush(self) -> None:
        client = self._client_or_none()
        flush = getattr(client, "flush", None)
        if callable(flush):
            with suppress(Exception):
                flush()

    def shutdown(self) -> None:
        client = self._client_or_none()
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            with suppress(Exception):
                shutdown()
            return
        self.flush()


def create_langfuse_tracer(
    *,
    public_key: str | None,
    secret_key: str | None,
    base_url: str | None,
    deployment_tag: str | None = None,
    environment: str | None = None,
    content_level: str = "full_debug",
) -> LangfuseTracer | None:
    public = _clean_text(public_key)
    secret = _clean_text(secret_key)
    url = _clean_text(base_url).rstrip("/")
    if not (public and secret and url):
        return None
    return LangfuseTracer(
        public_key=public,
        secret_key=secret,
        base_url=url,
        deployment_tag=deployment_tag,
        environment=environment,
        content_level=content_level,
    )
