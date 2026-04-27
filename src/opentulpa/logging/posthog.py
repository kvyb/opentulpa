"""Optional PostHog wiring for LangGraph and direct runtime-owned LLM calls."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

logger = logging.getLogger(__name__)

PROCESS_LOG_EVENT = "opentulpa.process_log"
_PROCESS_LOG_MESSAGE_LIMIT = 8000
_LOG_EXTRA_VALUE_LIMIT = 2000
_PROCESS_LOG_FIELD_VALUE_RE = r"[A-Za-z0-9_.:-]+"
_LOG_LEVEL_RE = re.compile(r"\b(CRITICAL|ERROR|WARNING|WARN|INFO|DEBUG|TRACE)\b", re.IGNORECASE)
_LOGGING_HANDLER_STATE = threading.local()
_STANDARD_LOG_RECORD_ATTRS = set(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__.keys()
) | {"asctime", "message"}


def _provider_value(provider: Callable[[], str] | None) -> str:
    if provider is None:
        return ""
    try:
        return str(provider() or "").strip()
    except Exception:
        return ""


def _extract_process_log_field(text: str, field_name: str) -> str:
    pattern = re.compile(
        rf"(?:^|[\s,{{])['\"]?{re.escape(field_name)}['\"]?\s*[:=]\s*['\"]?"
        rf"(?P<value>{_PROCESS_LOG_FIELD_VALUE_RE})",
        re.IGNORECASE,
    )
    match = pattern.search(str(text or ""))
    if not match:
        return ""
    return str(match.group("value") or "").strip()


def _detect_process_log_level(text: str) -> str:
    match = _LOG_LEVEL_RE.search(str(text or ""))
    if not match:
        return ""
    value = match.group(1).lower()
    return "warning" if value == "warn" else value


def _safe_log_extra_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:_LOG_EXTRA_VALUE_LIMIT]
    if isinstance(value, list | tuple):
        return [_safe_log_extra_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {
            str(key)[:80]: _safe_log_extra_value(item)
            for key, item in list(value.items())[:50]
            if str(key or "").strip()
        }
    return str(value)[:_LOG_EXTRA_VALUE_LIMIT]


def _log_record_extra_properties(record: logging.LogRecord) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        safe_key = str(key or "").strip()
        if (
            not safe_key
            or safe_key.startswith("_")
            or safe_key in _STANDARD_LOG_RECORD_ATTRS
            or safe_key in {"customer_id", "thread_id"}
        ):
            continue
        properties[f"extra_{safe_key}"] = _safe_log_extra_value(value)
    return properties


class PostHogLoggingHandler(logging.Handler):
    def __init__(
        self,
        *,
        posthog_logger: Any,
        public_base_url: str | None = None,
        customer_id_provider: Callable[[], str] | None = None,
        thread_id_provider: Callable[[], str] | None = None,
    ) -> None:
        super().__init__()
        self._posthog_logger = posthog_logger
        self._public_base_url = str(public_base_url or "").strip().rstrip("/")
        self._customer_id_provider = customer_id_provider
        self._thread_id_provider = thread_id_provider

    def emit(self, record: logging.LogRecord) -> None:
        if self._should_skip_record(record) or bool(getattr(_LOGGING_HANDLER_STATE, "active", False)):
            return
        try:
            _LOGGING_HANDLER_STATE.active = True
            raw_message = record.getMessage()
            message = raw_message[:_PROCESS_LOG_MESSAGE_LIMIT]
            customer_id = (
                _provider_value(self._customer_id_provider)
                or str(getattr(record, "customer_id", "") or "").strip()
                or _extract_process_log_field(raw_message, "customer_id")
            )
            thread_id = (
                _provider_value(self._thread_id_provider)
                or str(getattr(record, "thread_id", "") or "").strip()
                or _extract_process_log_field(raw_message, "thread_id")
            )
            properties: dict[str, Any] = {
                "source": "python_logging",
                "logger_name": str(record.name or "").strip(),
                "log_level": str(record.levelname or "").strip().lower(),
                "log_levelno": int(record.levelno),
                "message": message,
                "message_template": str(record.msg or "")[:_PROCESS_LOG_MESSAGE_LIMIT],
                "message_length": len(raw_message),
                "message_truncated": len(raw_message) > len(message),
                "customer_id": customer_id,
                "thread_id": thread_id,
                "public_base_url": self._public_base_url,
                "log_ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "module": str(record.module or "").strip(),
                "file": str(record.filename or "").strip(),
                "pathname": str(record.pathname or "").strip(),
                "line_number": int(record.lineno),
                "function": str(record.funcName or "").strip(),
                "process_id": int(record.process),
                "process_name": str(record.processName or "").strip(),
                "python_thread_id": int(record.thread),
                "python_thread_name": str(record.threadName or "").strip(),
            }
            properties.update(_log_record_extra_properties(record))
            if record.exc_info:
                exc_type, exc_value, _ = record.exc_info
                properties["exception_type"] = getattr(exc_type, "__name__", str(exc_type))
                properties["exception_message"] = str(exc_value or "")[:_LOG_EXTRA_VALUE_LIMIT]
                formatter = self.formatter or logging.Formatter()
                properties["traceback"] = formatter.formatException(record.exc_info)
            if record.stack_info:
                properties["stack_info"] = str(record.stack_info)[:_PROCESS_LOG_MESSAGE_LIMIT]
            if not customer_id:
                properties["$process_person_profile"] = False
            self._posthog_logger.capture_event(
                distinct_id=customer_id or "opentulpa_server",
                event=PROCESS_LOG_EVENT,
                properties=properties,
            )
        except Exception:
            return
        finally:
            _LOGGING_HANDLER_STATE.active = False

    @staticmethod
    def _should_skip_record(record: logging.LogRecord) -> bool:
        logger_name = str(record.name or "").strip()
        return (
            logger_name == __name__
            or logger_name.startswith("opentulpa.logging.posthog")
            or logger_name == "posthog"
            or logger_name.startswith("posthog.")
        )


def install_posthog_logging_handler(
    *,
    posthog_logger: Any,
    public_base_url: str | None = None,
    customer_id_provider: Callable[[], str] | None = None,
    thread_id_provider: Callable[[], str] | None = None,
    root_logger: logging.Logger | None = None,
) -> PostHogLoggingHandler:
    target_logger = root_logger or logging.getLogger()
    for handler in target_logger.handlers:
        if isinstance(handler, PostHogLoggingHandler) and handler._posthog_logger is posthog_logger:
            return handler
    handler = PostHogLoggingHandler(
        posthog_logger=posthog_logger,
        public_base_url=public_base_url,
        customer_id_provider=customer_id_provider,
        thread_id_provider=thread_id_provider,
    )
    target_logger.addHandler(handler)
    return handler


def uninstall_posthog_logging_handler(
    handler: logging.Handler | None,
    *,
    root_logger: logging.Logger | None = None,
) -> None:
    if handler is None:
        return
    target_logger = root_logger or logging.getLogger()
    try:
        target_logger.removeHandler(handler)
    except Exception:
        return
    try:
        handler.close()
    except Exception:
        return


def create_process_output_posthog_callback(
    *,
    posthog_logger: Any,
    public_base_url: str | None = None,
    customer_id_provider: Callable[[], str] | None = None,
    thread_id_provider: Callable[[], str] | None = None,
) -> Callable[[dict[str, Any]], None]:
    resolved_public_base_url = str(public_base_url or "").strip().rstrip("/")

    def capture_process_output(event: dict[str, Any]) -> None:
        raw_message = str(event.get("message", "") or "")
        if not raw_message:
            return
        message = raw_message[:_PROCESS_LOG_MESSAGE_LIMIT]
        customer_id = (
            _provider_value(customer_id_provider)
            or _extract_process_log_field(raw_message, "customer_id")
        )
        thread_id = (
            _provider_value(thread_id_provider)
            or _extract_process_log_field(raw_message, "thread_id")
        )
        properties: dict[str, Any] = {
            "source": "process_output",
            "stream": str(event.get("stream", "") or "").strip(),
            "message": message,
            "message_length": len(raw_message),
            "message_truncated": len(raw_message) > len(message),
            "customer_id": customer_id,
            "thread_id": thread_id,
            "public_base_url": resolved_public_base_url,
            "log_level": _detect_process_log_level(raw_message),
            "log_ts": str(event.get("ts", "") or "").strip(),
            "project_root": str(event.get("project_root", "") or "").strip(),
        }
        if not customer_id:
            properties["$process_person_profile"] = False
        posthog_logger.capture_event(
            distinct_id=customer_id or "opentulpa_server",
            event=PROCESS_LOG_EVENT,
            properties=properties,
        )

    return capture_process_output


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _usage_cost_fields_from_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    cost = _maybe_float(payload.get("cost"))
    cost_details = payload.get("cost_details")
    fields: dict[str, Any] = {}
    if cost is not None:
        fields["$ai_total_cost_usd"] = cost
    if isinstance(cost_details, dict) and cost_details:
        fields["openrouter_cost_details"] = dict(cost_details)
        prompt_cost = _maybe_float(cost_details.get("prompt"))
        completion_cost = _maybe_float(cost_details.get("completion"))
        if prompt_cost is not None:
            fields["openrouter_prompt_cost_usd"] = prompt_cost
        if completion_cost is not None:
            fields["openrouter_completion_cost_usd"] = completion_cost
    return fields


def _extract_openrouter_cost_fields(output: Any) -> dict[str, Any]:
    candidates: list[Any] = []
    llm_output = getattr(output, "llm_output", None)
    if isinstance(llm_output, dict):
        candidates.extend(
            [
                llm_output.get("usage"),
                llm_output.get("token_usage"),
                llm_output,
            ]
        )
    generations = getattr(output, "generations", None)
    if isinstance(generations, list):
        for generation in generations:
            if isinstance(generation, dict):
                candidates.append(generation.get("usage"))
                continue
            if isinstance(generation, list):
                for generation_chunk in generation:
                    if isinstance(generation_chunk, dict):
                        candidates.append(generation_chunk.get("usage"))
                    generation_info = getattr(generation_chunk, "generation_info", None)
                    if isinstance(generation_info, dict):
                        candidates.extend(
                            [
                                generation_info.get("usage"),
                                generation_info.get("usage_metadata"),
                            ]
                        )
                    message = getattr(generation_chunk, "message", None)
                    response_metadata = getattr(message, "response_metadata", None)
                    if isinstance(response_metadata, dict):
                        candidates.extend(
                            [
                                response_metadata.get("usage"),
                                response_metadata.get("token_usage"),
                                response_metadata,
                            ]
                        )
                    usage_metadata = getattr(message, "usage_metadata", None)
                    if isinstance(usage_metadata, dict):
                        candidates.append(usage_metadata)
    for candidate in candidates:
        fields = _usage_cost_fields_from_mapping(candidate)
        if fields:
            return fields
    return {}


class PostHogLangGraphLogger:
    """Lazy PostHog client/handler provider.

    This stays inert unless both POSTHOG_API_KEY and POSTHOG_HOST are configured.
    """

    def __init__(self, *, api_key: str, host: str) -> None:
        self._api_key = str(api_key or "").strip()
        self._host = str(host or "").strip()
        self._client: Any | None = None
        self._callback_handler_cls: type[Any] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._api_key and self._host)

    def _ensure_client(self) -> bool:
        if not self.enabled:
            return False
        if self._client is not None:
            return True
        try:
            from posthog import Posthog
        except Exception:
            logger.exception("Failed to import PostHog SDK; disabling PostHog client.")
            return False
        try:
            self._client = Posthog(self._api_key, host=self._host)
            return True
        except Exception:
            logger.exception("Failed to initialize PostHog client; disabling PostHog callbacks.")
            self._client = None
            return False

    def _ensure_imports(self) -> bool:
        if not self._ensure_client():
            return False
        if self._client is not None and self._callback_handler_cls is not None:
            return True
        try:
            from posthog.ai.langchain import CallbackHandler as BaseCallbackHandler
            from posthog.ai.langchain.callbacks import (
                ChatGeneration,
                _capture_exception_and_update_properties,
                _convert_message_to_dict,
                _extract_raw_response,
                _get_http_status,
                _parse_usage,
                _stringify_exception,
                sanitize_langchain,
                with_privacy_mode,
            )
        except Exception:
            logger.exception("Failed to import PostHog LangGraph integration; disabling PostHog callbacks.")
            return False
        try:
            class OpenTulpaPostHogCallbackHandler(BaseCallbackHandler):
                def _capture_generation(
                    self,
                    trace_id: Any,
                    run_id: Any,
                    run: Any,
                    output: Any,
                    parent_run_id: Any = None,
                ) -> None:
                    event_properties = {
                        "$ai_trace_id": trace_id,
                        "$ai_span_id": run_id,
                        "$ai_span_name": run.name,
                        "$ai_parent_id": parent_run_id,
                        "$ai_provider": run.provider,
                        "$ai_model": run.model,
                        "$ai_model_parameters": run.model_params,
                        "$ai_input": with_privacy_mode(
                            self._ph_client, self._privacy_mode, sanitize_langchain(run.input)
                        ),
                        "$ai_http_status": 200,
                        "$ai_latency": run.latency,
                        "$ai_base_url": run.base_url,
                        "$ai_framework": "langchain",
                    }

                    if isinstance(run.posthog_properties, dict):
                        event_properties.update(run.posthog_properties)

                    if run.tools:
                        event_properties["$ai_tools"] = run.tools

                    if self._properties:
                        event_properties.update(self._properties)

                    if self._distinct_id is None:
                        event_properties["$process_person_profile"] = False

                    if isinstance(output, BaseException):
                        event_properties["$ai_http_status"] = _get_http_status(output)
                        event_properties["$ai_error"] = _stringify_exception(output)
                        event_properties["$ai_is_error"] = True
                        event_properties = _capture_exception_and_update_properties(
                            self._ph_client,
                            output,
                            self._distinct_id,
                            self._groups,
                            event_properties,
                        )
                    else:
                        usage = _parse_usage(output, run.provider, run.model)
                        event_properties["$ai_input_tokens"] = usage.input_tokens
                        event_properties["$ai_output_tokens"] = usage.output_tokens
                        event_properties["$ai_cache_creation_input_tokens"] = usage.cache_write_tokens
                        event_properties["$ai_cache_read_input_tokens"] = usage.cache_read_tokens
                        event_properties["$ai_reasoning_tokens"] = usage.reasoning_tokens
                        event_properties.update(_extract_openrouter_cost_fields(output))

                        generation_result = output.generations[-1]
                        if isinstance(generation_result[-1], ChatGeneration):
                            completions = [
                                _convert_message_to_dict(cast(ChatGeneration, generation).message)
                                for generation in generation_result
                            ]
                        else:
                            completions = [
                                _extract_raw_response(generation)
                                for generation in generation_result
                            ]
                        event_properties["$ai_output_choices"] = with_privacy_mode(
                            self._ph_client, self._privacy_mode, completions
                        )

                    self._ph_client.capture(
                        distinct_id=self._distinct_id or trace_id,
                        event="$ai_generation",
                        properties=event_properties,
                        groups=self._groups,
                    )

            self._callback_handler_cls = OpenTulpaPostHogCallbackHandler
            return True
        except Exception:
            logger.exception("Failed to initialize PostHog client; disabling PostHog callbacks.")
            self._callback_handler_cls = None
            return self._client is not None

    def build_callbacks(
        self,
        *,
        distinct_id: str | None = None,
        trace_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> list[Any]:
        if not self._ensure_imports():
            return []
        callback_handler_cls = self._callback_handler_cls
        client = self._client
        if callback_handler_cls is None or client is None:
            return []
        kwargs: dict[str, Any] = {
            "client": client,
            "privacy_mode": False,
        }
        did = str(distinct_id or "").strip()
        tid = str(trace_id or "").strip()
        if did:
            kwargs["distinct_id"] = did
        if tid:
            kwargs["trace_id"] = tid
        cleaned_properties = {
            str(key): value
            for key, value in (properties or {}).items()
            if value not in (None, "", [], {}, ())
        }
        if cleaned_properties:
            kwargs["properties"] = cleaned_properties
        try:
            return [callback_handler_cls(**kwargs)]
        except Exception:
            logger.exception("Failed to build PostHog callback handler; skipping callbacks for this call.")
            return []

    def shutdown(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.shutdown()
        except Exception:
            logger.exception("Failed to shut down PostHog client cleanly.")
        finally:
            self._client = None
            self._callback_handler_cls = None

    def capture_event(
        self,
        *,
        distinct_id: str | None,
        event: str,
        properties: dict[str, Any] | None = None,
        groups: dict[str, Any] | None = None,
    ) -> None:
        event_name = str(event or "").strip()
        if not event_name or not self._ensure_client():
            return
        client = self._client
        if client is None:
            return
        cleaned_properties = {
            str(key): value
            for key, value in (properties or {}).items()
            if value not in (None, "", [], {}, ())
        }
        try:
            client.capture(
                distinct_id=str(distinct_id or "").strip() or event_name,
                event=event_name,
                properties=cleaned_properties or None,
                groups=groups,
            )
        except Exception:
            logger.exception("Failed to capture PostHog event '%s'.", event_name)


def create_posthog_logger(*, api_key: str | None, host: str | None) -> PostHogLangGraphLogger | None:
    key = str(api_key or "").strip()
    resolved_host = str(host or "").strip()
    if not key or not resolved_host:
        return None
    return PostHogLangGraphLogger(api_key=key, host=resolved_host)
