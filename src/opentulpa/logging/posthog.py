"""Optional PostHog wiring for LangGraph and direct runtime-owned LLM calls."""

from __future__ import annotations

import logging
from typing import cast
from typing import Any

logger = logging.getLogger(__name__)


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

    def _ensure_imports(self) -> bool:
        if not self.enabled:
            return False
        if self._client is not None and self._callback_handler_cls is not None:
            return True
        try:
            from posthog import Posthog
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

            self._client = Posthog(self._api_key, host=self._host)
            self._callback_handler_cls = OpenTulpaPostHogCallbackHandler
            return True
        except Exception:
            logger.exception("Failed to initialize PostHog client; disabling PostHog callbacks.")
            self._client = None
            self._callback_handler_cls = None
            return False

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


def create_posthog_logger(*, api_key: str | None, host: str | None) -> PostHogLangGraphLogger | None:
    key = str(api_key or "").strip()
    resolved_host = str(host or "").strip()
    if not key or not resolved_host:
        return None
    return PostHogLangGraphLogger(api_key=key, host=resolved_host)
