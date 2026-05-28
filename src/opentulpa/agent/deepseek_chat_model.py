"""DeepSeek chat model wiring for OpenRouter-hosted DeepSeek models."""

from __future__ import annotations

from typing import Any

from langchain_deepseek import ChatDeepSeek

from opentulpa.agent.lc_messages import AIMessage


class OpenRouterDeepSeekChatModel(ChatDeepSeek):
    """ChatDeepSeek variant that preserves DeepSeek thinking context across tool turns."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload_messages = payload.get("messages")
        if not isinstance(payload_messages, list):
            return payload
        for source, target in zip(messages, payload_messages, strict=False):
            if not isinstance(source, AIMessage) or not isinstance(target, dict):
                continue
            additional_kwargs = getattr(source, "additional_kwargs", {}) or {}
            reasoning_content = additional_kwargs.get("reasoning_content")
            if reasoning_content:
                target["reasoning_content"] = reasoning_content
            if reasoning_details := additional_kwargs.get("reasoning_details"):
                target["reasoning_details"] = reasoning_details
        return payload
