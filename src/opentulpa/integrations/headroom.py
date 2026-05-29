"""Headroom-backed tool result compression with a deterministic local fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opentulpa.agent.context_engine import trim_text_to_token_budget as _trim_text_to_token_budget
from opentulpa.agent.tool_parser import compact_tool_payload as _compact_tool_payload
from opentulpa.agent.utils import (
    approx_tokens as _approx_tokens,
)
from opentulpa.agent.utils import (
    safe_json as _safe_json,
)


def _normalize_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = str(item.get("text", "") or "").strip()
                if text:
                    parts.append(text)
            else:
                text = str(item or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(value or "").strip()


def _extract_messages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    messages = getattr(payload, "messages", None)
    if isinstance(messages, list):
        return [item for item in messages if isinstance(item, dict)]
    return []


@dataclass(slots=True)
class HeadroomService:
    model_name: str | None = None
    passthrough_token_limit: int = 160
    result_token_budget: int = 200
    result_value_char_limit: int | None = 160
    target_ratio: float = 0.2
    _compress_fn: Any | None = field(default=None, init=False, repr=False)
    _sdk_checked: bool = field(default=False, init=False, repr=False)

    def _compress_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        token_budget: int,
        model_name: str | None,
        target_ratio: float | None = None,
    ) -> str | None:
        compress = self._sdk()
        if compress is None:
            return None
        raw_tail_text = _normalize_message_content(messages[-1].get("content", "")) if messages else ""
        if not raw_tail_text:
            return None
        raw_tail_tokens = max(1, _approx_tokens(raw_tail_text))
        effective_ratio = float(target_ratio or self.target_ratio)
        effective_ratio = min(effective_ratio, max(0.05, min(0.95, float(token_budget) / float(raw_tail_tokens))))
        try:
            compressed_result = compress(
                messages,
                model=str(model_name or self.model_name or "").strip() or "gpt-4o-mini",
                model_limit=1,
                compress_user_messages=False,
                protect_recent=0,
                target_ratio=effective_ratio,
            )
        except Exception:
            return None
        if int(getattr(compressed_result, "tokens_saved", 0) or 0) <= 0:
            return None
        compressed_messages = _extract_messages(compressed_result)
        if not compressed_messages:
            return None
        candidate = _normalize_message_content(compressed_messages[-1].get("content", ""))
        if not candidate:
            return None
        candidate = _trim_text_to_token_budget(candidate, token_budget=token_budget)
        if not candidate or _approx_tokens(candidate) >= raw_tail_tokens:
            return None
        return candidate

    def _sdk(self) -> Any | None:
        if self._sdk_checked:
            return self._compress_fn
        self._sdk_checked = True
        try:
            from headroom import compress
        except ModuleNotFoundError as exc:
            if str(getattr(exc, "name", "") or "") == "headroom":
                self._compress_fn = None
                return None
            raise
        self._compress_fn = compress
        return self._compress_fn

    def _compress_with_headroom(
        self,
        *,
        tool_name: str,
        args: Any,
        raw_result_text: str,
        user_text: str,
        model_name: str | None,
    ) -> str | None:
        messages: list[dict[str, Any]] = []
        safe_user_text = str(user_text or "").strip()
        compact_args = _compact_tool_payload(args, value_char_limit=80).strip()
        if safe_user_text:
            messages.append({"role": "user", "content": safe_user_text[:2000]})
        elif compact_args:
            messages.append({"role": "user", "content": f"Tool args: {compact_args}"})
        messages.append({"role": "tool", "content": raw_result_text})
        return self._compress_messages(
            messages,
            token_budget=self.result_token_budget,
            model_name=model_name,
            target_ratio=float(self.target_ratio),
        )

    def compress_prompt_text(
        self,
        *,
        text: Any,
        field_name: str = "",
        tool_name: str = "",
        token_budget: int = 64,
        model_name: str | None = None,
    ) -> str:
        raw_text = _normalize_message_content(text)
        if not raw_text:
            return ""
        if _approx_tokens(raw_text) <= max(8, min(self.passthrough_token_limit, int(token_budget))):
            return raw_text
        messages: list[dict[str, Any]] = []
        context_bits: list[str] = []
        safe_tool_name = str(tool_name or "").strip()
        safe_field_name = str(field_name or "").strip()
        if safe_tool_name:
            context_bits.append(f"tool={safe_tool_name}")
        if safe_field_name:
            context_bits.append(f"field={safe_field_name}")
        if context_bits:
            messages.append({"role": "user", "content": "Compress for prompt history: " + ", ".join(context_bits)})
        messages.append({"role": "tool", "content": raw_text})
        candidate = self._compress_messages(
            messages,
            token_budget=max(8, int(token_budget)),
            model_name=model_name,
            target_ratio=float(self.target_ratio),
        )
        if candidate:
            return candidate
        return _trim_text_to_token_budget(raw_text, token_budget=max(8, int(token_budget)))

    def compress_tool_result(
        self,
        *,
        tool_name: str,
        args: Any,
        result: Any,
        user_text: str = "",
        model_name: str | None = None,
    ) -> str:
        raw_result_text = _safe_json(result).strip()
        if not raw_result_text:
            return ""
        if _approx_tokens(raw_result_text) <= self.passthrough_token_limit:
            return raw_result_text
        headroom_text = self._compress_with_headroom(
            tool_name=tool_name,
            args=args,
            raw_result_text=raw_result_text,
            user_text=user_text,
            model_name=model_name,
        )
        if headroom_text:
            return headroom_text
        compact = ""
        if isinstance(result, dict):
            status_value = str(result.get("status", "") or "").strip()
            compact_rest = _compact_tool_payload(
                {key: value for key, value in result.items() if str(key) != "status"},
                value_char_limit=self.result_value_char_limit,
            ).strip()
            compact_parts = [f"status={status_value}"] if status_value else []
            if compact_rest:
                compact_parts.append(compact_rest)
            compact = " | ".join(compact_parts).strip()
        if not compact:
            compact = _compact_tool_payload(result, value_char_limit=self.result_value_char_limit).strip()
        if not compact:
            compact = raw_result_text
        return _trim_text_to_token_budget(compact, token_budget=self.result_token_budget)


_PROMPT_HISTORY_SERVICE: HeadroomService | None = None


def compress_prompt_history_text(
    *,
    text: Any,
    field_name: str = "",
    tool_name: str = "",
    token_budget: int = 64,
) -> str:
    global _PROMPT_HISTORY_SERVICE
    if _PROMPT_HISTORY_SERVICE is None:
        _PROMPT_HISTORY_SERVICE = HeadroomService()
    return _PROMPT_HISTORY_SERVICE.compress_prompt_text(
        text=text,
        field_name=field_name,
        tool_name=tool_name,
        token_budget=token_budget,
    )
