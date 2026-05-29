"""History working-set selection for prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentulpa.agent.lc_messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from opentulpa.agent.tool_parser import compact_tool_call_record as _compact_tool_call_record
from opentulpa.agent.tool_parser import compact_tool_payload as _compact_tool_payload
from opentulpa.agent.utils import approx_tokens as _approx_tokens
from opentulpa.agent.utils import content_to_text as _content_to_text
from opentulpa.agent.utils import message_to_text as _message_to_text


def trim_text_to_token_budget(text: str, *, token_budget: int) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    budget = max(1, int(token_budget))
    if _approx_tokens(raw) <= budget:
        return raw
    max_chars = max(800, budget * 4)
    if len(raw) <= max_chars:
        return raw
    reserve = max(64, max_chars // 2 - 8)
    compact = f"{raw[:reserve]}\n...\n{raw[-reserve:]}"
    while _approx_tokens(compact) > budget and reserve > 64:
        reserve = max(64, int(reserve * 0.85))
        compact = f"{raw[:reserve]}\n...\n{raw[-reserve:]}"
    return compact.strip()


@dataclass(slots=True)
class HistoryWorkingSet:
    raw_messages: list[AnyMessage]
    summary_text: str
    raw_chat_count: int
    raw_tool_count: int
    protected_count: int


class ContextHistoryEngine:
    def __init__(
        self,
        *,
        raw_chat_limit: int = 20,
        raw_tool_limit: int = 5,
        stale_summary_token_budget: int = 900,
    ) -> None:
        self.raw_chat_limit = max(4, int(raw_chat_limit))
        self.raw_tool_limit = max(2, int(raw_tool_limit))
        self.stale_summary_token_budget = max(200, int(stale_summary_token_budget))

    @staticmethod
    def _is_chat_message(message: AnyMessage) -> bool:
        return isinstance(message, (HumanMessage, AIMessage))

    @staticmethod
    def _tool_call_ids(message: AIMessage) -> list[str]:
        raw_calls = getattr(message, "tool_calls", []) or []
        ids: list[str] = []
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id", "") or "").strip()
            if call_id:
                ids.append(call_id)
        return ids

    @staticmethod
    def _latest_tool_call_ids(messages: list[AnyMessage], *, limit: int = 5) -> set[str]:
        latest_ids: list[str] = []
        seen: set[str] = set()
        max_items = max(1, int(limit))
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                call_id = str(getattr(message, "tool_call_id", "") or "").strip()
                if call_id and call_id not in seen:
                    seen.add(call_id)
                    latest_ids.append(call_id)
            elif isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                for call_id in reversed(ContextHistoryEngine._tool_call_ids(message)):
                    if call_id and call_id not in seen:
                        seen.add(call_id)
                        latest_ids.append(call_id)
            if len(latest_ids) >= max_items:
                break
        return set(latest_ids[:max_items])

    def _protected_suffix_indices(self, messages: list[AnyMessage]) -> set[int]:
        protected: set[int] = set()
        idx = len(messages) - 1
        pending_ids: set[str] = set()
        while idx >= 0:
            message = messages[idx]
            if isinstance(message, ToolMessage):
                tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
                if not pending_ids and idx > 0 and not (
                    isinstance(messages[idx - 1], AIMessage)
                    and getattr(messages[idx - 1], "tool_calls", None)
                ):
                    break
                protected.add(idx)
                if tool_call_id:
                    pending_ids.add(tool_call_id)
                idx -= 1
                continue
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                call_ids = set(self._tool_call_ids(message))
                if not pending_ids or (call_ids and call_ids & pending_ids):
                    protected.add(idx)
                    pending_ids -= call_ids
                    idx -= 1
                    continue
            break
        return protected

    def _summarize_stale_messages(
        self,
        messages: list[AnyMessage],
        *,
        latest_tool_call_ids: set[str] | None = None,
    ) -> str:
        if not messages:
            return ""
        parts: list[str] = []
        tool_calls_by_id: dict[str, dict[str, Any]] = {}
        latest_tool_call_ids = set(latest_tool_call_ids or ())
        result_ids = {
            str(getattr(message, "tool_call_id", "") or "").strip()
            for message in messages
            if isinstance(message, ToolMessage)
            and str(getattr(message, "tool_call_id", "") or "").strip()
        }
        for message in messages:
            if isinstance(message, HumanMessage):
                text = trim_text_to_token_budget(
                    _content_to_text(getattr(message, "content", "")),
                    token_budget=80,
                )
                if text:
                    parts.append(f"User: {text}")
                continue
            if isinstance(message, AIMessage):
                raw_calls = getattr(message, "tool_calls", []) or []
                if raw_calls:
                    for call in raw_calls:
                        if not isinstance(call, dict):
                            continue
                        call_id = str(call.get("id", "") or "").strip()
                        if not call_id:
                            continue
                        args = call.get("args")
                        tool_calls_by_id[call_id] = {
                            "name": str(call.get("name", "") or "").strip() or "tool",
                            "args": _compact_tool_payload(
                                args,
                                value_char_limit=(
                                    None if call_id in latest_tool_call_ids else 100
                                ),
                            ),
                        }
                        if call_id in latest_tool_call_ids and call_id not in result_ids:
                            parts.append(
                                _compact_tool_call_record(
                                    tool_name=tool_calls_by_id[call_id]["name"],
                                    args=tool_calls_by_id[call_id]["args"],
                                    result="",
                                    args_value_char_limit=None,
                                    result_value_char_limit=None,
                                )
                            )
                    names = ", ".join(
                        sorted(
                            {
                                tool_calls_by_id[call_id]["name"]
                                for call_id in tool_calls_by_id
                                if call_id
                            }
                        )
                    )
                    if names:
                        parts.append(f"Assistant requested tools: {names}")
                    continue
                text = trim_text_to_token_budget(
                    _content_to_text(getattr(message, "content", "")),
                    token_budget=80,
                )
                if text:
                    parts.append(f"Assistant: {text}")
                continue
            if isinstance(message, ToolMessage):
                tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
                call = tool_calls_by_id.get(tool_call_id, {})
                tool_name = str(call.get("name", "") or "").strip() or "tool"
                parts.append(
                    _compact_tool_call_record(
                        tool_name=tool_name,
                        args=call.get("args", ""),
                        result=_content_to_text(getattr(message, "content", "")),
                        args_value_char_limit=None,
                        result_value_char_limit=(
                            None if tool_call_id in latest_tool_call_ids else 100
                        ),
                    )
                )
        return trim_text_to_token_budget(
            "\n".join(parts).strip(),
            token_budget=self.stale_summary_token_budget,
        )

    def build_history_working_set(
        self,
        messages: list[AnyMessage],
        *,
        token_budget: int,
    ) -> HistoryWorkingSet:
        if not messages:
            return HistoryWorkingSet([], "", 0, 0, 0)
        budget = max(400, int(token_budget))
        protected = self._protected_suffix_indices(messages)
        keep_indices: set[int] = set(protected)
        raw_chat = 0
        raw_tool = 0
        for idx in range(len(messages) - 1, -1, -1):
            message = messages[idx]
            if idx in keep_indices:
                if self._is_chat_message(message):
                    raw_chat += 1
                elif isinstance(message, ToolMessage):
                    raw_tool += 1
                continue
            if isinstance(message, ToolMessage):
                if raw_tool >= self.raw_tool_limit:
                    continue
                keep_indices.add(idx)
                raw_tool += 1
                continue
            if self._is_chat_message(message):
                if raw_chat >= self.raw_chat_limit:
                    continue
                keep_indices.add(idx)
                raw_chat += 1
        for idx, message in enumerate(messages):
            if (
                idx not in keep_indices
                or not isinstance(message, AIMessage)
                or not getattr(message, "tool_calls", None)
            ):
                continue
            call_ids = set(self._tool_call_ids(message))
            if not call_ids:
                continue
            matching_tool_indices = {
                tool_idx
                for tool_idx, tool_msg in enumerate(messages)
                if isinstance(tool_msg, ToolMessage)
                and str(getattr(tool_msg, "tool_call_id", "") or "").strip() in call_ids
            }
            if matching_tool_indices and not matching_tool_indices.issubset(keep_indices):
                keep_indices.discard(idx)
        stale_messages = [msg for idx, msg in enumerate(messages) if idx not in keep_indices]
        latest_tool_call_ids = self._latest_tool_call_ids(messages, limit=self.raw_tool_limit)
        summary_text = self._summarize_stale_messages(
            stale_messages,
            latest_tool_call_ids=latest_tool_call_ids,
        )
        summary_tokens = min(self.stale_summary_token_budget, max(0, budget // 4))
        summary_text = (
            trim_text_to_token_budget(summary_text, token_budget=summary_tokens)
            if summary_text
            else ""
        )
        raw_budget = max(200, budget - (_approx_tokens(summary_text) if summary_text else 0))
        selected_pairs = [(idx, msg) for idx, msg in enumerate(messages) if idx in keep_indices]

        while selected_pairs:
            used = sum(max(1, _approx_tokens(_message_to_text(msg))) for _, msg in selected_pairs)
            if used <= raw_budget:
                break
            dropped = False
            for idx, (original_index, _) in enumerate(selected_pairs):
                if original_index in protected:
                    continue
                selected_pairs.pop(idx)
                dropped = True
                break
            if not dropped:
                break
        selected = [msg for _, msg in selected_pairs]
        return HistoryWorkingSet(
            raw_messages=selected,
            summary_text=summary_text,
            raw_chat_count=sum(1 for msg in selected if self._is_chat_message(msg)),
            raw_tool_count=sum(1 for msg in selected if isinstance(msg, ToolMessage)),
            protected_count=len(protected),
        )
