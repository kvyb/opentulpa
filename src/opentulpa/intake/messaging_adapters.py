"""Messaging integration adapters used by intake workflows."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, NotRequired, Protocol, TypedDict

from opentulpa.core.ids import new_short_id

_DEFAULT_INSTAGRAM_SCAN_LIMIT = 20
_MAX_INSTAGRAM_SCAN_LIMIT = 20
_CONVERSATION_SUMMARY_STRING_FIELDS = (
    "conversation_id",
    "recipient_id",
    "platform",
    "username",
    "incoming_user_id",
    "latest_message_id",
    "latest_message_created_time",
    "latest_message_sender_id",
    "latest_message_sender_username",
    "latest_message_text_preview",
    "latest_inbound_message_id",
    "latest_inbound_message_created_time",
    "latest_inbound_message_text_preview",
    "latest_inbound_sender_id",
    "latest_inbound_sender_username",
    "latest_outbound_message_id",
    "latest_outbound_message_created_time",
    "conversation_updated_time",
    "business_connection_id",
    "reply_window_status",
    "reply_window_reason",
)


class ConversationSummary(TypedDict, total=False):
    conversation_id: str
    recipient_id: str
    platform: str
    username: str
    incoming_user_id: str
    latest_message_id: str
    latest_message_created_time: str
    latest_message_sender_id: str
    latest_message_sender_username: str
    latest_message_text_preview: str
    latest_inbound_message_id: str
    latest_inbound_message_created_time: str
    latest_inbound_message_text_preview: str
    latest_inbound_sender_id: str
    latest_inbound_sender_username: str
    latest_outbound_message_id: str
    latest_outbound_message_created_time: str
    conversation_updated_time: str
    business_connection_id: str
    reply_window_status: str
    reply_window_reason: str
    matched: bool
    participant_ids: list[str]
    participant_usernames: dict[str, str]
    recipient_id_verified: bool
    unanswered_customer_messages: NotRequired[list[dict[str, Any]]]


@dataclass(frozen=True)
class SourceItemsResult:
    items: list[ConversationSummary]
    error: str | None = None
    warnings: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert isinstance(self.items, list)
        assert self.error is None or self.error.strip()


@dataclass(frozen=True)
class ConversationLoadResult:
    summary: ConversationSummary
    conversation: dict[str, Any]
    error: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.summary, dict)
        assert isinstance(self.conversation, dict)


@dataclass(frozen=True)
class MessagingAdapterContext:
    workflow_name: str
    customer_id: str
    channel: str
    provider: str
    source_config: dict[str, Any]

    def __post_init__(self) -> None:
        assert self.workflow_name.strip()
        assert self.customer_id.strip()
        assert self.channel == self.channel.lower()
        assert self.provider == self.provider.lower()
        assert isinstance(self.source_config, dict)

    @property
    def key(self) -> tuple[str, str]:
        return self.channel, self.provider


class MessagingIntegrationAdapter(Protocol):
    channel: str
    provider: str

    def list_source_items(self, *, context: MessagingAdapterContext) -> SourceItemsResult: ...

    def load_conversation(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_id: str,
    ) -> ConversationLoadResult: ...

    async def send_reply(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_summary: ConversationSummary,
        reply_text: str,
    ) -> str | None: ...


AdapterRegistry = dict[tuple[str, str], MessagingIntegrationAdapter]


def messaging_adapter_context(workflow: dict[str, Any]) -> MessagingAdapterContext:
    channel = str(workflow.get("channel", "") or "").strip().lower()
    provider = str(workflow.get("provider", "") or "").strip().lower()
    assert channel == channel.lower()
    assert provider == provider.lower()
    return MessagingAdapterContext(
        workflow_name=str(workflow.get("name", "") or "").strip() or "<unnamed workflow>",
        customer_id=str(workflow.get("customer_id", "") or "").strip(),
        channel=channel,
        provider=provider,
        source_config=_safe_dict(workflow.get("source_config")),
    )


def build_messaging_adapter_registry(
    *,
    composio: Any | None,
    telegram_business: Any | None,
) -> AdapterRegistry:
    adapters: list[MessagingIntegrationAdapter] = [
        ComposioInstagramMessagingAdapter(composio=composio),
        TelegramBusinessMessagingAdapter(telegram_business=telegram_business),
    ]
    registry = {(adapter.channel, adapter.provider): adapter for adapter in adapters}
    assert len(registry) == len(adapters)
    assert ("instagram_dm", "composio") in registry
    assert ("telegram_business_dm", "telegram_bot_api") in registry
    return registry


class ComposioInstagramMessagingAdapter:
    channel = "instagram_dm"
    provider = "composio"

    def __init__(self, *, composio: Any | None) -> None:
        self._composio = composio

    def list_source_items(self, *, context: MessagingAdapterContext) -> SourceItemsResult:
        composio = self._composio
        if composio is None or not bool(getattr(composio, "enabled", False)):
            return SourceItemsResult(
                items=[],
                error=f"Workflow {context.workflow_name} failed: Composio is not available.",
            )

        connected_account_id = str(context.source_config.get("connected_account_id", "") or "").strip() or None
        scan_limit = _bounded_int(
            context.source_config.get("scan_limit", _DEFAULT_INSTAGRAM_SCAN_LIMIT),
            default=_DEFAULT_INSTAGRAM_SCAN_LIMIT,
            minimum=1,
            maximum=_MAX_INSTAGRAM_SCAN_LIMIT,
        )
        configured_conversation_ids = _configured_conversation_ids(context.source_config)
        warnings: list[dict[str, str]] = []

        try:
            if configured_conversation_ids:
                items = []
                for conversation_id in configured_conversation_ids:
                    detailed = composio.get_instagram_conversation(
                        customer_id=context.customer_id,
                        conversation_id=conversation_id,
                        connected_account_id=connected_account_id,
                    )
                    items.append(_conversation_summary(detailed.get("summary")))
            else:
                conversations_payload = composio.list_instagram_conversations(
                    customer_id=context.customer_id,
                    connected_account_id=connected_account_id,
                    limit=scan_limit,
                )
                items = _conversation_summaries(conversations_payload.get("items"))
                warnings = _conversation_warnings(conversations_payload)
        except Exception as exc:
            return SourceItemsResult(
                items=[],
                error=f"Workflow {context.workflow_name} failed while reading Instagram DMs: {exc}",
                warnings=warnings,
            )
        return SourceItemsResult(items=items, warnings=warnings)

    def load_conversation(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_id: str,
    ) -> ConversationLoadResult:
        composio = self._composio
        if composio is None:
            return ConversationLoadResult(summary={}, conversation={}, error="Composio is not configured")
        connected_account_id = str(context.source_config.get("connected_account_id", "") or "").strip() or None
        try:
            detailed = composio.get_instagram_conversation(
                customer_id=context.customer_id,
                conversation_id=conversation_id,
                connected_account_id=connected_account_id,
            )
        except Exception as exc:
            return ConversationLoadResult(summary={}, conversation={}, error=str(exc))
        return ConversationLoadResult(
            summary=_conversation_summary(detailed.get("summary")),
            conversation=_safe_dict(detailed.get("conversation")),
        )

    async def send_reply(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_summary: ConversationSummary,
        reply_text: str,
    ) -> str | None:
        if not reply_text:
            return "reply_action=send_reply requires non-empty reply_text"
        recipient_id = str(conversation_summary.get("recipient_id", "") or "").strip()
        conversation_id = str(conversation_summary.get("conversation_id", "") or "").strip()
        if not recipient_id or not conversation_id:
            return "Instagram reply requires verified recipient_id and conversation_id"

        arguments = {
            "recipient_id": recipient_id,
            "conversation_id": conversation_id,
            "text": reply_text,
        }
        latest_inbound = str(conversation_summary.get("latest_inbound_message_id", "") or "").strip()
        if latest_inbound:
            arguments["reply_to_message_id"] = latest_inbound

        connected_account_id = str(context.source_config.get("connected_account_id", "") or "").strip() or None
        composio = self._composio
        if composio is None:
            return "Composio is not configured"
        try:
            result = composio.execute_tool(
                customer_id=context.customer_id,
                tool_slug="INSTAGRAM_SEND_TEXT_MESSAGE",
                arguments=arguments,
                connected_account_id=connected_account_id,
            )
        except Exception as exc:
            return f"failed to send Instagram DM reply: {exc}"
        if not bool(result.get("successful", False)):
            return str(result.get("error") or "Instagram DM reply failed")
        return None


class TelegramBusinessMessagingAdapter:
    channel = "telegram_business_dm"
    provider = "telegram_bot_api"

    def __init__(self, *, telegram_business: Any | None) -> None:
        self._telegram_business = telegram_business

    def list_source_items(self, *, context: MessagingAdapterContext) -> SourceItemsResult:
        telegram_business = self._telegram_business
        if telegram_business is None:
            return SourceItemsResult(
                items=[],
                error=f"Workflow {context.workflow_name} failed: Telegram Business is not available.",
            )
        business_connection_id = str(context.source_config.get("business_connection_id", "") or "").strip()
        if not business_connection_id:
            return SourceItemsResult(
                items=[],
                error=f"Workflow {context.workflow_name} failed: source_config.business_connection_id is required.",
            )
        payload = telegram_business.list_conversations(
            customer_id=context.customer_id,
            business_connection_id=business_connection_id,
            limit=_bounded_int(context.source_config.get("scan_limit", 10), default=10, minimum=1, maximum=50),
            chat_ids=_configured_conversation_ids(context.source_config) or None,
        )
        return SourceItemsResult(items=_conversation_summaries(payload.get("items")))

    def load_conversation(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_id: str,
    ) -> ConversationLoadResult:
        telegram_business = self._telegram_business
        if telegram_business is None:
            return ConversationLoadResult(
                summary={},
                conversation={},
                error="Telegram Business service is not configured",
            )
        business_connection_id = str(context.source_config.get("business_connection_id", "") or "").strip()
        detailed = telegram_business.get_conversation(
            customer_id=context.customer_id,
            business_connection_id=business_connection_id,
            conversation_id=conversation_id,
        )
        if not bool(detailed.get("ok", False)):
            return ConversationLoadResult(
                summary={},
                conversation={},
                error=str(detailed.get("error") or "conversation not found"),
            )
        return ConversationLoadResult(
            summary=_conversation_summary(detailed.get("summary")),
            conversation=_safe_dict(detailed.get("conversation")),
        )

    async def send_reply(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_summary: ConversationSummary,
        reply_text: str,
    ) -> str | None:
        if not reply_text:
            return "reply_action=send_reply requires non-empty reply_text"
        telegram_business = self._telegram_business
        if telegram_business is None:
            return "Telegram Business is not available"
        business_connection_id = str(context.source_config.get("business_connection_id", "") or "").strip()
        conversation_id = str(conversation_summary.get("conversation_id", "") or "").strip()
        if not business_connection_id or not conversation_id:
            return "Telegram Business reply requires business_connection_id and conversation_id"
        client = getattr(telegram_business, "client", None)
        if client is None:
            return "Telegram Business client is not available"

        latest_inbound = str(conversation_summary.get("latest_inbound_message_id", "") or "").strip()
        try:
            sent = await client.send_message(
                chat_id=conversation_id,
                text=reply_text,
                parse_mode="HTML",
                business_connection_id=business_connection_id,
                reply_to_message_id=int(latest_inbound) if latest_inbound.isdigit() else None,
            )
        except Exception as exc:
            return f"failed to send Telegram Business reply: {exc}"
        if not sent:
            return "Telegram Business reply failed"

        for result_message in _telegram_result_messages(sent):
            result_message.setdefault("message_id", new_short_id("tgmsg"))
            result_message.setdefault("date", int(datetime.now(UTC).timestamp()))
            result_message.setdefault("chat", {"id": conversation_id, "type": "private"})
            result_message.setdefault("text", reply_text)
            result_message.setdefault("business_connection_id", business_connection_id)
            result_message.setdefault("sender_business_bot", {"id": "opentulpa"})
            with suppress(Exception):
                telegram_business.upsert_message(
                    business_connection_id=business_connection_id,
                    customer_id=context.customer_id,
                    message=result_message,
                )
        return None


def _telegram_result_messages(sent: Any) -> list[dict[str, Any]]:
    if not isinstance(sent, dict):
        return []
    candidates = sent.get("results")
    if isinstance(candidates, list):
        messages: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            result = candidate.get("result")
            if isinstance(result, dict):
                messages.append(dict(result))
        return messages
    result = sent.get("result")
    if isinstance(result, dict):
        return [dict(result)]
    return []


def _conversation_warnings(conversations_payload: dict[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for item in _safe_list(conversations_payload.get("warnings")):
        warning = _safe_dict(item)
        error = str(warning.get("error", "") or "").strip()
        if not error:
            continue
        warnings.append(
            {
                "conversation_id": str(warning.get("conversation_id", "") or ""),
                "error": error,
            }
        )
    return warnings


def _conversation_summaries(value: Any) -> list[ConversationSummary]:
    return [_conversation_summary(item) for item in _safe_list(value)]


def _conversation_summary(value: Any) -> ConversationSummary:
    raw = _safe_dict(value)
    summary: ConversationSummary = {}
    summary.update(raw)
    for key in _CONVERSATION_SUMMARY_STRING_FIELDS:
        if key in raw:
            summary[key] = str(raw.get(key, "") or "").strip()
    unanswered = raw.get("unanswered_customer_messages")
    if isinstance(unanswered, list):
        summary["unanswered_customer_messages"] = [_safe_dict(item) for item in unanswered]
    return summary


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _configured_conversation_ids(source_config: dict[str, Any]) -> list[str]:
    configured = (
        source_config.get("conversation_ids")
        if isinstance(source_config.get("conversation_ids"), list)
        else (
            [source_config.get("conversation_id")]
            if str(source_config.get("conversation_id", "")).strip()
            else []
        )
    )
    return _unique_string_list(configured)


def _unique_string_list(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    assert minimum <= maximum
    assert default >= minimum
    assert default <= maximum
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))
