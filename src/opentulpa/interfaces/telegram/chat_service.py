"""Telegram chat bridge orchestration service."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any

from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.config import get_openai_compatible_api_key_from_env
from opentulpa.core.debug_logs import (
    DEFAULT_DEBUG_LOG_LOOKBACK_DAYS,
    build_debug_logs_archive_bytes,
)
from opentulpa.core.ids import new_short_id
from opentulpa.core.shutdown_drain import ShutdownDrainingError
from opentulpa.interfaces.telegram.attachments import (
    build_uploaded_files_context,
    extract_attachments,
    ingest_attachments,
)
from opentulpa.interfaces.telegram.chat_routing import (
    TelegramAccessDecision,
    TelegramCommandRoute,
)
from opentulpa.interfaces.telegram.client import TelegramClient, parse_telegram_update
from opentulpa.interfaces.telegram.constants import STATE_PATH
from opentulpa.interfaces.telegram.env_management import (
    missing_key_prompt,
    status_text,
)
from opentulpa.interfaces.telegram.interactive_inbox import (
    InteractiveSession,
    InteractiveSubmissionResult,
    TelegramInteractiveInbox,
)
from opentulpa.interfaces.telegram.models import TelegramContext
from opentulpa.interfaces.telegram.relay import (
    _emit_typing_until_done,
    debug_log,
    stream_langgraph_reply_to_telegram,
)
from opentulpa.interfaces.telegram.relay import (
    relay_event_via_main_agent as _relay_event_via_main_agent,
)
from opentulpa.interfaces.telegram.relay import (
    relay_task_event_via_main_agent as _relay_task_event_via_main_agent,
)
from opentulpa.interfaces.telegram.security import is_user_allowed, parse_csv_set
from opentulpa.interfaces.telegram.state_store import TelegramStateStore

STATE_STORE = TelegramStateStore(STATE_PATH)
logger = logging.getLogger(__name__)
_BOT_USERNAME_BY_TOKEN: dict[str, str] = {}

_UPLOAD_WITHOUT_TEXT_PREFIX = "User uploaded one or more files without extra text."
_UPLOAD_CONTEXT_MARKER = "Internal uploaded-file context."
_UNCLEAR_UPLOAD_GUIDANCE_SNIPPETS = (
    "If intent is unclear, ask what the user wants done",
    "User-facing reply guidance: briefly acknowledge the upload",
)


def _clean_thread_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def find_session_slots_for_customer_id(customer_id: str) -> list[dict[str, Any]]:
    return STATE_STORE.find_session_slots(customer_id)


def get_session_slot_for_chat_id(chat_id: int) -> dict[str, Any] | None:
    return STATE_STORE.get_session_slot(chat_id)


def _format_agent_error_for_user(exc: Exception) -> str:
    """Convert backend/model failures into actionable Telegram-safe user messages."""
    text = str(exc)
    lowered = text.lower()
    if "401" in lowered and (
        "user not found" in lowered
        or "authentication" in lowered
        or "invalid api key" in lowered
        or "unauthorized" in lowered
    ):
        return (
            "Model authentication failed (the configured provider key is invalid or revoked). "
            "Set a valid OPENAI_COMPATIBLE_API_KEY for your OpenAI-compatible endpoint and restart OpenTulpa. "
            "OPENROUTER_API_KEY is still accepted as a legacy alias."
        )
    if "429" in lowered or "rate limit" in lowered:
        return "The model provider is rate-limiting requests right now. Please try again shortly."
    return "I hit a backend error while generating a reply. Please try again."


def _inject_voice_message_context(text: str, transcripts: list[str]) -> str:
    safe_lines = [str(item).strip() for item in transcripts if str(item).strip()]
    if not safe_lines:
        return str(text or "")
    voice_block = "\n".join(f"<user sent voice message>: {line}" for line in safe_lines)
    base = str(text or "").strip()
    if base:
        return f"{base}\n\n{voice_block}"
    return voice_block


def _reset_chat_session_context(
    state: dict[str, Any],
    *,
    chat_id: int,
    user_id: int,
    username: str | None = None,
    resolve_customer_id: Callable[[str], str] | None = None,
    resolve_telegram_customer_id: Callable[[int], str] | None = None,
) -> tuple[str, str]:
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
    chat_key = str(chat_id)
    slot = sessions.get(chat_key)
    if not isinstance(slot, dict):
        slot = {}
    existing_customer_id = str(slot.get("customer_id", "")).strip()
    if existing_customer_id and resolve_customer_id is not None:
        customer_id = str(resolve_customer_id(existing_customer_id) or "").strip()
    elif existing_customer_id:
        customer_id = existing_customer_id
    elif resolve_telegram_customer_id is not None:
        customer_id = str(resolve_telegram_customer_id(user_id) or "").strip()
    else:
        customer_id = f"telegram_{user_id}"
    if not customer_id:
        customer_id = f"telegram_{user_id}"
    now_utc_iso = datetime.now(UTC).isoformat()
    thread_id = new_short_id("chat")
    wake_thread_id = new_short_id("wake")
    sessions[chat_key] = {
        "user_id": int(user_id),
        "username": username or slot.get("username") or "",
        "customer_id": customer_id,
        "thread_id": thread_id,
        "wake_thread_id": wake_thread_id,
        "role": "owner",
        "last_user_message_at": now_utc_iso,
        "last_assistant_message_at": None,
    }
    state["sessions"] = sessions

    pending_map = state.get("pending_key_by_chat")
    if not isinstance(pending_map, dict):
        pending_map = {}
    pending_map.pop(chat_key, None)
    state["pending_key_by_chat"] = pending_map
    return thread_id, customer_id


def _start_help_text() -> str:
    return (
        "OpenTulpa is connected.\n\n"
        "What I can do:\n"
        "- Web + links: web search, read URLs, summarize current info\n"
        "- Interactive browsing: browser automation for dynamic sites (when configured)\n"
        "- Files: analyze PDFs/DOCX/text/images/voice notes you send\n"
        "- Code + automations: write/debug scripts, run checks, schedule recurring tasks\n"
        "- Memory + preferences: remember your style/process directives\n\n"
        "To personalize quickly, answer these:\n"
        "1. What are you struggling with right now?\n"
        "2. Which repetitive task should I automate first?\n"
        "3. Which services should I connect first (Gmail, Sheets, custom APIs, etc.)?\n\n"
        "Commands:\n"
        "/start\n"
        "/status\n"
        "/fresh\n"
        "/debug_logs"
    )


def _telegram_command_name(text: str) -> str:
    parts = str(text or "").strip().split(None, 1)
    if not parts:
        return ""
    head = parts[0].lower()
    if not head.startswith("/"):
        return ""
    return head.split("@", 1)[0]


async def _resolve_bot_username(bot_token: str | None) -> str:
    safe_token = str(bot_token or "").strip()
    if not safe_token:
        return ""
    cached = _BOT_USERNAME_BY_TOKEN.get(safe_token, "")
    if cached:
        return cached
    client = TelegramClient(safe_token)
    try:
        me = await client.get_me()
    finally:
        with suppress(Exception):
            await client.aclose()
    username = str((me or {}).get("username", "")).strip().lstrip("@")
    if username:
        _BOT_USERNAME_BY_TOKEN[safe_token] = username
    return username


def _telegram_chat_is_group(message: dict[str, Any]) -> bool:
    chat = message.get("chat")
    chat_type = str((chat or {}).get("type", "") if isinstance(chat, dict) else "").strip()
    return chat_type in {"group", "supergroup"}


def _telegram_message_mentions_bot(message: dict[str, Any], bot_username: str) -> bool:
    safe_username = str(bot_username or "").strip().lstrip("@").lower()
    if not safe_username:
        return False
    text = str(message.get("text") or message.get("caption") or "")
    if re.search(rf"(?<!\w)@{re.escape(safe_username)}(?!\w)", text, flags=re.IGNORECASE):
        return True
    for entity_key in ("entities", "caption_entities"):
        entities = message.get(entity_key)
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            entity_type = str(entity.get("type", "")).strip()
            user = entity.get("user")
            if (
                entity_type == "text_mention"
                and isinstance(user, dict)
                and user.get("is_bot") is True
                and str(user.get("username", "")).strip().lstrip("@").lower() == safe_username
            ):
                return True
    return False


def _strip_bot_mention(text: str, bot_username: str) -> str:
    safe_username = str(bot_username or "").strip().lstrip("@")
    if not safe_username:
        return str(text or "").strip()
    stripped = re.sub(
        rf"(?<!\w)@{re.escape(safe_username)}(?!\w)",
        " ",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r"[ \t]+", " ", stripped)
    stripped = re.sub(r" *\n *", "\n", stripped)
    return stripped.strip()


def _telegram_reply_to_context(message: dict[str, Any], *, require_bot_author: bool = True) -> str:
    reply = message.get("reply_to_message")
    if not isinstance(reply, dict):
        return ""
    author = reply.get("from")
    if require_bot_author and isinstance(author, dict) and author.get("is_bot") is not True:
        return ""

    message_id = reply.get("message_id")
    if require_bot_author:
        lines = ["Telegram reply context: the user replied to one of OpenTulpa's earlier messages."]
    else:
        lines = ["Telegram quoted message context: the user mentioned OpenTulpa while replying to this message."]
    if isinstance(message_id, int) and message_id > 0:
        lines.append(f"- replied_message_id: {message_id}")

    text = str(reply.get("text") or reply.get("caption") or "").strip()
    if text:
        lines.append(f"- replied_message_text_or_caption: {text[:2000]}")

    photos = reply.get("photo")
    if isinstance(photos, list) and photos:
        chosen = None
        for item in photos:
            if isinstance(item, dict) and (
                chosen is None
                or int(item.get("file_size") or 0) >= int(chosen.get("file_size") or 0)
            ):
                chosen = item
        if isinstance(chosen, dict):
            file_unique_id = str(chosen.get("file_unique_id", "")).strip()
            width = chosen.get("width")
            height = chosen.get("height")
            details = ["type=photo"]
            if file_unique_id:
                details.append(f"file_unique_id={file_unique_id}")
            if isinstance(width, int) and isinstance(height, int):
                details.append(f"size={width}x{height}")
            lines.append(f"- replied_message_media: {' '.join(details)}")

    document = reply.get("document")
    if isinstance(document, dict):
        name = str(document.get("file_name", "")).strip()
        mime_type = str(document.get("mime_type", "")).strip()
        file_unique_id = str(document.get("file_unique_id", "")).strip()
        details = ["type=document"]
        if name:
            details.append(f"name={name}")
        if mime_type:
            details.append(f"mime_type={mime_type}")
        if file_unique_id:
            details.append(f"file_unique_id={file_unique_id}")
        lines.append(f"- replied_message_media: {' '.join(details)}")

    if len(lines) == 1:
        lines.append("- replied_message_text_or_caption: unavailable")
    lines.append(
        "Use this as context for the user's current message; do not quote metadata verbatim."
    )
    return "\n".join(lines)


def _inject_telegram_reply_context(text: str, reply_context: str) -> str:
    clean_text = str(text or "").strip()
    clean_context = str(reply_context or "").strip()
    if not clean_context or not clean_text:
        return clean_text
    return f"{clean_context}\n\nCurrent user message:\n{clean_text}"


@asynccontextmanager
async def _active_turn_context(shutdown_drain: Any | None):
    if shutdown_drain is None or not hasattr(shutdown_drain, "active_turn"):
        yield
        return
    async with shutdown_drain.active_turn():
        yield


def support_bot_commands() -> list[dict[str, str]]:
    return [
        {"command": "support_customers", "description": "List customer tenants for support"},
        {"command": "support_bind", "description": "Act as a customer tenant"},
        {"command": "support_unbind", "description": "Clear support tenant binding"},
        {"command": "support_whoami", "description": "Show current support binding"},
    ]


def _is_support_command(command_name: str) -> bool:
    return str(command_name or "").strip().lower().startswith("/support_")


def _is_support_user(
    *,
    user_id: int,
    username: str | None,
    support_user_ids_csv: str | None,
    support_usernames_csv: str | None,
) -> bool:
    support_ids = parse_csv_set(support_user_ids_csv)
    support_usernames = parse_csv_set(support_usernames_csv, normalize_username=True)
    if not support_ids and not support_usernames:
        return False
    if str(user_id) in support_ids:
        return True
    return bool(username and username.lower() in support_usernames)


def _maybe_auto_bind_allowed_username(
    *,
    owner_customer_id: str | None,
    allowed_usernames_csv: str | None,
    username: str | None,
    user_id: int,
    bind_telegram_customer_id: Callable[..., Any] | None,
) -> None:
    owner_id = str(owner_customer_id or "").strip()
    if not owner_id or owner_id.startswith("telegram_") or bind_telegram_customer_id is None:
        return
    allowed_usernames = parse_csv_set(allowed_usernames_csv, normalize_username=True)
    safe_username = str(username or "").strip().removeprefix("@").lower()
    if len(allowed_usernames) != 1 or safe_username not in allowed_usernames:
        return
    try:
        bind_telegram_customer_id(user_id=owner_id, telegram_user_id=int(user_id))
    except ValueError as exc:
        logger.warning(
            "Telegram username bootstrap bind skipped for customer_id=%s username=%s user_id=%s: %s",
            owner_id,
            safe_username,
            user_id,
            exc,
        )


def _support_bindings(state: dict[str, Any]) -> dict[str, Any]:
    bindings = state.get("support_bindings")
    if not isinstance(bindings, dict):
        bindings = {}
        state["support_bindings"] = bindings
    return bindings


def _append_support_audit(state: dict[str, Any], event: dict[str, Any]) -> None:
    audit = state.get("support_audit")
    if not isinstance(audit, list):
        audit = []
    audit.append(event)
    state["support_audit"] = audit[-500:]


def _support_binding_for_chat(state: dict[str, Any], chat_id: int | str) -> dict[str, Any] | None:
    binding = _support_bindings(state).get(str(chat_id))
    return binding if isinstance(binding, dict) else None


def _current_support_binding(chat_id: int | str) -> dict[str, Any] | None:
    bindings = STATE_STORE.load().get("support_bindings", {})
    if not isinstance(bindings, dict):
        return None
    binding = bindings.get(str(chat_id))
    return binding if isinstance(binding, dict) else None


def _reset_support_thread_context(
    state: dict[str, Any],
    *,
    chat_id: int,
    user_id: int,
    username: str | None,
    customer_id: str,
) -> tuple[str, str]:
    now_utc_iso = datetime.now(UTC).isoformat()
    bindings = _support_bindings(state)
    chat_key = str(chat_id)
    binding = bindings.get(chat_key)
    if not isinstance(binding, dict):
        binding = {}
    thread_id = new_short_id("chat")
    wake_thread_id = new_short_id("wake")
    by_customer = binding.get("thread_id_by_customer")
    if not isinstance(by_customer, dict):
        by_customer = {}
    wake_by_customer = binding.get("wake_thread_id_by_customer")
    if not isinstance(wake_by_customer, dict):
        wake_by_customer = {}
    by_customer[customer_id] = thread_id
    wake_by_customer[customer_id] = wake_thread_id
    bindings[chat_key] = {
        **binding,
        "support_user_id": int(user_id),
        "support_username": username or "",
        "bound_customer_id": customer_id,
        "thread_id": thread_id,
        "wake_thread_id": wake_thread_id,
        "thread_id_by_customer": by_customer,
        "wake_thread_id_by_customer": wake_by_customer,
        "last_user_message_at": now_utc_iso,
        "last_assistant_message_at": None,
        "updated_at": now_utc_iso,
    }
    _append_support_audit(
        state,
        {
            "event": "support_thread_reset",
            "support_user_id": int(user_id),
            "support_username": username or "",
            "support_chat_id": int(chat_id),
            "bound_customer_id": customer_id,
            "thread_id": thread_id,
            "wake_thread_id": wake_thread_id,
            "created_at": now_utc_iso,
        },
    )
    return thread_id, customer_id


def _bind_support_customer(
    state: dict[str, Any],
    *,
    chat_id: int,
    user_id: int,
    username: str | None,
    customer_id: str,
) -> dict[str, Any]:
    now_utc_iso = datetime.now(UTC).isoformat()
    bindings = _support_bindings(state)
    chat_key = str(chat_id)
    binding = bindings.get(chat_key)
    if not isinstance(binding, dict):
        binding = {}
    by_customer = binding.get("thread_id_by_customer")
    if not isinstance(by_customer, dict):
        by_customer = {}
    wake_by_customer = binding.get("wake_thread_id_by_customer")
    if not isinstance(wake_by_customer, dict):
        wake_by_customer = {}
    thread_id = _clean_thread_id(by_customer.get(customer_id)) or new_short_id("chat")
    wake_thread_id = _clean_thread_id(wake_by_customer.get(customer_id)) or new_short_id("wake")
    by_customer[customer_id] = thread_id
    wake_by_customer[customer_id] = wake_thread_id
    next_binding = {
        **binding,
        "support_user_id": int(user_id),
        "support_username": username or "",
        "bound_customer_id": customer_id,
        "thread_id": thread_id,
        "wake_thread_id": wake_thread_id,
        "thread_id_by_customer": by_customer,
        "wake_thread_id_by_customer": wake_by_customer,
        "last_user_message_at": now_utc_iso,
        "last_assistant_message_at": binding.get("last_assistant_message_at"),
        "bound_at": binding.get("bound_at") or now_utc_iso,
        "updated_at": now_utc_iso,
    }
    bindings[chat_key] = next_binding
    _append_support_audit(
        state,
        {
            "event": "support_bound",
            "support_user_id": int(user_id),
            "support_username": username or "",
            "support_chat_id": int(chat_id),
            "bound_customer_id": customer_id,
            "thread_id": thread_id,
            "created_at": now_utc_iso,
        },
    )
    return next_binding


def _unbind_support_customer(
    state: dict[str, Any],
    *,
    chat_id: int,
    user_id: int,
    username: str | None,
) -> dict[str, Any] | None:
    now_utc_iso = datetime.now(UTC).isoformat()
    bindings = _support_bindings(state)
    chat_key = str(chat_id)
    binding = bindings.get(chat_key)
    if not isinstance(binding, dict):
        return None
    previous = dict(binding)
    binding["bound_customer_id"] = ""
    binding["thread_id"] = ""
    binding["wake_thread_id"] = ""
    binding["support_user_id"] = int(user_id)
    binding["support_username"] = username or ""
    binding["updated_at"] = now_utc_iso
    bindings[chat_key] = binding
    _append_support_audit(
        state,
        {
            "event": "support_unbound",
            "support_user_id": int(user_id),
            "support_username": username or "",
            "support_chat_id": int(chat_id),
            "previous_customer_id": str(previous.get("bound_customer_id", "") or ""),
            "created_at": now_utc_iso,
        },
    )
    return previous


def _touch_support_turn(
    state: dict[str, Any],
    *,
    chat_id: int,
    user_id: int,
    username: str | None,
    event: str,
    outcome: str = "",
) -> None:
    now_utc_iso = datetime.now(UTC).isoformat()
    binding = _support_binding_for_chat(state, chat_id)
    customer_id = str((binding or {}).get("bound_customer_id", "") or "")
    thread_id = str((binding or {}).get("thread_id", "") or "")
    if binding is not None:
        binding["support_user_id"] = int(user_id)
        binding["support_username"] = username or ""
        binding["last_user_message_at"] = now_utc_iso
        binding["updated_at"] = now_utc_iso
        _support_bindings(state)[str(chat_id)] = binding
    _append_support_audit(
        state,
        {
            "event": event,
            "support_user_id": int(user_id),
            "support_username": username or "",
            "support_chat_id": int(chat_id),
            "bound_customer_id": customer_id,
            "thread_id": thread_id,
            "outcome": outcome,
            "created_at": now_utc_iso,
        },
    )


async def _maybe_configure_support_commands_for_chat(
    *,
    bot_token: str | None,
    chat_id: int,
) -> None:
    if not str(bot_token or "").strip():
        return

    def _mark_if_needed(state: dict[str, Any]) -> bool:
        configured = state.get("support_command_chats")
        if not isinstance(configured, dict):
            configured = {}
        key = str(chat_id)
        if key in configured:
            state["support_command_chats"] = configured
            return False
        configured[key] = datetime.now(UTC).isoformat()
        state["support_command_chats"] = configured
        return True

    should_configure = bool(STATE_STORE.update(_mark_if_needed))
    if not should_configure:
        return
    client = TelegramClient(str(bot_token))
    try:
        setter = getattr(client, "set_my_commands", None)
        if callable(setter):
            await setter(
                commands=support_bot_commands(), scope={"type": "chat", "chat_id": int(chat_id)}
            )
    finally:
        if hasattr(client, "aclose"):
            with suppress(Exception):
                await client.aclose()


async def _send_debug_logs_file(*, chat_id: int, bot_token: str | None) -> str | None:
    if not str(bot_token or "").strip():
        return "Telegram file sending is unavailable because the bot token is not configured."
    archive = build_debug_logs_archive_bytes(lookback_days=DEFAULT_DEBUG_LOG_LOOKBACK_DAYS)
    if archive is None:
        return "Debug log file is not available yet."
    filename, raw_bytes = archive
    client = TelegramClient(str(bot_token))
    try:
        sent = await client.send_file(
            chat_id=chat_id,
            filename=filename,
            raw_bytes=raw_bytes,
            kind="document",
            mime_type="application/zip",
            caption=f"OpenTulpa debug logs dump (last {DEFAULT_DEBUG_LOG_LOOKBACK_DAYS} days)",
            parse_mode="HTML",
        )
    finally:
        if hasattr(client, "aclose"):
            with suppress(Exception):
                await client.aclose()
    if not sent:
        return "I couldn't send the debug log files right now."
    return None


def _customer_listing_items(
    customer_listing: Callable[[], list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    if customer_listing is None:
        return []
    try:
        items = customer_listing()
    except Exception:
        return []
    return [item for item in items if isinstance(item, dict)]


def _format_support_customer_line(index: int, item: dict[str, Any]) -> str:
    customer_id = str(item.get("customer_id", "") or "").strip()
    owner_username = str(item.get("owner_username", "") or "").strip()
    owner_chat_id = str(item.get("owner_chat_id", "") or "").strip()
    owner = owner_username or owner_chat_id or "unknown"
    if owner_username:
        owner = f"@{owner_username.lstrip('@')}"
        if owner_chat_id:
            owner = f"{owner} chat={owner_chat_id}"
    business = "connected" if bool(item.get("telegram_business_connected", False)) else "none"
    composio = "connected" if bool(item.get("composio_connected", False)) else "none"
    workflow_count = int(item.get("workflow_count") or 0)
    file_count = int(item.get("file_count") or 0)
    last_activity = str(item.get("last_activity", "") or "unknown")
    return (
        f"{index}. {customer_id} | owner={owner} | "
        f"business={business} | composio={composio} | "
        f"workflows={workflow_count} | files={file_count} | last={last_activity}"
    )


def _format_support_customers(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No customer tenants are known yet."
    lines = ["Support customers:"]
    lines.extend(
        _format_support_customer_line(index, item) for index, item in enumerate(items, start=1)
    )
    lines.append("")
    lines.append("Bind with /support_bind <number> or /support_bind <customer_id>.")
    return "\n".join(lines)


def _resolve_support_customer_arg(
    *,
    raw_arg: str,
    items: list[dict[str, Any]],
) -> str:
    arg = str(raw_arg or "").strip()
    if not arg:
        return ""
    if arg.isdigit():
        index = int(arg)
        if 1 <= index <= len(items):
            return str(items[index - 1].get("customer_id", "") or "").strip()
    known = {str(item.get("customer_id", "") or "").strip() for item in items}
    return arg if arg in known else ""


def _format_support_whoami(
    *,
    user_id: int,
    username: str | None,
    chat_id: int,
    binding: dict[str, Any] | None,
) -> str:
    bound_customer = str((binding or {}).get("bound_customer_id", "") or "").strip()
    thread_id = str((binding or {}).get("thread_id", "") or "").strip()
    wake_thread_id = str((binding or {}).get("wake_thread_id", "") or "").strip()
    username_text = f"@{username}" if username else "unknown"
    if not bound_customer:
        return (
            "Support operator: active\n"
            f"User: {user_id} ({username_text})\n"
            f"Chat: {chat_id}\n"
            "Bound customer: none\n"
            "Use /support_customers then /support_bind <number>."
        )
    return (
        "Support operator: active\n"
        f"User: {user_id} ({username_text})\n"
        f"Chat: {chat_id}\n"
        f"Bound customer: {bound_customer}\n"
        f"Support thread: {thread_id}\n"
        f"Wake thread: {wake_thread_id}"
    )


def _support_command_arg(text: str) -> str:
    parts = str(text or "").strip().split(None, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _resolve_turn_mode_for_thread(
    *,
    workflow_setup_status: Callable[..., dict[str, Any]] | None,
    customer_id: str,
    thread_id: str,
) -> str:
    if workflow_setup_status is None:
        return "interactive"
    try:
        state = workflow_setup_status(customer_id=customer_id, thread_id=thread_id)
    except Exception:
        logger.exception(
            "Failed to resolve Telegram workflow setup status (customer_id=%s, thread_id=%s)",
            customer_id,
            thread_id,
        )
        return "interactive"
    if str((state or {}).get("status", "") or "").strip().lower() == "active":
        return "workflow_setup"
    return "interactive"


def _apply_workflow_setup_after_reply(
    *,
    workflow_setup_after_reply: Callable[..., dict[str, Any]] | None,
    customer_id: str,
    thread_id: str,
    reply_text: str | None,
) -> None:
    if workflow_setup_after_reply is None or not str(reply_text or "").strip():
        return
    try:
        workflow_setup_after_reply(
            customer_id=customer_id,
            thread_id=thread_id,
            reply_text=str(reply_text or ""),
        )
    except Exception:
        logger.exception(
            "Failed to apply Telegram workflow setup reply hook (customer_id=%s, thread_id=%s)",
            customer_id,
            thread_id,
        )


async def _handle_support_command(
    *,
    command_name: str,
    ctx: TelegramContext,
    customer_listing: Callable[[], list[dict[str, Any]]] | None,
) -> str:
    if command_name == "/support_customers":
        return _format_support_customers(_customer_listing_items(customer_listing))
    if command_name == "/support_bind":
        items = _customer_listing_items(customer_listing)
        customer_id = _resolve_support_customer_arg(
            raw_arg=_support_command_arg(ctx.text),
            items=items,
        )
        if not customer_id:
            return "Customer not found. Run /support_customers and bind by number or exact customer_id."
        binding = STATE_STORE.update(
            lambda state: _bind_support_customer(
                state,
                chat_id=ctx.chat_id,
                user_id=ctx.user_id,
                username=ctx.username,
                customer_id=customer_id,
            )
        )
        return (
            f"Support bound to {customer_id}.\n"
            f"Support thread: {binding.get('thread_id')}\n"
            "Normal messages in this chat now act inside that customer tenant."
        )
    if command_name == "/support_unbind":
        previous = STATE_STORE.update(
            lambda state: _unbind_support_customer(
                state,
                chat_id=ctx.chat_id,
                user_id=ctx.user_id,
                username=ctx.username,
            )
        )
        previous_customer = str((previous or {}).get("bound_customer_id", "") or "").strip()
        if not previous_customer:
            return "Support chat was not bound to a customer."
        return f"Support unbound from {previous_customer}."
    if command_name == "/support_whoami":
        binding = _current_support_binding(ctx.chat_id)
        return _format_support_whoami(
            user_id=ctx.user_id,
            username=ctx.username,
            chat_id=ctx.chat_id,
            binding=binding if isinstance(binding, dict) else None,
        )
    return "Unknown support command."


async def relay_task_event_via_main_agent(
    *,
    customer_id: str,
    task_id: str,
    event_type: str,
    payload: dict[str, Any],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    return await _relay_task_event_via_main_agent(
        customer_id=customer_id,
        task_id=task_id,
        event_type=event_type,
        payload=payload,
        state_store=STATE_STORE,
        find_session_slots=find_session_slots_for_customer_id,
        agent_runtime=agent_runtime,
    )


async def relay_event_via_main_agent(
    *,
    customer_id: str,
    event_label: str,
    payload: dict[str, Any],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    return await _relay_event_via_main_agent(
        customer_id=customer_id,
        event_label=event_label,
        payload=payload,
        state_store=STATE_STORE,
        find_session_slots=find_session_slots_for_customer_id,
        agent_runtime=agent_runtime,
    )


async def _ingest_attachments_with_typing(
    *,
    attachments: list[Any],
    bot_token: str,
    file_vault: FileVaultService | None,
    memory: Any | None,
    agent_runtime: Any | None,
    customer_id: str,
    chat_id: int,
    caption: str | None,
) -> list[dict[str, Any]]:
    if not attachments or not bot_token or file_vault is None:
        return []
    typing_stop = asyncio.Event()
    typing_client = TelegramClient(str(bot_token))
    typing_task = asyncio.create_task(
        _emit_typing_until_done(
            client=typing_client,
            chat_id=chat_id,
            stop_event=typing_stop,
        )
    )
    try:
        return await ingest_attachments(
            attachments=attachments,
            bot_token=bot_token,
            file_vault=file_vault,
            memory=memory,
            agent_runtime=agent_runtime,
            customer_id=customer_id,
            chat_id=chat_id,
            caption=caption,
        )
    finally:
        typing_stop.set()
        with suppress(Exception):
            await typing_task
        if hasattr(typing_client, "aclose"):
            with suppress(Exception):
                await typing_client.aclose()


def _build_effective_telegram_text(
    *,
    user_text: str,
    attachments: list[Any],
    ingested_files: list[dict[str, Any]],
) -> tuple[str, str | None]:
    voice_transcripts = [
        str(item.get("voice_transcript", "")).strip()
        for item in ingested_files
        if str(item.get("kind", "")).strip() == "voice"
    ]
    non_voice_files = [
        item for item in ingested_files if str(item.get("kind", "")).strip() != "voice"
    ]
    effective_text = _inject_voice_message_context(user_text, voice_transcripts)
    context_blob = build_uploaded_files_context(
        non_voice_files,
        include_unclear_intent_guidance=not bool(effective_text.strip()),
    )
    if context_blob:
        if effective_text:
            effective_text = f"{effective_text}\n\n{context_blob}"
        else:
            effective_text = (
                "User uploaded one or more files without extra text.\n"
                "Use the internal file context below, but do not echo raw metadata. "
                "If recent conversation clearly says what to do with these files, do that. "
                "If intent is unclear, ask what the user wants done with them and mention options "
                "such as adding to user context, summarizing, analyzing, attaching to an intake workflow, "
                "using later, or archiving. Do not infer intent from filenames or content.\n\n"
                f"{context_blob}"
            )
    if effective_text:
        return effective_text, None
    has_voice = any(str(getattr(item, "kind", "")).strip() == "voice" for item in attachments)
    if has_voice:
        return "", (
            "I received your voice message but couldn't transcribe it. "
            "Please resend a shorter/clearer voice note or send text."
        )
    return "", None


def _drop_upload_without_text_guidance(fragment: str) -> str:
    text = str(fragment or "").strip()
    if not text.startswith(_UPLOAD_WITHOUT_TEXT_PREFIX):
        return text
    marker_index = text.find(_UPLOAD_CONTEXT_MARKER)
    if marker_index < 0:
        return text
    context = text[marker_index:].strip()
    lines = [
        line
        for line in context.splitlines()
        if not any(snippet in line for snippet in _UNCLEAR_UPLOAD_GUIDANCE_SNIPPETS)
    ]
    return "\n".join(lines).strip()


async def _send_direct_telegram_reply(
    *,
    bot_token: str,
    chat_id: int,
    text: str,
) -> bool:
    client = TelegramClient(str(bot_token))
    try:
        return bool(await client.send_message(chat_id=chat_id, text=text, parse_mode="HTML"))
    finally:
        if hasattr(client, "aclose"):
            with suppress(Exception):
                await client.aclose()


async def _materialize_interactive_submission(
    *,
    session: InteractiveSession,
    submission: Any,
    text: str,
    reply_context: str,
    caption: str | None,
    attachments: list[Any],
    bot_token: str,
    file_vault: FileVaultService | None,
    memory: Any | None,
    agent_runtime: Any | None,
    customer_id: str,
    chat_id: int,
) -> None:
    fragment = ""
    direct_reply = None
    try:
        ingested_files = await _ingest_attachments_with_typing(
            attachments=attachments,
            bot_token=bot_token,
            file_vault=file_vault,
            memory=memory,
            agent_runtime=agent_runtime,
            customer_id=customer_id,
            chat_id=chat_id,
            caption=caption,
        )
        fragment, direct_reply = _build_effective_telegram_text(
            user_text=text,
            attachments=attachments,
            ingested_files=ingested_files,
        )
        fragment = _inject_telegram_reply_context(fragment, reply_context)
    except Exception as exc:
        logger.exception(
            "Telegram interactive materialization failed (chat_id=%s, thread_id=%s): %s",
            chat_id,
            session.thread_id,
            exc,
        )
        direct_reply = _format_agent_error_for_user(exc)
    await session.publish(
        submission,
        fragment=fragment,
        direct_reply=direct_reply,
    )


async def _run_interactive_session(
    *,
    session: InteractiveSession,
    bot_token: str,
    agent_runtime: Any,
    workflow_setup_status: Callable[..., dict[str, Any]] | None = None,
    workflow_setup_after_reply: Callable[..., dict[str, Any]] | None = None,
    shutdown_drain: Any | None = None,
) -> None:
    while True:
        ready = await session.wait_for_ready_head()
        if not ready:
            if await session.finish_runner_if_idle():
                return
            continue
        ready_items = await session.consume_ready_batch()
        fragments = [
            str(item.fragment).strip()
            for item in ready_items
            if isinstance(item, InteractiveSubmissionResult) and str(item.fragment or "").strip()
        ]
        direct_replies = [
            str(item.direct_reply).strip()
            for item in ready_items
            if isinstance(item, InteractiveSubmissionResult)
            and str(item.direct_reply or "").strip()
        ]
        for reply_text in direct_replies:
            sent = await _send_direct_telegram_reply(
                bot_token=bot_token,
                chat_id=session.chat_id,
                text=reply_text,
            )
            if sent:
                STATE_STORE.touch_assistant_message(session.chat_id)
        if not fragments:
            if await session.finish_runner_if_idle():
                return
            continue
        if len(fragments) > 1 and any(
            not fragment.startswith(_UPLOAD_WITHOUT_TEXT_PREFIX) for fragment in fragments
        ):
            fragments = [_drop_upload_without_text_guidance(fragment) for fragment in fragments]
        effective_text = "\n\n".join(fragments).strip()
        if not effective_text:
            if await session.finish_runner_if_idle():
                return
            continue

        async def _send_interactive_owner_update(message: str) -> dict[str, Any]:
            sent = await _send_direct_telegram_reply(
                bot_token=bot_token,
                chat_id=session.chat_id,
                text=message,
            )
            if sent:
                STATE_STORE.touch_assistant_message(session.chat_id)
            return {"sent": bool(sent)}

        try:
            if hasattr(agent_runtime, "register_interactive_session"):
                await agent_runtime.register_interactive_session(
                    thread_id=session.thread_id,
                    session=session,
                )
            if hasattr(agent_runtime, "register_interactive_update_sender"):
                await agent_runtime.register_interactive_update_sender(
                    thread_id=session.thread_id,
                    sender=_send_interactive_owner_update,
                )
            turn_mode = _resolve_turn_mode_for_thread(
                workflow_setup_status=workflow_setup_status,
                customer_id=session.customer_id,
                thread_id=session.thread_id,
            )

            def _workflow_setup_late_reply(reply_text: str) -> None:
                _apply_workflow_setup_after_reply(
                    workflow_setup_after_reply=workflow_setup_after_reply,
                    customer_id=session.customer_id,
                    thread_id=session.thread_id,
                    reply_text=reply_text,
                )
                STATE_STORE.touch_assistant_message(session.chat_id)

            async with _active_turn_context(shutdown_drain):
                final, suppressed = await stream_langgraph_reply_to_telegram(
                    agent_runtime=agent_runtime,
                    thread_id=session.thread_id,
                    customer_id=session.customer_id,
                    text=effective_text,
                    bot_token=bot_token,
                    chat_id=session.chat_id,
                    turn_mode=turn_mode,
                    interactive_session=session,
                    final_reply_callback=(
                        _workflow_setup_late_reply if turn_mode == "workflow_setup" else None
                    ),
                )
                if turn_mode == "workflow_setup" and final and not suppressed:
                    _apply_workflow_setup_after_reply(
                        workflow_setup_after_reply=workflow_setup_after_reply,
                        customer_id=session.customer_id,
                        thread_id=session.thread_id,
                        reply_text=final,
                    )
                if final and not suppressed:
                    STATE_STORE.touch_assistant_message(session.chat_id)
                elif not suppressed:
                    debug_log(
                        hypothesis_id="telegram_chat",
                        location="interfaces/telegram/chat_service.py:_run_interactive_session",
                        message="fallback_no_final_reply",
                        data={"chat_id": session.chat_id, "thread_id": session.thread_id},
                    )
                    sent = await _send_direct_telegram_reply(
                        bot_token=bot_token,
                        chat_id=session.chat_id,
                        text=(
                            "I received your message but no final reply was available yet. "
                            "Ask again or use /status."
                        ),
                    )
                    if sent:
                        STATE_STORE.touch_assistant_message(session.chat_id)
        except ShutdownDrainingError:
            await _send_direct_telegram_reply(
                bot_token=bot_token,
                chat_id=session.chat_id,
                text="OpenTulpa is finishing a deploy. Please retry in a moment.",
            )
            return
        except Exception as exc:
            logger.exception(
                "Telegram interactive runner failed (chat_id=%s, thread_id=%s): %s",
                session.chat_id,
                session.thread_id,
                exc,
            )
            await _send_direct_telegram_reply(
                bot_token=bot_token,
                chat_id=session.chat_id,
                text=_format_agent_error_for_user(exc),
            )
            final = None
            suppressed = False
        finally:
            if hasattr(agent_runtime, "clear_interactive_update_sender"):
                await agent_runtime.clear_interactive_update_sender(
                    thread_id=session.thread_id,
                    sender=_send_interactive_owner_update,
                )
            if hasattr(agent_runtime, "clear_interactive_session"):
                await agent_runtime.clear_interactive_session(
                    thread_id=session.thread_id,
                    session=session,
                )
        if await session.finish_runner_if_idle():
            return


async def handle_telegram_text(
    *,
    body: dict[str, Any],
    bot_token: str | None = None,
    allowed_user_ids_csv: str | None = None,
    allowed_usernames_csv: str | None = None,
    support_user_ids_csv: str | None = None,
    support_usernames_csv: str | None = None,
    agent_runtime: Any | None = None,
    file_vault: FileVaultService | None = None,
    memory: Any | None = None,
    interactive_inbox: TelegramInteractiveInbox | None = None,
    support_customer_listing: Callable[[], list[dict[str, Any]]] | None = None,
    workflow_setup_status: Callable[..., dict[str, Any]] | None = None,
    workflow_setup_after_reply: Callable[..., dict[str, Any]] | None = None,
    resolve_customer_id: Callable[[str], str] | None = None,
    resolve_telegram_customer_id: Callable[[int], str] | None = None,
    owner_customer_id: str | None = None,
    bind_telegram_customer_id: Callable[..., Any] | None = None,
    shutdown_drain: Any | None = None,
) -> str | None:
    parsed = parse_telegram_update(body)
    if not parsed:
        return None
    chat_id, user_id, text = parsed
    if not chat_id or not user_id:
        return None

    message = body.get("message") or body.get("edited_message") or {}
    caption = str(message.get("caption", "")).strip() or None
    is_group_chat = _telegram_chat_is_group(message)
    bot_username = await _resolve_bot_username(bot_token) if is_group_chat else ""
    bot_was_mentioned = _telegram_message_mentions_bot(message, bot_username) if is_group_chat else False
    if is_group_chat and not bot_was_mentioned:
        return None
    if bot_was_mentioned:
        text = _strip_bot_mention(text or "", bot_username)
        if caption:
            caption = _strip_bot_mention(caption, bot_username) or None
    reply_context = _telegram_reply_to_context(message, require_bot_author=not bot_was_mentioned)
    attachments = extract_attachments(message)
    username = (message.get("from", {}) or {}).get("username")
    username = username.strip() or None if isinstance(username, str) else None
    ctx = TelegramContext(
        chat_id=chat_id,
        user_id=user_id,
        username=username,
        text=(text or "").strip(),
    )

    normal_allowed = is_user_allowed(
        user_id=ctx.user_id,
        username=ctx.username,
        allowed_user_ids_csv=allowed_user_ids_csv,
        allowed_usernames_csv=allowed_usernames_csv,
    )
    support_allowed = _is_support_user(
        user_id=ctx.user_id,
        username=ctx.username,
        support_user_ids_csv=support_user_ids_csv,
        support_usernames_csv=support_usernames_csv,
    )
    access = TelegramAccessDecision.from_flags(
        normal_allowed=normal_allowed,
        support_allowed=support_allowed,
    )
    if access.restricted_reply is not None:
        return access.restricted_reply
    if access.should_auto_bind_allowed_username:
        _maybe_auto_bind_allowed_username(
            owner_customer_id=owner_customer_id,
            allowed_usernames_csv=allowed_usernames_csv,
            username=ctx.username,
            user_id=ctx.user_id,
            bind_telegram_customer_id=bind_telegram_customer_id,
        )
    if access.should_configure_support_commands:
        await _maybe_configure_support_commands_for_chat(
            bot_token=bot_token,
            chat_id=ctx.chat_id,
        )

    def _ensure_admin(state: dict[str, Any]) -> Any:
        admin_user_id = state.get("admin_user_id")
        if admin_user_id is None:
            admin_user_id = ctx.user_id
            state["admin_user_id"] = admin_user_id
        return admin_user_id

    admin_user_id = STATE_STORE.update(_ensure_admin)
    _ = int(admin_user_id) == int(ctx.user_id)

    command_name = _telegram_command_name(ctx.text)
    command_route = TelegramCommandRoute.from_command(
        command_name=command_name,
        support_allowed=support_allowed,
        is_support_command=_is_support_command(command_name),
    )
    if command_route.kind == "restricted_support_command":
        return "This support command is restricted to configured support operators."
    if command_route.kind == "support_command":
        return await _handle_support_command(
            command_name=command_name,
            ctx=ctx,
            customer_listing=support_customer_listing,
        )
    if command_route.kind == "start_help":
        return _start_help_text()
    if command_route.kind == "status":
        agent_up = bool(agent_runtime and getattr(agent_runtime, "healthy", lambda: False)())
        return status_text(agent_up)
    if command_route.kind == "fresh":
        if support_allowed:
            binding = _current_support_binding(ctx.chat_id)
            bound_customer = str((binding or {}).get("bound_customer_id", "") or "").strip()
            if not bound_customer:
                return "Support mode is active. Use /support_customers and /support_bind before /fresh."
            thread_id, _ = STATE_STORE.update(
                lambda state: _reset_support_thread_context(
                    state,
                    chat_id=ctx.chat_id,
                    user_id=ctx.user_id,
                    username=ctx.username,
                    customer_id=bound_customer,
                )
            )
            if interactive_inbox is not None:
                await interactive_inbox.reset_chat(ctx.chat_id)
            return (
                "Started a fresh support chat context. "
                f"Bound customer: {bound_customer}. "
                f"New thread: {thread_id}. "
                "Customer owner chat history is unchanged."
            )
        thread_id, _ = STATE_STORE.update(
            lambda state: _reset_chat_session_context(
                state,
                chat_id=ctx.chat_id,
                user_id=ctx.user_id,
                username=ctx.username,
                resolve_customer_id=resolve_customer_id,
                resolve_telegram_customer_id=resolve_telegram_customer_id,
            )
        )
        if interactive_inbox is not None:
            await interactive_inbox.reset_chat(ctx.chat_id)
        return (
            "Started a fresh chat context. "
            f"New thread: {thread_id}. "
            "Your long-term memory is unchanged."
        )
    if command_route.kind == "debug_logs":
        return await _send_debug_logs_file(chat_id=ctx.chat_id, bot_token=bot_token)

    support_binding = None
    if support_allowed:
        support_binding = _current_support_binding(ctx.chat_id)
        bound_customer = str((support_binding or {}).get("bound_customer_id", "") or "").strip()
        if not bound_customer:
            STATE_STORE.update(
                lambda state: _touch_support_turn(
                    state,
                    chat_id=ctx.chat_id,
                    user_id=ctx.user_id,
                    username=ctx.username,
                    event="support_turn_rejected_unbound",
                    outcome="missing_binding",
                )
            )
            return "Support mode is active. Use /support_customers and /support_bind before chatting as a customer."

    if not get_openai_compatible_api_key_from_env():
        return missing_key_prompt()
    if agent_runtime is None:
        return "Agent runtime is unavailable. Restart OpenTulpa and try again."

    def _upsert_session(state: dict[str, Any]) -> tuple[str, str]:
        sessions = state.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
        slot = sessions.get(str(ctx.chat_id))
        if not isinstance(slot, dict):
            slot = {}
        thread_id = _clean_thread_id(slot.get("thread_id")) or f"chat-{ctx.chat_id}"
        wake_thread_id = _clean_thread_id(slot.get("wake_thread_id")) or None
        existing_customer_id = str(slot.get("customer_id", "")).strip()
        if existing_customer_id and resolve_customer_id is not None:
            customer_id = str(resolve_customer_id(existing_customer_id) or "").strip()
        elif existing_customer_id:
            customer_id = existing_customer_id
        elif resolve_telegram_customer_id is not None:
            customer_id = str(resolve_telegram_customer_id(ctx.user_id) or "").strip()
        else:
            customer_id = f"telegram_{ctx.user_id}"
        if not customer_id:
            customer_id = f"telegram_{ctx.user_id}"
        now_utc_iso = datetime.now(UTC).isoformat()
        sessions[str(ctx.chat_id)] = {
            "user_id": ctx.user_id,
            "username": ctx.username or "",
            "customer_id": customer_id,
            "thread_id": thread_id,
            "wake_thread_id": wake_thread_id,
            "role": "owner",
            "last_user_message_at": now_utc_iso,
            "last_assistant_message_at": slot.get("last_assistant_message_at"),
        }
        state["sessions"] = sessions
        return thread_id, customer_id

    if support_allowed:

        def _upsert_support_session(state: dict[str, Any]) -> tuple[str, str]:
            binding = _support_binding_for_chat(state, ctx.chat_id)
            if not isinstance(binding, dict):
                raise RuntimeError("support binding missing")
            bound_customer = str(binding.get("bound_customer_id", "") or "").strip()
            if not bound_customer:
                raise RuntimeError("support binding missing")
            now_utc_iso = datetime.now(UTC).isoformat()
            thread_id = _clean_thread_id(binding.get("thread_id")) or new_short_id("chat")
            wake_thread_id = _clean_thread_id(binding.get("wake_thread_id")) or new_short_id("wake")
            by_customer = binding.get("thread_id_by_customer")
            if not isinstance(by_customer, dict):
                by_customer = {}
            wake_by_customer = binding.get("wake_thread_id_by_customer")
            if not isinstance(wake_by_customer, dict):
                wake_by_customer = {}
            by_customer[bound_customer] = thread_id
            wake_by_customer[bound_customer] = wake_thread_id
            binding.update(
                {
                    "support_user_id": ctx.user_id,
                    "support_username": ctx.username or "",
                    "thread_id": thread_id,
                    "wake_thread_id": wake_thread_id,
                    "thread_id_by_customer": by_customer,
                    "wake_thread_id_by_customer": wake_by_customer,
                    "last_user_message_at": now_utc_iso,
                    "updated_at": now_utc_iso,
                }
            )
            _support_bindings(state)[str(ctx.chat_id)] = binding
            _append_support_audit(
                state,
                {
                    "event": "support_turn_started",
                    "support_user_id": ctx.user_id,
                    "support_username": ctx.username or "",
                    "support_chat_id": ctx.chat_id,
                    "bound_customer_id": bound_customer,
                    "thread_id": thread_id,
                    "created_at": now_utc_iso,
                },
            )
            return thread_id, bound_customer

        thread_id, customer_id = STATE_STORE.update(_upsert_support_session)
    else:
        thread_id, customer_id = STATE_STORE.update(_upsert_session)

    if interactive_inbox is not None and bot_token:
        if attachments and file_vault is None:
            return "I received your file, but file storage is not configured."
        session, submission, became_runner = await interactive_inbox.submit(
            chat_id=ctx.chat_id,
            customer_id=customer_id,
            thread_id=thread_id,
        )
        asyncio.create_task(
            _materialize_interactive_submission(
                session=session,
                submission=submission,
                text=ctx.text,
                reply_context=reply_context,
                caption=caption,
                attachments=attachments,
                bot_token=bot_token,
                file_vault=file_vault,
                memory=memory,
                agent_runtime=agent_runtime,
                customer_id=customer_id,
                chat_id=ctx.chat_id,
            )
        )
        if not became_runner:
            return None
        try:
            await _run_interactive_session(
                session=session,
                bot_token=bot_token,
                agent_runtime=agent_runtime,
                workflow_setup_status=workflow_setup_status,
                workflow_setup_after_reply=workflow_setup_after_reply,
                shutdown_drain=shutdown_drain,
            )
        finally:
            await interactive_inbox.prune_if_idle(session)
        return None

    ingested_files = await _ingest_attachments_with_typing(
        attachments=attachments,
        bot_token=str(bot_token or ""),
        file_vault=file_vault,
        memory=memory,
        agent_runtime=agent_runtime,
        customer_id=customer_id,
        chat_id=ctx.chat_id,
        caption=caption,
    )

    if attachments and not ctx.text and not ingested_files:
        if agent_runtime is None:
            return "I received your file, but agent runtime is unavailable right now."
        if file_vault is None:
            return "I received your file, but file storage is not configured."

    effective_text, direct_reply = _build_effective_telegram_text(
        user_text=ctx.text,
        attachments=attachments,
        ingested_files=ingested_files,
    )
    effective_text = _inject_telegram_reply_context(effective_text, reply_context)
    if direct_reply:
        return direct_reply
    if not effective_text:
        return None

    turn_mode = _resolve_turn_mode_for_thread(
        workflow_setup_status=workflow_setup_status,
        customer_id=customer_id,
        thread_id=thread_id,
    )
    if bot_token:
        try:

            def _workflow_setup_late_reply(reply_text: str) -> None:
                _apply_workflow_setup_after_reply(
                    workflow_setup_after_reply=workflow_setup_after_reply,
                    customer_id=customer_id,
                    thread_id=thread_id,
                    reply_text=reply_text,
                )
                STATE_STORE.touch_assistant_message(ctx.chat_id)

            async with _active_turn_context(shutdown_drain):
                final, suppressed = await stream_langgraph_reply_to_telegram(
                    agent_runtime=agent_runtime,
                    thread_id=thread_id,
                    customer_id=customer_id,
                    text=effective_text,
                    bot_token=bot_token,
                    chat_id=ctx.chat_id,
                    turn_mode=turn_mode,
                    final_reply_callback=(
                        _workflow_setup_late_reply if turn_mode == "workflow_setup" else None
                    ),
                )
                if suppressed:
                    return None
                if final:
                    if turn_mode == "workflow_setup":
                        _apply_workflow_setup_after_reply(
                            workflow_setup_after_reply=workflow_setup_after_reply,
                            customer_id=customer_id,
                            thread_id=thread_id,
                            reply_text=final,
                        )
                    STATE_STORE.touch_assistant_message(ctx.chat_id)
                    return None
                debug_log(
                    hypothesis_id="telegram_chat",
                    location="interfaces/telegram/chat_service.py:handle_telegram_text",
                    message="fallback_no_final_reply",
                    data={"chat_id": ctx.chat_id, "thread_id": thread_id},
                )
                return "I received your message but no final reply was available yet. Ask again or use /status."
        except ShutdownDrainingError:
            return "OpenTulpa is finishing a deploy. Please retry in a moment."
        except Exception as exc:
            logger.exception(
                "Telegram streaming reply failed (chat_id=%s, thread_id=%s): %s",
                ctx.chat_id,
                thread_id,
                exc,
            )
            return _format_agent_error_for_user(exc)

    try:
        async with _active_turn_context(shutdown_drain):
            response = await agent_runtime.ainvoke_text(
                thread_id=thread_id,
                customer_id=customer_id,
                text=effective_text,
                turn_mode=turn_mode,
            )
            if turn_mode == "workflow_setup":
                _apply_workflow_setup_after_reply(
                    workflow_setup_after_reply=workflow_setup_after_reply,
                    customer_id=customer_id,
                    thread_id=thread_id,
                    reply_text=str(response or ""),
                )
            return str(response) if response is not None else None
    except ShutdownDrainingError:
        return "OpenTulpa is finishing a deploy. Please retry in a moment."
    except Exception as exc:
        logger.exception(
            "Telegram non-streaming reply failed (chat_id=%s, thread_id=%s): %s",
            ctx.chat_id,
            thread_id,
            exc,
        )
        return _format_agent_error_for_user(exc)


class TelegramChatService:
    """Telegram chat orchestration service with injected dependencies."""

    def __init__(
        self,
        *,
        bot_token: str,
        file_vault: FileVaultService | None = None,
        memory: Any | None = None,
        support_customer_listing: Callable[[], list[dict[str, Any]]] | None = None,
        workflow_setup_status: Callable[..., dict[str, Any]] | None = None,
        workflow_setup_after_reply: Callable[..., dict[str, Any]] | None = None,
        resolve_customer_id: Callable[[str], str] | None = None,
        resolve_telegram_customer_id: Callable[[int], str] | None = None,
        owner_customer_id: str | None = None,
        bind_telegram_customer_id: Callable[..., Any] | None = None,
        alias_user_ids: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.bot_token = str(bot_token or "").strip()
        self.file_vault = file_vault
        self.memory = memory
        self.support_customer_listing = support_customer_listing
        self.workflow_setup_status = workflow_setup_status
        self.workflow_setup_after_reply = workflow_setup_after_reply
        self.resolve_customer_id = resolve_customer_id
        self.resolve_telegram_customer_id = resolve_telegram_customer_id
        self.owner_customer_id = str(owner_customer_id or "").strip()
        self.bind_telegram_customer_id = bind_telegram_customer_id
        self.alias_user_ids = alias_user_ids
        self._interactive_inbox = TelegramInteractiveInbox()

    def find_session_slots(self, customer_id: str) -> list[dict[str, Any]]:
        aliases = (
            self.alias_user_ids(customer_id)
            if self.alias_user_ids is not None
            else [customer_id]
        )
        slots: list[dict[str, Any]] = []
        seen: set[str] = set()
        for alias in aliases:
            for slot in find_session_slots_for_customer_id(alias):
                key = str(slot.get("chat_id", ""))
                if key in seen:
                    continue
                seen.add(key)
                slots.append(slot)
        return slots

    def get_session_slot(self, chat_id: int) -> dict[str, Any] | None:
        return get_session_slot_for_chat_id(chat_id)

    def touch_assistant_message(self, chat_id: int) -> None:
        STATE_STORE.touch_assistant_message(chat_id)

    def list_owner_customer_summaries(self) -> list[dict[str, Any]]:
        return STATE_STORE.list_owner_customer_summaries()

    async def relay_task_event(
        self,
        *,
        customer_id: str,
        task_id: str,
        event_type: str,
        payload: dict[str, Any],
        agent_runtime: Any | None = None,
    ) -> list[dict[str, Any]]:
        return await _relay_task_event_via_main_agent(
            customer_id=customer_id,
            task_id=task_id,
            event_type=event_type,
            payload=payload,
            state_store=STATE_STORE,
            find_session_slots=self.find_session_slots,
            agent_runtime=agent_runtime,
        )

    async def relay_event(
        self,
        *,
        customer_id: str,
        event_label: str,
        payload: dict[str, Any],
        agent_runtime: Any | None = None,
    ) -> list[dict[str, Any]]:
        return await _relay_event_via_main_agent(
            customer_id=customer_id,
            event_label=event_label,
            payload=payload,
            state_store=STATE_STORE,
            find_session_slots=self.find_session_slots,
            agent_runtime=agent_runtime,
        )

    async def handle_update(
        self,
        *,
        body: dict[str, Any],
        allowed_user_ids_csv: str | None = None,
        allowed_usernames_csv: str | None = None,
        support_user_ids_csv: str | None = None,
        support_usernames_csv: str | None = None,
        agent_runtime: Any | None = None,
        shutdown_drain: Any | None = None,
    ) -> str | None:
        return await handle_telegram_text(
            body=body,
            bot_token=self.bot_token,
            allowed_user_ids_csv=allowed_user_ids_csv,
            allowed_usernames_csv=allowed_usernames_csv,
            support_user_ids_csv=support_user_ids_csv,
            support_usernames_csv=support_usernames_csv,
            agent_runtime=agent_runtime,
            file_vault=self.file_vault,
            memory=self.memory,
            interactive_inbox=self._interactive_inbox,
            support_customer_listing=self.support_customer_listing,
            workflow_setup_status=self.workflow_setup_status,
            workflow_setup_after_reply=self.workflow_setup_after_reply,
            resolve_customer_id=self.resolve_customer_id,
            resolve_telegram_customer_id=self.resolve_telegram_customer_id,
            owner_customer_id=self.owner_customer_id,
            bind_telegram_customer_id=self.bind_telegram_customer_id,
            shutdown_drain=shutdown_drain,
        )
