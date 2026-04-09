"""Telegram chat bridge orchestration service."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.config import get_openai_compatible_api_key_from_env
from opentulpa.core.debug_logs import read_debug_log_bytes
from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.attachments import (
    build_uploaded_files_context,
    extract_attachments,
    ingest_attachments,
)
from opentulpa.interfaces.telegram.client import TelegramClient, parse_telegram_update
from opentulpa.interfaces.telegram.constants import STATE_PATH
from opentulpa.interfaces.telegram.env_management import (
    missing_key_prompt,
    status_text,
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
from opentulpa.interfaces.telegram.security import is_user_allowed
from opentulpa.interfaces.telegram.state_store import TelegramStateStore

STATE_STORE = TelegramStateStore(STATE_PATH)
logger = logging.getLogger(__name__)


def _clean_thread_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def find_session_slots_for_customer_id(customer_id: str) -> list[dict[str, Any]]:
    return STATE_STORE.find_session_slots(customer_id)


def _find_session_slots_for_customer_id(customer_id: str) -> list[dict[str, Any]]:
    """Backward-compatible alias."""
    return find_session_slots_for_customer_id(customer_id)


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
) -> tuple[str, str]:
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
    chat_key = str(chat_id)
    slot = sessions.get(chat_key)
    if not isinstance(slot, dict):
        slot = {}
    customer_id = str(slot.get("customer_id", "")).strip() or f"telegram_{user_id}"
    now_utc_iso = datetime.now(UTC).isoformat()
    thread_id = new_short_id("chat")
    wake_thread_id = new_short_id("wake")
    sessions[chat_key] = {
        "user_id": int(user_id),
        "customer_id": customer_id,
        "thread_id": thread_id,
        "wake_thread_id": wake_thread_id,
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


async def _send_debug_logs_file(*, chat_id: int, bot_token: str | None) -> str | None:
    if not str(bot_token or "").strip():
        return "Telegram file sending is unavailable because the bot token is not configured."
    raw_bytes = read_debug_log_bytes()
    if raw_bytes is None:
        return "Debug log file is not available yet."
    sent = await TelegramClient(str(bot_token)).send_file(
        chat_id=chat_id,
        filename="app.log",
        raw_bytes=raw_bytes,
        kind="document",
        mime_type="text/plain",
        caption="OpenTulpa debug log dump",
        parse_mode="HTML",
    )
    if not sent:
        return "I couldn't send the debug log file right now."
    return None


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


async def handle_telegram_text(
    *,
    body: dict[str, Any],
    bot_token: str | None = None,
    allowed_user_ids_csv: str | None = None,
    allowed_usernames_csv: str | None = None,
    agent_runtime: Any | None = None,
    file_vault: FileVaultService | None = None,
    memory: Any | None = None,
) -> str | None:
    parsed = parse_telegram_update(body)
    if not parsed:
        return None
    chat_id, user_id, text = parsed
    if not chat_id or not user_id:
        return None

    message = body.get("message") or body.get("edited_message") or {}
    caption = str(message.get("caption", "")).strip() or None
    attachments = extract_attachments(message)
    username = (message.get("from", {}) or {}).get("username")
    username = username.strip() or None if isinstance(username, str) else None
    ctx = TelegramContext(
        chat_id=chat_id,
        user_id=user_id,
        username=username,
        text=(text or "").strip(),
    )

    if not is_user_allowed(
        user_id=ctx.user_id,
        username=ctx.username,
        allowed_user_ids_csv=allowed_user_ids_csv,
        allowed_usernames_csv=allowed_usernames_csv,
    ):
        return "This bot is restricted and your Telegram account is not allowed."

    def _ensure_admin(state: dict[str, Any]) -> Any:
        admin_user_id = state.get("admin_user_id")
        if admin_user_id is None:
            admin_user_id = ctx.user_id
            state["admin_user_id"] = admin_user_id
        return admin_user_id

    admin_user_id = STATE_STORE.update(_ensure_admin)
    _ = int(admin_user_id) == int(ctx.user_id)

    command_name = _telegram_command_name(ctx.text)
    if command_name in {"/start", "/help"}:
        return _start_help_text()
    if command_name == "/status":
        agent_up = bool(agent_runtime and getattr(agent_runtime, "healthy", lambda: False)())
        return status_text(agent_up)
    if command_name == "/fresh":
        thread_id, _ = STATE_STORE.update(
            lambda state: _reset_chat_session_context(
                state,
                chat_id=ctx.chat_id,
                user_id=ctx.user_id,
            )
        )
        return (
            "Started a fresh chat context. "
            f"New thread: {thread_id}. "
            "Your long-term memory is unchanged."
        )
    if command_name == "/debug_logs":
        return await _send_debug_logs_file(chat_id=ctx.chat_id, bot_token=bot_token)

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
        customer_id = str(slot.get("customer_id", "")).strip() or f"telegram_{ctx.user_id}"
        now_utc_iso = datetime.now(UTC).isoformat()
        sessions[str(ctx.chat_id)] = {
            "user_id": ctx.user_id,
            "customer_id": customer_id,
            "thread_id": thread_id,
            "wake_thread_id": wake_thread_id,
            "last_user_message_at": now_utc_iso,
            "last_assistant_message_at": slot.get("last_assistant_message_at"),
        }
        state["sessions"] = sessions
        return thread_id, customer_id

    thread_id, customer_id = STATE_STORE.update(_upsert_session)

    ingested_files: list[dict[str, Any]] = []
    if attachments and bot_token and file_vault is not None:
        typing_stop = asyncio.Event()
        typing_task = asyncio.create_task(
            _emit_typing_until_done(
                client=TelegramClient(str(bot_token)),
                chat_id=ctx.chat_id,
                stop_event=typing_stop,
            )
        )
        try:
            ingested_files = await ingest_attachments(
                attachments=attachments,
                bot_token=bot_token,
                file_vault=file_vault,
                memory=memory,
                agent_runtime=agent_runtime,
                customer_id=customer_id,
                chat_id=ctx.chat_id,
                caption=caption,
            )
        finally:
            typing_stop.set()
            try:
                await typing_task
            except Exception:
                pass

    if attachments and not ctx.text and not ingested_files:
        if agent_runtime is None:
            return "I received your file, but agent runtime is unavailable right now."
        if file_vault is None:
            return "I received your file, but file storage is not configured."

    voice_transcripts = [
        str(item.get("voice_transcript", "")).strip()
        for item in ingested_files
        if str(item.get("kind", "")).strip() == "voice"
    ]
    non_voice_files = [
        item for item in ingested_files if str(item.get("kind", "")).strip() != "voice"
    ]

    context_blob = build_uploaded_files_context(non_voice_files)
    effective_text = _inject_voice_message_context(ctx.text, voice_transcripts)
    if context_blob:
        if effective_text:
            effective_text = f"{effective_text}\n\n{context_blob}"
        else:
            effective_text = (
                "User uploaded one or more files without extra text.\n"
                "Summarize what is available and ask a focused follow-up question.\n\n"
                f"{context_blob}"
            )

    if not effective_text:
        has_voice = any(str(getattr(item, "kind", "")).strip() == "voice" for item in attachments)
        if has_voice:
            return (
                "I received your voice message but couldn't transcribe it. "
                "Please resend a shorter/clearer voice note or send text."
            )
        return None

    if bot_token:
        try:
            final, suppressed = await stream_langgraph_reply_to_telegram(
                agent_runtime=agent_runtime,
                thread_id=thread_id,
                customer_id=customer_id,
                text=effective_text,
                bot_token=bot_token,
                chat_id=ctx.chat_id,
            )
            if suppressed:
                return None
        except Exception as exc:
            logger.exception(
                "Telegram streaming reply failed (chat_id=%s, thread_id=%s): %s",
                ctx.chat_id,
                thread_id,
                exc,
            )
            return _format_agent_error_for_user(exc)
        if final:
            STATE_STORE.touch_assistant_message(ctx.chat_id)
            return None
        debug_log(
            hypothesis_id="H4",
            location="interfaces/telegram/chat_service.py:handle_telegram_text",
            message="fallback_no_final_reply",
            data={"chat_id": ctx.chat_id, "thread_id": thread_id},
        )
        return "I received your message but no final reply was available yet. Ask again or use /status."

    try:
        response = await agent_runtime.ainvoke_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=effective_text,
            turn_mode="interactive",
        )
        return response
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
    ) -> None:
        self.bot_token = str(bot_token or "").strip()
        self.file_vault = file_vault
        self.memory = memory

    def find_session_slots(self, customer_id: str) -> list[dict[str, Any]]:
        return find_session_slots_for_customer_id(customer_id)

    def get_session_slot(self, chat_id: int) -> dict[str, Any] | None:
        return get_session_slot_for_chat_id(chat_id)

    def touch_assistant_message(self, chat_id: int) -> None:
        STATE_STORE.touch_assistant_message(chat_id)

    async def relay_task_event(
        self,
        *,
        customer_id: str,
        task_id: str,
        event_type: str,
        payload: dict[str, Any],
        agent_runtime: Any | None = None,
    ) -> list[dict[str, Any]]:
        return await relay_task_event_via_main_agent(
            customer_id=customer_id,
            task_id=task_id,
            event_type=event_type,
            payload=payload,
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
        return await relay_event_via_main_agent(
            customer_id=customer_id,
            event_label=event_label,
            payload=payload,
            agent_runtime=agent_runtime,
        )

    async def handle_update(
        self,
        *,
        body: dict[str, Any],
        allowed_user_ids_csv: str | None = None,
        allowed_usernames_csv: str | None = None,
        agent_runtime: Any | None = None,
    ) -> str | None:
        return await handle_telegram_text(
            body=body,
            bot_token=self.bot_token,
            allowed_user_ids_csv=allowed_user_ids_csv,
            allowed_usernames_csv=allowed_usernames_csv,
            agent_runtime=agent_runtime,
            file_vault=self.file_vault,
            memory=self.memory,
        )
