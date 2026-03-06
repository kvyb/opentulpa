"""Telegram chat bridge orchestration service."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.attachments import (
    build_uploaded_files_context,
    extract_attachments,
    ingest_attachments,
)
from opentulpa.interfaces.telegram.client import parse_telegram_update
from opentulpa.interfaces.telegram.constants import STATE_PATH
from opentulpa.interfaces.telegram.env_management import (
    extract_inline_key_value,
    extract_set_command,
    mask_secret,
    missing_key_prompt,
    status_text,
    upsert_env_key,
)
from opentulpa.interfaces.telegram.models import TelegramContext
from opentulpa.interfaces.telegram.relay import (
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
            "Model authentication failed (OpenRouter key is invalid or revoked). "
            "Set a valid OPENROUTER_API_KEY and restart OpenTulpa."
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
    now_utc_iso = datetime.now(timezone.utc).isoformat()
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
        "/status\n"
        "/setup\n"
        "/set KEY VALUE\n"
        "/setenv KEY VALUE\n"
        "/fresh\n"
        "/cancel"
    )


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

    def _ensure_admin_and_read_pending(state: dict[str, Any]) -> tuple[Any, str | None]:
        admin_user_id = state.get("admin_user_id")
        if admin_user_id is None:
            admin_user_id = ctx.user_id
            state["admin_user_id"] = admin_user_id
        pending_map = state.get("pending_key_by_chat")
        if not isinstance(pending_map, dict):
            pending_map = {}
            state["pending_key_by_chat"] = pending_map
        pending_key = pending_map.get(str(ctx.chat_id))
        if pending_key is None:
            return admin_user_id, None
        pending_key_text = str(pending_key).strip()
        return admin_user_id, (pending_key_text or None)

    admin_user_id, pending_key = STATE_STORE.update(_ensure_admin_and_read_pending)
    is_admin = int(admin_user_id) == int(ctx.user_id)

    text_lower = ctx.text.lower()
    if text_lower in {"/start", "/help"}:
        return _start_help_text()
    if text_lower == "/status":
        agent_up = bool(agent_runtime and getattr(agent_runtime, "healthy", lambda: False)())
        return status_text(agent_up)
    if text_lower == "/cancel":
        def _clear_pending(state: dict[str, Any]) -> None:
            pending_map = state.get("pending_key_by_chat")
            if not isinstance(pending_map, dict):
                pending_map = {}
            pending_map.pop(str(ctx.chat_id), None)
            state["pending_key_by_chat"] = pending_map

        STATE_STORE.update(_clear_pending)
        return "Cancelled pending setup."
    if text_lower == "/fresh":
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

    if text_lower == "/setup":
        if not is_admin:
            return "Only the bot admin can set keys."
        if not os.environ.get("OPENROUTER_API_KEY"):
            def _set_pending_openrouter(state: dict[str, Any]) -> None:
                pending_map = state.get("pending_key_by_chat")
                if not isinstance(pending_map, dict):
                    pending_map = {}
                pending_map[str(ctx.chat_id)] = "OPENROUTER_API_KEY"
                state["pending_key_by_chat"] = pending_map

            STATE_STORE.update(_set_pending_openrouter)
            return "Please send your OPENROUTER_API_KEY value now."
        return "Core key is already set. Use /set KEY VALUE for additional keys."

    if pending_key:
        if not is_admin:
            return "Only the bot admin can set keys."
        value = ctx.text.strip()
        if not value:
            return f"Please send the value for {pending_key}."
        try:
            upsert_env_key(pending_key, value)
        except Exception as exc:
            return f"Failed to save {pending_key}: {exc}"

        def _clear_pending_after_save(state: dict[str, Any]) -> None:
            pending_map = state.get("pending_key_by_chat")
            if not isinstance(pending_map, dict):
                pending_map = {}
            pending_map.pop(str(ctx.chat_id), None)
            state["pending_key_by_chat"] = pending_map

        STATE_STORE.update(_clear_pending_after_save)
        return f"Saved {pending_key}={mask_secret(value)}.\nRestart OpenTulpa to apply."

    kv = extract_set_command(ctx.text) or extract_inline_key_value(ctx.text)
    if kv:
        key, value = kv
        if not is_admin:
            return "Only the bot admin can set keys."
        try:
            upsert_env_key(key, value)
        except Exception as exc:
            return f"Failed to save {key}: {exc}"
        return f"Saved {key}={mask_secret(value)}."

    if not os.environ.get("OPENROUTER_API_KEY"):
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
        now_utc_iso = datetime.now(timezone.utc).isoformat()
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
