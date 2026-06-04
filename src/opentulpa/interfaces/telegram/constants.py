"""Telegram interface constants and filesystem paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
STATE_PATH = PROJECT_ROOT / ".opentulpa" / "telegram_state.json"
DEBUG_LOG_PATH = PROJECT_ROOT / ".cursor" / "debug.log"

TELEGRAM_WEBHOOK_ALLOWED_UPDATES = (
    "message",
    "edited_message",
    "callback_query",
    "my_chat_member",
    "business_connection",
    "business_message",
    "edited_business_message",
    "deleted_business_messages",
)

LOW_SIGNAL_REPLIES = {
    "i see",
    "understood",
    "let me see",
    "checking this",
    "checking",
    "working on it",
    "acknowledged",
}
