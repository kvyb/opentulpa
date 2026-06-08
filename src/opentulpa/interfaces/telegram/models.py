"""Typed models for Telegram chat handling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TelegramContext:
    chat_id: int
    user_id: int
    username: str | None
    text: str


@dataclass
class TelegramAttachment:
    kind: str
    file_id: str
    filename: str | None
    mime_type: str | None
    file_size: int | None = None
