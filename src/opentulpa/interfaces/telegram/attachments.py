"""Pure Telegram attachment extraction for the channel adapter."""

from __future__ import annotations

from typing import Any

from opentulpa.interfaces.telegram.models import TelegramAttachment


def extract_attachments(message: dict[str, Any]) -> list[TelegramAttachment]:
    attachments: list[TelegramAttachment] = []

    document = message.get("document")
    if isinstance(document, dict):
        fid = str(document.get("file_id", "")).strip()
        if fid:
            attachments.append(
                TelegramAttachment(
                    kind="document",
                    file_id=fid,
                    filename=str(document.get("file_name", "")).strip() or None,
                    mime_type=str(document.get("mime_type", "")).strip() or None,
                    file_size=int(document.get("file_size") or 0) or None,
                )
            )

    photos = message.get("photo")
    if isinstance(photos, list) and photos:
        chosen: dict[str, Any] | None = None
        for item in photos:
            if not isinstance(item, dict):
                continue
            if chosen is None or int(item.get("file_size") or 0) >= int(chosen.get("file_size") or 0):
                chosen = item
        if chosen:
            fid = str(chosen.get("file_id", "")).strip()
            if fid:
                unique = str(chosen.get("file_unique_id", "")).strip() or "photo"
                attachments.append(
                    TelegramAttachment(
                        kind="photo",
                        file_id=fid,
                        filename=f"{unique}.jpg",
                        mime_type="image/jpeg",
                        file_size=int(chosen.get("file_size") or 0) or None,
                    )
                )

    for key in ("video", "video_note", "audio", "voice"):
        item = message.get(key)
        if not isinstance(item, dict):
            continue
        fid = str(item.get("file_id", "")).strip()
        if not fid:
            continue
        unique = str(item.get("file_unique_id", "")).strip() or key
        ext = {
            "video": ".mp4",
            "video_note": ".mp4",
            "audio": ".mp3",
            "voice": ".ogg",
        }.get(key, "")
        filename = str(item.get("file_name", "")).strip() or f"{unique}{ext}"
        attachments.append(
            TelegramAttachment(
                kind=key,
                file_id=fid,
                filename=filename,
                mime_type=str(item.get("mime_type", "")).strip() or None,
                file_size=int(item.get("file_size") or 0) or None,
            )
        )
    return attachments
