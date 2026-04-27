from __future__ import annotations

from opentulpa.agent.file_analysis import _infer_audio_format
from opentulpa.interfaces.telegram.chat_service import (
    _inject_voice_message_context,
    _telegram_command_name,
)


def test_infer_audio_format_prefers_extension() -> None:
    assert _infer_audio_format(filename="note.ogg", mime_type="audio/ogg") == "ogg"
    assert _infer_audio_format(filename="clip.mp3", mime_type="audio/ogg") == "mp3"


def test_inject_voice_message_context_appends_to_text() -> None:
    out = _inject_voice_message_context("Hey", ["Hello from voice"])
    assert out == "Hey\n\n<user sent voice message>: Hello from voice"


def test_inject_voice_message_context_without_text() -> None:
    out = _inject_voice_message_context("", ["Voice only"])
    assert out == "<user sent voice message>: Voice only"


def test_telegram_command_name_handles_empty_or_whitespace_text() -> None:
    assert _telegram_command_name("") == ""
    assert _telegram_command_name("   ") == ""
    assert _telegram_command_name(None) == ""  # type: ignore[arg-type]
