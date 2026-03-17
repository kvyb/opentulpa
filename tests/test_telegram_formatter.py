from __future__ import annotations

from opentulpa.interfaces.telegram.formatter import (
    TELEGRAM_TEXT_CHAR_LIMIT,
    prepare_text_and_mode,
)


def test_prepare_text_and_mode_truncates_oversized_html_message() -> None:
    text = "A" * 10000

    formatted, mode = prepare_text_and_mode(text, "HTML")

    assert mode == "HTML"
    assert len(formatted) <= TELEGRAM_TEXT_CHAR_LIMIT
    assert "truncated" in formatted.lower()
