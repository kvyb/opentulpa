from __future__ import annotations

from opentulpa.interfaces.telegram.formatter import (
    TELEGRAM_TEXT_CHAR_LIMIT,
    prepare_text_and_mode,
    prepare_text_chunks_and_mode,
)


def test_prepare_text_and_mode_truncates_oversized_html_message() -> None:
    text = "A" * 10000

    formatted, mode = prepare_text_and_mode(text, "HTML")

    assert mode == "HTML"
    assert len(formatted) <= TELEGRAM_TEXT_CHAR_LIMIT
    assert "truncated" in formatted.lower()


def test_prepare_text_chunks_and_mode_splits_markdown_without_raw_fallback() -> None:
    section = (
        "## ВОПРОС 1\n\n"
        "---\n\n"
        "**1. «I retired my nice girl era» — Before/After**\n"
        "- Хук: резкий переход в чёрное платье\n"
        '- Титр: *"I retired my nice girl era"*\n'
        "- Почему вирусный: трансформация хорошо репостится\n\n"
    )
    text = section * 80

    chunks, mode = prepare_text_chunks_and_mode(text, "HTML")

    assert mode == "HTML"
    assert len(chunks) > 1
    assert all(len(chunk) <= TELEGRAM_TEXT_CHAR_LIMIT for chunk in chunks)
    assert all("[Truncated to fit Telegram.]" not in chunk for chunk in chunks)
    assert all("## " not in chunk for chunk in chunks)
    assert all("**" not in chunk for chunk in chunks)
    assert all("────────" not in chunk for chunk in chunks)
    assert all("\n---\n" not in chunk for chunk in chunks)
    assert all("<b>" in chunk or "• " in chunk for chunk in chunks)
