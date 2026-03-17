"""Telegram text formatting helpers."""

from __future__ import annotations

import re
from html import escape

TELEGRAM_TEXT_CHAR_LIMIT = 3800


def _truncate_plain_text(text: str, *, max_chars: int = TELEGRAM_TEXT_CHAR_LIMIT) -> str:
    raw = str(text or "").strip()
    if not raw or len(raw) <= max_chars:
        return raw
    suffix = "\n\n[Truncated to fit Telegram.]"
    keep = max(120, max_chars - len(suffix))
    clipped = raw[:keep].rstrip()
    boundary_floor = max(0, int(keep * 0.6))
    cut_positions = [
        clipped.rfind("\n\n", boundary_floor),
        clipped.rfind("\n", boundary_floor),
        clipped.rfind(". ", boundary_floor),
        clipped.rfind("! ", boundary_floor),
        clipped.rfind("? ", boundary_floor),
    ]
    best_cut = max(cut_positions)
    if best_cut > 0:
        clipped = clipped[:best_cut].rstrip()
    return clipped + suffix


def markdownish_to_html(text: str) -> str:
    """
    Convert common LLM markdown patterns to Telegram-safe HTML.
    Handles fenced code blocks, inline code, headings, links, emphasis, and list markers.
    """
    source = str(text or "")
    if not source:
        return ""

    code_blocks: list[str] = []
    inline_codes: list[str] = []
    links: list[str] = []

    def _stash_code_block(match: re.Match[str]) -> str:
        code = match.group(2) or ""
        placeholder = f"%%CODEBLOCK_{len(code_blocks)}%%"
        code_blocks.append(f"<pre><code>{escape(code)}</code></pre>")
        return placeholder

    def _stash_inline_code(match: re.Match[str]) -> str:
        code = match.group(1) or ""
        placeholder = f"%%INLINECODE_{len(inline_codes)}%%"
        inline_codes.append(f"<code>{escape(code)}</code>")
        return placeholder

    def _stash_link(match: re.Match[str]) -> str:
        label = (match.group(1) or "").strip()
        url = (match.group(2) or "").strip()
        placeholder = f"%%LINK_{len(links)}%%"
        links.append(f'<a href="{escape(url, quote=True)}">{escape(label)}</a>')
        return placeholder

    # Temporarily remove code regions so we don't mutate formatting inside code.
    working = re.sub(r"```(\w+)?\n(.*?)```", _stash_code_block, source, flags=re.DOTALL)
    working = re.sub(r"`([^`\n]+)`", _stash_inline_code, working)
    working = re.sub(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)", _stash_link, working)

    # Escape HTML first, then apply markdown-like transforms on the escaped text.
    working = escape(working)
    working = re.sub(r"\*\*([^\n]+?)\*\*", r"<b>\1</b>", working)
    working = re.sub(r"__([^\n]+?)__", r"<b>\1</b>", working)
    working = re.sub(r"~~([^\n]+?)~~", r"<s>\1</s>", working)
    working = re.sub(r"(?<!\*)\*([^\n*]+?)\*(?!\*)", r"<i>\1</i>", working)
    working = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", working)

    # Line-oriented transforms.
    lines = working.splitlines()
    out_lines: list[str] = []
    for line in lines:
        if re.fullmatch(r"\s*([-*_]\s*){3,}\s*", line):
            out_lines.append("────────")
            continue
        heading = re.match(r"^\s*#{1,6}\s+(.+)$", line)
        if heading:
            out_lines.append(f"<b>{heading.group(1).strip()}</b>")
            continue
        line = re.sub(r"^\s*>\s?", "│ ", line)
        line = re.sub(r"^\s*[*-]\s+", "• ", line)
        out_lines.append(line)
    working = "\n".join(out_lines)

    for idx, html_code in enumerate(inline_codes):
        working = working.replace(f"%%INLINECODE_{idx}%%", html_code)
    for idx, html_block in enumerate(code_blocks):
        working = working.replace(f"%%CODEBLOCK_{idx}%%", html_block)
    for idx, html_link in enumerate(links):
        working = working.replace(f"%%LINK_{idx}%%", html_link)

    return working


def prepare_text_and_mode(text: str, parse_mode: str | None) -> tuple[str, str | None]:
    raw = _truncate_plain_text(str(text or "").strip())
    if not raw:
        return "", parse_mode
    mode = (parse_mode or "HTML").upper()
    if mode == "HTML":
        formatted = markdownish_to_html(raw)
        if len(formatted) <= TELEGRAM_TEXT_CHAR_LIMIT:
            return formatted, "HTML"
        return escape(_truncate_plain_text(raw, max_chars=TELEGRAM_TEXT_CHAR_LIMIT - 32)), "HTML"
    return _truncate_plain_text(raw), parse_mode
