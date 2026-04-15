"""Utility helpers for LangGraph runtime orchestration."""

from __future__ import annotations

import html
import json
import re
import shlex
from typing import Any

from opentulpa.agent.lc_messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def normalize_model_name(model_name: str) -> str:
    return str(model_name or "").strip()


def safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(data)


def looks_like_shell_command(command: str) -> bool:
    text = (command or "").strip()
    if not text:
        return False
    try:
        parts = shlex.split(text)
    except Exception:
        parts = text.split()
    if not parts:
        return False
    first = parts[0].strip()
    first_l = first.lower()
    if first_l in {"search", "find", "check", "inspect", "review", "analyze", "look"}:
        return False
    return not (first[0].isupper() and len(parts) > 1 and first_l not in {"python", "bash", "sh"})


def latest_user_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
    return ""


def is_cron_like_schedule(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    parts = text.split()
    return len(parts) == 5 and all(parts)


def extract_relative_delay_minutes(text: str) -> int | None:
    t = str(text or "").lower()
    patterns = [
        (r"\bin\s+(\d+)\s*(minute|minutes|min|mins)\b", 1),
        (r"\bin\s+(\d+)\s*(hour|hours|hr|hrs)\b", 60),
        (r"\bin\s+(\d+)\s*m\b", 1),
        (r"\bin\s+(\d+)\s*h\b", 60),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, t)
        if match:
            try:
                return max(1, int(match.group(1)) * multiplier)
            except Exception:
                return None
    return None


def approx_tokens(text: str) -> int:
    return max(1, (len(str(text or "")) + 3) // 4)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        parts.append(f"[image:{image_url.get('url', '')}]")
                    else:
                        parts.append(f"[image:{image_url}]")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def message_to_text(message: Any) -> str:
    role = "message"
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, ToolMessage):
        role = "tool"
    content = content_to_text(getattr(message, "content", ""))
    parts = [f"[{role}]"]
    if content:
        parts.append(content)
    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        try:
            tool_text = json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            tool_text = str(tool_calls)
        if tool_text:
            parts.append(f"tool_calls={tool_text}")
    return " ".join(part for part in parts if part).strip()


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def extract_html_title(raw_html: str) -> str | None:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html)
    if not match:
        return None
    return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip() or None


def utc_offset_to_minutes(value: str) -> int | None:
    raw = str(value or "").strip()
    m = re.fullmatch(r"([+-])(\d{2}):(\d{2})", raw)
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    hours = int(m.group(2))
    minutes = int(m.group(3))
    if hours > 14 or minutes > 59:
        return None
    total = sign * (hours * 60 + minutes)
    if total < -12 * 60 or total > 14 * 60:
        return None
    return total


def minutes_to_utc_offset(total_minutes: int) -> str:
    sign = "+" if total_minutes >= 0 else "-"
    abs_m = abs(int(total_minutes))
    hours = abs_m // 60
    minutes = abs_m % 60
    return f"{sign}{hours:02d}:{minutes:02d}"
