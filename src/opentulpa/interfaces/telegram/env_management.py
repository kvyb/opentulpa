"""Environment-key setup helpers for Telegram onboarding."""

from __future__ import annotations

import os
import re
from contextlib import suppress

from opentulpa.interfaces.telegram.constants import BLOCKED_ENV_KEYS, ENV_PATH


def is_allowed_env_key(key: str) -> bool:
    if key in BLOCKED_ENV_KEYS:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", key))


def upsert_env_key(key: str, value: str) -> None:
    if not is_allowed_env_key(key):
        raise ValueError(
            "Unsupported key name. Use ENV-style uppercase names (A-Z, 0-9, _) and avoid system keys."
        )
    existing_lines: list[str] = []
    if ENV_PATH.exists():
        existing_lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    updated = False
    out: list[str] = []
    prefix = f"{key}="
    for line in existing_lines:
        if line.startswith(prefix):
            out.append(f"{key}={value}")
            updated = True
        else:
            out.append(line)
    if not updated:
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out).strip() + "\n", encoding="utf-8")
    with suppress(Exception):
        os.chmod(ENV_PATH, 0o600)
    os.environ[key] = value


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def extract_set_command(text: str) -> tuple[str, str] | None:
    parts = text.strip().split(" ", 2)
    if len(parts) < 3:
        return None
    if parts[0].lower() not in {"/set", "/setenv"}:
        return None
    key = parts[1].strip().upper()
    value = parts[2].strip()
    if not key or not value:
        return None
    return key, value


def extract_inline_key_value(text: str) -> tuple[str, str] | None:
    t = text.strip()
    # Strict form only: KEY=VALUE (no free-form sentence before '=').
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]{1,127})\s*=\s*(.+)", t)
    if not match:
        return None
    key = match.group(1).upper().strip()
    value = match.group(2).strip()
    if not value:
        return None
    return key, value


def missing_key_prompt() -> str:
    return (
        "I can help set keys from Telegram.\n\n"
        "OpenTulpa uses OPENROUTER_* env names for the model backend, even if you are "
        "pointing them at another OpenAI-compatible provider.\n\n"
        "Use:\n"
        "/set OPENROUTER_API_KEY <value>\n"
        "or send OPENROUTER_API_KEY=<value>\n\n"
        "Then restart OpenTulpa to start/refresh the chat backend."
    )


def status_text(agent_up: bool) -> str:
    keys = {
        "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
        "TELEGRAM_BOT_TOKEN": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        "BROWSER_USE_HEADLESS": bool(os.environ.get("BROWSER_USE_HEADLESS")),
        "BROWSER_USE_MODEL": bool(os.environ.get("BROWSER_USE_MODEL")),
    }
    lines = [
        "OpenTulpa status:",
        f"- Agent backend: {'up' if agent_up else 'down'}",
        f"- OPENROUTER_API_KEY (model provider key): {'set' if keys['OPENROUTER_API_KEY'] else 'missing'}",
        f"- TELEGRAM_BOT_TOKEN: {'set' if keys['TELEGRAM_BOT_TOKEN'] else 'missing'}",
        f"- BROWSER_USE_HEADLESS: {'set' if keys['BROWSER_USE_HEADLESS'] else 'default(true)'}",
        f"- BROWSER_USE_MODEL: {'set' if keys['BROWSER_USE_MODEL'] else 'default(LLM_MODEL)'}",
        "",
        "Commands: /setup, /status, /set KEY VALUE, /cancel",
    ]
    return "\n".join(lines)
