"""Environment-key setup helpers for Telegram onboarding."""

from __future__ import annotations

import os

from opentulpa.core.config import (
    LEGACY_OPENROUTER_API_KEY_ENV,
    PRIMARY_OPENAI_COMPATIBLE_API_KEY_ENV,
    get_openai_compatible_api_key_from_env,
)


def missing_key_prompt() -> str:
    return (
        "The model backend is not configured yet.\n\n"
        "Set OPENAI_COMPATIBLE_API_KEY in the deployment or local environment, then restart OpenTulpa. "
        "OPENROUTER_API_KEY is still accepted as a legacy alias."
    )


def status_text(agent_up: bool) -> str:
    keys = {
        "OPENAI_COMPATIBLE_API_KEY": bool(get_openai_compatible_api_key_from_env()),
        "TELEGRAM_BOT_TOKEN": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        "BROWSER_USE_HEADLESS": bool(os.environ.get("BROWSER_USE_HEADLESS")),
        "BROWSER_USE_MODEL": bool(os.environ.get("BROWSER_USE_MODEL")),
        "CAPSOLVER_API_KEY": bool(os.environ.get("CAPSOLVER_API_KEY")),
    }
    lines = [
        "OpenTulpa status:",
        f"- Agent backend: {'up' if agent_up else 'down'}",
        (
            f"- {PRIMARY_OPENAI_COMPATIBLE_API_KEY_ENV} "
            f"(model provider key; {LEGACY_OPENROUTER_API_KEY_ENV} also accepted): "
            f"{'set' if keys['OPENAI_COMPATIBLE_API_KEY'] else 'missing'}"
        ),
        f"- TELEGRAM_BOT_TOKEN: {'set' if keys['TELEGRAM_BOT_TOKEN'] else 'missing'}",
        f"- BROWSER_USE_HEADLESS: {'set' if keys['BROWSER_USE_HEADLESS'] else 'default(true)'}",
        (
            f"- BROWSER_USE_MODEL: "
            f"{'set' if keys['BROWSER_USE_MODEL'] else 'default(MULTIMODAL_LLM)'}"
        ),
        f"- CAPSOLVER_API_KEY: {'set' if keys['CAPSOLVER_API_KEY'] else 'disabled'}",
        "",
        "Commands: /start, /status, /fresh, /debug_logs",
    ]
    return "\n".join(lines)
