"""Helpers for resolving canonical public base URLs."""

from __future__ import annotations

import os
from collections.abc import Mapping


def resolve_public_base_url(env: Mapping[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    raw = str(source.get("PUBLIC_BASE_URL", "")).strip()
    if not raw:
        railway_domain = str(source.get("RAILWAY_PUBLIC_DOMAIN", "")).strip()
        if railway_domain:
            raw = f"https://{railway_domain}"
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    return raw.rstrip("/")


def build_public_composio_callback_path() -> str:
    return "/webhook/composio/callback"


def build_public_composio_callback_url(env: Mapping[str, str] | None = None) -> str:
    public_base_url = resolve_public_base_url(env)
    if not public_base_url:
        return ""
    return f"{public_base_url}{build_public_composio_callback_path()}"
