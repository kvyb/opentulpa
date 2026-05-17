"""Customer id alias helpers for API route boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def resolve_customer_id(
    customer_id: Any,
    resolver: Callable[[str], str] | None,
) -> str:
    cid = str(customer_id or "").strip()
    if not cid or resolver is None:
        return cid
    resolved = str(resolver(cid) or "").strip()
    return resolved or cid


def resolve_body_customer_id(
    body: dict[str, Any],
    resolver: Callable[[str], str] | None,
) -> str:
    return resolve_customer_id(body.get("customer_id", ""), resolver)

