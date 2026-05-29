"""Composio prompt-context discovery."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

COMPOSIO_PROMPT_TOOLKIT_CACHE_SECONDS = 300
COMPOSIO_PROMPT_TOOLKIT_LIMIT = 20
COMPOSIO_PROMPT_TOOLKIT_TIMEOUT_SECONDS = 2.0


async def load_connected_composio_toolkits_context(
    *,
    composio: Any,
    cache: dict[str, Any],
    customer_id: str,
) -> str:
    cid = str(customer_id or "").strip()
    if not cid:
        return ""
    now = asyncio.get_running_loop().time()
    cached = cache.get(cid)
    if (
        isinstance(cached, tuple)
        and len(cached) == 2
        and isinstance(cached[0], int | float)
        and isinstance(cached[1], list)
        and now - float(cached[0]) < COMPOSIO_PROMPT_TOOLKIT_CACHE_SECONDS
    ):
        toolkits = [str(item) for item in cached[1] if str(item).strip()]
    else:
        toolkits = await _load_connected_composio_toolkits(composio, cid)
        cache[cid] = (now, toolkits)
    if not toolkits:
        return ""
    assert len(toolkits) <= COMPOSIO_PROMPT_TOOLKIT_LIMIT
    assert all(toolkit == toolkit.strip().lower() for toolkit in toolkits)
    return (
        "Available via Composio tool for this customer: "
        f"{', '.join(toolkits)}. "
        "For private or authenticated resources in these services, prefer "
        'tool_group_exec(group="composio") with composio_tool_search/composio_tool_execute '
        "before anonymous web or browser access."
    )


async def _load_connected_composio_toolkits(composio: Any, customer_id: str) -> list[str]:
    list_connected_accounts = (
        getattr(composio, "list_connected_accounts", None)
        if bool(getattr(composio, "enabled", False))
        else None
    )
    if not callable(list_connected_accounts):
        return []
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                list_connected_accounts,
                customer_id=customer_id,
                statuses=["ACTIVE"],
                limit=COMPOSIO_PROMPT_TOOLKIT_LIMIT,
            ),
            timeout=COMPOSIO_PROMPT_TOOLKIT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.debug("Composio prompt toolkit lookup failed: %s", exc)
        return []
    items = response.get("items") if isinstance(response, dict) else []
    visible_items = items[:COMPOSIO_PROMPT_TOOLKIT_LIMIT] if isinstance(items, list) else []
    seen: set[str] = set()
    toolkits: list[str] = []
    for item in visible_items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "") or "").strip().upper()
        slug = str(item.get("toolkit_slug", "") or "").strip().lower()
        if (not status or status == "ACTIVE") and slug and slug not in seen:
            seen.add(slug)
            toolkits.append(slug)
    assert len(toolkits) <= COMPOSIO_PROMPT_TOOLKIT_LIMIT
    return toolkits
