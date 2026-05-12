"""Server time tools."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain.tools import tool


def register_server_time_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def server_time() -> Any:
        """Get current server time, local timezone, UTC time, and Unix timestamp."""
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(UTC)
        return {
            "server_time_local_iso": now_local.isoformat(),
            "server_timezone": str(now_local.tzinfo),
            "server_time_utc_iso": now_utc.isoformat(),
            "unix_timestamp": int(now_utc.timestamp()),
        }

    return {
        "server_time": server_time,
    }
