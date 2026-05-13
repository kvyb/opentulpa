"""Tulpa workspace tools."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import (
    normalize_command_for_working_dir,
    normalize_execution_origin,
    require_customer_id,
)
from opentulpa.agent.tools.core_tools import _decorate_python_dependency_failure
from opentulpa.agent.utils import looks_like_shell_command as _looks_like_shell_command


def register_tulpa_workspace_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def tulpa_write_file(path: str, content: str) -> Any:
        """Write or update a file in approved tulpa_stuff workspace paths."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/write_file",
            json_body={"path": path, "content": content},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"write failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_validate_file(path: str) -> Any:
        """Validate generated file syntax/contracts in approved paths."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/validate_file",
            json_body={"path": path},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"validation failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_reload() -> Any:
        """Reload tulpa_stuff routers so newly written connectors become active."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/reload",
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"reload failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_run_terminal(
        command: str,
        working_dir: str = "tulpa_stuff",
        timeout_seconds: int = 90,
        thread_id: str = "",
        execution_origin: str | None = None,
    ) -> Any:
        """Run a concrete shell/script command inside the tulpa_stuff agent venv."""
        safe_working_dir = str(working_dir or "").strip() or "tulpa_stuff"
        safe_command = normalize_command_for_working_dir(
            command=str(command or "").strip(),
            working_dir=safe_working_dir,
        )
        if not _looks_like_shell_command(safe_command):
            return {
                "error": (
                    "Command rejected: provide a concrete shell command (executable + args), "
                    "not natural language."
                )
            }
        safe_timeout = max(5, min(int(timeout_seconds), 600))
        require_customer_id(runtime)
        safe_thread = str(thread_id or "").strip()
        normalized_origin = normalize_execution_origin(
            thread_id=safe_thread,
            execution_origin=execution_origin,
        )

        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/run_terminal",
            json_body={
                "command": safe_command,
                "working_dir": safe_working_dir,
                "timeout_seconds": safe_timeout,
            },
            timeout=max(10.0, float(safe_timeout) + 10.0),
            retries=1,
        )
        if r.status_code != 200:
            return {"error": f"terminal failed: {r.text}"}
        payload = r.json()
        if isinstance(payload, dict):
            payload["execution_origin"] = normalized_origin
            payload = _decorate_python_dependency_failure(payload)
        return payload

    @tool
    async def tulpa_read_file(path: str, max_chars: int = 12000) -> Any:
        """Read a bounded text excerpt from approved tulpa_stuff workspace paths."""
        safe_max_chars = max(500, min(int(max_chars), 20000))
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/tulpa/read_file",
            params={"path": path, "max_chars": safe_max_chars},
            timeout=15.0,
        )
        if r.status_code != 200:
            return {"error": f"read failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_catalog() -> Any:
        """List tracked tulpa_stuff workspace files, generated artifacts, and metadata."""
        r = await runtime._request_with_backoff("GET", "/internal/tulpa/catalog", timeout=10.0)
        if r.status_code != 200:
            return {"error": f"catalog failed: {r.text}"}
        return r.json().get("catalog", {})

    return {
        "tulpa_write_file": tulpa_write_file,
        "tulpa_validate_file": tulpa_validate_file,
        "tulpa_reload": tulpa_reload,
        "tulpa_run_terminal": tulpa_run_terminal,
        "tulpa_read_file": tulpa_read_file,
        "tulpa_catalog": tulpa_catalog,
    }
