"""Routine tool registration."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import (
    normalize_cleanup_paths,
    normalize_command_for_working_dir,
    require_customer_id,
)
from opentulpa.agent.utils import looks_like_shell_command as _looks_like_shell_command


def register_routine_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def routine_create(
        name: str,
        schedule: str,
        implementation_command: str,
        instruction: str,
        notify_user: bool = True,
        cleanup_paths: list[str] | None = None,
        thread_id: str = "",
        execution_origin: str | None = None,
    ) -> Any:
        """
        Create a scheduled routine.
        - Recurring: cron (e.g. "0 9 * * *")
        - One-time: local ISO datetime (e.g. "2026-02-18T23:45:00+08:00")
        - Routines are clock-driven jobs. They are different from intake workflows,
          which react to inbound message events.
        - Do not create a routine to poll or "fix" a Telegram Business intake workflow.
          telegram_business_dm workflows are webhook-driven; empty routine_id/schedule is expected.
        - instruction: explicit schedule-time scratchpad for each run. Include required scripts,
          files/paths, keys to read from storage, and expected output/action.
        - implementation_command: planned shell/script command for the routine execution.
        - cleanup_paths: optional repo-relative file paths to remove when deleting this automation.
        """
        safe_name = str(name or "").strip()
        safe_schedule = str(schedule or "").strip()
        safe_instruction = str(instruction or "").strip()
        safe_command = normalize_command_for_working_dir(
            command=str(implementation_command or "").strip(),
            working_dir="tulpa_stuff",
        )
        safe_customer = require_customer_id(runtime)
        if not safe_name:
            return {"error": "routine_create failed: name is required"}
        if not safe_schedule:
            return {"error": "routine_create failed: schedule is required"}
        if not safe_instruction:
            return {"error": "routine_create failed: instruction is required"}
        if not safe_command:
            return {
                "error": (
                    "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create requires "
                    "implementation_command (concrete shell/script command)."
                )
            }
        if not _looks_like_shell_command(safe_command):
            return {
                "error": (
                    "ROUTINE_IMPLEMENTATION_COMMAND_INVALID: implementation_command must be a "
                    "concrete shell command (executable + args)."
                )
            }

        auto_notify = bool(notify_user)
        safe_cleanup_paths = normalize_cleanup_paths(cleanup_paths)

        r = await runtime._request_with_backoff(
            "POST",
            "/internal/scheduler/routine",
            json_body={
                "name": safe_name,
                "schedule": safe_schedule,
                "payload": {
                    "instruction": safe_instruction,
                    "customer_id": safe_customer,
                    "notify_user": auto_notify,
                    "notification_opt_out": not auto_notify,
                    "cleanup_paths": safe_cleanup_paths,
                },
                "is_cron": " " in safe_schedule and len(safe_schedule.split()) >= 5,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_create failed: {r.text}"}
        return r.json()

    @tool
    async def routine_list() -> Any:
        """List scheduled routines, reminders, and clock-driven automations for this user."""
        customer_id = require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/scheduler/routines",
            params={"customer_id": customer_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_list failed: {r.text}"}
        return r.json().get("routines", [])

    @tool
    async def routine_delete(routine_id: str) -> Any:
        """Delete or stop one scheduled routine automation by routine_id."""
        customer_id = require_customer_id(runtime)
        rid = str(routine_id or "").strip()
        if not rid:
            return {"error": "routine_delete failed: routine_id is required"}

        r = await runtime._request_with_backoff(
            "DELETE",
            f"/internal/scheduler/routine/{rid}",
            params={"customer_id": customer_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_delete failed: {r.text}"}
        payload = r.json() if r.content else {}
        if not bool(payload.get("ok")):
            return {
                "error": "routine_delete failed: routine not found or not accessible",
                "routine_id": rid,
            }

        verify = await runtime._request_with_backoff(
            "GET",
            "/internal/scheduler/routines",
            params={"customer_id": customer_id},
            timeout=10.0,
        )
        if verify.status_code != 200:
            return {
                "ok": True,
                "routine_id": rid,
                "verified_removed": False,
                "warning": "delete succeeded but verification list failed",
            }
        routines = verify.json().get("routines", [])
        still_present = any(str(item.get("id", "")) == rid for item in routines if isinstance(item, dict))
        return {
            "ok": not still_present,
            "routine_id": rid,
            "verified_removed": not still_present,
            "remaining_routines": routines,
        }

    return {
        "routine_create": routine_create,
        "routine_list": routine_list,
        "routine_delete": routine_delete,
    }
