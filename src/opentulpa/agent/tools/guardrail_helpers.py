"""Shared guardrail and command-shaping helpers for tool modules."""

from __future__ import annotations

import shlex
from typing import Any

from opentulpa.policy.execution_boundary import ExecutionBoundaryGuard


def normalize_cleanup_paths(paths: list[str] | None) -> list[str]:
    if not isinstance(paths, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in paths:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


_WORKING_DIR_PREFIXES: dict[str, str] = {
    "tulpa_stuff": "tulpa_stuff",
    "integrations": "src/opentulpa/integrations",
    "interfaces": "src/opentulpa/interfaces",
    "tools": "src/opentulpa/tools",
    "skills": "src/opentulpa/skills",
    "opentulpa": "src/opentulpa",
}


def normalize_command_for_working_dir(command: str, working_dir: str) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    prefix = _WORKING_DIR_PREFIXES.get(str(working_dir or "").strip())
    if not prefix:
        return text
    try:
        parts = shlex.split(text)
    except Exception:
        return text
    if len(parts) <= 1:
        return text

    markers = (f"{prefix}/", f"./{prefix}/")

    def _strip_one(token: str) -> str:
        raw = str(token)
        for marker in markers:
            if raw.startswith(marker):
                return raw[len(marker) :]
        if raw.startswith("--") and "=" in raw:
            key, value = raw.split("=", 1)
            for marker in markers:
                if value.startswith(marker):
                    return f"{key}={value[len(marker):]}"
        return raw

    normalized = [parts[0], *(_strip_one(item) for item in parts[1:])]
    return shlex.join(normalized)


def normalize_execution_origin(
    *,
    thread_id: str | None,
    execution_origin: str | None,
) -> str:
    return ExecutionBoundaryGuard.normalize_execution_origin(
        thread_id=str(thread_id or "").strip(),
        execution_origin=str(execution_origin or "").strip(),
    )


def approval_pending_payload(
    *,
    action_name: str,
    command_preview: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    approval_id = str(decision.get("approval_id", "")).strip()
    if approval_id.lower() in {"none", "null"}:
        approval_id = ""
    summary = str(decision.get("summary", f"execute {action_name}")).strip()
    reason = str(decision.get("reason", "approval_required")).strip()
    if not approval_id:
        return {
            "ok": False,
            "status": "guardrail_unavailable",
            "action_name": action_name,
            "command_preview": command_preview[:300],
            "approval_id": None,
            "delivery_mode": str(decision.get("delivery_mode", "")).strip() or None,
            "summary": summary,
            "reason": reason or "approval_challenge_unavailable",
            "message": (
                "GUARDRAIL_BLOCKED: Approval is required but the approval challenge "
                "could not be created. Please retry."
            ),
            "gate": "require_approval",
            "retryable": True,
        }
    message = (
        "APPROVAL_PENDING: This executable action is waiting for user approval "
        f"(approval_id={approval_id}; summary={summary}; reason={reason})."
    )
    return {
        "ok": False,
        "status": "approval_pending",
        "action_name": action_name,
        "command_preview": command_preview[:300],
        "approval_id": approval_id or None,
        "delivery_mode": str(decision.get("delivery_mode", "")).strip() or None,
        "summary": summary,
        "reason": reason,
        "message": message,
        "gate": "require_approval",
    }
