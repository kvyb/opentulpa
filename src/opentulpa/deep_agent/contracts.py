"""Public contracts for Deep Agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from opentulpa.tooling.contract import AgentRunContext

AgentRunStatus = Literal[
    "running",
    "interrupted",
    "resume_pending",
    "completed",
    "failed",
    "cancelled",
]
AgentApprovalStatus = Literal["pending", "approve", "edit", "reject"]
AgentRunEventType = Literal[
    "run.started",
    "message.delta",
    "tool.started",
    "tool.completed",
    "approval.required",
    "artifact.ready",
    "run.completed",
    "run.failed",
]


class AgentRunIdempotencyConflictError(RuntimeError):
    """An interface reused an idempotency key for a different run request."""


class AgentRunCheckpointConflictError(RuntimeError):
    """A checkpoint thread already has a run that requires an explicit outcome."""


class AgentRunCapabilityConflictError(RuntimeError):
    """A run's pinned dynamic capability bundle is no longer active."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class AgentRunRequest:
    context: AgentRunContext
    text: str
    file_ids: tuple[str, ...] = ()
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if not str(self.text or "").strip():
            raise ValueError("text is required")
        if self.idempotency_key is not None:
            key = str(self.idempotency_key).strip()
            if not key or len(key) > 300 or any(ord(char) < 32 for char in key):
                raise ValueError("idempotency_key is invalid")


@dataclass(frozen=True, slots=True)
class AgentRunEvent:
    type: AgentRunEventType
    run_id: str
    sequence: int
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentApproval:
    id: str
    tool_name: str
    description: str
    arguments: dict[str, Any]
    allowed_decisions: tuple[str, ...]
    status: AgentApprovalStatus = "pending"
    edited_arguments: dict[str, Any] | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    approval_id: str
    decision: Literal["approve", "edit", "reject"]
    edited_arguments: dict[str, Any] | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.decision == "edit" and self.edited_arguments is None:
            raise ValueError("edited_arguments are required for edit decisions")


@dataclass(frozen=True, slots=True)
class AgentRunSnapshot:
    run_id: str
    context: AgentRunContext
    status: AgentRunStatus
    final_text: str = ""
    error: str = ""
    approvals: tuple[AgentApproval, ...] = ()
    created_at: str = ""
    updated_at: str = ""
