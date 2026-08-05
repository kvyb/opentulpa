"""Typed bridge from durable evolution attempts to the immutable bootstrap."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from opentulpa.bootstrap.models import (
    ActivationKind,
    ActivationStatus,
    ReleaseOrigin,
    ReleaseRecord,
)
from opentulpa.bootstrap.supervisor import BootstrapSupervisor


class ReleaseActivationStatus(StrEnum):
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class ReleaseActivationResult(BaseModel):
    """Terminal bootstrap result consumed by the evolution archive."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    activation_id: str = Field(min_length=1, max_length=100)
    status: ReleaseActivationStatus
    failure_code: str | None = Field(default=None, min_length=1, max_length=100)
    failure_message: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def _consistent(self) -> ReleaseActivationResult:
        has_failure = self.failure_code is not None or self.failure_message is not None
        if self.status is ReleaseActivationStatus.ACTIVE and has_failure:
            raise ValueError("active release activation cannot contain a failure")
        if self.status is not ReleaseActivationStatus.ACTIVE and (
            self.failure_code is None or self.failure_message is None
        ):
            raise ValueError("non-active release activation requires a sanitized failure")
        return self


class ReleaseActivator(Protocol):
    async def activate(
        self,
        release: ReleaseRecord,
        *,
        activation_id: str,
        origin: ReleaseOrigin | None,
        reason: str,
        rollback: bool,
    ) -> ReleaseActivationResult: ...


class BootstrapReleaseActivator:
    """Idempotently run one release through bootstrap staging and probation."""

    def __init__(self, supervisor: BootstrapSupervisor) -> None:
        self._supervisor = supervisor

    async def activate(
        self,
        release: ReleaseRecord,
        *,
        activation_id: str,
        origin: ReleaseOrigin | None,
        reason: str,
        rollback: bool,
    ) -> ReleaseActivationResult:
        existing = self._supervisor.store.get_activation(activation_id)
        if existing is None:
            if rollback:
                rollback_target = str(release.metadata.get("rollback_target") or "")
                if not rollback_target:
                    raise RuntimeError("synthetic rollback release has no historical target")
                self._supervisor.store.add_release_alias(
                    release,
                    artifact_release_id=rollback_target,
                )
            queued = await self._supervisor.request_activation(
                release,
                origin=origin,
                reason=reason,
                kind=ActivationKind.ROLLBACK if rollback else ActivationKind.DEPLOY,
                activation_id=activation_id,
            )
        else:
            queued = existing
            if queued.target_release_id != release.id:
                raise RuntimeError("bootstrap activation identity is bound to another release")
        terminal = await self._supervisor.activate(queued.id)
        return self._terminal_result(terminal.status, terminal.id, terminal.failure_code, terminal.failure_message)

    @staticmethod
    def _terminal_result(
        status: ActivationStatus,
        activation_id: str,
        failure_code: str | None,
        failure_message: str | None,
    ) -> ReleaseActivationResult:
        if status is ActivationStatus.ACTIVE:
            return ReleaseActivationResult(
                activation_id=activation_id,
                status=ReleaseActivationStatus.ACTIVE,
            )
        public_code = failure_code or "activation_failed"
        public_message = failure_message or "Release activation did not become active."
        if status is ActivationStatus.ROLLED_BACK:
            terminal = ReleaseActivationStatus.ROLLED_BACK
        elif status is ActivationStatus.CANCELLED:
            terminal = ReleaseActivationStatus.CANCELLED
        else:
            terminal = ReleaseActivationStatus.FAILED
        return ReleaseActivationResult(
            activation_id=activation_id,
            status=terminal,
            failure_code=public_code,
            failure_message=public_message,
        )


__all__ = [
    "BootstrapReleaseActivator",
    "ReleaseActivationResult",
    "ReleaseActivationStatus",
    "ReleaseActivator",
]
