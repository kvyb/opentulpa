"""Application service for revisioned intake workflow drafts."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import JsonValue, TypeAdapter, ValidationError

from opentulpa.core.ids import new_short_id
from opentulpa.intake.drafts.models import (
    ActivatedIntakeDraft,
    IntakeDraft,
    IntakeWorkflowProposal,
    PreparedIntakeDraft,
)
from opentulpa.intake.drafts.store import (
    IntakeDraftConfirmationError,
    IntakeDraftConflictError,
    IntakeDraftNotFoundError,
    IntakeDraftStore,
)

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_RESERVED_PAYLOAD_KEYS = {
    "actor_id",
    "confirmation_token",
    "created_at",
    "customer_id",
    "id",
    "prepared_revision",
    "proposal_hash",
    "revision",
    "routine_id",
    "tenant_id",
    "updated_at",
    "workflow_id",
}


class IntakeDraftValidationError(ValueError):
    """Draft content cannot be normalized into an active workflow proposal."""


class IntakeDraftActivationError(RuntimeError):
    """The active workflow replacement failed without consuming confirmation."""


class IntakeWorkflowActivator(Protocol):
    """Atomically replace one tenant workflow or raise without changing the prior one."""

    def activate_draft(
        self,
        *,
        draft_store: IntakeDraftStore,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        confirmation_token_hash: str,
        proposal: dict[str, JsonValue],
        now: datetime,
    ) -> ActivatedIntakeDraft | Awaitable[ActivatedIntakeDraft]: ...


ProposalNormalizer = Callable[
    [str, dict[str, JsonValue]],
    IntakeWorkflowProposal | Mapping[str, Any],
]


class IntakeDraftService:
    def __init__(
        self,
        store: IntakeDraftStore,
        *,
        workflow_activator: IntakeWorkflowActivator,
        proposal_normalizer: ProposalNormalizer | None = None,
        clock: Callable[[], datetime] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self._store = store
        self._workflow_activator = workflow_activator
        self._proposal_normalizer = proposal_normalizer or self._default_normalizer
        self._clock = clock or (lambda: datetime.now(UTC))
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(32))

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("intake draft clock must return an aware datetime")
        return now.astimezone(UTC)

    @staticmethod
    def _identifier(value: str, *, field: str, maximum: int) -> str:
        identifier = str(value or "").strip()
        if not identifier:
            raise IntakeDraftValidationError(f"{field} is required")
        if len(identifier) > maximum:
            raise IntakeDraftValidationError(f"{field} must be at most {maximum} characters")
        return identifier

    @classmethod
    def _scope(cls, tenant_id: str, actor_id: str | None = None) -> tuple[str, str | None]:
        tenant = cls._identifier(tenant_id, field="tenant_id", maximum=200)
        actor = (
            cls._identifier(actor_id or "", field="actor_id", maximum=200)
            if actor_id is not None
            else None
        )
        return tenant, actor

    @staticmethod
    def _validated_patch(patch: Mapping[str, Any]) -> dict[str, JsonValue]:
        if not isinstance(patch, Mapping) or not patch:
            raise IntakeDraftValidationError("draft patch must be a non-empty object")
        reserved = sorted(_RESERVED_PAYLOAD_KEYS.intersection(str(key) for key in patch))
        if reserved:
            raise IntakeDraftValidationError(
                f"draft payload contains reserved fields: {', '.join(reserved)}"
            )
        try:
            return _JSON_OBJECT.validate_python(dict(patch))
        except ValidationError as exc:
            raise IntakeDraftValidationError("draft payload must contain JSON values") from exc

    @staticmethod
    def _default_normalizer(
        workflow_id: str,
        payload: dict[str, JsonValue],
    ) -> IntakeWorkflowProposal:
        return IntakeWorkflowProposal.model_validate({"workflow_id": workflow_id, **payload})

    @staticmethod
    def _canonical_proposal(proposal: IntakeWorkflowProposal) -> tuple[dict[str, JsonValue], str]:
        payload = _JSON_OBJECT.validate_python(proposal.model_dump(mode="json"))
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return payload, hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _token_hash(
        *,
        tenant_id: str,
        draft_id: str,
        revision: int,
        proposal_hash: str,
        token: str,
    ) -> str:
        bound = "\0".join(
            (
                "intake-draft-confirmation-v1",
                tenant_id,
                draft_id,
                str(revision),
                proposal_hash,
                token,
            )
        )
        return hashlib.sha256(bound.encode("utf-8")).hexdigest()

    def get(self, *, tenant_id: str, draft_id: str) -> IntakeDraft | None:
        tenant, _ = self._scope(tenant_id)
        return self._store.get(
            tenant_id=tenant,
            draft_id=self._identifier(draft_id, field="draft_id", maximum=100),
        )

    def list(
        self,
        *,
        tenant_id: str,
        workflow_id: str | None = None,
    ) -> list[IntakeDraft]:
        tenant, _ = self._scope(tenant_id)
        workflow = (
            self._identifier(workflow_id, field="workflow_id", maximum=100) if workflow_id else None
        )
        return self._store.list(tenant_id=tenant, workflow_id=workflow)

    def save(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        patch: Mapping[str, Any],
        draft_id: str | None = None,
        workflow_id: str | None = None,
        expected_revision: int | None = None,
    ) -> IntakeDraft:
        tenant, actor = self._scope(tenant_id, actor_id)
        assert actor is not None
        safe_draft_id = self._identifier(
            draft_id or new_short_id("idft"),
            field="draft_id",
            maximum=100,
        )
        existing = self._store.get(tenant_id=tenant, draft_id=safe_draft_id)
        if existing is not None:
            safe_workflow_id = self._identifier(
                workflow_id or existing.workflow_id,
                field="workflow_id",
                maximum=100,
            )
        else:
            safe_workflow_id = self._identifier(
                workflow_id or new_short_id("iwf"),
                field="workflow_id",
                maximum=100,
            )
        return self._store.save(
            tenant_id=tenant,
            actor_id=actor,
            draft_id=safe_draft_id,
            workflow_id=safe_workflow_id,
            patch=self._validated_patch(patch),
            expected_revision=expected_revision,
            now=self._now(),
        )

    def prepare(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
    ) -> PreparedIntakeDraft:
        tenant, actor = self._scope(tenant_id, actor_id)
        assert actor is not None
        safe_draft_id = self._identifier(draft_id, field="draft_id", maximum=100)
        draft = self._store.get(tenant_id=tenant, draft_id=safe_draft_id)
        if draft is None:
            raise IntakeDraftNotFoundError(safe_draft_id)
        if draft.revision != expected_revision:
            raise IntakeDraftConflictError(
                f"expected revision {expected_revision}, found {draft.revision}"
            )
        try:
            normalized = self._proposal_normalizer(draft.workflow_id, dict(draft.payload))
            proposal = (
                normalized
                if isinstance(normalized, IntakeWorkflowProposal)
                else IntakeWorkflowProposal.model_validate(normalized)
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise IntakeDraftValidationError(f"workflow draft is invalid: {exc}") from exc
        if proposal.workflow_id != draft.workflow_id:
            raise IntakeDraftValidationError("normalizer changed the draft workflow_id")
        proposal_payload, proposal_hash = self._canonical_proposal(proposal)
        token = str(self._token_factory() or "").strip()
        if len(token) < 32:
            raise RuntimeError("confirmation token generator returned an unsafe token")
        token_hash = self._token_hash(
            tenant_id=tenant,
            draft_id=safe_draft_id,
            revision=expected_revision,
            proposal_hash=proposal_hash,
            token=token,
        )
        prepared = self._store.prepare(
            tenant_id=tenant,
            actor_id=actor,
            draft_id=safe_draft_id,
            expected_revision=expected_revision,
            proposal=proposal_payload,
            proposal_hash=proposal_hash,
            confirmation_token_hash=token_hash,
            now=self._now(),
        )
        return PreparedIntakeDraft(
            draft=prepared,
            proposal=proposal,
            proposal_hash=proposal_hash,
            confirmation_token=token,
        )

    async def activate(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        confirmation_token: str,
    ) -> ActivatedIntakeDraft:
        tenant, actor = self._scope(tenant_id, actor_id)
        assert actor is not None
        safe_draft_id = self._identifier(draft_id, field="draft_id", maximum=100)
        draft = self._store.get(tenant_id=tenant, draft_id=safe_draft_id)
        if draft is None:
            raise IntakeDraftNotFoundError(safe_draft_id)
        if draft.proposal is None or not draft.proposal_hash:
            raise IntakeDraftConfirmationError("draft has no prepared proposal")
        proposal_payload, proposal_hash = self._canonical_proposal(draft.proposal)
        if proposal_hash != draft.proposal_hash:
            raise IntakeDraftConfirmationError("prepared proposal hash is invalid")
        token = str(confirmation_token or "").strip()
        token_hash = self._token_hash(
            tenant_id=tenant,
            draft_id=safe_draft_id,
            revision=expected_revision,
            proposal_hash=proposal_hash,
            token=token,
        )
        try:
            result = self._workflow_activator.activate_draft(
                draft_store=self._store,
                tenant_id=tenant,
                actor_id=actor,
                draft_id=safe_draft_id,
                expected_revision=expected_revision,
                confirmation_token_hash=token_hash,
                proposal=proposal_payload,
                now=self._now(),
            )
            activated = await result if isinstance(result, Awaitable) else result
        except (
            IntakeDraftConfirmationError,
            IntakeDraftConflictError,
            IntakeDraftNotFoundError,
        ):
            raise
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            raise IntakeDraftActivationError("workflow activation failed") from exc
        if (
            activated.draft.tenant_id != tenant
            or activated.draft.id != safe_draft_id
            or activated.draft.revision != expected_revision
            or activated.draft.status != "activated"
            or str(activated.workflow.get("workflow_id") or "") != draft.workflow_id
        ):
            raise IntakeDraftActivationError("workflow activator returned invalid state")
        return activated

    def delete(
        self,
        *,
        tenant_id: str,
        draft_id: str,
        expected_revision: int,
    ) -> None:
        tenant, _ = self._scope(tenant_id)
        self._store.delete(
            tenant_id=tenant,
            draft_id=self._identifier(draft_id, field="draft_id", maximum=100),
            expected_revision=expected_revision,
        )


__all__ = [
    "IntakeDraftActivationError",
    "IntakeDraftService",
    "IntakeDraftValidationError",
    "IntakeWorkflowActivator",
    "ProposalNormalizer",
]
