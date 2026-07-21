"""Atomic active-workflow replacement behind intake draft activation."""

from __future__ import annotations

import sqlite3
from contextlib import closing, suppress
from datetime import datetime

from pydantic import JsonValue

from opentulpa.core.ids import new_short_id
from opentulpa.intake.drafts.models import ActivatedIntakeDraft, IntakeWorkflowProposal
from opentulpa.intake.drafts.store import IntakeDraftConfirmationError, IntakeDraftStore
from opentulpa.intake.service import IntakeWorkflowService


class IntakeWorkflowActivator:
    """Commit the workflow row and consumed draft confirmation together."""

    def __init__(self, workflows: IntakeWorkflowService) -> None:
        self._workflows = workflows

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
    ) -> ActivatedIntakeDraft:
        validated = IntakeWorkflowProposal.model_validate(proposal)
        workflow_id = validated.workflow_id
        existing = self._workflows.get_workflow(
            customer_id=tenant_id,
            workflow_id=workflow_id,
        )
        existing_revision = int(existing.get("revision") or 0) if existing else 0
        if validated.channel == "telegram_business_dm":
            competing = [
                workflow
                for workflow in self._workflows.list_workflows(
                    customer_id=tenant_id,
                    include_disabled=True,
                )
                if str(workflow.get("channel") or "") == "telegram_business_dm"
                and str(workflow.get("workflow_id") or "") != workflow_id
            ]
            if competing:
                raise ValueError(
                    "telegram_business_dm supports only one active workflow per customer"
                )
        payload = validated.model_dump(mode="python", exclude={"workflow_id"})
        workflow = self._workflows._normalize_workflow_payload(
            workflow_id=workflow_id,
            customer_id=tenant_id,
            existing=existing,
            **payload,
        )
        timestamp = now.isoformat()
        created_at = str(existing.get("created_at") or "").strip() if existing else ""
        created_at = created_at or timestamp
        attempt_id = new_short_id("iact")
        workflow_db_path = self._workflows._db_path
        same_database = workflow_db_path == draft_store.db_path
        schema = "main" if same_database else "intake_drafts_db"

        with closing(self._workflows._store.conn()) as conn:
            attached = False
            try:
                conn.execute("PRAGMA synchronous=FULL")
                if not same_database:
                    conn.execute(
                        "ATTACH DATABASE ? AS intake_drafts_db",
                        (str(draft_store.db_path),),
                    )
                    attached = True
                    conn.execute("PRAGMA intake_drafts_db.synchronous=FULL")
                conn.execute("BEGIN IMMEDIATE")
                claimed = draft_store.claim_activation_in_transaction(
                    conn,
                    schema=schema,
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    draft_id=draft_id,
                    expected_revision=expected_revision,
                    confirmation_token_hash=confirmation_token_hash,
                    activation_attempt_id=attempt_id,
                    now=now,
                )
                if claimed.workflow_id != workflow_id or claimed.proposal != validated:
                    raise IntakeDraftConfirmationError(
                        "prepared proposal changed before activation"
                    )
                workflow_revision = (
                    self._workflows._store.upsert_workflow_record_in_transaction(
                        conn,
                        workflow=workflow,
                        created_at=created_at,
                        updated_at=timestamp,
                        expected_revision=existing_revision,
                    )
                )
                activated = draft_store.finish_activation_in_transaction(
                    conn,
                    schema=schema,
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    draft_id=draft_id,
                    expected_revision=expected_revision,
                    activation_attempt_id=attempt_id,
                    now=now,
                )
                workflow["revision"] = workflow_revision
                response = {
                    key: value for key, value in workflow.items() if key != "routine_id"
                }
                response["created_at"] = created_at
                response["updated_at"] = timestamp
                result = ActivatedIntakeDraft(draft=activated, workflow=response)
                conn.commit()
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise
            finally:
                if attached:
                    with suppress(sqlite3.Error):
                        conn.execute("DETACH DATABASE intake_drafts_db")

        self._workflows._index_workflow_knowledge(workflow)
        return result


__all__ = ["IntakeWorkflowActivator"]
