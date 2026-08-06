from __future__ import annotations

from opentulpa.evolution.context import EvolutionAuditContext, SourceSessionContext
from opentulpa.evolution.evaluation_metadata import EvaluationMetadata


def test_audit_context_preserves_sanitization_and_stable_session_identity() -> None:
    audit = EvolutionAuditContext.from_mapping(
        {
            "tenant_id": " tenant_1 ",
            "actor_id": "owner_1",
            "unknown": "ignored",
            "reason": "x" * 5_000,
        }
    )

    session = SourceSessionContext.create(audit)

    assert audit.tenant_id == "tenant_1"
    assert audit.reason == "x" * 4_000
    assert session.session_key == "cc3bd2964c5853d917a8b7188da2fbc36931ea5433c0c5a97193e5d6fc8d1ff6"
    assert session.matches(session_key="other", tenant_id="tenant_1")


def test_source_session_context_reads_legacy_fallback_and_preserves_unknown_metadata() -> None:
    metadata = {
        "source_session": True,
        "requested_by": {"tenant_id": "tenant_1", "thread_id": "thread_1"},
        "label": "preserved",
    }

    session = SourceSessionContext.from_metadata(metadata)

    assert session.matches(session_key="missing", tenant_id="tenant_1")
    assert session.apply_to_metadata(metadata) == metadata


def test_evaluation_metadata_preserves_unknown_extensions() -> None:
    metadata = {"source_release_operation_id": "operation_1", "runner": "isolated"}

    parsed = EvaluationMetadata.from_metadata(metadata)

    assert parsed.source_release_operation_id == "operation_1"
    assert parsed.apply_to_metadata(metadata) == metadata
