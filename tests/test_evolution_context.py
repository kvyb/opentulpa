from opentulpa.evolution.context import EvolutionAuditContext


def test_evolution_audit_context_sanitizes_known_fields() -> None:
    context = EvolutionAuditContext.from_mapping(
        {
            "tenant_id": " owner ",
            "thread_id": "thread-1",
            "unknown": "ignored",
        }
    )

    assert context.as_metadata() == {"tenant_id": "owner", "thread_id": "thread-1"}
