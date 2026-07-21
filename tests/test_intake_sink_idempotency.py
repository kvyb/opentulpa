from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.intake.sink_writer import SinkWriter
from opentulpa.integrations.tenant_composio import (
    ComposioProviderError,
    IntegrationConnectionNotFoundError,
    TenantComposioIntakePort,
)
from opentulpa.persistence.idempotency import IdempotencyStore
from opentulpa.persistence.tenant_namespace import tenant_namespace_label


class _Composio:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.successful = True
        self.raise_on_execute = False
        self.result_data: dict[str, Any] = {"external_id": "record-1"}
        self.tool_slug = "CRM_UPSERT_LEAD"
        self.toolkit = "crm"
        self.schema_properties = {
            "external_key": {"type": "string"},
            "name": {"type": "string"},
        }

    def list_connected_accounts(self, **_: Any) -> dict[str, Any]:
        return {
            "items": [
                {
                    "id": "connection-1",
                    "status": "ACTIVE",
                    "user_id": "tenant-1",
                    "toolkit_slug": "crm",
                },
                {
                    "id": "connection-foreign",
                    "status": "ACTIVE",
                    "user_id": "tenant-2",
                    "toolkit_slug": "crm",
                },
            ]
        }

    def _tool(self, slug: str | None = None) -> dict[str, Any]:
        return {
            "slug": slug or self.tool_slug,
            "toolkit_slug": self.toolkit,
            "input_schema": {
                "type": "object",
                "properties": dict(self.schema_properties),
            },
        }

    def search_tools(self, **_: Any) -> dict[str, Any]:
        return {"items": [self._tool()]}

    def get_tool_schema(self, *, tool_slug: str) -> dict[str, Any]:
        return {"tool": self._tool(tool_slug)}

    def execute_tool(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raise_on_execute:
            raise RuntimeError("token=provider-secret")
        return {
            "successful": self.successful,
            "data": dict(self.result_data),
        }


def test_generic_sink_requires_booking_id_mapping(tmp_path: Path) -> None:
    service = IntakeWorkflowService(
        db_path=tmp_path / "intake.sqlite3",
        project_root=tmp_path,
    )

    with pytest.raises(ValueError, match="booking_id"):
        service._normalize_sink_config(  # noqa: SLF001
            sink_type="generic_composio_write",
            sink_config={
                "toolkit": "crm",
                "operation_hint": "upsert lead",
                "field_mapping": {"name": "customer_name"},
            },
            workflow_id="workflow_1",
            customer_id="tenant_1",
        )


def test_generic_sink_rejects_non_idempotent_action(tmp_path: Path) -> None:
    service = IntakeWorkflowService(
        db_path=tmp_path / "intake.sqlite3",
        project_root=tmp_path,
    )

    with pytest.raises(ValueError, match="upsert or update"):
        service._normalize_sink_config(  # noqa: SLF001
            sink_type="generic_composio_write",
            sink_config={
                "toolkit": "messaging",
                "tool_slug": "MESSAGING_SEND_MESSAGE",
                "operation_hint": "send message",
                "field_mapping": {"external_key": "booking_id"},
            },
            workflow_id="workflow-1",
            customer_id="tenant-1",
        )


def test_generic_sink_passes_booking_id_to_provider_arguments(tmp_path: Path) -> None:
    writer = SinkWriter(
        sink_root=tmp_path / "sinks",
        composio=None,
        idempotency=IdempotencyStore(tmp_path / "effects.db"),
    )

    arguments = writer._build_composio_arguments(  # noqa: SLF001
        workflow={
            "workflow_id": "workflow_1",
            "customer_id": "tenant_1",
            "sink_type": "generic_composio_write",
            "sink_config": {
                "field_mapping": {
                    "external_key": "booking_id",
                    "customer_name": "name",
                }
            },
        },
        booking={"booking_id": "booking_1", "conversation_id": "conversation_1"},
        conversation_summary={},
        payload={"name": "Alice"},
        record_status="completed",
    )

    assert isinstance(arguments, dict)
    assert arguments["external_key"] == "booking_1"
    assert arguments["customer_name"] == "Alice"


def test_intake_composio_binding_is_owned_exact_and_schema_checked() -> None:
    provider = _Composio()
    port = TenantComposioIntakePort(provider=provider)

    binding = port.resolve_sink_binding(
        tenant_id="tenant-1",
        toolkit="crm",
        connected_account_id="connection-1",
        tool_slug="CRM_UPSERT_LEAD",
        operation_hint="upsert lead",
        required_arguments={"external_key", "name"},
        allow_discovery=False,
    )
    assert binding.connected_account_id == "connection-1"
    assert binding.tool_slug == "CRM_UPSERT_LEAD"

    with pytest.raises(IntegrationConnectionNotFoundError, match="connection not found"):
        port.resolve_sink_binding(
            tenant_id="tenant-1",
            toolkit="crm",
            connected_account_id="connection-foreign",
            tool_slug="CRM_UPSERT_LEAD",
            operation_hint="upsert lead",
            required_arguments={"external_key", "name"},
            allow_discovery=False,
        )

    provider.tool_slug = "CRM_UPDATE_AND_SEND_MESSAGE"
    with pytest.raises(ComposioProviderError, match="not an approved"):
        port.resolve_sink_binding(
            tenant_id="tenant-1",
            toolkit="crm",
            connected_account_id="connection-1",
            tool_slug=provider.tool_slug,
            operation_hint="update and send",
            required_arguments={"external_key", "name"},
            allow_discovery=False,
        )

    provider.tool_slug = "CRM_UPSERT_LEAD"
    provider.schema_properties.pop("external_key")
    with pytest.raises(ComposioProviderError, match="external_key"):
        port.resolve_sink_binding(
            tenant_id="tenant-1",
            toolkit="crm",
            connected_account_id="connection-1",
            tool_slug=provider.tool_slug,
            operation_hint="upsert lead",
            required_arguments={"external_key", "name"},
            allow_discovery=False,
        )


def test_intake_composio_revalidates_connection_before_execution() -> None:
    provider = _Composio()
    port = TenantComposioIntakePort(provider=provider)
    original_list = provider.list_connected_accounts

    def no_longer_owned(**kwargs: Any) -> dict[str, Any]:
        result = original_list(**kwargs)
        result["items"][0]["user_id"] = "tenant-2"
        return result

    provider.list_connected_accounts = no_longer_owned  # type: ignore[method-assign]
    with pytest.raises(IntegrationConnectionNotFoundError, match="connection not found"):
        port.execute_sink(
            tenant_id="tenant-1",
            toolkit="crm",
            connected_account_id="connection-1",
            tool_slug="CRM_UPSERT_LEAD",
            arguments={"external_key": "booking-1", "name": "Alice"},
        )
    assert provider.calls == []


def test_intake_composio_sanitizes_provider_results_and_errors() -> None:
    provider = _Composio()
    provider.result_data = {
        "external_id": "record-1",
        "access_token": "provider-secret",
        "note": "api_key=provider-secret",
    }
    port = TenantComposioIntakePort(provider=provider)

    result = port.execute_sink(
        tenant_id="tenant-1",
        toolkit="crm",
        connected_account_id="connection-1",
        tool_slug="CRM_UPSERT_LEAD",
        arguments={"external_key": "booking-1", "name": "Alice"},
    )
    assert result["data"]["access_token"] == "[redacted]"
    assert result["data"]["note"] == "api_key=[redacted]"

    provider.raise_on_execute = True
    with pytest.raises(ComposioProviderError) as error:
        port.execute_sink(
            tenant_id="tenant-1",
            toolkit="crm",
            connected_account_id="connection-1",
            tool_slug="CRM_UPSERT_LEAD",
            arguments={"external_key": "booking-1", "name": "Alice"},
        )
    assert "provider-secret" not in str(error.value)


@pytest.mark.parametrize(
    "file_path",
    ["/tmp/leads.csv", "../leads.csv", "nested/../leads.csv", "nested\\leads.csv"],
)
def test_local_csv_rejects_paths_outside_tenant_sink_root(
    tmp_path: Path,
    file_path: str,
) -> None:
    service = IntakeWorkflowService(
        db_path=tmp_path / "intake.sqlite3",
        project_root=tmp_path,
        sink_root=tmp_path / "sinks",
    )

    with pytest.raises(ValueError, match="local_csv"):
        service._normalize_sink_config(  # noqa: SLF001
            sink_type="local_csv",
            sink_config={"file_path": file_path},
            workflow_id="workflow-1",
            customer_id="tenant-1",
        )


def test_local_csv_is_tenant_scoped_and_rejects_symlink_targets(tmp_path: Path) -> None:
    effects = IdempotencyStore(tmp_path / "effects.db")
    root = tmp_path / "sinks"
    writer = SinkWriter(sink_root=root, composio=None, idempotency=effects)
    workflow = {
        "workflow_id": "workflow-1",
        "name": "Intake",
        "sink_type": "local_csv",
        "sink_config": {"file_path": "leads.csv"},
    }
    booking = {"booking_id": "booking-1", "conversation_id": "conversation-1"}

    for tenant_id in ("tenant-a", "tenant-b"):
        result, error = writer.write_to_local_csv(
            workflow={**workflow, "customer_id": tenant_id},
            booking=booking,
            payload={"name": tenant_id},
        )
        assert error is None
        assert result["file_path"] == "leads.csv"
        tenant_file = root / tenant_namespace_label(tenant_id) / "leads.csv"
        contents = tenant_file.read_text(encoding="utf-8")
        assert f",{tenant_id},completed," in contents

    outside = tmp_path / "outside.csv"
    outside.write_text("do-not-touch", encoding="utf-8")
    tenant_file = root / tenant_namespace_label("tenant-c") / "leads.csv"
    tenant_file.parent.mkdir(mode=0o700)
    tenant_file.symlink_to(outside)
    result, error = writer.write_to_local_csv(
        workflow={**workflow, "customer_id": "tenant-c"},
        booking=booking,
        payload={"name": "Mallory"},
    )

    assert result == {}
    assert error == "intake sink execution failed"
    assert outside.read_text(encoding="utf-8") == "do-not-touch"


def test_local_csv_neutralizes_formulas_and_rejects_symlink_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sinks"
    writer = SinkWriter(
        sink_root=root,
        composio=None,
        idempotency=IdempotencyStore(tmp_path / "effects.db"),
    )
    workflow = {
        "workflow_id": "workflow-1",
        "customer_id": "tenant-a",
        "name": "=malicious-name",
        "sink_type": "local_csv",
        "sink_config": {"file_path": "leads.csv"},
    }
    booking = {"booking_id": "booking-1", "conversation_id": "conversation-1"}

    _, error = writer.write_to_local_csv(
        workflow=workflow,
        booking=booking,
        payload={"name": "  =2+2", "note": "\tSUM(A1:A2)"},
    )
    assert error is None
    target = root / tenant_namespace_label("tenant-a") / "leads.csv"
    with target.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["name"] == "'  =2+2"
    assert row["note"] == "'\tSUM(A1:A2)"
    assert row["workflow_name"] == "'=malicious-name"

    outside = tmp_path / "outside"
    outside.mkdir()
    tenant_root = root / tenant_namespace_label("tenant-b")
    tenant_root.mkdir(mode=0o700)
    (tenant_root / "nested").symlink_to(outside, target_is_directory=True)
    result, error = writer.write_to_local_csv(
        workflow={
            **workflow,
            "customer_id": "tenant-b",
            "sink_config": {"file_path": "nested/leads.csv"},
        },
        booking=booking,
        payload={"name": "Mallory"},
    )
    assert result == {}
    assert error == "intake sink execution failed"
    assert not (outside / "leads.csv").exists()


def test_invalid_legacy_sink_is_preserved_for_migration_but_fails_closed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "intake.sqlite3"
    service = IntakeWorkflowService(db_path=database, project_root=tmp_path)
    workflow = service.upsert_workflow(
        customer_id="tenant-a",
        workflow_id="workflow-1",
        name="Legacy",
        channel="instagram_dm",
        provider="composio",
        source_config={},
        intent_description="Capture a lead",
        required_fields=["name"],
        sink_type="local_csv",
        sink_config={"file_path": "leads.csv"},
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE intake_workflows SET sink_config_json = ? WHERE workflow_id = ?",
            ('{"file_path":"../../outside.csv"}', "workflow-1"),
        )
        connection.commit()

    restarted = IntakeWorkflowService(db_path=database, project_root=tmp_path)
    legacy = restarted.get_workflow(customer_id="tenant-a", workflow_id="workflow-1")
    assert legacy is not None
    assert legacy["sink_config"]["file_path"] == "../../outside.csv"
    result, error = restarted._write_to_local_csv(  # noqa: SLF001
        workflow={**workflow, "sink_config": legacy["sink_config"]},
        booking={"booking_id": "booking-1", "conversation_id": "conversation-1"},
        conversation_summary={},
        payload={"name": "Alice"},
    )
    assert result == {}
    assert error == "local_csv sink path is invalid"
    assert not (tmp_path.parent / "outside.csv").exists()


@pytest.mark.asyncio
async def test_external_sink_crash_reuses_persisted_booking_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composio = _Composio()
    sink_composio = TenantComposioIntakePort(provider=composio)
    effects = IdempotencyStore(tmp_path / "effects.db")
    service = IntakeWorkflowService(
        db_path=tmp_path / "intake.sqlite3",
        project_root=tmp_path,
        sink_root=tmp_path / "sinks",
        idempotency=effects,
        sink_composio=sink_composio,
    )
    workflow = {
        "workflow_id": "workflow-1",
        "customer_id": "tenant-1",
        "name": "Lead intake",
        "required_fields": ["name"],
        "sink_type": "generic_composio_write",
        "sink_config": {
            "toolkit": "crm",
            "tool_slug": "CRM_UPSERT_LEAD",
            "connected_account_id": "connection-1",
            "field_mapping": {
                "external_key": "booking_id",
                "name": "name",
            },
        },
        "notify_user": False,
    }
    summary = {"conversation_id": "conversation-1"}
    decision = {
        "booking_action": "create_new_booking",
        "ready_to_save": True,
        "reply_action": "none",
        "save_payload": {"name": "Alice"},
    }

    def crash_after_provider_success(**_: Any) -> None:
        raise RuntimeError("simulated persistence crash")

    monkeypatch.setattr(effects, "complete", crash_after_provider_success)
    _, first_error, _ = await service._apply_decision(  # noqa: SLF001
        workflow=workflow,
        conversation_summary=summary,
        conversation={},
        active_booking=None,
        recent_completed_booking=None,
        decision=decision,
    )
    bookings = service.list_bookings(
        customer_id="tenant-1",
        workflow_id="workflow-1",
        conversation_id="conversation-1",
    )
    assert first_error == "intake sink outcome is indeterminate"
    assert len(bookings) == 1
    booking_id = bookings[0]["booking_id"]
    assert len(composio.calls) == 1

    monkeypatch.undo()
    _, second_error, _ = await service._apply_decision(  # noqa: SLF001
        workflow=workflow,
        conversation_summary=summary,
        conversation={},
        active_booking=bookings[0],
        recent_completed_booking=None,
        decision=decision,
    )
    retried = service.list_bookings(
        customer_id="tenant-1",
        workflow_id="workflow-1",
        conversation_id="conversation-1",
    )
    assert second_error == "intake sink outcome is indeterminate"
    assert [item["booking_id"] for item in retried] == [booking_id]
    assert len(composio.calls) == 1


@pytest.mark.asyncio
async def test_failed_provider_response_requires_owner_reconciliation_before_retry(
    tmp_path: Path,
) -> None:
    composio = _Composio()
    composio.successful = False
    effects = IdempotencyStore(tmp_path / "effects.db")
    service = IntakeWorkflowService(
        db_path=tmp_path / "intake.sqlite3",
        project_root=tmp_path,
        sink_root=tmp_path / "sinks",
        idempotency=effects,
        sink_composio=TenantComposioIntakePort(provider=composio),
    )
    workflow = service.upsert_workflow(
        customer_id="tenant-1",
        workflow_id="workflow-1",
        name="Lead intake",
        intent_description="Capture leads",
        required_fields=["name"],
        sink_type="generic_composio_write",
        sink_config={
            "toolkit": "crm",
            "tool_slug": "CRM_UPSERT_LEAD",
            "connected_account_id": "connection-1",
            "operation_hint": "upsert lead",
            "field_mapping": {
                "external_key": "booking_id",
                "name": "name",
            },
        },
        notify_user=False,
    )
    decision = {
        "booking_action": "create_new_booking",
        "ready_to_save": True,
        "reply_action": "none",
        "save_payload": {"name": "Alice"},
    }
    summary = {"conversation_id": "conversation-1"}

    _, first_error, _ = await service._apply_decision(  # noqa: SLF001
        workflow=workflow,
        conversation_summary=summary,
        conversation={},
        active_booking=None,
        recent_completed_booking=None,
        decision=decision,
    )
    booking = service.list_bookings(
        customer_id="tenant-1",
        workflow_id="workflow-1",
        conversation_id="conversation-1",
    )[0]
    assert first_error == "intake sink outcome is indeterminate"
    assert booking["sink_write_status"] == "indeterminate"
    assert len(composio.calls) == 1

    _, blocked_error, _ = await service._apply_decision(  # noqa: SLF001
        workflow=workflow,
        conversation_summary=summary,
        conversation={},
        active_booking=booking,
        recent_completed_booking=None,
        decision=decision,
    )
    assert blocked_error == "intake sink outcome is indeterminate"
    assert len(composio.calls) == 1

    reconciliation = service.reconcile_sink_effect(
        customer_id="tenant-1",
        actor_id="owner-1",
        workflow_id="workflow-1",
        booking_id=str(booking["booking_id"]),
        effect_revision=int(booking["sink_effect_revision"]),
        decision="retry_no_effect",
        reason="provider confirmed no record was written",
    )
    assert reconciliation["sink_write_status"] == "failed"

    composio.successful = True
    reconciled_booking = service.list_bookings(
        customer_id="tenant-1",
        workflow_id="workflow-1",
        conversation_id="conversation-1",
    )[0]
    _, retry_error, _ = await service._apply_decision(  # noqa: SLF001
        workflow=workflow,
        conversation_summary=summary,
        conversation={},
        active_booking=reconciled_booking,
        recent_completed_booking=None,
        decision=decision,
    )
    completed = service.list_bookings(
        customer_id="tenant-1",
        workflow_id="workflow-1",
        conversation_id="conversation-1",
    )[0]
    assert retry_error is None
    assert completed["sink_write_status"] == "succeeded"
    assert completed["sink_effect_revision"] == booking["sink_effect_revision"]
    assert len(composio.calls) == 2
