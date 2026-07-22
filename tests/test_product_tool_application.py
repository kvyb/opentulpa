from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from opentulpa.application.product_tools import ProductToolApplication
from opentulpa.integrations.web_search import WebSearchProviderError
from opentulpa.logging.langfuse import redact_for_langfuse
from opentulpa.persistence.idempotency import IdempotencyStore
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling.adapters import (
    ProductToolApplicationError,
    ProductToolInvocation,
    build_product_tools,
)
from opentulpa.tooling.contract import (
    TOOL_SPEC_BY_NAME,
    TOOL_SPECS,
    AgentChannel,
    AgentRunContext,
    AgentRunKind,
)


class _Port:
    def __init__(self, **responses: Any) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(**kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            response = self.responses.get(name, {"ok": True})
            if isinstance(response, Exception):
                raise response
            return response(**kwargs) if callable(response) else response

        return call


class _IdempotencyPort(_Port):
    async def execute(self, **kwargs: Any) -> Any:
        invoke = kwargs.pop("invoke")
        self.calls.append(("execute", kwargs))
        value = invoke()
        return await value if inspect.isawaitable(value) else value


def _application(**overrides: Any) -> tuple[ProductToolApplication, dict[str, Any]]:
    ports = {
        "profiles": _Port(),
        "files": _Port(),
        "artifacts": _Port(),
        "knowledge": _Port(),
        "research": _Port(),
        "browser": _Port(),
        "integrations": _Port(),
        "intake": _Port(),
        "schedules": _Port(),
        "jobs": _Port(create={"tenant_id": "tenant-a", "id": "job-1", "status": "queued"}),
        "idempotency": _IdempotencyPort(),
    }
    if overrides.get("evolution") is not None:
        overrides.setdefault("evolution_owner_tenant_id", "tenant-a")
    ports.update(overrides)
    return ProductToolApplication(**ports), ports  # type: ignore[arg-type]


def _invocation(
    name: str,
    arguments: dict[str, Any],
    *,
    tenant_id: str = "tenant-a",
    idempotency_key: str | None = "request-1",
) -> ProductToolInvocation:
    return ProductToolInvocation(
        spec=TOOL_SPEC_BY_NAME[name],
        context=AgentRunContext(
            tenant_id=tenant_id,
            actor_id="owner-1",
            thread_id="thread-1",
            channel=AgentChannel.WEB,
            run_kind=AgentRunKind.OWNER,
            correlation_id="correlation-1",
            origin=OriginRef(interface="web", source_id="test"),
            agent_spec=AgentSpecRef(tenant_id=tenant_id, spec_id="owner", revision=1),
            trust_class="owner",
        ),
        arguments=arguments,
        idempotency_key=idempotency_key,
        audit_id="audit-1",
    )


def test_concrete_application_covers_registry_without_legacy_runtime_or_http() -> None:
    application, _ = _application()

    assert tuple(tool.name for tool in build_product_tools(application)) == tuple(
        spec.name for spec in TOOL_SPECS
    )
    source = Path(ProductToolApplication.__module__.replace(".", "/") + ".py")
    text = (Path(__file__).resolve().parents[1] / "src" / source).read_text()
    assert "opentulpa" + ".agent" not in text
    assert "httpx" not in text
    assert "/internal/" not in text


@pytest.mark.asyncio
async def test_web_search_provider_failure_is_a_sanitized_retryable_tool_error() -> None:
    research = _Port(
        search=WebSearchProviderError(
            "Web search returned no grounded sources.",
            retryable=True,
        )
    )
    application, _ = _application(research=research)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.web_search(_invocation("web_search", {"query": "OpenTulpa", "limit": 8}))

    assert error.value.code == "web_search_failed"
    assert error.value.public_message == "Web search returned no grounded sources."
    assert error.value.retryable is True


@pytest.mark.asyncio
async def test_resource_owner_is_injected_and_mismatched_service_data_fails_closed() -> None:
    files = _Port(
        get={
            "file_id": "file-1",
            "customer_id": "tenant-b",
            "original_filename": "private.pdf",
        }
    )
    application, _ = _application(files=files)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.file_get(_invocation("file_get", {"file_id": "file-1"}))

    assert error.value.code == "not_found"
    assert files.calls == [("get", {"tenant_id": "tenant-a", "file_id": "file-1"})]


@pytest.mark.asyncio
async def test_model_output_removes_internal_identity_paths_and_secrets() -> None:
    files = _Port(
        get={
            "file_id": "file-1",
            "customer_id": "tenant-a",
            "original_filename": "report.pdf",
            "stored_path": "/srv/opentulpa/tenant-a/report.pdf",
            "api_key": "must-not-leak",
            "metadata": {
                "tenant_id": "tenant-a",
                "token": "also-secret",
                "pages": 4,
            },
        }
    )
    application, _ = _application(files=files)

    output = await application.file_get(_invocation("file_get", {"file_id": "file-1"}))

    assert output.data == {
        "file_id": "file-1",
        "original_filename": "report.pdf",
        "api_key": "[redacted]",
        "metadata": {"token": "[redacted]", "pages": 4},
    }


@pytest.mark.asyncio
async def test_background_operation_preflights_tenant_resource_then_submits_direct_job() -> None:
    integrations = _Port(
        get_connection={
            "connection_id": "connection-1",
            "tenant_id": "tenant-a",
            "status": "active",
        }
    )
    jobs = _Port(
        create={
            "id": "job-42",
            "tenant_id": "tenant-a",
            "status": "queued",
            "arguments": {"connection_id": "connection-1"},
        }
    )
    application, _ = _application(integrations=integrations, jobs=jobs)
    invocation = _invocation(
        "integration_invoke",
        {
            "connection_id": "connection-1",
            "action_name": "GOOGLEDRIVE_FIND_FILE",
            "parameters": {"query": "proposal"},
        },
        idempotency_key="invoke-1",
    )

    output = await application.integration_invoke(invocation)

    assert integrations.calls == [
        (
            "get_connection",
            {"tenant_id": "tenant-a", "connection_id": "connection-1"},
        )
    ]
    assert jobs.calls == [
        (
            "create",
            {
                "tenant_id": "tenant-a",
                "handler_name": "integration_invoke",
                "arguments": dict(invocation.arguments),
                "idempotency_key": "invoke-1",
            },
        )
    ]
    assert output.job_id == "job-42"
    assert output.data == {"job_id": "job-42", "status": "queued"}


@pytest.mark.asyncio
async def test_background_operation_does_not_submit_when_tenant_preflight_fails() -> None:
    integrations = _Port(get_connection=None)
    jobs = _Port(create={"id": "job-should-not-exist"})
    application, _ = _application(integrations=integrations, jobs=jobs)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.integration_invoke(
            _invocation(
                "integration_invoke",
                {
                    "connection_id": "foreign-connection",
                    "action_name": "ACTION",
                    "parameters": {},
                },
            )
        )

    assert error.value.code == "not_found"
    assert jobs.calls == []


@pytest.mark.asyncio
async def test_connected_account_user_id_must_match_tenant_before_mutation() -> None:
    integrations = _Port(
        get_connection={
            "id": "foreign-connection",
            "user_id": "tenant-b",
            "status": "active",
        }
    )
    application, _ = _application(integrations=integrations)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.connection_disconnect(
            _invocation(
                "connection_disconnect",
                {"connection_id": "foreign-connection"},
                idempotency_key="disconnect-1",
            )
        )

    assert error.value.code == "not_found"
    assert [name for name, _ in integrations.calls] == ["get_connection"]


@pytest.mark.asyncio
async def test_delivery_uses_trusted_actor_channel_and_thread_not_model_arguments() -> None:
    artifacts = _Port(
        get={"artifact_id": "artifact-1", "tenant_id": "tenant-a"},
        deliver={
            "ok": True,
            "tenant_id": "tenant-a",
            "delivery_id": "delivery-1",
            "local_path": "/private/artifact.txt",
        },
    )
    application, _ = _application(artifacts=artifacts)

    output = await application.artifact_deliver(
        _invocation(
            "artifact_deliver",
            {"artifact_id": "artifact-1", "caption": "Result"},
            idempotency_key="deliver-1",
        )
    )

    assert artifacts.calls[-1] == (
        "deliver",
        {
            "tenant_id": "tenant-a",
            "actor_id": "owner-1",
            "thread_id": "thread-1",
            "channel": "web",
            "artifact_id": "artifact-1",
            "caption": "Result",
            "idempotency_key": "deliver-1",
        },
    )
    assert output.data == {"ok": True, "delivery_id": "delivery-1"}


@pytest.mark.asyncio
async def test_source_operations_are_owner_only_iterative_and_hide_worktree_paths() -> None:
    evolution = _Port(
        source_status={
            "active": True,
            "candidate_id": "candidate-1",
            "diff_sha256": "d" * 64,
            "current_release_id": "release-current",
            "rollback_target_release_id": "release-prior",
        },
        source_shell={
            "candidate": {
                "id": "candidate-1",
                "status": "building",
                "worktree_path": "/private/evolution/candidate-1",
            },
            "exit_code": 0,
            "output": "tests passed",
        },
        source_release={
            "candidate": {"id": "candidate-1", "status": "ready"},
            "promotion": {"id": "promotion-1", "status": "queued"},
        },
        source_rollback={"id": "rollback-1", "status": "queued"},
    )
    application, ports = _application(evolution=evolution)

    status = await application.source_status(_invocation("source_status", {}, idempotency_key=None))
    shell = await application.source_shell(
        _invocation(
            "source_shell",
            {"command": "pytest -q", "timeout_seconds": 300},
            idempotency_key=None,
        )
    )
    released = await application.source_release(
        _invocation(
            "source_release",
            {
                "expected_candidate_id": "candidate-1",
                "expected_diff_sha256": "d" * 64,
                "message": "Improve website interface",
            },
            idempotency_key="release-1",
        )
    )
    assert ports["idempotency"].calls == []
    rolled_back = await application.source_rollback(
        _invocation(
            "source_rollback",
            {
                "expected_current_release_id": "release-current",
                "expected_target_release_id": "release-prior",
                "reason": "Regression found",
            },
            idempotency_key="rollback-1",
        )
    )

    assert status.data["candidate_id"] == "candidate-1"
    assert shell.data["candidate"] == {"id": "candidate-1", "status": "building"}
    assert shell.data["output"] == "tests passed"
    assert released.data["promotion"]["id"] == "promotion-1"
    assert rolled_back.data == {"id": "rollback-1", "status": "queued"}
    audit_context = {
        "tenant_id": "tenant-a",
        "actor_id": "owner-1",
        "thread_id": "thread-1",
        "correlation_id": "correlation-1",
        "channel": "web",
        "run_kind": "owner",
        "origin": (
            '{"interface":"web","source_id":"test","conversation_id":null,"message_id":null}'
        ),
    }
    assert evolution.calls == [
        ("source_status", {"audit_context": audit_context}),
        (
            "source_shell",
            {
                "command": "pytest -q",
                "timeout_seconds": 300,
                "audit_context": audit_context,
            },
        ),
        (
            "source_release",
            {
                "idempotency_key": "release-1",
                "expected_candidate_id": "candidate-1",
                "expected_diff_sha256": "d" * 64,
                "message": "Improve website interface",
                "audit_context": audit_context,
            },
        ),
        (
            "source_rollback",
            {
                "idempotency_key": "rollback-1",
                "expected_current_release_id": "release-current",
                "expected_target_release_id": "release-prior",
                "reason": "Regression found",
                "audit_context": audit_context,
            },
        ),
    ]
    assert ports["idempotency"].calls == []

    unavailable, _ = _application()
    with pytest.raises(ProductToolApplicationError, match="unavailable"):
        await unavailable.source_status(_invocation("source_status", {}, idempotency_key=None))

    with pytest.raises(ProductToolApplicationError, match="unavailable"):
        await application.source_status(
            _invocation(
                "source_status",
                {},
                tenant_id="tenant-b",
                idempotency_key=None,
            )
        )


@pytest.mark.asyncio
async def test_trace_reads_are_tenant_scoped_and_direct() -> None:
    traces = _Port(
        trace_list=[{"run_id": "run-1", "status": "failed"}],
        trace_get={"run_id": "run-1", "events": []},
    )
    application, _ = _application(traces=traces)

    listed = await application.trace_list(
        _invocation(
            "trace_list",
            {"status": "failed", "limit": 10, "before_run_id": "run-2"},
            idempotency_key=None,
        )
    )
    fetched = await application.trace_get(
        _invocation(
            "trace_get",
            {
                "run_id": "run-1",
                "after_sequence": 0,
                "limit": 200,
                "include_messages": True,
            },
            idempotency_key=None,
        )
    )

    assert listed.data == [{"run_id": "run-1", "status": "failed"}]
    assert fetched.data == {"run_id": "run-1", "events": []}
    assert traces.calls == [
        (
            "trace_list",
            {
                "tenant_id": "tenant-a",
                "status": "failed",
                "limit": 10,
                "before_run_id": "run-2",
            },
        ),
        (
            "trace_get",
            {
                "tenant_id": "tenant-a",
                "run_id": "run-1",
                "after_sequence": 0,
                "limit": 200,
                "include_messages": True,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_external_delivery_replays_from_durable_idempotency_store(tmp_path: Path) -> None:
    artifacts = _Port(
        get={"artifact_id": "artifact-1", "tenant_id": "tenant-a"},
        deliver={"ok": True, "tenant_id": "tenant-a", "delivery_id": "delivery-1"},
    )
    application, _ = _application(
        artifacts=artifacts,
        idempotency=IdempotencyStore(tmp_path / "effects.sqlite"),
    )
    invocation = _invocation(
        "artifact_deliver",
        {"artifact_id": "artifact-1", "caption": None},
        idempotency_key="delivery-replay-1",
    )

    first = await application.artifact_deliver(invocation)
    replay = await application.artifact_deliver(invocation)

    assert first == replay
    assert [name for name, _ in artifacts.calls] == ["get", "deliver"]


@pytest.mark.asyncio
async def test_intake_confirmation_token_is_exposed_only_as_model_handle(
    tmp_path: Path,
) -> None:
    handle = "hash-bound-confirmation-handle-at-least-32-chars"
    intake = _Port(
        get_draft={"id": "draft-1", "tenant_id": "tenant-a", "revision": 2},
        prepare_draft={
            "draft": {"id": "draft-1", "tenant_id": "tenant-a", "revision": 2},
            "proposal": {"workflow_id": "workflow-1", "name": "Lead capture"},
            "confirmation_token": handle,
        },
        activate_draft={
            "draft": {"id": "draft-1", "tenant_id": "tenant-a", "status": "activated"},
            "workflow": {"workflow_id": "workflow-1", "tenant_id": "tenant-a"},
        },
    )
    application, _ = _application(
        intake=intake,
        idempotency=IdempotencyStore(tmp_path / "intake-effects.sqlite"),
    )
    prepare = _invocation(
        "intake_draft_prepare",
        {"draft_id": "draft-1", "expected_revision": 2},
        idempotency_key="prepare-1",
    )

    prepared = await application.intake_draft_prepare(prepare)
    replayed = await application.intake_draft_prepare(prepare)
    activated = await application.intake_draft_activate(
        _invocation(
            "intake_draft_activate",
            {
                "draft_id": "draft-1",
                "expected_revision": 2,
                "confirmation_handle": handle,
            },
            idempotency_key="activate-1",
        )
    )

    assert prepared == replayed
    assert prepared.data["confirmation_handle"] == handle
    assert redact_for_langfuse(prepared.data)["confirmation_handle"] == handle
    assert "confirmation_token" not in prepared.data
    assert [name for name, _ in intake.calls].count("prepare_draft") == 1
    activate_call = next(kwargs for name, kwargs in intake.calls if name == "activate_draft")
    assert activate_call["confirmation_token"] == handle
    assert "confirmation_handle" not in activate_call
    assert activated.data["workflow"]["workflow_id"] == "workflow-1"


@pytest.mark.asyncio
async def test_schedule_list_filters_disabled_and_save_passes_typed_write_and_idempotency() -> None:
    schedules = _Port(
        list=[
            {"id": "enabled", "tenant_id": "tenant-a", "enabled": True},
            {"id": "disabled", "tenant_id": "tenant-a", "enabled": False},
        ],
        save=lambda **kwargs: {
            "id": kwargs["schedule_id"] or "new-schedule",
            "tenant_id": kwargs["tenant_id"],
            "revision": 1,
            **kwargs["write"].model_dump(mode="json"),
        },
    )
    application, _ = _application(schedules=schedules)

    listed = await application.schedule_list(
        _invocation("schedule_list", {"include_disabled": False}, idempotency_key=None)
    )
    saved = await application.schedule_save(
        _invocation(
            "schedule_save",
            {
                "schedule_id": None,
                "expected_revision": None,
                "schedule": {
                    "name": "Morning reminder",
                    "trigger": {
                        "kind": "cron",
                        "expression": "0 9 * * *",
                        "timezone": "Europe/Moscow",
                    },
                    "action": {"kind": "reminder", "message": "Check leads"},
                    "notify_owner": True,
                    "enabled": True,
                },
            },
            idempotency_key="derived-schedule-1",
        )
    )

    assert listed.data == [{"id": "enabled", "enabled": True}]
    save_call = schedules.calls[-1]
    assert save_call[0] == "save"
    assert save_call[1]["tenant_id"] == "tenant-a"
    assert save_call[1]["idempotency_key"] == "derived-schedule-1"
    assert save_call[1]["write"].name == "Morning reminder"
    assert saved.data["id"] == "new-schedule"
    assert "tenant_id" not in saved.data


@pytest.mark.asyncio
async def test_capability_lifecycle_uses_hidden_owner_context_and_opaque_secret_handles() -> None:
    manifest = {
        "name": "telegram",
        "revision": 1,
        "secrets": [{"name": "TELEGRAM_BOT_TOKEN", "required": True}],
    }
    activation = {
        "namespace": "tenant-a",
        "capability_name": "telegram",
        "revision": 1,
        "generation": 1,
        "secret_handles": {"TELEGRAM_BOT_TOKEN": "telegram_bot"},
    }
    capabilities = _Port(
        list=[{"manifest": manifest, "activation": None, "test": None}],
        seed_bundled=(manifest,),
        test={
            "namespace": "tenant-a",
            "capability_name": "telegram",
            "revision": 1,
            "status": "passed",
        },
        activate=activation,
        rollback={**activation, "revision": 0, "generation": 2},
        deactivate={**activation, "generation": 2},
    )
    application, ports = _application(capabilities=capabilities)

    listed = await application.capability_list(
        _invocation("capability_list", {}, idempotency_key=None)
    )
    seeded = await application.capability_seed_bundled(
        _invocation("capability_seed_bundled", {}, idempotency_key="derived-seed")
    )
    tested = await application.capability_test(
        _invocation(
            "capability_test",
            {"capability_name": "telegram", "revision": 1},
            idempotency_key=None,
        )
    )
    activated = await application.capability_activate(
        _invocation(
            "capability_activate",
            {
                "capability_name": "telegram",
                "revision": 1,
                "expected_generation": None,
                "config": {"pairing_mode": "one_time"},
                "secret_handles": {"TELEGRAM_BOT_TOKEN": "telegram_bot"},
            },
            idempotency_key="activate-telegram-v1",
        )
    )
    await application.capability_rollback(
        _invocation(
            "capability_rollback",
            {"capability_name": "telegram", "expected_generation": 1},
            idempotency_key="rollback-telegram",
        )
    )
    await application.capability_deactivate(
        _invocation(
            "capability_deactivate",
            {"capability_name": "telegram", "expected_generation": 2},
            idempotency_key="deactivate-telegram",
        )
    )

    assert listed.data[0]["manifest"]["credential_requirements"] == [
        {"name": "TELEGRAM_BOT_TOKEN", "required": True}
    ]
    assert seeded.data["capabilities"][0]["name"] == "telegram"
    assert "namespace" not in tested.data
    assert activated.data["credential_bindings"] == [
        {"name": "TELEGRAM_BOT_TOKEN", "handle_id": "telegram_bot"}
    ]
    assert capabilities.calls == [
        ("list", {"tenant_id": "tenant-a"}),
        ("seed_bundled", {"tenant_id": "tenant-a", "actor_id": "owner-1"}),
        (
            "test",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "capability_name": "telegram",
                "revision": 1,
            },
        ),
        (
            "activate",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "capability_name": "telegram",
                "revision": 1,
                "expected_generation": None,
                "config": {"pairing_mode": "one_time"},
                "secret_handles": {"TELEGRAM_BOT_TOKEN": "telegram_bot"},
            },
        ),
        (
            "rollback",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "capability_name": "telegram",
                "expected_generation": 1,
                "config": None,
                "secret_handles": None,
            },
        ),
        (
            "deactivate",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "capability_name": "telegram",
                "expected_generation": 2,
            },
        ),
    ]
    assert [name for name, _ in ports["idempotency"].calls] == [
        "execute",
        "execute",
        "execute",
        "execute",
    ]
