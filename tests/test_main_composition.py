from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from opentulpa import __main__ as main_module
from opentulpa.bootstrap.capability_worker_api import CapabilityWorkerClient
from opentulpa.bootstrap.evolution_api import EvolutionClient
from opentulpa.bootstrap.sandbox_api import SandboxExecutionClient
from opentulpa.capabilities import SubprocessWorkerHost
from opentulpa.capability_workers.state import TelegramWorkerState
from opentulpa.core.config import Settings
from opentulpa.deep_agent.process_sandbox import RestrictedProcessExecutionProvider
from opentulpa.deep_agent.railway_sandbox import RailwaySandboxExecutionProvider
from opentulpa.deep_agent.service import DeepAgentService
from opentulpa.integrations.browser_use_cloud import BrowserUseCloudSessionProvider
from opentulpa.mcp import SQLiteMCPAuditSink, SQLiteMCPIdempotencyStore
from opentulpa.notifications import TriggerNotificationSink
from opentulpa.sandbox.client import SandboxWorkerExecutionProvider
from opentulpa.secrets.host_key import load_or_create_host_cipher
from opentulpa.tooling import TOOL_SPECS


def _settings(root: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "OPENAI_COMPATIBLE_API_KEY": "test-model-key",
        "opentulpa_owner_token": "test-owner-token",
        "telegram_bot_token": "",
        "composio_api_key": "",
        "deepagents_checkpoint_db_path": str(root / "runtime" / "checkpoints.db"),
        "deepagents_store_db_path": str(root / "runtime" / "store.db"),
        "deepagents_runs_db_path": str(root / "runtime" / "runs.db"),
        "deepagents_workspaces_root": str(root / "runtime" / "workspaces"),
        "intake_drafts_db_path": str(root / "runtime" / "intake_drafts.db"),
        "browser_use_user_data_dir": str(root / "runtime" / "browser_profiles"),
        "sandbox_image": "opentulpa-tenant-sandbox:test",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(autouse=True)
def _resolve_reviewed_sandbox_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_module,
        "resolve_local_oci_image",
        lambda **_: f"sha256:{'a' * 64}",
    )


def test_resolve_path_is_project_relative_and_preserves_absolute_paths(tmp_path: Path) -> None:
    assert main_module._resolve_path(tmp_path, "state/store.db") == (
        tmp_path / "state" / "store.db"
    ).resolve()
    absolute = (tmp_path / "elsewhere" / "runs.db").resolve()
    assert main_module._resolve_path(tmp_path / "project", absolute) == absolute


def test_dynamic_child_owns_telegram_identity_seeding_and_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir(mode=0o700)
    capability_state_root = data_root / ".opentulpa" / "deepagents" / "capability_state"
    monkeypatch.setenv("OPENTULPA_TELEGRAM_OWNER_ID", "7")

    main_module._seed_dynamic_host_telegram_identity(
        data_root=data_root,
        capability_state_root=capability_state_root,
        tenant_id="owner",
    )

    state_path = next(capability_state_root.glob("*/telegram.json"))
    assert TelegramWorkerState(state_path).paired_identity() == (7, 7)
    assert state_path.stat().st_mode & 0o777 == 0o600

    monkeypatch.setenv("OPENTULPA_TELEGRAM_OWNER_ID", "")
    main_module._seed_dynamic_host_telegram_identity(
        data_root=data_root,
        capability_state_root=capability_state_root,
        tenant_id="owner",
    )

    assert not state_path.exists()


def test_dynamic_child_storage_parents_are_private_before_telegram_seeding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / ".opentulpa"
    deepagents_root = data_root / "deepagents"
    artifacts_root = deepagents_root / "artifacts"
    for directory in (data_root, deepagents_root, artifacts_root):
        main_module._private_runtime_directory(directory)
    capability_state_root = deepagents_root / "capability_state"
    capability_state_root.mkdir(mode=0o700)
    monkeypatch.setenv("OPENTULPA_TELEGRAM_OWNER_ID", "")

    main_module._seed_dynamic_host_telegram_identity(
        data_root=data_root,
        capability_state_root=capability_state_root,
        tenant_id="owner",
    )

    assert data_root.stat().st_mode & 0o777 == 0o700
    assert deepagents_root.stat().st_mode & 0o777 == 0o700
    assert artifacts_root.stat().st_mode & 0o777 == 0o700


def test_missing_model_key_fails_before_creating_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = _settings(tmp_path, OPENAI_COMPATIBLE_API_KEY="")

    with pytest.raises(RuntimeError, match="OPENAI_COMPATIBLE_API_KEY is required"):
        main_module.build_application(project_root=tmp_path, settings=settings)

    assert list(tmp_path.iterdir()) == []


def test_storage_bootstrap_refuses_to_delete_conflicting_source_data(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "data"
    source = project_root / ".opentulpa"
    target = data_root / ".opentulpa"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    (source / "state.db").write_bytes(b"source")
    (target / "state.db").write_bytes(b"target")

    with pytest.raises(RuntimeError, match="file conflict"):
        main_module._bootstrap_persistent_storage(project_root, str(data_root))

    assert source.is_dir()
    assert (source / "state.db").read_bytes() == b"source"
    assert (target / "state.db").read_bytes() == b"target"


def test_secret_vault_host_key_is_created_once_with_private_permissions(
    tmp_path: Path,
) -> None:
    first = load_or_create_host_cipher(tmp_path)
    encrypted = first.encrypt(b"credential", associated_data=b"tenant")
    second = load_or_create_host_cipher(tmp_path)
    key_path = tmp_path / "bootstrap" / "secret-vault.key"

    assert key_path.stat().st_mode & 0o777 == 0o600
    assert second.decrypt(encrypted, associated_data=b"tenant") == b"credential"


def test_secret_vault_refuses_a_group_readable_host_key(tmp_path: Path) -> None:
    key_path = tmp_path / "bootstrap" / "secret-vault.key"
    key_path.parent.mkdir(parents=True)
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o640)

    with pytest.raises(RuntimeError, match="permissions"):
        load_or_create_host_cipher(tmp_path)


def test_owner_tenant_id_is_persisted_across_profile_changes(tmp_path: Path) -> None:
    class Profiles:
        storage_ids = ["tenant-a"]

        def resolve_customer_id(self, customer_id: str) -> str:
            return customer_id

        def resolve_telegram_customer_id(self, telegram_user_id: str) -> str:
            return f"telegram_{telegram_user_id}"

        def list_profiles(self) -> list[Any]:
            return [SimpleNamespace(storage_user_id=value) for value in self.storage_ids]

    settings = _settings(tmp_path)
    data_root = tmp_path / ".opentulpa"
    profiles = Profiles()

    assert main_module._resolve_owner_tenant_id(settings, profiles, data_root=data_root) == "tenant-a"
    assert (data_root / "bootstrap" / "owner-tenant-id").read_text(encoding="utf-8").strip() == "tenant-a"

    profiles.storage_ids = ["tenant-a", "tenant-b"]

    assert main_module._resolve_owner_tenant_id(settings, profiles, data_root=data_root) == "tenant-a"


def test_configured_owner_tenant_id_updates_persisted_identity(tmp_path: Path) -> None:
    class Profiles:
        def resolve_customer_id(self, customer_id: str) -> str:
            return {"configured-owner": "tenant-b"}.get(customer_id, customer_id)

        def resolve_telegram_customer_id(self, telegram_user_id: str) -> str:
            return f"telegram_{telegram_user_id}"

        def list_profiles(self) -> list[Any]:
            return [SimpleNamespace(storage_user_id="tenant-a")]

    data_root = tmp_path / ".opentulpa"
    settings = _settings(tmp_path, opentulpa_owner_customer_id="configured-owner")

    assert main_module._resolve_owner_tenant_id(settings, Profiles(), data_root=data_root) == "tenant-b"
    assert (data_root / "bootstrap" / "owner-tenant-id").read_text(encoding="utf-8").strip() == "tenant-b"


def test_owner_tenant_id_adopts_single_existing_codex_credential(tmp_path: Path) -> None:
    class Profiles:
        def resolve_customer_id(self, customer_id: str) -> str:
            return customer_id

        def resolve_telegram_customer_id(self, telegram_user_id: str) -> str:
            return f"telegram_{telegram_user_id}"

        def list_profiles(self) -> list[Any]:
            return [
                SimpleNamespace(storage_user_id="tenant-a"),
                SimpleNamespace(storage_user_id="tenant-b"),
            ]

    data_root = tmp_path / ".opentulpa"
    db_path = data_root / "deepagents" / "inference.db"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE codex_credentials (tenant_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO codex_credentials VALUES ('tenant-codex')")

    assert (
        main_module._resolve_owner_tenant_id(_settings(tmp_path), Profiles(), data_root=data_root)
        == "tenant-codex"
    )
    assert (
        (data_root / "bootstrap" / "owner-tenant-id").read_text(encoding="utf-8").strip()
        == "tenant-codex"
    )


def test_build_application_composes_only_v2_product_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "get_web_search_provider", lambda: None)
    composition = main_module.build_application(
        project_root=tmp_path,
        settings=_settings(tmp_path),
    )
    app = composition.app

    assert app.state.owner_tenant_id == "owner"
    tool_names = {tool.name for tool in app.state.product_tools}
    assert tool_names == {spec.name for spec in TOOL_SPECS} - {"web_search"}
    assert "content_fetch" in tool_names
    assert app.state.evolution_service is None
    assert app.state.job_registry.names() == (
        "browser_act",
        "browser_start",
        "file_analyze",
        "intake_workflow_test",
        "integration_invoke",
        "knowledge_attach",
        "knowledge_reindex",
    )
    paths = {str(getattr(route, "path", "")) for route in app.routes}
    assert "/healthz" in paths
    assert "/webhook/telegram" in paths
    assert "/webhook/composio/callback" in paths
    assert "/v2/agent/runs" in paths
    assert "/v2/evolution/candidates" not in paths
    assert not any(path.startswith("/web/") or path.startswith("/internal/") for path in paths)

    agent = app.state.agent_service
    assert agent._checkpoint_db_path == tmp_path / "runtime" / "checkpoints.db"
    assert agent._store_db_path == tmp_path / "runtime" / "store.db"
    assert agent._runs_db_path == tmp_path / "runtime" / "runs.db"
    assert agent._workspaces_root == tmp_path / "runtime" / "workspaces"
    assert app.state.schedule_service._trigger_specs._store.db_path == (  # noqa: SLF001
        tmp_path / ".opentulpa" / "deepagents" / "trigger_specs.db"
    )
    assert (
        app.state.intake_draft_service._store.db_path
        == tmp_path / "runtime" / "intake_drafts.db"
    )
    assert app.state.notifications._store.db_path == (  # noqa: SLF001
        tmp_path / ".opentulpa" / "notifications.db"
    )
    capability_host = app.state.capability_service._workers._host  # noqa: SLF001
    assert isinstance(capability_host, SubprocessWorkerHost)
    assert app.state.capability_service._release_state_path == (  # noqa: SLF001
        tmp_path
        / ".opentulpa"
        / "deepagents"
        / "capability_state"
        / "seed_activations.json"
    )
    mcp_runtime = app.state.capability_service._tool_host  # noqa: SLF001
    mcp_broker = mcp_runtime._broker  # noqa: SLF001
    assert isinstance(mcp_broker._audit, SQLiteMCPAuditSink)  # noqa: SLF001
    assert isinstance(mcp_broker._idempotency, SQLiteMCPIdempotencyStore)  # noqa: SLF001
    assert mcp_broker._audit.db_path == (  # noqa: SLF001
        tmp_path / ".opentulpa" / "deepagents" / "mcp.db"
    )
    assert mcp_broker._idempotency.db_path == mcp_broker._audit.db_path  # noqa: SLF001


def test_build_application_exposes_web_search_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "get_web_search_provider", object)

    composition = main_module.build_application(
        project_root=tmp_path,
        settings=_settings(tmp_path),
    )

    assert "web_search" in {tool.name for tool in composition.app.state.product_tools}


def test_build_application_injects_browser_use_cloud_session_provider(tmp_path: Path) -> None:
    composition = main_module.build_application(
        project_root=tmp_path,
        settings=_settings(tmp_path, browser_use_api_key="browser-use-test-key"),
    )

    provider = composition.app.state.browser_service._session_provider  # noqa: SLF001
    assert isinstance(provider, BrowserUseCloudSessionProvider)
    assert provider._profile_metadata_root == (  # noqa: SLF001
        tmp_path / "runtime" / "browser_profiles" / ".browser-use-cloud"
    )


def test_build_application_wires_meta_messenger_webhook_configuration(
    tmp_path: Path,
) -> None:
    composition = main_module.build_application(
        project_root=tmp_path,
        settings=_settings(
            tmp_path,
            meta_messenger_verify_token="verify-token",
            meta_app_secret="app-secret",
            meta_messenger_trigger_id="messenger-intake",
        ),
    )

    routes = {
        (route.path, method)
        for route in composition.app.routes
        for method in getattr(route, "methods", set())
    }
    assert ("/webhook/meta/messenger", "GET") in routes
    assert ("/webhook/meta/messenger", "POST") in routes


def test_build_application_separates_business_webhook_from_owner_notifications(
    tmp_path: Path,
) -> None:
    composition = main_module.build_application(
        project_root=tmp_path,
        settings=_settings(
            tmp_path,
            telegram_bot_token="test-bot-token",
            telegram_webhook_secret="test-webhook-secret",
        ),
    )
    app = composition.app
    relay = app.state.telegram_business_relay
    delivery = app.state.trigger_dispatcher._deliver  # noqa: SLF001

    assert relay is not None
    assert isinstance(delivery, TriggerNotificationSink)
    assert delivery._notifications is app.state.notifications  # noqa: SLF001


@pytest.mark.asyncio
async def test_managed_release_health_includes_configured_consumer_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SyncHealthProbe:
        def __init__(self, healthy: bool) -> None:
            self.value = healthy

        def healthy(self) -> bool:
            return self.value

    class AsyncHealthProbe:
        def __init__(self, healthy: bool) -> None:
            self.value = healthy

        async def healthy(self) -> bool:
            return self.value

    agent = SyncHealthProbe(True)
    capabilities = AsyncHealthProbe(False)
    monkeypatch.setenv("OPENTULPA_RELEASE_ID", "release-health-test")
    monkeypatch.setenv("OPENTULPA_CONTROL_TOKEN", "t" * 32)
    monkeypatch.delenv("OPENTULPA_DISABLE_CONSUMERS", raising=False)

    service = main_module._build_release_control_service(
        agent_service=cast(DeepAgentService, agent),
        resolve_agent_spec=lambda *_: None,
        secret_ingress=cast(Any, object()),
        notifications=cast(Any, object()),
        capabilities=cast(Any, capabilities),
    )

    assert service is not None
    report = await service.health()
    assert report.healthy is False
    assert report.components == {
        "runtime": True,
        "agent_api": True,
        "capabilities": False,
    }

    monkeypatch.setenv("OPENTULPA_DISABLE_CONSUMERS", "true")
    staging_report = await service.health()
    assert staging_report.healthy is True
    assert staging_report.components["capabilities"] is True


@pytest.mark.asyncio
async def test_deferred_agent_fails_closed_then_delegates() -> None:
    class FakeAgent:
        async def run(self, request: Any) -> Any:
            return request

    proxy = main_module._DeferredAgentService()
    with pytest.raises(RuntimeError, match="not bound"):
        proxy.require()

    agent = FakeAgent()
    proxy.bind(cast(DeepAgentService, agent))
    sentinel = cast(Any, object())
    assert await proxy.run(sentinel) is sentinel


def test_runtime_and_migration_paths_share_defaults() -> None:
    settings = Settings(openai_compatible_api_key="test-model-key")

    assert settings.deepagents_store_db_path == ".opentulpa/deepagents/store.db"
    assert settings.intake_drafts_db_path == ".opentulpa/deepagents/intake_drafts.db"


def test_container_cli_uses_same_deployment_alias_as_start_script() -> None:
    settings = Settings(
        openai_compatible_api_key="test-model-key",
        OPENTULPA_CONTAINER_CLI="podman",
    )

    assert settings.sandbox_container_cli == "podman"


def test_direct_runtime_resolves_configured_sandbox_to_local_image_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def resolve(**kwargs: Any) -> str:
        calls.append(kwargs)
        return f"sha256:{'b' * 64}"

    monkeypatch.setattr(main_module, "resolve_local_oci_image", resolve)

    image, execution = main_module._sandbox_execution_configuration(  # noqa: SLF001
        project_root=tmp_path,
        settings=_settings(tmp_path, OPENTULPA_CONTAINER_CLI="podman"),
    )

    assert image == f"sha256:{'b' * 64}"
    assert execution is None
    assert calls == [
        {
            "container_cli": "podman",
            "image": "opentulpa-tenant-sandbox:test",
            "cwd": tmp_path,
            "allow_desktop_vm": False,
        }
    ]


def test_direct_runtime_fails_closed_when_sandbox_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**_: Any) -> str:
        raise RuntimeError("configured OCI engine must operate in rootless mode")

    monkeypatch.setattr(main_module, "resolve_local_oci_image", unavailable)
    monkeypatch.setattr(
        main_module.RestrictedProcessExecutionProvider,
        "supported",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="requires a healthy sandbox worker"):
        main_module._sandbox_execution_configuration(  # noqa: SLF001
            project_root=tmp_path,
            settings=_settings(tmp_path),
        )


def test_direct_runtime_allows_missing_sandbox_only_in_explicit_dev_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**_: Any) -> str:
        raise RuntimeError("configured OCI engine must operate in rootless mode")

    monkeypatch.setenv("OPENTULPA_DEV_ALLOW_NO_SANDBOX", "1")
    monkeypatch.setattr(main_module, "resolve_local_oci_image", unavailable)
    monkeypatch.setattr(
        main_module.RestrictedProcessExecutionProvider,
        "supported",
        lambda: False,
    )

    image, execution = main_module._sandbox_execution_configuration(  # noqa: SLF001
        project_root=tmp_path,
        settings=_settings(tmp_path),
    )

    assert image == "opentulpa-tenant-sandbox:test"
    assert execution is not None
    with pytest.raises(RuntimeError, match="sandbox execution is unavailable"):
        execution.execute(tenant_id="tenant", command="pwd", timeout=1)


def test_direct_runtime_uses_restricted_process_sandbox_without_oci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**_: Any) -> str:
        raise RuntimeError("configured OCI engine must operate in rootless mode")

    monkeypatch.setattr(main_module, "resolve_local_oci_image", unavailable)
    monkeypatch.setattr(
        main_module.RestrictedProcessExecutionProvider,
        "supported",
        lambda: True,
    )

    image, execution = main_module._sandbox_execution_configuration(  # noqa: SLF001
        project_root=tmp_path,
        settings=_settings(tmp_path),
    )

    assert image == "opentulpa-tenant-sandbox:test"
    assert isinstance(execution, RestrictedProcessExecutionProvider)


def test_direct_railway_runtime_uses_hosted_sandbox_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAILWAY_TOKEN", "project-token")
    monkeypatch.setenv(
        "OPENTULPA_SANDBOX_RAILWAY_ENVIRONMENT_ID",
        "environment-id",
    )
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "app-environment-id")
    bridge = tmp_path / "bridge" / "bridge.mjs"
    bridge.parent.mkdir()
    bridge.write_text("", encoding="utf-8")
    (bridge.parent / "node_modules" / "railway").mkdir(parents=True)
    (bridge.parent / "node_modules" / "railway" / "package.json").write_text(
        "{}",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENTULPA_RAILWAY_SANDBOX_BRIDGE_PATH", str(bridge))

    image, execution = main_module._sandbox_execution_configuration(  # noqa: SLF001
        project_root=tmp_path,
        settings=_settings(tmp_path),
    )

    assert image == "opentulpa-tenant-sandbox:test"
    assert isinstance(execution, RailwaySandboxExecutionProvider)
    assert execution._environment_id == "environment-id"  # noqa: SLF001


def test_direct_runtime_uses_mandatory_sandbox_worker_when_wired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENTULPA_SANDBOX_RPC_URL",
        "http://127.0.0.1:8787/internal/v1/sandbox",
    )
    monkeypatch.setenv("OPENTULPA_SANDBOX_RPC_TOKEN", "s" * 48)

    image, execution = main_module._sandbox_execution_configuration(  # noqa: SLF001
        project_root=tmp_path,
        settings=_settings(tmp_path),
    )

    assert image == "opentulpa-tenant-sandbox:test"
    assert isinstance(execution, SandboxWorkerExecutionProvider)


def test_explicit_railway_runtime_fails_when_project_credentials_are_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAILWAY_TOKEN", "project-token")
    monkeypatch.delenv(
        "OPENTULPA_SANDBOX_RAILWAY_ENVIRONMENT_ID",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="RAILWAY_TOKEN and OPENTULPA_SANDBOX_RAILWAY_ENVIRONMENT_ID",
    ):
        main_module._sandbox_execution_configuration(  # noqa: SLF001
            project_root=tmp_path,
            settings=_settings(tmp_path, sandbox_provider="railway"),
        )


def test_managed_production_uses_lease_bound_stable_sandbox_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_MANAGED_RELEASE", "true")
    monkeypatch.setenv("OPENTULPA_RELEASE_MODE", "production")
    monkeypatch.setenv("OPENTULPA_RELEASE_ID", "release-test")
    monkeypatch.setenv("OPENTULPA_LEASE_EPOCH", "4")
    monkeypatch.setenv("OPENTULPA_CONTROL_TOKEN", "c" * 48)
    monkeypatch.setenv(
        "OPENTULPA_BOOTSTRAP_SANDBOX_URL",
        "http://host.docker.internal:8000/bootstrap/internal/v1/sandbox",
    )
    monkeypatch.setenv("OPENTULPA_BOOTSTRAP_SANDBOX_TOKEN", "s" * 48)

    image, execution = main_module._sandbox_execution_configuration(  # noqa: SLF001
        project_root=tmp_path,
        settings=_settings(tmp_path),
    )

    assert image == "opentulpa-tenant-sandbox:test"
    assert isinstance(execution, SandboxExecutionClient)


def test_managed_production_fails_without_stable_sandbox_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_MANAGED_RELEASE", "true")
    monkeypatch.setenv("OPENTULPA_RELEASE_MODE", "production")
    monkeypatch.delenv("OPENTULPA_BOOTSTRAP_SANDBOX_URL", raising=False)
    monkeypatch.delenv("OPENTULPA_BOOTSTRAP_SANDBOX_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="stable sandbox execution service"):
        main_module._sandbox_execution_configuration(  # noqa: SLF001
            project_root=tmp_path,
            settings=_settings(tmp_path),
        )


def test_managed_production_uses_lease_bound_stable_capability_worker_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_MANAGED_RELEASE", "true")
    monkeypatch.setenv("OPENTULPA_RELEASE_MODE", "production")
    monkeypatch.setenv("OPENTULPA_RELEASE_ID", "release-test")
    monkeypatch.setenv("OPENTULPA_LEASE_EPOCH", "4")
    monkeypatch.setenv("OPENTULPA_CONTROL_TOKEN", "c" * 48)
    monkeypatch.setenv(
        "OPENTULPA_BOOTSTRAP_CAPABILITY_WORKER_URL",
        "http://host.docker.internal:8000/bootstrap/internal/v1/capability-workers",
    )
    monkeypatch.setenv("OPENTULPA_BOOTSTRAP_CAPABILITY_WORKER_TOKEN", "w" * 48)

    host = main_module._capability_worker_host(tmp_path)  # noqa: SLF001

    assert isinstance(host, CapabilityWorkerClient)


def test_managed_production_fails_without_stable_capability_worker_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_MANAGED_RELEASE", "true")
    monkeypatch.setenv("OPENTULPA_RELEASE_MODE", "production")
    monkeypatch.delenv("OPENTULPA_BOOTSTRAP_CAPABILITY_WORKER_URL", raising=False)
    monkeypatch.delenv("OPENTULPA_BOOTSTRAP_CAPABILITY_WORKER_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="stable capability worker service"):
        main_module._capability_worker_host(tmp_path)  # noqa: SLF001


def test_managed_release_registers_control_plane_and_uses_bootstrap_evolution_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_RELEASE_ID", "release_test")
    monkeypatch.setenv("OPENTULPA_LEASE_EPOCH", "7")
    monkeypatch.setenv("OPENTULPA_CONTROL_TOKEN", "c" * 48)
    monkeypatch.setenv(
        "OPENTULPA_BOOTSTRAP_EVOLUTION_URL",
        "http://host.docker.internal:8000/bootstrap/internal/v1/evolution",
    )
    monkeypatch.setenv("OPENTULPA_BOOTSTRAP_EVOLUTION_TOKEN", "e" * 48)

    composition = main_module.build_application(
        project_root=tmp_path,
        settings=_settings(tmp_path, evolution_enabled=True),
    )

    app = composition.app
    paths = {str(getattr(route, "path", "")) for route in app.routes}
    assert app.state.release_control is not None
    assert app.state.release_control.release_id == "release_test"
    assert app.state.release_control.lease_epoch == 7
    assert app.state.evolution_service is not None
    assert isinstance(app.state.evolution_service, EvolutionClient)
    assert "/_control/v1/health" in paths
    assert "/_control/v1/ingress" in paths
    assert "/_control/v1/events" in paths
