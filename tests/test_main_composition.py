from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from opentulpa import __main__ as main_module
from opentulpa.bootstrap.capability_worker_api import CapabilityWorkerClient
from opentulpa.bootstrap.evolution_api import EvolutionClient
from opentulpa.bootstrap.sandbox_api import SandboxExecutionClient
from opentulpa.capabilities import SubprocessWorkerHost
from opentulpa.core.config import Settings
from opentulpa.deep_agent.service import DeepAgentService
from opentulpa.integrations.browser_use_cloud import BrowserUseCloudSessionProvider
from opentulpa.mcp import SQLiteMCPAuditSink, SQLiteMCPIdempotencyStore
from opentulpa.notifications import TriggerNotificationSink
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
    assert "/v2/evolution/candidates" in paths
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


def test_auto_configure_webhook_reuses_composed_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        status_code = 200
        text = ""
        content = b"{}"

        @staticmethod
        def json() -> dict[str, bool]:
            return {"ok": True}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            del args

        def post(self, url: str, *, data: dict[str, Any]) -> FakeResponse:
            calls.append({"url": url, "data": data})
            return FakeResponse()

    monkeypatch.setattr(main_module, "_resolve_public_base_url", lambda: "https://example.test")
    monkeypatch.setattr(httpx, "Client", FakeClient)
    settings = _settings(tmp_path, telegram_bot_token="bot-token")

    main_module._auto_configure_telegram_webhook(
        settings,
        webhook_secret="composed-secret",
    )

    assert len(calls) == 1
    assert calls[0]["url"] == "https://api.telegram.org/botbot-token/setWebhook"
    assert calls[0]["data"]["url"] == "https://example.test/webhook/telegram"
    assert calls[0]["data"]["secret_token"] == "composed-secret"
    assert calls[0]["data"]["allowed_updates"]


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


def test_direct_runtime_keeps_chat_available_when_sandbox_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**_: Any) -> str:
        raise RuntimeError("configured OCI engine must operate in rootless mode")

    monkeypatch.setattr(main_module, "resolve_local_oci_image", unavailable)

    image, execution = main_module._sandbox_execution_configuration(  # noqa: SLF001
        project_root=tmp_path,
        settings=_settings(tmp_path),
    )

    assert image == "opentulpa-tenant-sandbox:test"
    assert execution is not None
    with pytest.raises(RuntimeError, match="sandbox execution is unavailable"):
        execution.execute(tenant_id="tenant", command="pwd", timeout=1)


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
