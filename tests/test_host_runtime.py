from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from opentulpa.capability_workers.state import TelegramWorkerState
from opentulpa.host.models import HostConfig
from opentulpa.host.runtime import RuntimeSupervisor, RuntimeUnavailableError
from opentulpa.persistence.tenant_namespace import tenant_namespace_label


def _config() -> HostConfig:
    return HostConfig(
        revision=1,
        status="active",
        api_key=SecretStr("provider-secret-value"),
        base_url="https://models.example/v1",
        model="moonshotai/kimi-k3",
        telegram_bot_token=SecretStr("telegram-secret-value"),
        telegram_user_id=7,
        internal_runtime_token=SecretStr("internal-owner-secret-value"),
        telegram_pairing_code=SecretStr("pairing-secret-value"),
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_child_environment_hides_interface_secrets_and_logs_redact_exact_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "host-telegram-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "host-webhook-secret")
    monkeypatch.setenv("OPENTULPA_OWNER_CUSTOMER_ID", "opentulpa-gf")
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    runtime.configure_evolution_control(
        base_url="http://127.0.0.1:8000/bootstrap/internal/v1/evolution",
        token="e" * 48,
    )
    runtime.configure_sandbox_worker(
        base_url="http://127.0.0.1:8787/internal/v1/sandbox",
        token="s" * 48,
    )
    config = _config()

    environment = runtime._child_environment(config, port=8123)
    runtime._redaction_values = {
        config.api_key.get_secret_value(),
        config.internal_runtime_token.get_secret_value(),
        "s" * 48,
    }
    runtime._append_log(
        "stderr",
        "provider-secret-value Authorization=internal-owner-secret-value "
        f"password=hunter2 sandbox={'s' * 48}",
    )

    assert environment["OPENAI_COMPATIBLE_API_KEY"] == "provider-secret-value"
    assert environment["OPENTULPA_OWNER_TOKEN"] == "internal-owner-secret-value"
    assert environment["OPENTULPA_OWNER_CUSTOMER_ID"] == "opentulpa-gf"
    assert environment["OPENTULPA_BOOTSTRAP_EVOLUTION_TOKEN"] == "e" * 48
    assert environment["OPENTULPA_BOOTSTRAP_EVOLUTION_URL"].endswith(
        "/bootstrap/internal/v1/evolution"
    )
    assert environment["OPENTULPA_SANDBOX_RPC_URL"].endswith("/internal/v1/sandbox")
    assert environment["OPENTULPA_SANDBOX_RPC_TOKEN"] == "s" * 48
    assert environment["PYTHONPATH"] == str(tmp_path / "src")
    assert "TELEGRAM_BOT_TOKEN" not in environment
    assert "TELEGRAM_WEBHOOK_SECRET" not in environment
    line = runtime.logs()[0].text
    assert runtime.logs()[0].stream_id == runtime.log_stream_id
    assert "provider-secret-value" not in line
    assert "internal-owner-secret-value" not in line
    assert "hunter2" not in line
    assert "s" * 48 not in line
    assert line.count("[redacted]") == 4

    monkeypatch.delenv("OPENTULPA_OWNER_CUSTOMER_ID")
    assert runtime._child_environment(config, port=8124)["OPENTULPA_OWNER_CUSTOMER_ID"] == "owner"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_evolved_runtime_uses_stable_host_railway_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = tmp_path / "railway_sandbox_bridge" / "bridge.mjs"
    bridge.parent.mkdir()
    bridge.write_text("", encoding="utf-8")
    candidate_root = tmp_path / "candidate"
    package = candidate_root / "src" / "opentulpa"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv(
        "OPENTULPA_RAILWAY_SANDBOX_BRIDGE_PATH",
        "/untrusted/inherited/bridge.mjs",
    )
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")

    runtime.set_project_root(candidate_root)
    environment = runtime._child_environment(
        _config(),
        port=8123,
        project_root=candidate_root,
    )

    assert environment["PYTHONPATH"] == str(candidate_root / "src")
    assert environment["OPENTULPA_RAILWAY_SANDBOX_BRIDGE_PATH"] == str(bridge)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_host_seeds_owner_identity_in_runtime_telegram_state(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=data_root)
    expected_path = (
        data_root
        / ".opentulpa"
        / "deepagents"
        / "capability_state"
        / tenant_namespace_label("owner")
        / "telegram.json"
    )

    runtime._seed_telegram_identity(_config())

    assert runtime._telegram_state_path() == expected_path
    assert TelegramWorkerState(expected_path).paired_identity() == (7, 7)

    runtime.clear_telegram_identity()

    assert not expected_path.exists()
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_unhealthy_source_swap_restores_previous_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_root = tmp_path / "previous"
    candidate_root = tmp_path / "candidate"
    for root in (previous_root, candidate_root):
        package = root / "src" / "opentulpa"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    runtime = RuntimeSupervisor(project_root=previous_root, data_root=tmp_path / "data")
    previous = SimpleNamespace(project_root=previous_root, config=_config())
    runtime._child = previous  # type: ignore[assignment]
    spawned: list[Path] = []

    async def stop_child(child: Any) -> None:
        assert child is previous

    async def spawn(config: HostConfig, *, project_root: Path | None = None) -> Any:
        del config
        assert project_root is not None
        spawned.append(project_root)
        if project_root == candidate_root:
            raise RuntimeUnavailableError("candidate is unhealthy")
        return previous

    monkeypatch.setattr(runtime, "_stop_child", stop_child)
    monkeypatch.setattr(runtime, "_spawn", spawn)

    with pytest.raises(RuntimeUnavailableError, match="candidate is unhealthy"):
        await runtime.replace_source(candidate_root)

    assert spawned == [candidate_root, previous_root]
    assert runtime._child is previous
    assert runtime.project_root == previous_root
    runtime._child = None
