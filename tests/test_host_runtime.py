from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from opentulpa.host.models import HostConfig
from opentulpa.host.runtime import RuntimeSupervisor


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
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    config = _config()

    environment = runtime._child_environment(config, port=8123)
    runtime._redaction_values = {
        config.api_key.get_secret_value(),
        config.internal_runtime_token.get_secret_value(),
    }
    runtime._append_log(
        "stderr",
        "provider-secret-value Authorization=internal-owner-secret-value password=hunter2",
    )

    assert environment["OPENAI_COMPATIBLE_API_KEY"] == "provider-secret-value"
    assert environment["OPENTULPA_OWNER_TOKEN"] == "internal-owner-secret-value"
    assert "TELEGRAM_BOT_TOKEN" not in environment
    assert "TELEGRAM_WEBHOOK_SECRET" not in environment
    line = runtime.logs()[0].text
    assert "provider-secret-value" not in line
    assert "internal-owner-secret-value" not in line
    assert "hunter2" not in line
    assert line.count("[redacted]") == 3
    await runtime.shutdown()
