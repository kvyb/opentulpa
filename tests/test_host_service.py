from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from opentulpa.host.models import HostConfig, HostConfigInput
from opentulpa.host.runtime import RuntimeUnavailableError
from opentulpa.host.service import HostActivationError, HostService
from opentulpa.host.store import HostStore
from opentulpa.secrets.cipher import AesGcmHostKeyCipher


class _Runtime:
    def __init__(self) -> None:
        self.current: HostConfig | None = None
        self.replacements: list[int] = []
        self.status = "stopped"
        self.error = None
        self.endpoint = "http://runtime.test"

    @property
    def revision(self) -> int | None:
        return self.current.revision if self.current else None

    async def replace(self, config: HostConfig, *, rollback: HostConfig | None) -> None:
        self.current = config
        self.replacements.append(config.revision)
        self.status = "ready"

    async def start(self, config: HostConfig) -> None:
        self.current = config

    async def stop(self) -> None:
        self.current = None

    async def shutdown(self) -> None:
        self.current = None

    async def restart_current(self) -> None:
        return None

    def clear_telegram_identity(self) -> None:
        return None


class _Service(HostService):
    def __init__(self, *, store: HostStore, runtime: _Runtime) -> None:
        super().__init__(
            store=store,
            runtime=runtime,  # type: ignore[arg-type]
        )
        self.fail_revision: int | None = None
        self.configured_revisions: list[int] = []

    async def _validate_external(
        self, value: HostConfigInput, *, previous: HostConfig | None
    ) -> None:
        return None

    async def _configure_telegram(self, config: HostConfig) -> None:
        self.configured_revisions.append(config.revision)
        if config.revision == self.fail_revision:
            raise HostActivationError("Telegram worker failed readiness")


class _Evolution:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def prepare(self) -> None:
        self.events.append("prepare")

    async def start(self) -> None:
        self.events.append("start")

    async def shutdown(self) -> None:
        self.events.append("shutdown")


@pytest.mark.asyncio
async def test_shutdown_contains_runtime_after_evolution_shutdown_failure(tmp_path: Path) -> None:
    store = HostStore(tmp_path / "host.db", cipher=AesGcmHostKeyCipher(b"q" * 32))
    runtime = _Runtime()
    staged = store.stage(HostConfigInput(api_key=SecretStr("provider-secret")))
    runtime.current = store.activate(staged.revision)

    class FailingEvolution(_Evolution):
        async def shutdown(self) -> None:
            self.events.append("shutdown")
            raise RuntimeError("configured evolution shutdown failure")

    evolution = FailingEvolution()
    service = HostService(
        store=store,
        runtime=runtime,  # type: ignore[arg-type]
        evolution=evolution,
    )

    with pytest.raises(RuntimeError, match="configured evolution shutdown failure"):
        await service.shutdown()

    assert runtime.current is None
    assert evolution.events == ["shutdown"]


@pytest.mark.asyncio
async def test_host_prepares_evolution_before_runtime_and_starts_it_afterward(
    tmp_path: Path,
) -> None:
    store = HostStore(tmp_path / "host.db", cipher=AesGcmHostKeyCipher(b"e" * 32))
    staged = store.stage(HostConfigInput(api_key=SecretStr("provider-secret")))
    store.activate(staged.revision)
    runtime = _Runtime()
    evolution = _Evolution()
    service = HostService(
        store=store,
        runtime=runtime,  # type: ignore[arg-type]
        evolution=evolution,
    )

    await service.start()

    assert runtime.current is not None
    assert evolution.events == ["prepare", "start"]
    await service.shutdown()
    assert evolution.events == ["prepare", "start", "shutdown"]


@pytest.mark.asyncio
async def test_host_start_applies_bootstrap_overlay_when_active_config_lacks_telegram(
    tmp_path: Path,
) -> None:
    store = HostStore(tmp_path / "host.db", cipher=AesGcmHostKeyCipher(b"b" * 32))
    first = store.stage(HostConfigInput(api_key=SecretStr("provider-secret")))
    first = store.activate(first.revision)
    runtime = _Runtime()
    service = _Service(store=store, runtime=runtime)
    service._bootstrap_config = HostConfigInput(  # noqa: SLF001
        expected_revision=first.revision,
        base_url=first.base_url,
        model=first.model,
        telegram_bot_token=SecretStr("123:telegram-secret"),
        telegram_user_id=7,
    )

    await service.start()

    active = store.active()
    assert active is not None
    assert active.revision == first.revision + 1
    assert active.api_key.get_secret_value() == "provider-secret"
    assert active.telegram_bot_token == SecretStr("123:telegram-secret")
    assert active.telegram_user_id == 7
    assert runtime.current is not None
    assert runtime.current.revision == active.revision
    assert runtime.replacements == [active.revision]
    assert service.configured_revisions == [active.revision]


@pytest.mark.asyncio
async def test_failed_candidate_keeps_previous_revision_and_runtime(tmp_path: Path) -> None:
    store = HostStore(tmp_path / "host.db", cipher=AesGcmHostKeyCipher(b"s" * 32))
    first = store.stage(HostConfigInput(api_key=SecretStr("first-secret")))
    first = store.activate(first.revision)
    runtime = _Runtime()
    runtime.current = first
    service = _Service(store=store, runtime=runtime)
    service.fail_revision = first.revision + 1

    with pytest.raises(HostActivationError, match="Telegram worker failed readiness"):
        await service.apply(
            HostConfigInput(
                expected_revision=first.revision,
                api_key=SecretStr("candidate-secret"),
            )
        )

    assert store.active().revision == first.revision  # type: ignore[union-attr]
    assert runtime.current is not None
    assert runtime.current.revision == first.revision
    assert runtime.replacements == [first.revision + 1, first.revision]
    assert service.configured_revisions == [first.revision + 1, first.revision]
    assert store.get(first.revision + 1).status == "failed"  # type: ignore[union-attr]
    await service.shutdown()


@pytest.mark.asyncio
async def test_failed_candidate_preserves_runtime_unavailable_message(tmp_path: Path) -> None:
    store = HostStore(tmp_path / "host.db", cipher=AesGcmHostKeyCipher(b"r" * 32))
    first = store.stage(HostConfigInput(api_key=SecretStr("first-secret")))
    first = store.activate(first.revision)

    class FailingRuntime(_Runtime):
        async def replace(self, config: HostConfig, *, rollback: HostConfig | None) -> None:
            del config, rollback
            raise RuntimeUnavailableError("recorded runtime process could not be inspected")

    runtime = FailingRuntime()
    runtime.current = first
    service = _Service(store=store, runtime=runtime)

    with pytest.raises(HostActivationError, match="recorded runtime process"):
        await service.apply(
            HostConfigInput(
                expected_revision=first.revision,
                api_key=SecretStr("candidate-secret"),
            )
        )

    failed = store.get(first.revision + 1)
    assert failed is not None
    assert failed.error == "recorded runtime process could not be inspected"
    await service.shutdown()


@pytest.mark.asyncio
async def test_telegram_token_only_is_validated_stored_and_activated(tmp_path: Path) -> None:
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "moonshotai/kimi-k3"}]})
        if request.url.host == "api.telegram.org":
            return httpx.Response(200, json={"ok": True, "result": {"id": 1}})
        if request.url.path == "/v2/capabilities/telegram":
            return httpx.Response(
                200,
                json={"manifest": {"revision": 1}, "activation": None, "test": None},
            )
        if request.url.path == "/v2/secrets/telegram-bot-token" and request.method == "GET":
            return httpx.Response(404, json={"detail": "not found"})
        if request.url.path == "/v2/secrets/pending":
            return httpx.Response(201, json={"secret": {"revision": 1}})
        if request.url.path == "/v2/secrets/telegram-bot-token" and request.method == "PUT":
            if body is not None and "scopes" in body:
                return httpx.Response(
                    409,
                    json={
                        "detail": (
                            "pending secret scopes cannot change while storing its first value"
                        )
                    },
                )
            return httpx.Response(200, json={"secret": {"revision": 2}})
        if request.url.path.endswith("/test"):
            return httpx.Response(200, json={"test": {"status": "passed"}})
        if request.url.path.endswith("/activate"):
            return httpx.Response(200, json={"activation": {"generation": 1}})
        return httpx.Response(500, json={"detail": "unexpected test request"})

    store = HostStore(tmp_path / "host.db", cipher=AesGcmHostKeyCipher(b"t" * 32))
    runtime = _Runtime()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = HostService(
        store=store,
        runtime=runtime,  # type: ignore[arg-type]
        client=client,
    )

    view = await service.apply(
        HostConfigInput(
            api_key=SecretStr("provider-secret"),
            base_url="https://models.example",
            telegram_bot_token=SecretStr("123:telegram-secret"),
        )
    )

    assert view.telegram_configured is True
    assert view.telegram_pairing_required is True
    pending = next(body for method, path, body in requests if path == "/v2/secrets/pending")
    assert pending == {
        "id": "telegram-bot-token",
        "name": "telegram-bot-token",
        "scopes": ["telegram.receive", "telegram.send"],
    }
    stored = next(
        body
        for method, path, body in requests
        if method == "PUT" and path == "/v2/secrets/telegram-bot-token"
    )
    assert stored == {
        "expected_revision": 1,
        "value": "123:telegram-secret",
    }
    activation = next(body for method, path, body in requests if path.endswith("/activate"))
    assert activation["secret_handles"] == {"TELEGRAM_BOT_TOKEN": "telegram-bot-token"}
    assert "telegram-secret" not in json.dumps(activation)
    await client.aclose()
