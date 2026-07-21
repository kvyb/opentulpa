from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from opentulpa.integrations.browser_sessions import (
    BrowserSessionHandle,
    BrowserStartJobArguments,
    TenantBrowserService,
)
from opentulpa.integrations.browser_use_cloud import (
    BrowserUseCloudBrowserSession,
    BrowserUseCloudClient,
    BrowserUseCloudError,
    BrowserUseCloudSessionProvider,
)
from opentulpa.jobs import JobExecutionContext
from opentulpa.persistence.idempotency import IdempotencyStore


class _Resolver:
    def __init__(self, addresses: list[str]) -> None:
        self.addresses = addresses

    async def resolve(self, hostname: str, port: int) -> list[str]:
        assert hostname
        assert port > 0
        return self.addresses


class _CloudClient:
    def __init__(self, *, cdp_url: str = "wss://cdp.browser-use.example/session") -> None:
        self.cdp_url = cdp_url
        self.profiles: list[str] = []
        self.sessions: list[str] = []
        self.stopped: list[str] = []

    async def create_profile(self, *, name: str) -> str:
        self.profiles.append(name)
        return f"profile-{len(self.profiles)}"

    async def create_browser_session(self, *, profile_id: str) -> BrowserUseCloudBrowserSession:
        self.sessions.append(profile_id)
        return BrowserUseCloudBrowserSession(
            id=f"remote-{len(self.sessions)}",
            cdp_url=self.cdp_url,
            profile_id=profile_id,
            live_url="https://live.browser-use.example/session",
        )

    async def stop_browser_session(self, session_id: str) -> None:
        self.stopped.append(session_id)


class _Session:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = 0
        self.stopped = 0
        self.current_url = "about:blank"
        self.navigations: list[str] = []

    async def start(self) -> None:
        self.started += 1

    async def navigate_to(self, url: str) -> None:
        self.navigations.append(url)
        self.current_url = url

    async def get_current_page(self) -> Any:
        return object()

    async def get_current_page_url(self) -> str:
        return self.current_url

    async def get_current_page_title(self) -> str:
        return "Test page"

    async def get_state_as_text(self) -> str:
        return "Visible text"

    async def stop(self) -> None:
        self.stopped += 1


class _SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[_Session] = []

    def __call__(self, **kwargs: Any) -> _Session:
        session = _Session(**kwargs)
        self.sessions.append(session)
        return session


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed_domains", [[], ["*"], [" * "]])
async def test_browser_use_provider_requires_explicit_domains_before_cloud_side_effects(
    tmp_path: Path,
    allowed_domains: list[str],
) -> None:
    cloud = _CloudClient()
    provider = BrowserUseCloudSessionProvider(
        client=cast(BrowserUseCloudClient, cloud),
        profile_metadata_root=tmp_path / "cloud-profiles",
        session_factory=_SessionFactory(),
        host_resolver=_Resolver(["93.184.216.34"]),
    )

    with pytest.raises(BrowserUseCloudError, match="explicit allowed_domains"):
        await provider.create(tenant_id="tenant-a", allowed_domains=allowed_domains)

    assert cloud.profiles == []
    assert cloud.sessions == []


@pytest.mark.asyncio
async def test_browser_use_provider_reuses_profile_across_session_dirs_and_restart(
    tmp_path: Path,
) -> None:
    cloud = _CloudClient()
    sessions = _SessionFactory()
    profiles_root = tmp_path / "browser-profiles"
    metadata_root = profiles_root / ".browser-use-cloud"
    provider = BrowserUseCloudSessionProvider(
        client=cast(BrowserUseCloudClient, cloud),
        profile_metadata_root=metadata_root,
        session_factory=sessions,
        host_resolver=_Resolver(["93.184.216.34"]),
    )

    first = await provider.create(
        tenant_id="tenant-secret-name",
        allowed_domains=["example.com"],
    )

    restarted_provider = BrowserUseCloudSessionProvider(
        client=cast(BrowserUseCloudClient, cloud),
        profile_metadata_root=metadata_root,
        session_factory=sessions,
        host_resolver=_Resolver(["93.184.216.34"]),
    )
    second = await restarted_provider.create(
        tenant_id="tenant-secret-name",
        allowed_domains=["example.com"],
    )
    other_tenant = await restarted_provider.create(
        tenant_id="other-secret-tenant",
        allowed_domains=["example.com"],
    )

    assert first.backend == second.backend == other_tenant.backend == "browser-use-cloud"
    assert cloud.sessions == ["profile-1", "profile-1", "profile-2"]
    assert len(cloud.profiles) == 2
    assert cloud.profiles[0] != cloud.profiles[1]
    assert all("secret" not in name for name in cloud.profiles)
    metadata_files = list(metadata_root.rglob(".browser-use-cloud.json"))
    assert len(metadata_files) == 2
    assert all("secret" not in path.as_posix() for path in metadata_files)
    assert all("secret" not in path.read_text(encoding="utf-8") for path in metadata_files)
    assert sessions.sessions[0].kwargs == {
        "cdp_url": "wss://cdp.browser-use.example/session",
        "allowed_domains": ["example.com"],
    }

    await first.session.stop()
    await first.session.stop()
    await second.session.stop()
    await other_tenant.session.stop()
    assert cloud.stopped == ["remote-1", "remote-2", "remote-3"]


@pytest.mark.asyncio
async def test_browser_start_reuses_cloud_profile_across_service_restart(
    tmp_path: Path,
) -> None:
    cloud = _CloudClient()
    sessions = _SessionFactory()
    profiles_root = tmp_path / "browser-profiles"
    metadata_root = profiles_root / ".browser-use-cloud"
    browser_db = tmp_path / "browser.db"
    idempotency_db = tmp_path / "idempotency.db"

    async def progress(_: dict[str, Any]) -> None:
        return None

    def context(*, tenant_id: str, job_id: str, key: str) -> JobExecutionContext:
        return JobExecutionContext(
            tenant_id=tenant_id,
            job_id=job_id,
            idempotency_key=key,
            attempt=1,
            _emit_progress=progress,
        )

    first_service = TenantBrowserService(
        db_path=browser_db,
        idempotency=IdempotencyStore(idempotency_db),
        session_provider=BrowserUseCloudSessionProvider(
            client=cast(BrowserUseCloudClient, cloud),
            profile_metadata_root=metadata_root,
            session_factory=sessions,
            host_resolver=_Resolver(["93.184.216.34"]),
        ),
    )
    arguments = BrowserStartJobArguments(allowed_domains=["example.com"])
    first = await first_service._start_job(  # noqa: SLF001
        arguments,
        context(tenant_id="tenant-secret-name", job_id="job-1", key="start-1"),
    )
    first_session_id = str(first.data["session_id"])
    await first_service.shutdown()

    restarted_service = TenantBrowserService(
        db_path=browser_db,
        idempotency=IdempotencyStore(idempotency_db),
        session_provider=BrowserUseCloudSessionProvider(
            client=cast(BrowserUseCloudClient, cloud),
            profile_metadata_root=metadata_root,
            session_factory=sessions,
            host_resolver=_Resolver(["93.184.216.34"]),
        ),
    )
    second = await restarted_service._start_job(  # noqa: SLF001
        arguments,
        context(tenant_id="tenant-secret-name", job_id="job-2", key="start-2"),
    )
    other_tenant = await restarted_service._start_job(  # noqa: SLF001
        arguments,
        context(tenant_id="other-secret-tenant", job_id="job-3", key="start-3"),
    )
    second_session_id = str(second.data["session_id"])
    other_session_id = str(other_tenant.data["session_id"])
    assert first_session_id != second_session_id
    assert first_session_id != other_session_id
    assert second_session_id != other_session_id
    assert cloud.sessions == ["profile-1", "profile-1", "profile-2"]
    assert len(cloud.profiles) == 2

    await restarted_service.shutdown()
    assert cloud.stopped == ["remote-1", "remote-2", "remote-3"]


@pytest.mark.asyncio
async def test_browser_use_provider_rejects_private_cdp_without_local_fallback(
    tmp_path: Path,
) -> None:
    cloud = _CloudClient(cdp_url="wss://internal.browser-use.example/session")
    sessions = _SessionFactory()
    provider = BrowserUseCloudSessionProvider(
        client=cast(BrowserUseCloudClient, cloud),
        profile_metadata_root=tmp_path / "cloud-profiles",
        session_factory=sessions,
        host_resolver=_Resolver(["127.0.0.1"]),
    )

    with pytest.raises(BrowserUseCloudError, match="unsafe endpoint"):
        await provider.create(
            tenant_id="tenant-a",
            allowed_domains=["example.com"],
        )

    assert cloud.stopped == ["remote-1"]
    assert sessions.sessions == []


@pytest.mark.asyncio
async def test_tenant_browser_service_without_isolated_provider_fails_closed(
    tmp_path: Path,
) -> None:
    service = TenantBrowserService(
        db_path=tmp_path / "browser.db",
        idempotency=IdempotencyStore(tmp_path / "idempotency.db"),
    )

    with pytest.raises(RuntimeError, match="no isolated browser provider"):
        await service._create_session(  # noqa: SLF001
            tenant_id="tenant-a",
            allowed_domains=["example.com"],
        )


@pytest.mark.asyncio
async def test_tenant_browser_service_uses_injected_provider_and_idempotent_job(
    tmp_path: Path,
) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.session = _Session()

        async def create(self, **kwargs: Any) -> BrowserSessionHandle:
            self.calls.append(kwargs)
            return BrowserSessionHandle(
                session=self.session,
                backend="browser-use-cloud",
            )

    provider = Provider()
    service = TenantBrowserService(
        db_path=tmp_path / "browser.db",
        idempotency=IdempotencyStore(tmp_path / "idempotency.db"),
        session_provider=provider,
    )

    async def progress(_: dict[str, Any]) -> None:
        return None

    context = JobExecutionContext(
        tenant_id="tenant-a",
        job_id="job-1",
        idempotency_key="start-once",
        attempt=1,
        _emit_progress=progress,
    )
    arguments = BrowserStartJobArguments(
        start_url="https://example.com/start",
        allowed_domains=["example.com"],
    )

    first = await service._start_job(arguments, context)  # noqa: SLF001
    second = await service._start_job(arguments, context)  # noqa: SLF001
    session_id = str(first.data["session_id"])

    assert first == second
    assert len(provider.calls) == 1
    assert provider.calls[0]["tenant_id"] == "tenant-a"
    assert provider.calls[0]["allowed_domains"] == ["example.com"]
    assert service.get(tenant_id="tenant-a", session_id=session_id)["backend"] == (
        "browser-use-cloud"
    )
    with pytest.raises(KeyError):
        service.get(tenant_id="tenant-b", session_id=session_id)

    await service.stop(
        tenant_id="tenant-a",
        session_id=session_id,
        idempotency_key="stop-once",
    )
    assert provider.session.stopped == 1
