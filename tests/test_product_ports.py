from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from opentulpa.application.product_ports import (
    ArtifactDeliveryProductPort,
    CustomerProfileProductPort,
    FileVaultProductPort,
    JobProductPort,
    ResearchProductPort,
)
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.files.analysis import FileAnalysisService
from opentulpa.integrations.content_fetch import ContentFetchResult
from opentulpa.integrations.web_search import WebSearchResult
from opentulpa.jobs import JobArtifact


def test_profile_port_reads_and_updates_only_the_requested_tenant(tmp_path: Path) -> None:
    profiles = CustomerProfileService(tmp_path / "profiles.sqlite")
    port = CustomerProfileProductPort(profiles)

    created = port.update(
        tenant_id="tenant-a",
        actor_id="owner-1",
        updates={"directive_text": "Be concise", "locale": "en-US", "utc_offset": "+03:00"},
        idempotency_key="profile-1",
    )
    updated = port.update(
        tenant_id="tenant-a",
        actor_id="owner-2",
        updates={"locale": "ru-RU"},
        idempotency_key="profile-2",
    )

    assert created.customer_id == "tenant-a"
    assert updated.customer_id == "tenant-a"
    assert updated.directive_text == "Be concise"
    assert updated.locale == "ru-RU"
    assert updated.utc_offset == "+03:00"
    assert updated.source == "agent:owner-2"
    assert port.get(tenant_id="tenant-a") == updated
    with pytest.raises(KeyError):
        port.get(tenant_id="tenant-b")


def _ingest(
    files: FileVaultService,
    *,
    tenant_id: str,
    filename: str,
    content: bytes,
) -> dict[str, Any]:
    return files.ingest_file(
        customer_id=tenant_id,
        chat_id=None,
        kind="document",
        telegram_file_id=None,
        original_filename=filename,
        mime_type="text/plain",
        caption=None,
        raw_bytes=content,
    )


def test_file_port_scopes_search_get_and_inspection_to_tenant(tmp_path: Path) -> None:
    files = FileVaultService(
        root_dir=tmp_path / "vault",
        db_path=tmp_path / "files.sqlite",
    )
    own = _ingest(
        files,
        tenant_id="tenant-a",
        filename="owner-notes.txt",
        content=b"Owner launch plan and budget",
    )
    foreign = _ingest(
        files,
        tenant_id="tenant-b",
        filename="private-notes.txt",
        content=b"Private tenant data",
    )
    port = FileVaultProductPort(files=files, analysis=FileAnalysisService(files))

    assert [record["id"] for record in port.search(tenant_id="tenant-a", query="", limit=20)] == [
        own["id"]
    ]
    assert port.get(tenant_id="tenant-a", file_id=str(own["id"]))["customer_id"] == "tenant-a"
    inspection = port.inspect(
        tenant_id="tenant-a",
        file_id=str(own["id"]),
        question="launch budget",
    )
    assert inspection["tenant_id"] == "tenant-a"
    assert inspection["file_id"] == own["id"]
    assert "Owner launch plan and budget" in inspection["sections"][0]["preview"]

    with pytest.raises(KeyError):
        port.get(tenant_id="tenant-a", file_id=str(foreign["id"]))
    with pytest.raises(KeyError):
        port.inspect(
            tenant_id="tenant-a",
            file_id=str(foreign["id"]),
            question=None,
        )


class _SearchProvider:
    name = "test-search"

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str) -> WebSearchResult:
        self.queries.append(query)
        return WebSearchResult(
            answer="Answer",
            sources=[
                {"url": "https://one.example", "domain": "one.example"},
                {"url": "https://two.example", "domain": "two.example"},
                {"url": "https://three.example", "domain": "three.example"},
            ],
            provider=self.name,
            model="test-model",
        )


class _ContentFetch:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def fetch(self, url: str) -> ContentFetchResult:
        self.urls.append(url)
        return ContentFetchResult(
            url=url,
            status_code=200,
            content_type="text/plain",
            charset="utf-8",
            title=None,
            text="Fetched text",
            bytes_read=12,
            redirects=0,
        )


@pytest.mark.asyncio
async def test_research_port_bounds_sources_and_uses_secure_fetch_adapter() -> None:
    search = _SearchProvider()
    fetch = _ContentFetch()
    port = ResearchProductPort(
        web_search=cast(Any, search),
        content_fetch=cast(Any, fetch),
    )

    result = await port.search(tenant_id="tenant-a", query="current answer", limit=2)
    fetched = await port.fetch(tenant_id="tenant-a", url="https://public.example/report")

    assert search.queries == ["current answer"]
    assert result["source_count"] == 2
    assert [source["domain"] for source in result["sources"]] == [
        "one.example",
        "two.example",
    ]
    assert fetch.urls == ["https://public.example/report"]
    assert fetched["text"] == "Fetched text"
    assert fetched["bytes_read"] == 12

    unavailable = ResearchProductPort(
        web_search=None,
        content_fetch=cast(Any, fetch),
    )
    assert (await unavailable.search(tenant_id="tenant-a", query="anything", limit=2)) == {
        "available": False,
        "answer": "Web search is not configured.",
        "sources": [],
        "source_count": 0,
    }


def _artifact(*, tenant_id: str, artifact_id: str, path: Path) -> JobArtifact:
    return JobArtifact(
        id=artifact_id,
        tenant_id=tenant_id,
        job_id="job-1",
        name="report.txt",
        media_type="text/plain",
        uri=path.as_uri(),
        size_bytes=path.stat().st_size,
        sha256=None,
        metadata={},
        created_at=datetime.now(UTC),
    )


class _ArtifactJobs:
    def __init__(self, artifacts: list[JobArtifact]) -> None:
        self._artifacts = {(item.tenant_id, item.id): item for item in artifacts}
        self.calls: list[tuple[str, str]] = []

    def get_artifact(self, *, tenant_id: str, artifact_id: str) -> JobArtifact:
        self.calls.append((tenant_id, artifact_id))
        try:
            return self._artifacts[(tenant_id, artifact_id)]
        except KeyError as exc:
            raise KeyError(artifact_id) from exc


class _Delivery:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def deliver_artifact(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"delivered": True}


@pytest.mark.asyncio
async def test_artifact_port_requires_ownership_and_contained_regular_file(tmp_path: Path) -> None:
    allowed = tmp_path / "artifacts"
    allowed.mkdir()
    valid_path = allowed / "report.txt"
    valid_path.write_text("report", encoding="utf-8")
    outside_path = tmp_path / "private.txt"
    outside_path.write_text("private", encoding="utf-8")
    link_path = allowed / "report-link.txt"
    link_path.symlink_to(valid_path)

    jobs = _ArtifactJobs(
        [
            _artifact(tenant_id="tenant-a", artifact_id="valid", path=valid_path),
            _artifact(tenant_id="tenant-a", artifact_id="outside", path=outside_path),
            _artifact(tenant_id="tenant-a", artifact_id="link", path=link_path),
        ]
    )
    delivery = _Delivery()
    port = ArtifactDeliveryProductPort(
        jobs=cast(Any, jobs),
        delivery=delivery,
        allowed_roots=[allowed],
    )

    with pytest.raises(KeyError):
        port.get(tenant_id="tenant-b", artifact_id="valid")

    delivered = await port.deliver(
        tenant_id="tenant-a",
        actor_id="owner-1",
        thread_id="thread-1",
        channel="web",
        artifact_id="valid",
        caption="Result",
        idempotency_key="delivery-1",
    )
    assert delivered == {
        "artifact_id": "valid",
        "job_id": "job-1",
        "channel": "telegram",
        "delivered": True,
    }
    assert delivery.calls == [
        {
            "tenant_id": "tenant-a",
            "path": valid_path,
            "filename": "report.txt",
            "media_type": "text/plain",
            "caption": "Result",
        }
    ]

    with pytest.raises(PermissionError):
        await port.deliver(
            tenant_id="tenant-a",
            actor_id="owner-1",
            thread_id="thread-1",
            channel="web",
            artifact_id="outside",
            caption=None,
            idempotency_key="delivery-2",
        )
    with pytest.raises(ValueError, match="symbolic link"):
        await port.deliver(
            tenant_id="tenant-a",
            actor_id="owner-1",
            thread_id="thread-1",
            channel="web",
            artifact_id="link",
            caption=None,
            idempotency_key="delivery-3",
        )
    assert len(delivery.calls) == 1


class _Jobs:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create", kwargs))
        return {"id": "job-1", "tenant_id": kwargs["tenant_id"]}

    def get(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", kwargs))
        return {"id": kwargs["job_id"], "tenant_id": kwargs["tenant_id"]}

    def events(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("events", kwargs))
        return []

    def artifacts(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("artifacts", kwargs))
        return []

    async def cancel(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("cancel", kwargs))
        return {"id": kwargs["job_id"], "tenant_id": kwargs["tenant_id"], "status": "cancelled"}


@pytest.mark.asyncio
async def test_job_port_forwards_exact_tenant_handler_arguments_and_cursor() -> None:
    jobs = _Jobs()
    port = JobProductPort(cast(Any, jobs))

    await port.create(
        tenant_id="tenant-a",
        handler_name="file_analyze",
        arguments={"file_id": "file-1", "instruction": "summarize"},
        idempotency_key="job-key-1",
    )
    port.get(tenant_id="tenant-a", job_id="job-1")
    port.events(tenant_id="tenant-a", job_id="job-1", after_sequence=7, limit=25)
    port.artifacts(tenant_id="tenant-a", job_id="job-1")
    await port.cancel(
        tenant_id="tenant-a",
        job_id="job-1",
        idempotency_key="cancel-key-1",
    )

    assert jobs.calls == [
        (
            "create",
            {
                "tenant_id": "tenant-a",
                "handler_name": "file_analyze",
                "arguments": {"file_id": "file-1", "instruction": "summarize"},
                "idempotency_key": "job-key-1",
            },
        ),
        ("get", {"tenant_id": "tenant-a", "job_id": "job-1"}),
        (
            "events",
            {
                "tenant_id": "tenant-a",
                "job_id": "job-1",
                "after_sequence": 7,
                "limit": 25,
            },
        ),
        ("artifacts", {"tenant_id": "tenant-a", "job_id": "job-1"}),
        ("cancel", {"tenant_id": "tenant-a", "job_id": "job-1"}),
    ]
