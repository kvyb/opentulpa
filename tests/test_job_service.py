from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import Field

from opentulpa.jobs import (
    Job,
    JobArguments,
    JobArtifactWrite,
    JobExecutionContext,
    JobHandlerRegistrationError,
    JobHandlerRegistry,
    JobHandlerResult,
    JobIdempotencyConflictError,
    JobNotFoundError,
    JobService,
    JobValidationError,
)


class _EchoArguments(JobArguments):
    text: str = Field(min_length=1, max_length=200)


async def _wait_terminal(
    service: JobService,
    *,
    tenant_id: str,
    job_id: str,
    timeout: float = 2,
) -> Job:
    async with asyncio.timeout(timeout):
        while True:
            job = service.get(tenant_id=tenant_id, job_id=job_id)
            if job.status in {"succeeded", "failed", "cancelled"}:
                return job
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_registered_job_is_idempotent_owned_and_persists_events_artifacts(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    async def echo(
        arguments: _EchoArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        calls.append(arguments.text)
        await context.progress({"phase": "rendering"})
        return JobHandlerResult(
            summary="Rendered",
            data={"echo": arguments.text},
            artifacts=[
                JobArtifactWrite(
                    name="echo.txt",
                    media_type="text/plain",
                    uri=f"workspace://artifacts/{context.job_id}/echo.txt",
                    size_bytes=len(arguments.text),
                    sha256="a" * 64,
                )
            ],
        )

    registry = JobHandlerRegistry()
    registry.register(name="render_echo", arguments_model=_EchoArguments, handler=echo)
    service = JobService(tmp_path / "jobs.sqlite", registry=registry)
    await service.start()
    first, duplicate = await asyncio.gather(
        service.create(
            tenant_id="tenant-a",
            handler_name="render_echo",
            arguments={"text": "hello"},
            idempotency_key="echo-1",
        ),
        service.create(
            tenant_id="tenant-a",
            handler_name="render_echo",
            arguments={"text": "hello"},
            idempotency_key="echo-1",
        ),
    )

    assert first.id == duplicate.id
    completed = await _wait_terminal(
        service,
        tenant_id="tenant-a",
        job_id=first.id,
    )
    assert completed.status == "succeeded"
    assert completed.result is not None
    assert completed.result.data == {"echo": "hello"}
    assert completed.attempt_count == 1
    assert calls == ["hello"]
    events = service.events(tenant_id="tenant-a", job_id=first.id)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert [event.event_type for event in events] == [
        "queued",
        "running",
        "progress",
        "artifact.ready",
        "completed",
    ]
    artifacts = service.artifacts(tenant_id="tenant-a", job_id=first.id)
    assert len(artifacts) == 1
    assert artifacts[0].tenant_id == "tenant-a"
    assert artifacts[0].name == "echo.txt"

    with pytest.raises(JobNotFoundError):
        service.get(tenant_id="tenant-b", job_id=first.id)
    with pytest.raises(JobNotFoundError):
        service.events(tenant_id="tenant-b", job_id=first.id)
    with pytest.raises(JobNotFoundError):
        service.artifacts(tenant_id="tenant-b", job_id=first.id)
    with pytest.raises(JobNotFoundError):
        await service.cancel(tenant_id="tenant-b", job_id=first.id)

    other_tenant = await service.create(
        tenant_id="tenant-b",
        handler_name="render_echo",
        arguments={"text": "hello"},
        idempotency_key="echo-1",
    )
    assert other_tenant.id != first.id
    with pytest.raises(JobIdempotencyConflictError):
        await service.create(
            tenant_id="tenant-a",
            handler_name="render_echo",
            arguments={"text": "different"},
            idempotency_key="echo-1",
        )
    await service.shutdown()


@pytest.mark.asyncio
async def test_job_cancel_stops_running_handler_and_is_durable(tmp_path: Path) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def blocking(
        arguments: _EchoArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        del arguments, context
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return JobHandlerResult()  # pragma: no cover

    registry = JobHandlerRegistry()
    registry.register(name="wait_for_signal", arguments_model=_EchoArguments, handler=blocking)
    service = JobService(tmp_path / "jobs.sqlite", registry=registry)
    await service.start()
    job = await service.create(
        tenant_id="tenant-a",
        handler_name="wait_for_signal",
        arguments={"text": "wait"},
        idempotency_key="wait-1",
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    cancelled = await service.cancel(tenant_id="tenant-a", job_id=job.id)

    assert cancelled.status == "cancelled"
    assert stopped.is_set()
    assert service.artifacts(tenant_id="tenant-a", job_id=job.id) == []
    assert [event.event_type for event in service.events(tenant_id="tenant-a", job_id=job.id)][
        -2:
    ] == ["cancel_requested", "cancelled"]
    await service.shutdown()


@pytest.mark.asyncio
async def test_shutdown_requeues_and_restart_recovers_registered_job(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite"
    started = asyncio.Event()

    async def interrupted(
        arguments: _EchoArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        del arguments, context
        started.set()
        await asyncio.Event().wait()
        return JobHandlerResult()  # pragma: no cover

    first_registry = JobHandlerRegistry()
    first_registry.register(
        name="build_report",
        arguments_model=_EchoArguments,
        handler=interrupted,
    )
    first_service = JobService(db_path, registry=first_registry)
    await first_service.start()
    created = await first_service.create(
        tenant_id="tenant-a",
        handler_name="build_report",
        arguments={"text": "quarterly"},
        idempotency_key="report-1",
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    await first_service.shutdown()
    interrupted_job = first_service.get(tenant_id="tenant-a", job_id=created.id)
    assert interrupted_job.status == "queued"
    assert interrupted_job.attempt_count == 1

    resumed_calls: list[int] = []

    async def resumed(
        arguments: _EchoArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        assert arguments.text == "quarterly"
        resumed_calls.append(context.attempt)
        return JobHandlerResult(summary="Report complete", data={"resumed": True})

    second_registry = JobHandlerRegistry()
    second_registry.register(
        name="build_report",
        arguments_model=_EchoArguments,
        handler=resumed,
    )
    second_service = JobService(db_path, registry=second_registry)
    await second_service.start()
    completed = await _wait_terminal(
        second_service,
        tenant_id="tenant-a",
        job_id=created.id,
    )

    assert completed.status == "succeeded"
    assert completed.attempt_count == 2
    assert resumed_calls == [2]
    assert "recovered" in [
        event.event_type for event in second_service.events(tenant_id="tenant-a", job_id=created.id)
    ]
    duplicate = await second_service.create(
        tenant_id="tenant-a",
        handler_name="build_report",
        arguments={"text": "quarterly"},
        idempotency_key="report-1",
    )
    assert duplicate.id == created.id
    await second_service.shutdown()


@pytest.mark.asyncio
async def test_registry_and_input_schema_fail_closed_for_arbitrary_execution(
    tmp_path: Path,
) -> None:
    class _CommandArguments(JobArguments):
        command: str

    async def unused(
        arguments: _CommandArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        del arguments, context
        return JobHandlerResult()

    registry = JobHandlerRegistry()
    with pytest.raises(JobHandlerRegistrationError, match="must be async"):
        registry.register(
            name="sync_handler",
            arguments_model=_EchoArguments,
            handler=lambda arguments, context: JobHandlerResult(),  # type: ignore[arg-type,return-value]
        )
    with pytest.raises(JobHandlerRegistrationError, match="forbidden fields: command"):
        registry.register(
            name="run_operation",
            arguments_model=_CommandArguments,
            handler=unused,
        )
    with pytest.raises(JobHandlerRegistrationError, match="source editing"):
        registry.register(
            name="source_edit",
            arguments_model=_EchoArguments,
            handler=unused,  # type: ignore[arg-type]
        )

    async def echo(
        arguments: _EchoArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        del arguments, context
        return JobHandlerResult()

    registry.register(name="safe_echo", arguments_model=_EchoArguments, handler=echo)
    service = JobService(tmp_path / "jobs.sqlite", registry=registry)
    with pytest.raises(JobValidationError, match="invalid registered job request"):
        await service.create(
            tenant_id="tenant-a",
            handler_name="safe_echo",
            arguments={"text": "hello", "command": "rm -rf /"},
            idempotency_key="unsafe-1",
        )
