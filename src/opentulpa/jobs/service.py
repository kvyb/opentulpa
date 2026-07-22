"""Durable in-process execution of registered product background jobs."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import JsonValue, TypeAdapter, ValidationError

from opentulpa.core.ids import new_short_id
from opentulpa.jobs.models import Job, JobArtifact, JobError, JobEvent, JobHandlerResult
from opentulpa.jobs.registry import (
    JobExecutionContext,
    JobHandlerNotFoundError,
    JobHandlerRegistry,
)
from opentulpa.jobs.store import JobStore

logger = logging.getLogger(__name__)
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class JobValidationError(ValueError):
    """A create request does not match its registered handler contract."""


class JobService:
    """Execute only typed registry handlers with durable tenant-owned state."""

    def __init__(
        self,
        db_path: Path,
        *,
        registry: JobHandlerRegistry,
        max_concurrency: int = 4,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = JobStore(db_path)
        self._registry = registry
        self._clock = clock or (lambda: datetime.now(UTC))
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._running: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._started = False
        self._stopping = False

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("job service clock must return an aware datetime")
        return now.astimezone(UTC)

    @staticmethod
    def _identifier(value: str, *, field: str, maximum: int) -> str:
        identifier = str(value or "").strip()
        if not identifier:
            raise JobValidationError(f"{field} is required")
        if len(identifier) > maximum:
            raise JobValidationError(f"{field} must be at most {maximum} characters")
        return identifier

    @staticmethod
    def _request_hash(
        *,
        handler_name: str,
        handler_version: int,
        arguments: dict[str, JsonValue],
    ) -> str:
        canonical = json.dumps(
            {
                "handler_name": handler_name,
                "handler_version": handler_version,
                "arguments": arguments,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            self._started = True
            self._stopping = False
            recovered = self._store.recover(now=self._now())
            for job in recovered:
                self._spawn(job)

    async def shutdown(self) -> None:
        async with self._lock:
            if not self._started:
                return
            self._stopping = True
            tasks = list(self._running.values())
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            self._running.clear()
            self._started = False
            self._stopping = False

    async def create(
        self,
        *,
        tenant_id: str,
        handler_name: str,
        arguments: Mapping[str, Any],
        idempotency_key: str,
        handler_version: int | None = None,
    ) -> Job:
        tenant = self._identifier(tenant_id, field="tenant_id", maximum=200)
        key = self._identifier(
            idempotency_key,
            field="idempotency_key",
            maximum=200,
        )
        try:
            registered = self._registry.get(handler_name, handler_version)
            parsed = registered.parse_arguments(dict(arguments))
            normalized = _JSON_OBJECT.validate_python(parsed.model_dump(mode="json"))
        except (JobHandlerNotFoundError, ValidationError, TypeError, ValueError) as exc:
            raise JobValidationError(f"invalid registered job request: {exc}") from exc
        request_hash = self._request_hash(
            handler_name=registered.name,
            handler_version=registered.version,
            arguments=normalized,
        )
        async with self._lock:
            job, created = self._store.create(
                tenant_id=tenant,
                job_id=new_short_id("job"),
                handler_name=registered.name,
                handler_version=registered.version,
                arguments=normalized,
                idempotency_key=key,
                request_hash=request_hash,
                now=self._now(),
            )
            if self._started and job.status == "queued":
                self._spawn(job)
        if not created:
            return self._store.get(tenant_id=tenant, job_id=job.id)
        return job

    def get(self, *, tenant_id: str, job_id: str) -> Job:
        return self._store.get(
            tenant_id=self._identifier(tenant_id, field="tenant_id", maximum=200),
            job_id=self._identifier(job_id, field="job_id", maximum=100),
        )

    def events(
        self,
        *,
        tenant_id: str,
        job_id: str,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[JobEvent]:
        return self._store.events(
            tenant_id=self._identifier(tenant_id, field="tenant_id", maximum=200),
            job_id=self._identifier(job_id, field="job_id", maximum=100),
            after_sequence=after_sequence,
            limit=limit,
        )

    def artifacts(self, *, tenant_id: str, job_id: str) -> list[JobArtifact]:
        return self._store.artifacts(
            tenant_id=self._identifier(tenant_id, field="tenant_id", maximum=200),
            job_id=self._identifier(job_id, field="job_id", maximum=100),
        )

    def get_artifact(self, *, tenant_id: str, artifact_id: str) -> JobArtifact:
        return self._store.get_artifact(
            tenant_id=self._identifier(tenant_id, field="tenant_id", maximum=200),
            artifact_id=self._identifier(artifact_id, field="artifact_id", maximum=100),
        )

    async def cancel(self, *, tenant_id: str, job_id: str) -> Job:
        tenant = self._identifier(tenant_id, field="tenant_id", maximum=200)
        safe_job_id = self._identifier(job_id, field="job_id", maximum=100)
        job = self._store.request_cancel(
            tenant_id=tenant,
            job_id=safe_job_id,
            now=self._now(),
        )
        task = self._running.get((tenant, safe_job_id))
        if job.status in {"cancel_requested", "cancelled"} and task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self._store.get(tenant_id=tenant, job_id=safe_job_id)

    def _spawn(self, job: Job) -> None:
        key = (job.tenant_id, job.id)
        current = self._running.get(key)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._run(job.tenant_id, job.id),
            name=f"opentulpa-job:{job.id}",
        )
        self._running[key] = task

        def on_done(finished: asyncio.Task[None]) -> None:
            self._runner_finished(key, finished)

        task.add_done_callback(on_done)

    def _runner_finished(
        self,
        key: tuple[str, str],
        task: asyncio.Task[None],
    ) -> None:
        if self._running.get(key) is task:
            self._running.pop(key, None)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.error(
                "job runner exited unexpectedly",
                exc_info=(type(exception), exception, exception.__traceback__),
            )

    async def _run(self, tenant_id: str, job_id: str) -> None:
        async with self._semaphore:
            claimed = self._store.claim(
                tenant_id=tenant_id,
                job_id=job_id,
                now=self._now(),
            )
            if claimed is None:
                return
            try:
                registered = self._registry.get(
                    claimed.handler_name,
                    claimed.handler_version,
                )
            except JobHandlerNotFoundError:
                self._store.fail(
                    tenant_id=tenant_id,
                    job_id=job_id,
                    error=JobError(
                        code="handler_unavailable",
                        message="The registered background operation is unavailable.",
                        retryable=False,
                    ),
                    now=self._now(),
                )
                return
            try:
                parsed = registered.parse_arguments(dict(claimed.arguments))

                async def emit_progress(payload: dict[str, Any]) -> None:
                    normalized = _JSON_OBJECT.validate_python(payload)
                    self._store.progress(
                        tenant_id=tenant_id,
                        job_id=job_id,
                        payload=normalized,
                        now=self._now(),
                    )

                context = JobExecutionContext(
                    tenant_id=tenant_id,
                    job_id=job_id,
                    idempotency_key=claimed.idempotency_key,
                    attempt=claimed.attempt_count,
                    _emit_progress=emit_progress,
                )
                async with asyncio.timeout(registered.timeout_seconds):
                    pending = registered.handler(parsed, context)
                    if not inspect.isawaitable(pending):
                        raise TypeError("registered job handler must be async")
                    result = await pending
                if not isinstance(result, JobHandlerResult):
                    raise TypeError("registered job handler must return JobHandlerResult")
                if self._stopping:
                    self._store.requeue_interrupted(
                        tenant_id=tenant_id,
                        job_id=job_id,
                        now=self._now(),
                    )
                    return
                self._store.complete(
                    tenant_id=tenant_id,
                    job_id=job_id,
                    result=result,
                    now=self._now(),
                )
            except asyncio.CancelledError:
                if self._stopping:
                    self._store.requeue_interrupted(
                        tenant_id=tenant_id,
                        job_id=job_id,
                        now=self._now(),
                    )
                else:
                    self._store.mark_cancelled(
                        tenant_id=tenant_id,
                        job_id=job_id,
                        reason="cancelled_by_owner",
                        now=self._now(),
                    )
                raise
            except TimeoutError:
                self._store.fail(
                    tenant_id=tenant_id,
                    job_id=job_id,
                    error=JobError(
                        code="handler_timeout",
                        message="The background operation exceeded its time limit.",
                        retryable=True,
                    ),
                    now=self._now(),
                )
            except Exception:
                logger.exception(
                    "registered background operation failed",
                    extra={"job_id": job_id, "handler_name": claimed.handler_name},
                )
                self._store.fail(
                    tenant_id=tenant_id,
                    job_id=job_id,
                    error=JobError(
                        code="handler_failed",
                        message="The background operation failed.",
                        retryable=False,
                    ),
                    now=self._now(),
                )


__all__ = ["JobService", "JobValidationError"]
