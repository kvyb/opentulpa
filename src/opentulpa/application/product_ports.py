"""Concrete adapters from the product tool boundary to OpenTulpa services."""

from __future__ import annotations

import inspect
from builtins import list as list_type
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from pydantic import Field, JsonValue

from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.files.analysis import FileAnalysisService
from opentulpa.intake.drafts import IntakeDraftService, IntakeWorkflowProposal
from opentulpa.intake.poller import IntakePollDispatcher
from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.integrations.content_fetch import ContentFetchService
from opentulpa.integrations.web_search import WebSearchProvider, WebSearchResult
from opentulpa.jobs import (
    JobArguments,
    JobExecutionContext,
    JobHandlerRegistry,
    JobHandlerResult,
    JobService,
)
from opentulpa.schedules import ScheduleService, ScheduleWrite


class ArtifactDelivery(Protocol):
    async def deliver_artifact(
        self,
        *,
        tenant_id: str,
        path: Path,
        filename: str,
        media_type: str | None = None,
        caption: str | None = None,
    ) -> dict[str, Any]: ...


class CustomerProfileProductPort:
    def __init__(self, profiles: CustomerProfileService) -> None:
        self._profiles = profiles

    def get(self, *, tenant_id: str) -> Any:
        profile = self._profiles.get_profile(tenant_id)
        if profile is None:
            raise KeyError(tenant_id)
        return profile

    def update(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        updates: Mapping[str, Any],
        idempotency_key: str,
    ) -> Any:
        del idempotency_key
        return self._profiles.update_profile(
            tenant_id,
            updates=updates,
            source=f"agent:{str(actor_id or 'owner')[:100]}",
        )


class FileVaultProductPort:
    def __init__(
        self,
        *,
        files: FileVaultService,
        analysis: FileAnalysisService,
    ) -> None:
        self._files = files
        self._analysis = analysis

    def search(self, *, tenant_id: str, query: str, limit: int) -> Any:
        return self._files.search(tenant_id, query, limit)

    def get(self, *, tenant_id: str, file_id: str) -> Any:
        record = self._files.get_file(tenant_id, file_id)
        if record is None:
            raise KeyError(file_id)
        return record

    def inspect(self, *, tenant_id: str, file_id: str, question: str | None) -> Any:
        return self._analysis.inspect(
            tenant_id=tenant_id,
            file_id=file_id,
            question=question,
        )


class ArtifactDeliveryProductPort:
    """Resolve only tenant-owned job artifacts under declared storage roots."""

    def __init__(
        self,
        *,
        jobs: JobService,
        delivery: ArtifactDelivery,
        allowed_roots: Sequence[Path],
    ) -> None:
        roots = tuple(root.expanduser().resolve() for root in allowed_roots)
        if not roots:
            raise ValueError("at least one artifact root is required")
        self._jobs = jobs
        self._delivery = delivery
        self._allowed_roots = roots

    def get(self, *, tenant_id: str, artifact_id: str) -> Any:
        return self._jobs.get_artifact(tenant_id=tenant_id, artifact_id=artifact_id)

    async def deliver(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        thread_id: str,
        channel: str,
        artifact_id: str,
        caption: str | None,
        idempotency_key: str,
    ) -> dict[str, Any]:
        del actor_id, thread_id, channel, idempotency_key
        artifact = self._jobs.get_artifact(tenant_id=tenant_id, artifact_id=artifact_id)
        path = self._artifact_path(artifact.uri)
        await self._delivery.deliver_artifact(
            tenant_id=tenant_id,
            path=path,
            filename=artifact.name,
            media_type=artifact.media_type,
            caption=caption,
        )
        return {
            "artifact_id": artifact.id,
            "job_id": artifact.job_id,
            "channel": "telegram",
            "delivered": True,
        }

    def _artifact_path(self, uri: str) -> Path:
        parsed = urlsplit(str(uri or ""))
        if parsed.scheme and parsed.scheme != "file":
            raise ValueError("artifact URI is not deliverable")
        if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
            raise ValueError("artifact URI host is not allowed")
        raw_path = unquote(parsed.path) if parsed.scheme == "file" else str(uri or "")
        unresolved_path = Path(raw_path).expanduser()
        if unresolved_path.is_symlink():
            raise ValueError("artifact path must not be a symbolic link")
        path = unresolved_path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("artifact path is not a regular file")
        for root in self._allowed_roots:
            try:
                path.relative_to(root)
            except ValueError:
                continue
            return path
        raise PermissionError("artifact path is outside tenant artifact storage")


class ResearchProductPort:
    def __init__(
        self,
        *,
        web_search: WebSearchProvider | None,
        content_fetch: ContentFetchService,
    ) -> None:
        self._web_search = web_search
        self._content_fetch = content_fetch

    async def search(self, *, tenant_id: str, query: str, limit: int) -> Any:
        del tenant_id
        if self._web_search is None:
            return {
                "available": False,
                "answer": "Web search is not configured.",
                "sources": [],
                "source_count": 0,
            }
        result = await self._web_search.search(query)
        if isinstance(result, WebSearchResult):
            payload = result.to_payload()
            raw_sources = payload.get("sources")
            sources = raw_sources[:limit] if isinstance(raw_sources, list_type) else []
            payload["sources"] = sources
            payload["source_count"] = len(sources)
            return payload
        return {
            "available": True,
            "answer": str(result or "No response from web search."),
            "sources": [],
            "source_count": 0,
            "provider": self._web_search.name,
        }

    async def fetch(self, *, tenant_id: str, url: str) -> Any:
        del tenant_id
        return (await self._content_fetch.fetch(url)).to_payload()


class IntakeWorkflowTestJobArguments(JobArguments):
    workflow_id: str | None = Field(default=None, max_length=300)
    draft_id: str | None = Field(default=None, max_length=300)
    sample: dict[str, JsonValue] = Field(default_factory=dict)


class IntakeProductPort:
    def __init__(
        self,
        *,
        workflows: IntakeWorkflowService,
        drafts: IntakeDraftService,
        poller: IntakePollDispatcher,
    ) -> None:
        self._workflows = workflows
        self._drafts = drafts
        self._poller = poller

    def register_handlers(self, registry: JobHandlerRegistry) -> None:
        registry.register(
            name="intake_workflow_test",
            arguments_model=IntakeWorkflowTestJobArguments,
            handler=self._test_job,
            timeout_seconds=120,
        )

    def list_workflows(self, *, tenant_id: str, include_inactive: bool) -> Any:
        return self._workflows.list_workflows(
            customer_id=tenant_id,
            include_disabled=include_inactive,
        )

    def get_workflow(self, *, tenant_id: str, workflow_id: str | None) -> Any:
        if workflow_id:
            workflow = self._workflows.get_workflow(
                customer_id=tenant_id,
                workflow_id=workflow_id,
            )
            if workflow is None:
                raise KeyError(workflow_id)
            return workflow
        active = self._workflows.list_workflows(
            customer_id=tenant_id,
            include_disabled=False,
        )
        if not active:
            raise KeyError("active intake workflow")
        if len(active) > 1:
            raise ValueError("multiple active workflows exist; specify workflow_id")
        return active[0]

    def get_draft(self, *, tenant_id: str, draft_id: str) -> Any:
        draft = self._drafts.get(tenant_id=tenant_id, draft_id=draft_id)
        if draft is None:
            raise KeyError(draft_id)
        return draft

    def save_draft(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str | None,
        expected_revision: int | None,
        patch: Mapping[str, Any],
        idempotency_key: str,
    ) -> Any:
        del idempotency_key
        return self._drafts.save(
            tenant_id=tenant_id,
            actor_id=actor_id,
            draft_id=draft_id,
            expected_revision=expected_revision,
            patch=patch,
        )

    def prepare_draft(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> Any:
        del idempotency_key
        return self._drafts.prepare(
            tenant_id=tenant_id,
            actor_id=actor_id,
            draft_id=draft_id,
            expected_revision=expected_revision,
        )

    async def activate_draft(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        confirmation_token: str,
        idempotency_key: str,
    ) -> Any:
        del idempotency_key
        activated = await self._drafts.activate(
            tenant_id=tenant_id,
            actor_id=actor_id,
            draft_id=draft_id,
            expected_revision=expected_revision,
            confirmation_token=confirmation_token,
        )
        self._poller.upsert(dict(activated.workflow))
        return activated

    def delete_workflow(
        self,
        *,
        tenant_id: str,
        workflow_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> Any:
        del idempotency_key
        result = self._workflows.delete_workflow(
            customer_id=tenant_id,
            workflow_id=workflow_id,
            expected_revision=expected_revision,
        )
        if not result.get("deleted"):
            raise KeyError(workflow_id)
        self._poller.remove(tenant_id=tenant_id, workflow_id=workflow_id)
        return result

    async def _test_job(
        self,
        arguments: IntakeWorkflowTestJobArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        if bool(arguments.workflow_id) == bool(arguments.draft_id):
            raise ValueError("provide exactly one workflow_id or draft_id")
        if arguments.draft_id:
            draft = self.get_draft(
                tenant_id=context.tenant_id,
                draft_id=arguments.draft_id,
            )
            workflow_id = str(draft.workflow_id)
            proposal = IntakeWorkflowProposal.model_validate(
                {"workflow_id": workflow_id, **dict(draft.payload)}
            )
            source = "draft"
        else:
            workflow = self.get_workflow(
                tenant_id=context.tenant_id,
                workflow_id=arguments.workflow_id,
            )
            workflow_id = str(workflow["workflow_id"])
            proposal = IntakeWorkflowProposal.model_validate(
                {
                    field: workflow[field]
                    for field in IntakeWorkflowProposal.model_fields
                    if field in workflow
                }
            )
            source = "workflow"
        await context.progress({"stage": "validated"})
        missing = [
            field
            for field in proposal.required_fields
            if not str(arguments.sample.get(field, "") or "").strip()
        ]
        missing_values: list_type[JsonValue] = list(missing)
        data: dict[str, JsonValue] = {
            "configuration": source,
            "workflow_id": workflow_id,
            "valid": True,
            "sample_missing_fields": missing_values,
            "would_request_fields": bool(missing),
        }
        return JobHandlerResult(
            summary="Intake workflow configuration is valid",
            data=data,
        )


class ScheduleProductPort:
    def __init__(self, *, schedules: ScheduleService) -> None:
        self._schedules = schedules

    def list(self, *, tenant_id: str) -> Any:
        return self._schedules.list(tenant_id=tenant_id)

    def get(self, *, tenant_id: str, schedule_id: str) -> Any:
        schedule = self._schedules.get(tenant_id=tenant_id, schedule_id=schedule_id)
        if schedule is None:
            raise KeyError(schedule_id)
        return schedule

    def save(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        write: ScheduleWrite,
        schedule_id: str | None,
        expected_revision: int | None,
        idempotency_key: str,
    ) -> Any:
        del idempotency_key
        schedule = self._schedules.save(
            tenant_id=tenant_id,
            actor_id=actor_id,
            write=write,
            schedule_id=schedule_id,
            expected_revision=expected_revision,
        )
        return schedule

    def delete(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        schedule_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        del idempotency_key
        self._schedules.delete(
            tenant_id=tenant_id,
            actor_id=actor_id,
            schedule_id=schedule_id,
            expected_revision=expected_revision,
        )
        return {"tenant_id": tenant_id, "schedule_id": schedule_id, "deleted": True}


class JobProductPort:
    def __init__(self, jobs: JobService) -> None:
        self._jobs = jobs

    async def create(
        self,
        *,
        tenant_id: str,
        handler_name: str,
        arguments: Mapping[str, Any],
        idempotency_key: str,
    ) -> Any:
        return await self._jobs.create(
            tenant_id=tenant_id,
            handler_name=handler_name,
            arguments=arguments,
            idempotency_key=idempotency_key,
        )

    def get(self, *, tenant_id: str, job_id: str) -> Any:
        return self._jobs.get(tenant_id=tenant_id, job_id=job_id)

    def events(
        self,
        *,
        tenant_id: str,
        job_id: str,
        after_sequence: int,
        limit: int,
    ) -> Any:
        return self._jobs.events(
            tenant_id=tenant_id,
            job_id=job_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    def artifacts(self, *, tenant_id: str, job_id: str) -> Any:
        return self._jobs.artifacts(tenant_id=tenant_id, job_id=job_id)

    async def cancel(
        self,
        *,
        tenant_id: str,
        job_id: str,
        idempotency_key: str,
    ) -> Any:
        del idempotency_key
        return await self._jobs.cancel(tenant_id=tenant_id, job_id=job_id)


async def maybe_shutdown(service: Any) -> None:
    """Call a dependency's optional shutdown hook during app teardown."""

    shutdown = getattr(service, "shutdown", None)
    if not callable(shutdown):
        return
    result = shutdown()
    if inspect.isawaitable(result):
        await result


__all__ = [
    "ArtifactDeliveryProductPort",
    "CustomerProfileProductPort",
    "FileVaultProductPort",
    "IntakeProductPort",
    "IntakeWorkflowTestJobArguments",
    "JobProductPort",
    "ResearchProductPort",
    "ScheduleProductPort",
    "maybe_shutdown",
]
