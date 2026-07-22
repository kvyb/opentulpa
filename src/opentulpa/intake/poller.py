"""Internal APScheduler trigger for intake sources that require polling."""

from __future__ import annotations

import logging
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

from opentulpa.intake.service import IntakeWorkflowService

logger = logging.getLogger(__name__)


class IntakePollDispatcher:
    """Rebuild internal polling jobs from active workflow configuration."""

    def __init__(
        self,
        workflows: IntakeWorkflowService,
        *,
        scheduler: AsyncIOScheduler | None = None,
        timezone: str = "UTC",
    ) -> None:
        self._workflows = workflows
        self._scheduler = scheduler or AsyncIOScheduler(timezone=timezone)
        self._timezone = timezone
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._scheduler.start()
        self._started = True
        self.sync_all()

    def shutdown(self, *, wait: bool = True) -> None:
        if not self._started:
            return
        self._scheduler.shutdown(wait=wait)
        self._started = False

    def sync_all(self) -> None:
        seen: set[str] = set()
        for summary in self._workflows.list_customer_summaries():
            tenant_id = str(summary.get("customer_id", "") or "").strip()
            if not tenant_id:
                continue
            for workflow in self._workflows.list_workflows(
                customer_id=tenant_id,
                include_disabled=True,
            ):
                seen.add(self._job_id(tenant_id, str(workflow.get("workflow_id", "") or "")))
                self.upsert(workflow)
        for job in self._scheduler.get_jobs():
            if job.id.startswith("intake-poll:") and job.id not in seen:
                self._scheduler.remove_job(job.id)

    def upsert(self, workflow: dict[str, Any]) -> None:
        tenant_id = str(workflow.get("customer_id", "") or "").strip()
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        if not tenant_id or not workflow_id:
            raise ValueError("intake poll requires customer_id and workflow_id")
        job_id = self._job_id(tenant_id, workflow_id)
        schedule = str(workflow.get("schedule", "") or "").strip()
        enabled = bool(workflow.get("enabled", False))
        channel = str(workflow.get("channel", "") or "").strip()
        if not enabled or not schedule or channel == "telegram_business_dm":
            self.remove(tenant_id=tenant_id, workflow_id=workflow_id)
            return
        trigger = CronTrigger.from_crontab(schedule, timezone=self._timezone)
        self._scheduler.add_job(
            self._run,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            kwargs={"tenant_id": tenant_id, "workflow_id": workflow_id},
            coalesce=True,
            max_instances=1,
            misfire_grace_time=1,
        )

    def remove(self, *, tenant_id: str, workflow_id: str) -> None:
        job_id = self._job_id(tenant_id, workflow_id)
        if self._scheduler.get_job(job_id) is not None:
            self._scheduler.remove_job(job_id)

    async def _run(self, *, tenant_id: str, workflow_id: str) -> None:
        try:
            await self._workflows.run_workflow(
                customer_id=tenant_id,
                workflow_id=workflow_id,
                event_type="scheduled",
                force=False,
            )
        except Exception:
            logger.exception(
                "Intake poll failed: tenant_id=%s workflow_id=%s",
                tenant_id,
                workflow_id,
            )

    @staticmethod
    def _job_id(tenant_id: str, workflow_id: str) -> str:
        return f"intake-poll:{tenant_id}:{workflow_id}"


__all__ = ["IntakePollDispatcher"]
