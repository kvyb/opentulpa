"""Tenant knowledge library operations and registered indexing jobs."""

from __future__ import annotations

import json
from builtins import list as list_type
from typing import Any

from pydantic import Field, JsonValue

from opentulpa.business_knowledge.service import BusinessKnowledgeService
from opentulpa.jobs import (
    JobArguments,
    JobExecutionContext,
    JobHandlerRegistry,
    JobHandlerResult,
)

_SCOPE_TYPE = "customer_business"
_SCOPE_ID = "library"


class KnowledgeAttachJobArguments(JobArguments):
    file_id: str = Field(min_length=1, max_length=300)
    title: str | None = Field(default=None, max_length=500)
    tags: list_type[str] = Field(default_factory=list_type, max_length=30)


class KnowledgeReindexJobArguments(JobArguments):
    source_id: str | None = Field(default=None, max_length=300)


class TenantKnowledgeService:
    def __init__(self, knowledge: BusinessKnowledgeService) -> None:
        self._knowledge = knowledge
        self._repository = knowledge.repository

    def register_handlers(self, registry: JobHandlerRegistry) -> None:
        registry.register(
            name="knowledge_attach",
            arguments_model=KnowledgeAttachJobArguments,
            handler=self._attach_job,
            timeout_seconds=300,
        )
        registry.register(
            name="knowledge_reindex",
            arguments_model=KnowledgeReindexJobArguments,
            handler=self._reindex_job,
            timeout_seconds=300,
        )

    def list(
        self,
        *,
        tenant_id: str,
        include_archived: bool,
        limit: int,
    ) -> list_type[dict[str, Any]]:
        rows = self._repository.product_source_rows(
            customer_id=tenant_id,
            include_archived=include_archived,
            limit=limit,
        )
        return [self._source(row) for row in rows]

    def find(self, *, tenant_id: str, query: str, limit: int) -> list_type[dict[str, Any]]:
        rows = self._repository.product_source_rows(
            customer_id=tenant_id,
            query=query,
            limit=limit,
        )
        return [self._source(row) for row in rows]

    def get(self, *, tenant_id: str, source_id: str) -> dict[str, Any]:
        rows = self._repository.product_source_rows(
            customer_id=tenant_id,
            include_archived=True,
            limit=200,
        )
        row = next((item for item in rows if str(item["file_id"]) == source_id), None)
        if row is None:
            raise KeyError(source_id)
        return self._source(row)

    def archive(
        self,
        *,
        tenant_id: str,
        source_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        del idempotency_key
        self._repository.archive_product_source(
            customer_id=tenant_id,
            file_id=source_id,
        )
        return self.get(tenant_id=tenant_id, source_id=source_id)

    def query(
        self,
        *,
        tenant_id: str,
        query: str,
        source_ids: list_type[str],
        limit: int,
    ) -> Any:
        del limit
        active = {
            str(row["file_id"])
            for row in self._repository.product_source_rows(
                customer_id=tenant_id,
                include_archived=False,
                limit=200,
            )
        }
        requested = source_ids or sorted(active)
        if any(source_id not in active for source_id in requested):
            raise KeyError("knowledge source not found")
        return self._knowledge.query(
            customer_id=tenant_id,
            scope_type=_SCOPE_TYPE,
            scope_id=_SCOPE_ID,
            query=query,
            file_ids=requested,
        )

    async def _attach_job(
        self,
        arguments: KnowledgeAttachJobArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        await context.progress({"stage": "indexing"})
        result = self._knowledge.index_sources(
            customer_id=context.tenant_id,
            scope_type=_SCOPE_TYPE,
            scope_id=_SCOPE_ID,
            file_ids=[arguments.file_id],
        )
        self._repository.set_product_metadata(
            customer_id=context.tenant_id,
            file_id=arguments.file_id,
            title=arguments.title,
            tags=arguments.tags,
            archived=False,
        )
        return JobHandlerResult(
            summary="Knowledge source attached",
            data={
                "source_id": arguments.file_id,
                "source_count": len(result.get("sources") or []),
            },
        )

    async def _reindex_job(
        self,
        arguments: KnowledgeReindexJobArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        rows = self._repository.product_source_rows(
            customer_id=context.tenant_id,
            include_archived=False,
            limit=200,
        )
        file_ids = [str(row["file_id"]) for row in rows]
        if arguments.source_id:
            if arguments.source_id not in file_ids:
                raise KeyError(arguments.source_id)
            file_ids = [arguments.source_id]
        if not file_ids:
            return JobHandlerResult(summary="No knowledge sources to reindex", data={"count": 0})
        await context.progress({"stage": "reindexing", "count": len(file_ids)})
        self._knowledge.index_sources(
            customer_id=context.tenant_id,
            scope_type=_SCOPE_TYPE,
            scope_id=_SCOPE_ID,
            file_ids=file_ids,
        )
        source_ids: list_type[JsonValue] = [str(file_id) for file_id in file_ids]
        data: dict[str, JsonValue] = {
            "count": len(file_ids),
            "source_ids": source_ids,
        }
        return JobHandlerResult(
            summary="Knowledge sources reindexed",
            data=data,
        )

    @staticmethod
    def _source(row: Any) -> dict[str, Any]:
        try:
            tags = json.loads(str(row["tags_json"] or "[]"))
        except (TypeError, ValueError):
            tags = []
        return {
            "tenant_id": str(row["customer_id"]),
            "source_id": str(row["file_id"]),
            "file_id": str(row["file_id"]),
            "title": str(row["title"] or row["filename"]),
            "filename": str(row["filename"]),
            "mime_type": str(row["mime_type"]),
            "status": "archived" if bool(row["archived"]) else str(row["status"]),
            "tags": tags if isinstance(tags, list) else [],
            "section_count": int(row["section_count"] or 0),
            "char_count": int(row["char_count"] or 0),
            "indexed_at": str(row["indexed_at"]),
        }


__all__ = [
    "KnowledgeAttachJobArguments",
    "KnowledgeReindexJobArguments",
    "TenantKnowledgeService",
]
