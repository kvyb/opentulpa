"""Use-case helpers for internal user-context HTTP routes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse

from opentulpa.api.customer_ids import resolve_body_customer_id


@dataclass(frozen=True)
class UserContextRouteUseCases:
    get_user_context_service: Callable[[], Any]
    get_file_vault: Callable[[], Any]
    get_agent_runtime: Callable[[], Any]
    resolve_customer_id: Callable[[str], str] | None = None

    def _customer_id(self, body: dict[str, Any]) -> str:
        return resolve_body_customer_id(body, self.resolve_customer_id)

    def _record_event(self, event: str, **fields: Any) -> None:
        runtime = self.get_agent_runtime()
        if runtime is None:
            return
        recorder = getattr(runtime, "record_observability_event", None)
        if callable(recorder):
            recorder(event=event, **fields)
            return
        logger = getattr(runtime, "log_behavior_event", None)
        if callable(logger):
            logger(event=event, **fields)

    async def add_files(self, body: dict[str, Any]) -> Any:
        service = self.get_user_context_service()
        try:
            started = time.monotonic()
            customer_id = self._customer_id(body)
            file_ids: list[Any] = _list_body_value(body, "file_ids")
            prep = await self.prepare_files(customer_id=customer_id, file_ids=file_ids)
            result = service.add_files(customer_id=customer_id, file_ids=file_ids)
            self._record_event(
                "user_context.add_files",
                customer_id=customer_id,
                file_count=len(file_ids),
                source_count=source_count(result),
                warning_count=warning_count(result),
                prepared_count=int(prep.get("prepared_count") or 0),
                failed_count=int(prep.get("failed_count") or 0),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def list_sources(self, body: dict[str, Any]) -> Any:
        service = self.get_user_context_service()
        try:
            return service.list_sources(
                customer_id=self._customer_id(body),
                include_archived=bool(body.get("include_archived", False)),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def find_sources(self, body: dict[str, Any]) -> Any:
        service = self.get_user_context_service()
        try:
            return service.find_sources(
                customer_id=self._customer_id(body),
                query=str(body.get("query", "")).strip(),
                limit=int(body.get("limit", 10) or 10),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def query(self, body: dict[str, Any]) -> Any:
        service = self.get_user_context_service()
        try:
            started = time.monotonic()
            customer_id = self._customer_id(body)
            result = service.query(
                customer_id=customer_id,
                query=str(body.get("query", "")).strip(),
                max_extract_chars=int(body.get("max_extract_chars", 3000) or 3000),
            )
            self._record_event(
                "user_context.query",
                customer_id=customer_id,
                ok=bool(result.get("ok")) if isinstance(result, dict) else False,
                source_count=int(result.get("source_count") or 0) if isinstance(result, dict) else 0,
                section_count=int(result.get("section_count") or 0)
                if isinstance(result, dict)
                else 0,
                warning_count=warning_count(result),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def reindex(self, body: dict[str, Any]) -> Any:
        service = self.get_user_context_service()
        try:
            started = time.monotonic()
            customer_id = self._customer_id(body)
            raw_file_ids = body.get("file_ids")
            file_ids = raw_file_ids if isinstance(raw_file_ids, list) else None
            prep = {"prepared_count": 0, "failed_count": 0}
            if file_ids:
                prep = await self.prepare_files(customer_id=customer_id, file_ids=file_ids)
            result = service.reindex(customer_id=customer_id, file_ids=file_ids)
            self._record_event(
                "user_context.reindex",
                customer_id=customer_id,
                file_count=len(file_ids or []),
                source_count=source_count(result),
                warning_count=warning_count(result),
                prepared_count=int(prep.get("prepared_count") or 0),
                failed_count=int(prep.get("failed_count") or 0),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def archive_sources(self, body: dict[str, Any]) -> Any:
        service = self.get_user_context_service()
        try:
            started = time.monotonic()
            customer_id = self._customer_id(body)
            file_ids = _list_body_value(body, "file_ids")
            result = service.archive_sources(customer_id=customer_id, file_ids=file_ids)
            self._record_event(
                "user_context.archive_sources",
                customer_id=customer_id,
                file_count=len(file_ids),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def promote_to_intake(self, body: dict[str, Any]) -> Any:
        service = self.get_user_context_service()
        try:
            started = time.monotonic()
            customer_id = self._customer_id(body)
            file_ids = _list_body_value(body, "file_ids")
            workflow_id = str(body.get("workflow_id", "")).strip()
            result = service.promote_to_intake(
                customer_id=customer_id,
                workflow_id=workflow_id,
                file_ids=file_ids,
            )
            self._record_event(
                "user_context.promote_to_intake",
                customer_id=customer_id,
                workflow_id=workflow_id,
                file_count=len(file_ids),
                source_count=source_count(result),
                warning_count=warning_count(result),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def prepare_files(self, *, customer_id: str, file_ids: list[Any]) -> dict[str, Any]:
        runtime = self.get_agent_runtime()
        if runtime is None or not hasattr(runtime, "summarize_uploaded_blob"):
            return {"prepared_count": 0, "failed_count": 0}
        vault = self.get_file_vault()
        prepared_count = 0
        failed_count = 0
        for raw_file_id in file_ids[:50]:
            file_id = str(raw_file_id or "").strip()
            if not file_id:
                continue
            record = vault.get_file(customer_id, file_id)
            if not isinstance(record, dict) or not record:
                continue
            if not needs_model_processing(record):
                continue
            raw_bytes = vault.read_file_bytes(customer_id, file_id)
            if raw_bytes is None:
                continue
            try:
                analysis = await runtime.summarize_uploaded_blob(
                    filename=str(record.get("original_filename", "") or "").strip() or None,
                    mime_type=str(record.get("mime_type", "") or "").strip().lower() or None,
                    kind=str(record.get("kind", "") or "").strip().lower() or None,
                    raw_bytes=raw_bytes,
                    caption=str(record.get("caption", "") or "").strip() or None,
                    question=(
                        "Prepare this file for durable user_context retrieval. Extract transcript-like "
                        "speech, visible text, visual facts, document layout facts, hooks, offers, claims, "
                        "style cues, and concrete details. Return concise source-grounding notes only."
                    ),
                )
            except Exception as exc:
                failed_count += 1
                self._record_prepare_failure(
                    customer_id=customer_id,
                    file_id=file_id,
                    record=record,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
                continue
            if str(analysis or "").strip():
                vault.set_ai_summary(customer_id, file_id, str(analysis).strip())
                prepared_count += 1
                self._record_prepare_success(
                    customer_id=customer_id,
                    file_id=file_id,
                    record=record,
                    analysis=str(analysis or ""),
                )
            else:
                failed_count += 1
        return {"prepared_count": prepared_count, "failed_count": failed_count}

    def _record_prepare_failure(
        self,
        *,
        customer_id: str,
        file_id: str,
        record: dict[str, Any],
        error: str,
    ) -> None:
        self._record_event(
            "user_context.media_prepare_failed",
            customer_id=customer_id,
            file_id=file_id,
            filename=str(record.get("original_filename", "") or "").strip(),
            mime_type=str(record.get("mime_type", "") or "").strip().lower(),
            kind=str(record.get("kind", "") or "").strip().lower(),
            error=error,
        )

    def _record_prepare_success(
        self,
        *,
        customer_id: str,
        file_id: str,
        record: dict[str, Any],
        analysis: str,
    ) -> None:
        self._record_event(
            "user_context.media_prepare_succeeded",
            customer_id=customer_id,
            file_id=file_id,
            filename=str(record.get("original_filename", "") or "").strip(),
            mime_type=str(record.get("mime_type", "") or "").strip().lower(),
            kind=str(record.get("kind", "") or "").strip().lower(),
            analysis_chars=len(analysis),
        )


def warning_count(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        return len(warnings)
    indexed = payload.get("indexed")
    if isinstance(indexed, dict):
        return sum(
            len(source.get("warnings") or [])
            for source in indexed.get("sources", [])
            if isinstance(source, dict)
        )
    return 0


def source_count(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    sources = payload.get("sources")
    if isinstance(sources, list):
        return len(sources)
    indexed = payload.get("indexed")
    if isinstance(indexed, dict):
        index = indexed.get("index")
        if isinstance(index, dict):
            return int(index.get("source_count") or 0)
    return 0


def needs_model_processing(record: dict[str, Any]) -> bool:
    filename = str(record.get("original_filename", "") or "").strip()
    mime_type = str(record.get("mime_type", "") or "").strip().lower()
    kind = str(record.get("kind", "") or "").strip().lower()
    lower_name = filename.lower()
    return (
        kind in {"photo", "video", "video_note", "audio", "voice"}
        or mime_type.startswith(("image/", "video/", "audio/"))
        or mime_type == "application/pdf"
        or lower_name.endswith(".pdf")
    )


def _list_body_value(body: dict[str, Any], key: str) -> list[Any]:
    value = body.get(key)
    return value if isinstance(value, list) else []
