"""Authenticated tenant-scoped v2 file routes."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, ConfigDict

from opentulpa.api.routes.v2_principal import (
    V2Principal,
    require_v2_scope,
    resolve_v2_principal,
)
from opentulpa.capabilities.credentials import CapabilityAPIScope
from opentulpa.persistence.idempotency import (
    IdempotencyConflictError,
    IdempotencyPendingError,
    IdempotencyStore,
)

logger = logging.getLogger(__name__)


class FileVaultPort(Protocol):
    def search(self, customer_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]: ...

    def get_file(self, customer_id: str, file_id: str) -> dict[str, Any] | None: ...

    def ingest_file(
        self,
        *,
        customer_id: str,
        chat_id: int | None,
        kind: str,
        telegram_file_id: str | None,
        original_filename: str | None,
        mime_type: str | None,
        caption: str | None,
        raw_bytes: bytes,
    ) -> dict[str, Any]: ...

    def delete_file(self, customer_id: str, file_id: str) -> bool: ...


class PublicFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    original_filename: str
    mime_type: str | None = None
    size_bytes: int
    caption: str | None = None
    summary: str = ""
    text_excerpt: str | None = None
    created_at: str


class FileListResponse(BaseModel):
    files: list[PublicFile]


class FileResponse(BaseModel):
    file: PublicFile


class FileDeleteResponse(BaseModel):
    deleted: bool
    file_id: str


def _public_file(
    record: dict[str, Any],
    *,
    include_excerpt: bool,
    max_excerpt_chars: int = 16_000,
) -> PublicFile:
    payload: dict[str, Any] = {
        "id": str(record.get("id", "") or ""),
        "kind": str(record.get("kind", "file") or "file"),
        "original_filename": str(record.get("original_filename", "file.bin") or "file.bin"),
        "mime_type": str(record.get("mime_type", "") or "").strip() or None,
        "size_bytes": max(0, int(record.get("size_bytes", 0) or 0)),
        "caption": str(record.get("caption", "") or "").strip() or None,
        "summary": str(record.get("summary", "") or ""),
        "created_at": str(record.get("created_at", "") or ""),
    }
    if include_excerpt:
        payload["text_excerpt"] = str(record.get("text_excerpt", "") or "")[:max_excerpt_chars]
    return PublicFile.model_validate(payload)


def register_v2_file_routes(
    app: FastAPI,
    *,
    get_file_vault: Callable[[], FileVaultPort | None],
    get_idempotency_store: Callable[[], IdempotencyStore | None],
    resolve_principal: Callable[[Request], V2Principal | Awaitable[V2Principal]],
    max_upload_bytes: int = 45_000_000,
) -> None:
    """Register file create/read/delete routes without exposing tenant IDs or paths."""

    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")

    def vault_or_503() -> FileVaultPort:
        vault = get_file_vault()
        if vault is None:
            raise HTTPException(status_code=503, detail="file service unavailable")
        return vault

    def idempotency_or_503() -> IdempotencyStore:
        store = get_idempotency_store()
        if store is None:
            raise HTTPException(status_code=503, detail="idempotency service unavailable")
        return store

    @app.get("/v2/files", response_model=FileListResponse)
    async def list_files(
        request: Request,
        query: Annotated[str, Query(max_length=2_000)] = "",
        limit: Annotated[int, Query(ge=1, le=20)] = 20,
    ) -> FileListResponse:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, "files.read")
        records = vault_or_503().search(
            principal.tenant_id,
            query=str(query or "").strip(),
            limit=limit,
        )
        return FileListResponse(
            files=[_public_file(record, include_excerpt=False) for record in records]
        )

    @app.get("/v2/files/{file_id}", response_model=FileResponse)
    async def get_file(
        file_id: str,
        request: Request,
        max_excerpt_chars: Annotated[int, Query(ge=500, le=60_000)] = 16_000,
    ) -> FileResponse:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, "files.read")
        record = vault_or_503().get_file(principal.tenant_id, file_id)
        if record is None:
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(
            file=_public_file(
                record,
                include_excerpt=True,
                max_excerpt_chars=max_excerpt_chars,
            )
        )

    @app.post("/v2/files", response_model=FileResponse, status_code=201)
    async def create_file(
        request: Request,
        upload: Annotated[UploadFile, File()],
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1, max_length=300),
        ],
        kind: Annotated[str, Form(min_length=1, max_length=50)] = "document",
        caption: Annotated[str | None, Form(max_length=4_000)] = None,
    ) -> FileResponse:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.FILE_UPLOAD.value)
        form = await request.form()
        if {"tenant_id", "customer_id", "actor_id"}.intersection(form):
            raise HTTPException(status_code=422, detail="identity fields are not accepted")
        filename = str(upload.filename or "file.bin")
        mime_type = str(upload.content_type or "").strip() or None
        try:
            raw_bytes = await upload.read(max_upload_bytes + 1)
        finally:
            await upload.close()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="file is empty")
        if len(raw_bytes) > max_upload_bytes:
            raise HTTPException(status_code=413, detail="file is too large")
        safe_key = str(idempotency_key or "").strip()
        if not safe_key:
            raise HTTPException(status_code=422, detail="Idempotency-Key is required")
        safe_kind = str(kind or "document").strip()
        safe_caption = str(caption or "").strip() or None
        request_hash = IdempotencyStore.request_hash(
            "v2_file_upload_payload",
            {
                "filename": filename,
                "mime_type": mime_type,
                "kind": safe_kind,
                "caption": safe_caption,
                "size_bytes": len(raw_bytes),
                "content_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            },
        )
        vault = vault_or_503()

        def ingest() -> dict[str, Any]:
            return vault.ingest_file(
                customer_id=principal.tenant_id,
                chat_id=None,
                kind=safe_kind,
                telegram_file_id=None,
                original_filename=filename,
                mime_type=mime_type,
                caption=safe_caption,
                raw_bytes=raw_bytes,
            )

        try:
            result = await idempotency_or_503().execute(
                tenant_id=principal.tenant_id,
                operation="v2_file_upload",
                idempotency_key=safe_key,
                request_hash=request_hash,
                invoke=ingest,
            )
            file_id = str(result.get("id", "") if isinstance(result, dict) else "").strip()
            record = vault.get_file(principal.tenant_id, file_id) if file_id else None
            if record is None:
                raise HTTPException(
                    status_code=409,
                    detail="idempotent file result is no longer available",
                )
        except (IdempotencyConflictError, IdempotencyPendingError) as exc:
            raise HTTPException(status_code=409, detail="idempotency key conflict") from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("v2 file ingest failed: exception=%s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="file could not be stored") from exc
        return FileResponse(file=_public_file(record, include_excerpt=True))

    @app.delete("/v2/files/{file_id}", response_model=FileDeleteResponse)
    async def delete_file(file_id: str, request: Request) -> FileDeleteResponse:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, "files.delete")
        vault = vault_or_503()
        if vault.get_file(principal.tenant_id, file_id) is None:
            raise HTTPException(status_code=404, detail="file not found")
        if not vault.delete_file(principal.tenant_id, file_id):
            raise HTTPException(status_code=500, detail="file could not be deleted")
        return FileDeleteResponse(deleted=True, file_id=file_id)


__all__ = [
    "FileDeleteResponse",
    "FileListResponse",
    "FileResponse",
    "FileVaultPort",
    "PublicFile",
    "register_v2_file_routes",
]
