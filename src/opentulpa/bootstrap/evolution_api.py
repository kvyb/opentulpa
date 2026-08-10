"""Authenticated boundary between a mutable runtime and stable source control."""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from typing import Annotated, Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field


class EvolutionControlError(RuntimeError):
    """Sanitized failure returned across the stable control boundary."""


class _ControlModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceContextRequest(_ControlModel):
    audit_context: dict[str, str] = Field(default_factory=dict)


class SourceReadRequest(SourceContextRequest):
    path: str = Field(min_length=1, max_length=4_096)
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=2_000, ge=1, le=2_000)


class SourceWriteRequest(SourceContextRequest):
    path: str = Field(min_length=1, max_length=4_096)
    content: str = Field(max_length=2_000_000)


class SourceEditRequest(SourceContextRequest):
    path: str = Field(min_length=1, max_length=4_096)
    old_text: str = Field(min_length=1, max_length=1_000_000)
    new_text: str = Field(max_length=1_000_000)
    replace_all: bool = False


class SourceBashRequest(SourceContextRequest):
    command: str = Field(min_length=1, max_length=100_000)
    timeout_seconds: int = Field(default=300, ge=1, le=600)


class SourceActivateRequest(SourceContextRequest):
    idempotency_key: str = Field(min_length=1, max_length=200)
    message: str = Field(default="OpenTulpa self-update", min_length=1, max_length=500)
    reason: str = Field(default="Trusted source activation", max_length=4_000)


class SourceRollbackRequest(SourceContextRequest):
    idempotency_key: str = Field(min_length=1, max_length=200)
    expected_active_release_id: str = Field(min_length=1, max_length=100)
    reason: str = Field(default="Owner requested rollback", max_length=4_000)


class SourceRuntimeEnvSetRequest(SourceContextRequest):
    idempotency_key: str = Field(min_length=1, max_length=200)
    name: str = Field(pattern=r"^[A-Z_][A-Z0-9_]{0,127}$")
    value: str = Field(max_length=65_536)


def register_evolution_control_api(
    app: FastAPI,
    *,
    service: Any,
    token: str,
    prefix: str = "/bootstrap/internal/v1/evolution",
) -> None:
    """Expose the small trusted source surface through a fixed host route."""

    expected_token = str(token or "").strip()
    if len(expected_token) < 32:
        raise ValueError("evolution control token must contain at least 32 characters")

    async def authorize(
        supplied: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Evolution-Token", max_length=500),
        ] = None,
    ) -> None:
        if not hmac.compare_digest(str(supplied or ""), expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid evolution control credentials are required",
            )

    router = APIRouter(prefix=prefix, dependencies=[Depends(authorize)], include_in_schema=False)

    @router.post("/source/status")
    async def source_status(body: SourceContextRequest) -> dict[str, Any]:
        return dict(await service.source_status(audit_context=body.audit_context))

    @router.post("/source/read")
    async def source_read(body: SourceReadRequest) -> dict[str, Any]:
        return dict(
            await service.source_read(
                path=body.path,
                offset=body.offset,
                limit=body.limit,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/write")
    async def source_write(body: SourceWriteRequest) -> dict[str, Any]:
        return dict(
            await service.source_write(
                path=body.path,
                content=body.content,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/edit")
    async def source_edit(body: SourceEditRequest) -> dict[str, Any]:
        return dict(
            await service.source_edit(
                path=body.path,
                old_text=body.old_text,
                new_text=body.new_text,
                replace_all=body.replace_all,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/bash")
    async def source_bash(body: SourceBashRequest) -> dict[str, Any]:
        return dict(
            await service.source_bash(
                command=body.command,
                timeout_seconds=body.timeout_seconds,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/activate", status_code=status.HTTP_202_ACCEPTED)
    async def source_activate(body: SourceActivateRequest) -> dict[str, Any]:
        return dict(
            await service.source_activate(
                idempotency_key=body.idempotency_key,
                message=body.message,
                reason=body.reason,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/rollback", status_code=status.HTTP_202_ACCEPTED)
    async def source_rollback(body: SourceRollbackRequest) -> dict[str, Any]:
        return dict(
            await service.source_rollback(
                idempotency_key=body.idempotency_key,
                expected_active_release_id=body.expected_active_release_id,
                reason=body.reason,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/runtime-env/read")
    async def source_runtime_env_get(body: SourceContextRequest) -> dict[str, Any]:
        return dict(await service.source_runtime_env_get(audit_context=body.audit_context))

    @router.post("/source/runtime-env")
    async def source_runtime_env_set(body: SourceRuntimeEnvSetRequest) -> dict[str, Any]:
        return dict(
            await service.source_set_runtime_env(
                name=body.name,
                value=body.value,
                idempotency_key=body.idempotency_key,
                audit_context=body.audit_context,
            )
        )

    app.include_router(router)


class EvolutionClient:
    """Mutable-runtime client for the stable source control API."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        cleaned_url = str(base_url or "").strip().rstrip("/")
        parsed = urlsplit(cleaned_url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("evolution control URL must be an authenticated HTTP endpoint")
        safe_token = str(token or "").strip()
        if len(safe_token) < 32:
            raise ValueError("evolution control token must contain at least 32 characters")
        self._base_url = cleaned_url
        self._headers = {"X-OpenTulpa-Evolution-Token": safe_token}
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(60, read=300),
            trust_env=False,
        )
        self._owns_client = client is None
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def shutdown(self) -> None:
        self._started = False
        if self._owns_client:
            await self._client.aclose()

    async def source_status(
        self,
        *,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post("/source/status", audit_context=audit_context)

    async def source_read(
        self,
        *,
        path: str,
        offset: int = 1,
        limit: int = 2_000,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "/source/read",
            audit_context=audit_context,
            path=path,
            offset=offset,
            limit=limit,
        )

    async def source_write(
        self,
        *,
        path: str,
        content: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "/source/write", audit_context=audit_context, path=path, content=content
        )

    async def source_edit(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "/source/edit",
            audit_context=audit_context,
            path=path,
            old_text=old_text,
            new_text=new_text,
            replace_all=replace_all,
        )

    async def source_bash(
        self,
        *,
        command: str,
        timeout_seconds: int = 300,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "/source/bash",
            audit_context=audit_context,
            timeout=httpx.Timeout(60, read=max(660, timeout_seconds + 60)),
            command=command,
            timeout_seconds=timeout_seconds,
        )

    async def source_activate(
        self,
        *,
        idempotency_key: str,
        message: str = "OpenTulpa self-update",
        reason: str = "Trusted source activation",
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "/source/activate",
            audit_context=audit_context,
            idempotency_key=idempotency_key,
            message=message,
            reason=reason,
        )

    async def source_rollback(
        self,
        *,
        idempotency_key: str,
        expected_active_release_id: str,
        reason: str = "Owner requested rollback",
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "/source/rollback",
            audit_context=audit_context,
            idempotency_key=idempotency_key,
            expected_active_release_id=expected_active_release_id,
            reason=reason,
        )

    async def source_runtime_env_get(
        self,
        *,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post("/source/runtime-env/read", audit_context=audit_context)

    async def source_set_runtime_env(
        self,
        *,
        name: str,
        value: str,
        idempotency_key: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "/source/runtime-env",
            audit_context=audit_context,
            name=name,
            value=value,
            idempotency_key=idempotency_key,
        )

    async def _post(
        self,
        endpoint: str,
        *,
        audit_context: Mapping[str, str] | None,
        timeout: httpx.Timeout | None = None,
        **values: Any,
    ) -> dict[str, Any]:
        payload = {**values, "audit_context": dict(audit_context or {})}
        response = await self._request("POST", endpoint, json=payload, timeout=timeout)
        try:
            result = response.json()
        except ValueError as exc:
            raise EvolutionControlError("evolution control returned an invalid response") from exc
        if not isinstance(result, dict):
            raise EvolutionControlError("evolution control returned an invalid object")
        return result

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if not self._started:
            raise EvolutionControlError("evolution control client is not started")
        if kwargs.get("timeout") is None:
            kwargs.pop("timeout", None)
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}{path}",
                headers=self._headers,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise EvolutionControlError("stable evolution control is unavailable") from exc
        if response.status_code >= 400:
            raise EvolutionControlError("stable evolution control rejected the operation")
        return response


__all__ = ["EvolutionClient", "EvolutionControlError", "register_evolution_control_api"]
