"""Authenticated control boundary between a mutable release and stable evolution."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.evolution.models import Candidate, CandidateStatus, PromotionAttempt

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}\Z")


class EvolutionControlError(RuntimeError):
    """Sanitized failure returned across the stable control boundary."""


class _ControlModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ContributionRequest(_ControlModel):
    expected_revision: int | None = Field(default=None, ge=1)
    audit_context: dict[str, str] = Field(default_factory=dict)


class SourceContextRequest(_ControlModel):
    audit_context: dict[str, str] = Field(default_factory=dict)


class SourceShellRequest(SourceContextRequest):
    command: str = Field(min_length=1, max_length=100_000)
    timeout_seconds: int = Field(default=300, ge=1, le=3_600)


class SourceSyncRequest(SourceContextRequest):
    expected_active_release_id: str = Field(min_length=1, max_length=100)


class SourceResolveDependenciesRequest(SourceContextRequest):
    expected_candidate_id: str = Field(min_length=1, max_length=100)
    expected_diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceReleaseRequest(SourceContextRequest):
    idempotency_key: str = Field(min_length=1, max_length=200)
    expected_candidate_id: str = Field(min_length=1, max_length=100)
    expected_diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    message: str = Field(default="OpenTulpa self-update", min_length=1, max_length=500)


class SourceRollbackRequest(SourceContextRequest):
    idempotency_key: str = Field(min_length=1, max_length=200)
    expected_current_release_id: str = Field(min_length=1, max_length=100)
    expected_target_release_id: str = Field(min_length=1, max_length=100)
    reason: str = Field(default="Owner requested rollback", max_length=4_000)


def register_evolution_control_api(
    app: FastAPI,
    *,
    service: Any,
    token: str,
    prefix: str = "/bootstrap/internal/v1/evolution",
) -> None:
    """Expose only typed evolution operations; no host, Git, or OCI primitive escapes."""

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

    @router.post("/source/shell")
    async def source_shell(body: SourceShellRequest) -> dict[str, Any]:
        return dict(
            await service.source_shell(
                command=body.command,
                timeout_seconds=body.timeout_seconds,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/sync-upstream")
    async def source_sync_upstream(body: SourceSyncRequest) -> dict[str, Any]:
        return dict(
            await service.source_sync_upstream(
                expected_active_release_id=body.expected_active_release_id,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/resolve-dependencies")
    async def source_resolve_dependencies(
        body: SourceResolveDependenciesRequest,
    ) -> dict[str, Any]:
        return dict(
            await service.source_resolve_dependencies(
                expected_candidate_id=body.expected_candidate_id,
                expected_diff_sha256=body.expected_diff_sha256,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/release", status_code=status.HTTP_202_ACCEPTED)
    async def source_release(body: SourceReleaseRequest) -> dict[str, Any]:
        return dict(
            await service.source_release(
                idempotency_key=body.idempotency_key,
                expected_candidate_id=body.expected_candidate_id,
                expected_diff_sha256=body.expected_diff_sha256,
                message=body.message,
                audit_context=body.audit_context,
            )
        )

    @router.post("/source/rollback", status_code=status.HTTP_202_ACCEPTED)
    async def source_rollback(body: SourceRollbackRequest) -> dict[str, Any]:
        attempt = await service.source_rollback(
            idempotency_key=body.idempotency_key,
            expected_current_release_id=body.expected_current_release_id,
            expected_target_release_id=body.expected_target_release_id,
            reason=body.reason,
            audit_context=body.audit_context,
        )
        return dict(attempt.model_dump(mode="json"))

    @router.get("/candidates")
    async def list_candidates(
        candidate_status: Annotated[CandidateStatus | None, Query(alias="status")] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 100,
    ) -> list[dict[str, Any]]:
        candidates = await service.list_candidates(status=candidate_status, limit=limit)
        return [candidate.model_dump(mode="json") for candidate in candidates]

    @router.get("/candidates/{candidate_id}")
    async def get_candidate(candidate_id: str) -> dict[str, Any]:
        candidate = await service.get_candidate(candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        return dict(candidate.model_dump(mode="json"))

    @router.get("/promotions/{attempt_id}")
    async def promotion(attempt_id: str) -> dict[str, Any]:
        attempt = await service.get_promotion_attempt(attempt_id)
        if attempt is None:
            raise HTTPException(status_code=404, detail="promotion attempt not found")
        return dict(attempt.model_dump(mode="json"))

    @router.post("/candidates/{candidate_id}/contribution")
    async def contribution(candidate_id: str, body: ContributionRequest) -> dict[str, Any]:
        candidate = await service.prepare_contribution(
            candidate_id,
            expected_revision=body.expected_revision,
            audit_context=body.audit_context,
        )
        return dict(candidate.model_dump(mode="json"))

    @router.get("/candidates/{candidate_id}/patch", response_model=None)
    async def patch(candidate_id: str) -> FileResponse:
        path = await service.review_patch(candidate_id)
        return FileResponse(
            path=path,
            media_type="text/x-patch",
            headers={"X-Content-SHA256": await _file_sha256(path)},
        )

    app.include_router(router)


async def _file_sha256(path: Path) -> str:
    import asyncio

    def digest() -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    return await asyncio.to_thread(digest)


class EvolutionClient:
    """Mutable-release client for the fixed evolution control API."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        review_cache_root: Path,
        client: httpx.AsyncClient | None = None,
        max_patch_bytes: int = 20 * 1024 * 1024,
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
        if not 1_024 <= max_patch_bytes <= 100 * 1024 * 1024:
            raise ValueError("evolution patch byte limit is invalid")
        self._base_url = cleaned_url
        self._headers = {"X-OpenTulpa-Evolution-Token": safe_token}
        self._review_cache_root = review_cache_root.expanduser().resolve()
        self._max_patch_bytes = max_patch_bytes
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(60.0, read=300.0),
            trust_env=False,
        )
        self._owns_client = client is None
        self._started = False

    async def start(self) -> None:
        self._review_cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
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
        return self._mapping(
            await self._json(
                "POST",
                "/source/status",
                json={"audit_context": dict(audit_context or {})},
            )
        )

    async def source_shell(
        self,
        *,
        command: str,
        timeout_seconds: int = 300,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._mapping(
            await self._json(
                "POST",
                "/source/shell",
                json={
                    "command": command,
                    "timeout_seconds": timeout_seconds,
                    "audit_context": dict(audit_context or {}),
                },
                timeout=httpx.Timeout(60.0, read=max(660.0, timeout_seconds + 60.0)),
            )
        )

    async def source_sync_upstream(
        self,
        *,
        expected_active_release_id: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._mapping(
            await self._json(
                "POST",
                "/source/sync-upstream",
                json={
                    "expected_active_release_id": expected_active_release_id,
                    "audit_context": dict(audit_context or {}),
                },
                timeout=httpx.Timeout(60.0, read=300.0),
            )
        )

    async def source_resolve_dependencies(
        self,
        *,
        expected_candidate_id: str,
        expected_diff_sha256: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._mapping(
            await self._json(
                "POST",
                "/source/resolve-dependencies",
                json={
                    "expected_candidate_id": expected_candidate_id,
                    "expected_diff_sha256": expected_diff_sha256,
                    "audit_context": dict(audit_context or {}),
                },
                timeout=httpx.Timeout(60.0, read=1_800.0),
            )
        )

    async def source_release(
        self,
        *,
        idempotency_key: str,
        expected_candidate_id: str,
        expected_diff_sha256: str,
        message: str = "OpenTulpa self-update",
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._mapping(
            await self._json(
                "POST",
                "/source/release",
                json={
                    "idempotency_key": idempotency_key,
                    "expected_candidate_id": expected_candidate_id,
                    "expected_diff_sha256": expected_diff_sha256,
                    "message": message,
                    "audit_context": dict(audit_context or {}),
                },
                timeout=httpx.Timeout(60.0, read=1_800.0),
            )
        )

    async def source_rollback(
        self,
        *,
        idempotency_key: str,
        expected_current_release_id: str,
        expected_target_release_id: str,
        reason: str = "Owner requested rollback",
        audit_context: Mapping[str, str] | None = None,
    ) -> PromotionAttempt:
        result = await self._json(
            "POST",
            "/source/rollback",
            json={
                "idempotency_key": idempotency_key,
                "expected_current_release_id": expected_current_release_id,
                "expected_target_release_id": expected_target_release_id,
                "reason": reason,
                "audit_context": dict(audit_context or {}),
            },
        )
        return PromotionAttempt.model_validate(result)

    async def get_candidate(self, candidate_id: str) -> Candidate | None:
        response = await self._request("GET", f"/candidates/{self._identifier(candidate_id)}")
        if response.status_code == 404:
            return None
        return Candidate.model_validate(self._response_json(response))

    async def list_candidates(
        self,
        *,
        status: CandidateStatus | str | None = None,
        limit: int = 100,
    ) -> list[Candidate]:
        params: dict[str, str | int] = {"limit": limit}
        if status is not None:
            params["status"] = CandidateStatus(status).value
        payload = await self._json("GET", "/candidates", params=params)
        if not isinstance(payload, list):
            raise EvolutionControlError("evolution control returned an invalid candidate list")
        return [Candidate.model_validate(item) for item in payload]

    async def get_promotion_attempt(self, attempt_id: str) -> PromotionAttempt | None:
        response = await self._request(
            "GET", f"/promotions/{self._identifier(attempt_id)}"
        )
        if response.status_code == 404:
            return None
        return PromotionAttempt.model_validate(self._response_json(response))

    async def prepare_contribution(
        self,
        candidate_id: str,
        *,
        expected_revision: int | None = None,
        audit_context: Mapping[str, str] | None = None,
    ) -> Candidate:
        result = await self._json(
            "POST",
            f"/candidates/{self._identifier(candidate_id)}/contribution",
            json={
                "expected_revision": expected_revision,
                "audit_context": dict(audit_context or {}),
            },
        )
        return Candidate.model_validate(result)

    async def review_patch(self, candidate_id: str) -> Path:
        safe_id = self._identifier(candidate_id)
        response = await self._request("GET", f"/candidates/{safe_id}/patch")
        content = response.content
        if len(content) > self._max_patch_bytes:
            raise EvolutionControlError("candidate patch exceeded its byte limit")
        expected = str(response.headers.get("x-content-sha256") or "").strip().lower()
        actual = hashlib.sha256(content).hexdigest()
        if not hmac.compare_digest(expected, actual):
            raise EvolutionControlError("candidate patch failed digest validation")
        self._review_cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = self._review_cache_root / f"{safe_id}-{actual}.patch"
        temporary = self._review_cache_root / f".{safe_id}-{os.getpid()}.tmp"
        temporary.write_bytes(content)
        temporary.chmod(0o600)
        os.replace(temporary, target)
        return target

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._response_json(await self._request(method, path, **kwargs))

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if not self._started:
            raise EvolutionControlError("evolution control client is not started")
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}{path}",
                headers=self._headers,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise EvolutionControlError("stable evolution control is unavailable") from exc
        if response.status_code >= 400 and response.status_code != 404:
            raise EvolutionControlError("stable evolution control rejected the operation")
        return response

    @staticmethod
    def _response_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise EvolutionControlError("evolution control returned an invalid response") from exc

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise EvolutionControlError("evolution control returned an invalid response")
        return dict(value)

    @staticmethod
    def _identifier(value: str) -> str:
        safe = str(value or "").strip()
        if _IDENTIFIER_RE.fullmatch(safe) is None:
            raise ValueError("evolution identifier is invalid")
        return safe


__all__ = [
    "EvolutionClient",
    "EvolutionControlError",
    "register_evolution_control_api",
]
