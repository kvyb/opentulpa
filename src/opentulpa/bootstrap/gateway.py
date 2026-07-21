"""Stable public gateway that survives mutable release replacement."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi import Path as ApiPath
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from opentulpa.bootstrap.capability_worker_api import (
    CapabilityWorkerLease,
    StableCapabilityWorkerService,
    register_capability_worker_api,
)
from opentulpa.bootstrap.evolution_api import register_evolution_control_api
from opentulpa.bootstrap.evolution_composition import build_managed_evolution_runtime
from opentulpa.bootstrap.evolution_runtime import ManagedEvolutionRuntime
from opentulpa.bootstrap.host import ReleaseHost
from opentulpa.bootstrap.models import IngressEnvelope, OutboxEvent, ReleaseRecord, RunningRelease
from opentulpa.bootstrap.oci_host import OciMount, OciReleasePolicy, RootlessOciReleaseHost
from opentulpa.bootstrap.recovery import RecoveryService, create_recovery_router
from opentulpa.bootstrap.sandbox_api import (
    SandboxExecutionLease,
    TenantSandboxExecutionService,
    register_sandbox_execution_api,
)
from opentulpa.bootstrap.store import BootstrapConflictError, BootstrapStore, LeaseFenceError
from opentulpa.bootstrap.supervisor import BootstrapSupervisor, OutboxSink, SupervisorPolicy
from opentulpa.capabilities.oci_workers import OciCapabilityPolicy, OciCapabilityWorkerHost
from opentulpa.core.config import Settings, get_settings
from opentulpa.deep_agent.sandbox import TenantContainerPolicy
from opentulpa.evolution.sandbox import resolve_local_oci_image

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_INTERNAL_HEADERS = frozenset(
    {
        "x-opentulpa-control-token",
        "x-opentulpa-ingress-token",
        "x-opentulpa-release-id",
        "x-opentulpa-lease-epoch",
    }
)
_UNTRUSTED_FORWARDING_HEADERS = frozenset(
    {
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-proto",
    }
)

logger = logging.getLogger(__name__)


class GatewayUnavailableError(RuntimeError):
    pass


class GatewayIngressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tenant_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=300)
    payload: dict[str, JsonValue]


class ActiveReleaseTransport(OutboxSink):
    """Resolve only the fenced serving release and deliver internal envelopes."""

    def __init__(
        self,
        *,
        store: BootstrapStore,
        host: ReleaseHost,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._store = store
        self._host = host
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
            trust_env=False,
        )
        self._owns_client = client is None

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def resolve(self) -> tuple[RunningRelease, ReleaseRecord, int]:
        state = self._store.get_state()
        release_id = state.serving_release_id
        epoch = state.active_lease_epoch
        if state.safe_mode or state.ingress_paused or release_id is None or epoch is None:
            raise GatewayUnavailableError("release traffic is paused")
        self._store.assert_active_lease(release_id, epoch)
        release = self._store.get_release(release_id)
        if release is None:
            raise GatewayUnavailableError("serving release metadata is unavailable")
        running = await self._host.discover(release_id, mode="production")
        if (
            running is None
            or running.endpoint is None
            or running.control_token is None
            or running.lease_epoch != epoch
            or running.release_id != release_id
        ):
            raise GatewayUnavailableError("serving release process is unavailable")
        return running, release, epoch

    async def deliver_ingress(self, envelope: IngressEnvelope) -> None:
        running, release, epoch = await self.resolve()
        async with self._client.stream(
            "POST",
            f"{running.endpoint}{release.ingress_path}",
            json=envelope.model_dump(mode="json"),
            headers=self._delivery_headers(
                running,
                epoch=epoch,
                idempotency_key=envelope.idempotency_key,
            ),
        ) as response:
            if not 200 <= response.status_code < 300:
                raise GatewayUnavailableError("serving release rejected durable ingress")

    async def deliver(self, event: OutboxEvent) -> None:
        running, release, epoch = await self.resolve()
        async with self._client.stream(
            "POST",
            f"{running.endpoint}{release.event_path}",
            json=event.model_dump(mode="json"),
            headers=self._delivery_headers(
                running,
                epoch=epoch,
                idempotency_key=event.event_key,
            ),
        ) as response:
            if not 200 <= response.status_code < 300:
                raise GatewayUnavailableError("serving release rejected bootstrap event")

    @staticmethod
    def _gateway_headers(running: RunningRelease, epoch: int) -> dict[str, str]:
        headers = {
            "X-OpenTulpa-Release-ID": running.release_id,
            "X-OpenTulpa-Lease-Epoch": str(epoch),
        }
        if running.control_token is not None:
            headers["X-OpenTulpa-Control-Token"] = running.control_token
        return headers

    @classmethod
    def _delivery_headers(
        cls,
        running: RunningRelease,
        *,
        epoch: int,
        idempotency_key: str,
    ) -> dict[str, str]:
        headers = {
            **cls._gateway_headers(running, epoch),
            "Idempotency-Key": idempotency_key,
        }
        if running.control_token is not None:
            headers["Authorization"] = f"Bearer {running.control_token}"
        return headers


class BootstrapGateway:
    """Persist interface ingress and proxy public API traffic to the lease holder."""

    def __init__(
        self,
        *,
        store: BootstrapStore,
        transport: ActiveReleaseTransport,
        retry_interval_seconds: float = 0.5,
        max_ingress_bytes: int = 1_048_576,
    ) -> None:
        if not 0.05 <= retry_interval_seconds <= 60:
            raise ValueError("gateway retry interval must be between 0.05 and 60 seconds")
        if not 1_024 <= max_ingress_bytes <= 10_485_760:
            raise ValueError("gateway ingress limit must be between 1 KiB and 10 MiB")
        self._store = store
        self._transport = transport
        self._retry_interval_seconds = retry_interval_seconds
        self._max_ingress_bytes = max_ingress_bytes
        self._wake = asyncio.Event()
        self._dispatcher: asyncio.Task[None] | None = None

    @property
    def max_ingress_bytes(self) -> int:
        return self._max_ingress_bytes

    async def start(self) -> None:
        if self._dispatcher is not None and not self._dispatcher.done():
            return
        self._dispatcher = asyncio.create_task(
            self._dispatch_loop(),
            name="opentulpa-bootstrap-ingress-dispatcher",
        )

    async def shutdown(self) -> None:
        task = self._dispatcher
        self._dispatcher = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self._transport.aclose()

    def accept(self, envelope: IngressEnvelope) -> IngressEnvelope:
        if len(envelope.model_dump_json().encode("utf-8")) > self._max_ingress_bytes:
            raise ValueError("durable ingress exceeded its byte limit")
        persisted = self._store.enqueue_ingress(envelope)
        self._wake.set()
        return persisted

    async def dispatch_once(self, *, limit: int = 20) -> int:
        try:
            _, _, epoch = await self._transport.resolve()
        except (GatewayUnavailableError, LeaseFenceError):
            return 0
        state = self._store.get_state()
        release_id = state.serving_release_id
        if release_id is None:
            return 0
        claimed = self._store.claim_ingress(
            release_id=release_id,
            lease_epoch=epoch,
            limit=limit,
        )
        completed = 0
        for envelope in claimed:
            try:
                await self._transport.deliver_ingress(envelope)
                self._store.complete_ingress(
                    envelope.id,
                    release_id=release_id,
                    lease_epoch=epoch,
                )
                completed += 1
            except (GatewayUnavailableError, LeaseFenceError, httpx.HTTPError):
                with suppress(LeaseFenceError):
                    self._store.requeue_ingress_claim(
                        envelope.id,
                        release_id=release_id,
                        lease_epoch=epoch,
                    )
                break
        return completed

    async def proxy(
        self,
        request: Request,
        *,
        forbidden_bearer_tokens: tuple[str, ...] = (),
    ) -> StreamingResponse:
        try:
            running, release, epoch = await self._transport.resolve()
        except (GatewayUnavailableError, LeaseFenceError) as exc:
            raise HTTPException(status_code=503, detail="OpenTulpa release is unavailable") from exc
        assert running.endpoint is not None
        url = f"{running.endpoint}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        request_hop_headers = _HOP_BY_HOP_HEADERS | self._connection_tokens(
            request.headers.getlist("connection")
        )
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.casefold() not in request_hop_headers
            and not self._is_internal_header(key)
            and key.casefold()
            not in _UNTRUSTED_FORWARDING_HEADERS | {"host", "content-length"}
            and not (
                key.casefold() == "authorization"
                and self._contains_bearer(value, forbidden_bearer_tokens)
            )
        }
        headers.update(ActiveReleaseTransport._gateway_headers(running, epoch))
        upstream = self._transport.client.build_request(
            request.method,
            url,
            headers=headers,
            content=request.stream(),
        )
        try:
            response = await self._transport.client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="OpenTulpa release is unavailable") from exc

        async def response_body() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        proxied = StreamingResponse(
            response_body(),
            status_code=response.status_code,
        )
        response_hop_headers = _HOP_BY_HOP_HEADERS | self._connection_tokens(
            response.headers.get_list("connection")
        )
        proxied.raw_headers = [
            (key, value)
            for key, value in response.headers.raw
            if key.decode("ascii", errors="ignore").casefold()
            not in response_hop_headers | {"content-length"}
            and not self._is_internal_header(key.decode("ascii", errors="ignore"))
        ]
        return proxied

    @staticmethod
    def _contains_bearer(value: str, forbidden_tokens: tuple[str, ...]) -> bool:
        scheme, _, supplied = value.partition(" ")
        return scheme.casefold() == "bearer" and any(
            hmac.compare_digest(supplied, token) for token in forbidden_tokens
        )

    @staticmethod
    def _connection_tokens(values: list[str]) -> frozenset[str]:
        return frozenset(
            token.strip().casefold()
            for value in values
            for token in value.split(",")
            if token.strip()
        )

    @staticmethod
    def _is_internal_header(name: str) -> bool:
        normalized = name.casefold()
        return normalized in _INTERNAL_HEADERS or normalized.startswith("x-opentulpa-")

    async def _dispatch_loop(self) -> None:
        while True:
            self._wake.clear()
            try:
                await self.dispatch_once()
            except Exception:
                # The envelope remains durable; the next wake/retry tries again.
                logger.exception("bootstrap ingress dispatch failed")
            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=self._retry_interval_seconds,
                )
            except TimeoutError:
                continue


def create_gateway_app(
    *,
    supervisor: BootstrapSupervisor,
    gateway: BootstrapGateway,
    recovery_token: str,
    ingress_token: str,
    managed_evolution: ManagedEvolutionRuntime | None = None,
    evolution_token: str | None = None,
    sandbox_execution: TenantSandboxExecutionService | None = None,
    sandbox_token: str | None = None,
    capability_workers: StableCapabilityWorkerService | None = None,
    capability_worker_token: str | None = None,
) -> FastAPI:
    """Create the stable public process; mutable release routes are last-resort proxied."""

    durable_ingress_token = str(ingress_token or "").strip()
    if len(durable_ingress_token) < 32:
        raise ValueError("ingress token must contain at least 32 characters")
    recovery = RecoveryService(supervisor)
    if sandbox_execution is not None or capability_workers is not None:

        async def reconcile_stable_lease_authorities(lease: Any) -> None:
            errors: list[Exception] = []
            if sandbox_execution is not None:
                try:
                    await sandbox_execution.reconcile_lease(
                        SandboxExecutionLease(
                            release_id=lease.release_id,
                            lease_epoch=lease.epoch,
                        )
                        if lease is not None
                        else None
                    )
                except Exception as exc:
                    errors.append(exc)
            if capability_workers is not None:
                try:
                    await capability_workers.reconcile_lease(
                        CapabilityWorkerLease(
                            release_id=lease.release_id,
                            lease_epoch=lease.epoch,
                        )
                        if lease is not None
                        else None
                    )
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise RuntimeError("stable lease authority reconciliation failed") from errors[0]

        supervisor.set_lease_change_hook(reconcile_stable_lease_authorities)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await supervisor.start()
            if managed_evolution is not None:
                await managed_evolution.start()
            await gateway.start()
            yield
        finally:
            try:
                await recovery.shutdown()
            finally:
                try:
                    if managed_evolution is not None:
                        await managed_evolution.shutdown()
                finally:
                    try:
                        await gateway.shutdown()
                    finally:
                        try:
                            if capability_workers is not None:
                                await capability_workers.aclose()
                        finally:
                            close = getattr(supervisor.host, "aclose", None)
                            if callable(close):
                                await close()

    app = FastAPI(
        title="OpenTulpa Bootstrap",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.include_router(create_recovery_router(recovery, recovery_token=recovery_token))
    if managed_evolution is not None:
        register_evolution_control_api(
            app,
            service=managed_evolution.service,
            token=str(evolution_token or ""),
        )
    if sandbox_execution is not None or capability_workers is not None:

        async def authorize_release_lease(
            release_id: str,
            lease_epoch: int,
            control_token: str,
        ) -> None:
            try:
                state = supervisor.store.get_state()
                if (
                    state.safe_mode
                    or state.serving_release_id != release_id
                    or state.active_lease_epoch != lease_epoch
                ):
                    raise LeaseFenceError("release is not the active lease holder")
                supervisor.store.assert_active_lease(release_id, lease_epoch)
                running = await supervisor.host.discover(release_id, mode="production")
                if (
                    running is None
                    or running.control_token is None
                    or not hmac.compare_digest(running.control_token, control_token)
                ):
                    raise LeaseFenceError("release control credential is invalid")
            except LeaseFenceError as exc:
                raise HTTPException(
                    status_code=401,
                    detail="active release credentials are required",
                ) from exc

    if sandbox_execution is not None:
        register_sandbox_execution_api(
            app,
            service=sandbox_execution,
            token=str(sandbox_token or ""),
            authorize_lease=authorize_release_lease,
        )
    if capability_workers is not None:
        register_capability_worker_api(
            app,
            service=capability_workers,
            token=str(capability_worker_token or ""),
            authorize_lease=authorize_release_lease,
        )

    def authorize_ingress(value: str | None) -> None:
        if not hmac.compare_digest(str(value or ""), durable_ingress_token):
            raise HTTPException(status_code=401, detail="valid ingress credentials are required")

    async def read_ingress_body(request: Request) -> GatewayIngressRequest:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
            if declared_length < 0 or declared_length > gateway.max_ingress_bytes:
                raise HTTPException(status_code=413, detail="ingress body is too large")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > gateway.max_ingress_bytes:
                raise HTTPException(status_code=413, detail="ingress body is too large")
            body.extend(chunk)
        try:
            return GatewayIngressRequest.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail="invalid ingress body") from exc

    @app.get("/bootstrap/v1/live")
    async def bootstrap_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/bootstrap/v1/ingress/{channel}", status_code=status.HTTP_202_ACCEPTED)
    async def accept_ingress(
        channel: Annotated[str, ApiPath(pattern=r"^[a-z][a-z0-9_.-]{0,49}$")],
        request: Request,
        idempotency_key: Annotated[
            str | None,
            Header(alias="Idempotency-Key", max_length=300),
        ] = None,
        x_ingress_token: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Ingress-Token", max_length=500),
        ] = None,
    ) -> dict[str, Any]:
        authorize_ingress(x_ingress_token)
        body = await read_ingress_body(request)
        key = str(idempotency_key or "").strip()
        if not key:
            raise HTTPException(status_code=400, detail="Idempotency-Key is required")
        try:
            envelope = gateway.accept(
                IngressEnvelope(
                    tenant_id=body.tenant_id,
                    thread_id=body.thread_id,
                    channel=channel,
                    idempotency_key=key,
                    payload=body.payload,
                )
            )
        except BootstrapConflictError as exc:
            raise HTTPException(status_code=409, detail="idempotency key conflict") from exc
        except ValueError as exc:
            raise HTTPException(status_code=413, detail="ingress body is too large") from exc
        return {
            "ingress_id": envelope.id,
            "status": envelope.status,
            "created_at": envelope.created_at,
        }

    @app.get("/bootstrap/v1/ingress/{ingress_id}")
    async def ingress_status(
        ingress_id: Annotated[str, ApiPath(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")],
        x_ingress_token: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Ingress-Token", max_length=500),
        ] = None,
    ) -> dict[str, Any]:
        authorize_ingress(x_ingress_token)
        envelope = supervisor.store.get_ingress(ingress_id)
        if envelope is None:
            raise HTTPException(status_code=404, detail="ingress was not found")
        return envelope.model_dump(mode="json")

    @app.api_route(
        "/bootstrap",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    @app.api_route(
        "/bootstrap/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def reject_unknown_bootstrap_route(path: str = "") -> None:
        del path
        raise HTTPException(status_code=404, detail="bootstrap route was not found")

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        response_class=StreamingResponse,
    )
    async def proxy_release(path: str, request: Request) -> StreamingResponse:
        del path
        return await gateway.proxy(request, forbidden_bearer_tokens=(recovery_token,))

    app.state.bootstrap_supervisor = supervisor
    app.state.bootstrap_gateway = gateway
    app.state.recovery_service = recovery
    app.state.managed_evolution = managed_evolution
    app.state.capability_workers = capability_workers
    return app


def build_standalone_gateway(
    *,
    db_path: Path,
    recovery_token: str,
    ingress_token: str,
    host_policy: OciReleasePolicy | None = None,
    supervisor_policy: SupervisorPolicy | None = None,
    evolution_factory: Callable[[BootstrapSupervisor], ManagedEvolutionRuntime | None]
    | None = None,
    evolution_token: str | None = None,
    sandbox_execution: TenantSandboxExecutionService | None = None,
    sandbox_token: str | None = None,
    capability_worker_factory: Callable[[BootstrapStore], StableCapabilityWorkerService]
    | None = None,
    capability_worker_token: str | None = None,
) -> FastAPI:
    store = BootstrapStore(db_path)
    host = RootlessOciReleaseHost(policy=host_policy, release_loader=store.get_release)
    transport = ActiveReleaseTransport(store=store, host=host)
    supervisor = BootstrapSupervisor(
        store=store,
        host=host,
        policy=supervisor_policy,
        outbox_sink=transport,
    )
    gateway = BootstrapGateway(store=store, transport=transport)
    managed_evolution = evolution_factory(supervisor) if evolution_factory is not None else None
    capability_workers = (
        capability_worker_factory(store)
        if capability_worker_factory is not None
        else None
    )
    return create_gateway_app(
        supervisor=supervisor,
        gateway=gateway,
        recovery_token=recovery_token,
        ingress_token=ingress_token,
        managed_evolution=managed_evolution,
        evolution_token=evolution_token,
        sandbox_execution=sandbox_execution,
        sandbox_token=sandbox_token,
        capability_workers=capability_workers,
        capability_worker_token=capability_worker_token,
    )


def _load_or_create_token(path: Path) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise RuntimeError("bootstrap token directory cannot be a symlink")
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(secrets.token_urlsafe(48))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise RuntimeError("bootstrap token must be a private regular file")
    token = path.read_text(encoding="ascii").strip()
    if len(token) < 32 or any(character.isspace() for character in token):
        raise RuntimeError("bootstrap token is invalid")
    return token


def _production_environment(
    settings: Settings,
    *,
    evolution_url: str,
    evolution_token: str,
    sandbox_url: str = "",
    sandbox_token: str = "",
    capability_worker_url: str = "",
    capability_worker_token: str = "",
    internal_agent_api_url: str = "",
) -> tuple[tuple[str, str], ...]:
    """Select the only host settings allowed to cross into a production release."""

    values: dict[str, str | None] = {
        "OPENAI_COMPATIBLE_API_KEY": settings.openai_compatible_api_key,
        "OPENAI_COMPATIBLE_BASE_URL": settings.openai_compatible_base_url,
        "LLM_MODEL": settings.llm_model,
        "LLM_FALLBACK_MODELS": json.dumps(
            settings.llm_fallback_models,
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "LLM_PROVIDER_ORDER": json.dumps(
            settings.llm_provider_order,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "LLM_REASONING_EFFORT": settings.llm_reasoning_effort,
        "MODEL_ALIASES": json.dumps(
            settings.model_aliases,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "BUSINESS_KNOWLEDGE_ORACLE_MODEL": settings.business_knowledge_oracle_model,
        "OPENTULPA_OWNER_TOKEN": settings.opentulpa_owner_token,
        "OPENTULPA_OWNER_CUSTOMER_ID": settings.opentulpa_owner_customer_id,
        "TELEGRAM_BOT_TOKEN": settings.telegram_bot_token,
        "TELEGRAM_WEBHOOK_SECRET": settings.telegram_webhook_secret,
        "TELEGRAM_ALLOWED_USERNAMES": settings.telegram_allowed_usernames,
        "TELEGRAM_ALLOWED_USER_IDS": settings.telegram_allowed_user_ids,
        "COMPOSIO_API_KEY": settings.composio_api_key,
        "COMPOSIO_DEFAULT_CALLBACK_URL": settings.composio_default_callback_url,
        "BROWSER_USE_API_KEY": settings.browser_use_api_key,
        "LANGFUSE_PUBLIC_KEY": settings.langfuse_public_key,
        "LANGFUSE_SECRET_KEY": settings.langfuse_secret_key,
        "LANGFUSE_BASE_URL": settings.langfuse_base_url,
        "LANGFUSE_DEPLOYMENT_TAG": settings.langfuse_deployment_tag,
        "LANGFUSE_TRACING_ENVIRONMENT": settings.langfuse_environment,
        "LANGFUSE_CONTENT_LEVEL": settings.langfuse_content_level,
        "PUBLIC_BASE_URL": os.environ.get("PUBLIC_BASE_URL"),
        "RAILWAY_PUBLIC_DOMAIN": os.environ.get("RAILWAY_PUBLIC_DOMAIN"),
        "EXA_API_KEY": os.environ.get("EXA_API_KEY"),
        "OPENROUTER_WEB_SEARCH_MODEL": os.environ.get("OPENROUTER_WEB_SEARCH_MODEL"),
        "OPENTULPA_BOOTSTRAP_EVOLUTION_URL": evolution_url,
        "OPENTULPA_BOOTSTRAP_EVOLUTION_TOKEN": evolution_token,
        "OPENTULPA_BOOTSTRAP_SANDBOX_URL": sandbox_url,
        "OPENTULPA_BOOTSTRAP_SANDBOX_TOKEN": sandbox_token,
        "OPENTULPA_BOOTSTRAP_CAPABILITY_WORKER_URL": capability_worker_url,
        "OPENTULPA_BOOTSTRAP_CAPABILITY_WORKER_TOKEN": capability_worker_token,
        "OPENTULPA_INTERNAL_AGENT_API_URL": internal_agent_api_url,
        "EVOLUTION_ENABLED": "true" if settings.evolution_enabled else "false",
    }
    return tuple(
        sorted(
            (name, str(value))
            for name, value in values.items()
            if value is not None and str(value).strip()
        )
    )


def _validate_release_workspace(
    workspace: Path,
    *,
    state_root: Path,
    source_root: Path,
) -> Path:
    """Keep the mutable release mount disjoint from stable state and source."""

    configured = workspace.expanduser()
    if configured.is_symlink():
        raise RuntimeError("release workspace cannot be a symlink")
    workspace = configured.resolve()
    state_root = state_root.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    protected = (state_root, source_root)
    if workspace.is_symlink() or any(
        workspace == root
        or workspace.is_relative_to(root)
        or root.is_relative_to(workspace)
        for root in protected
    ):
        raise RuntimeError("release workspace cannot expose bootstrap state or source metadata")
    return workspace


def main() -> None:
    recovery_token = str(os.environ.get("OPENTULPA_RECOVERY_TOKEN") or "").strip()
    ingress_token = str(os.environ.get("OPENTULPA_INGRESS_TOKEN") or "").strip()
    settings = get_settings()
    state_root = Path(
        os.environ.get(
            "OPENTULPA_BOOTSTRAP_STATE_ROOT",
            ".opentulpa/bootstrap",
        )
    ).expanduser().resolve()
    state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    evolution_token = _load_or_create_token(state_root / "evolution-api.token")
    sandbox_token = _load_or_create_token(state_root / "sandbox-api.token")
    capability_worker_token = _load_or_create_token(
        state_root / "capability-worker-api.token"
    )
    port = int(os.environ.get("PORT", "8000"))
    gateway_host = str(os.environ.get("OPENTULPA_RELEASE_HOST_GATEWAY") or "host.docker.internal")
    evolution_url = f"http://{gateway_host}:{port}/bootstrap/internal/v1/evolution"
    sandbox_url = f"http://{gateway_host}:{port}/bootstrap/internal/v1/sandbox"
    capability_worker_url = (
        f"http://{gateway_host}:{port}/bootstrap/internal/v1/capability-workers"
    )
    internal_agent_api_url = f"http://{gateway_host}:{port}"
    source_root = Path(__file__).resolve().parents[3]
    workspace = Path(
        os.environ.get("OPENTULPA_RELEASE_WORKSPACE")
        or state_root.parent / "release-data"
    )
    workspace = _validate_release_workspace(
        workspace,
        state_root=state_root,
        source_root=source_root,
    )
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    production_network = str(
        os.environ.get("OPENTULPA_RELEASE_EGRESS_NETWORK") or ""
    ).strip()
    if not production_network:
        raise RuntimeError("OPENTULPA_RELEASE_EGRESS_NETWORK must be configured explicitly")
    runtime_base_image = str(
        os.environ.get("OPENTULPA_RELEASE_BASE_IMAGE") or ""
    ).strip()
    if settings.evolution_enabled and not runtime_base_image:
        raise RuntimeError("OPENTULPA_RELEASE_BASE_IMAGE is required for managed evolution")
    container_cli = os.environ.get("OPENTULPA_CONTAINER_CLI", settings.sandbox_container_cli)
    sandbox_image = resolve_local_oci_image(
        container_cli=container_cli,
        image=settings.sandbox_image,
        cwd=source_root,
    )
    configured_workspaces = Path(settings.deepagents_workspaces_root)
    if configured_workspaces.is_absolute() or ".." in configured_workspaces.parts:
        raise RuntimeError("managed tenant workspaces must be relative to release storage")
    tenant_workspaces = (workspace / configured_workspaces).resolve()
    if not tenant_workspaces.is_relative_to(workspace):
        raise RuntimeError("managed tenant workspaces escaped release storage")
    sandbox_execution = TenantSandboxExecutionService(
        workspaces_root=tenant_workspaces,
        allowed_root=workspace,
        policy=TenantContainerPolicy(
            image=sandbox_image,
            cpu_limit=settings.sandbox_cpu_limit,
            memory_limit=settings.sandbox_memory_limit,
            pid_limit=settings.sandbox_pid_limit,
            timeout_seconds=settings.sandbox_timeout_seconds,
            max_output_bytes=settings.sandbox_max_output_bytes,
            network_enabled=True,
        ),
        container_cli=container_cli,
    )
    app = build_standalone_gateway(
        db_path=Path(
            os.environ.get(
                "OPENTULPA_BOOTSTRAP_DB",
                ".opentulpa/bootstrap/control.db",
            )
        ),
        recovery_token=recovery_token,
        ingress_token=ingress_token,
        host_policy=OciReleasePolicy(
            container_cli=container_cli,
            state_root=state_root,
            production_network_name=production_network,
            host_gateway_name=gateway_host,
            runtime_user=f"{os.getuid()}:{os.getgid()}",
            require_persistent_data_mount=True,
            production_environment=_production_environment(
                settings,
                evolution_url=evolution_url,
                evolution_token=evolution_token,
                sandbox_url=sandbox_url,
                sandbox_token=sandbox_token,
                capability_worker_url=capability_worker_url,
                capability_worker_token=capability_worker_token,
                internal_agent_api_url=internal_agent_api_url,
            ),
            mounts=(OciMount(source=workspace, target="/workspace", read_only=False),),
            allowed_mount_roots=(workspace,),
        ),
        evolution_factory=(
            (
                lambda supervisor: build_managed_evolution_runtime(
                    bootstrap=supervisor,
                    project_root=source_root,
                    state_root=state_root,
                    settings=settings,
                    runtime_base_image=runtime_base_image,
                )
            )
            if settings.evolution_enabled
            else None
        ),
        evolution_token=evolution_token,
        sandbox_execution=sandbox_execution,
        sandbox_token=sandbox_token,
        capability_worker_factory=lambda store: StableCapabilityWorkerService(
            host=OciCapabilityWorkerHost(
                policy=OciCapabilityPolicy(
                    container_cli=container_cli,
                    state_root=state_root / "capability-workers" / "runtime",
                    restricted_egress_network=production_network,
                    restricted_allowed_hosts=tuple(
                        sorted(
                            {
                                "api.telegram.org:443",
                                *(
                                    value.strip()
                                    for value in str(
                                        os.environ.get(
                                            "OPENTULPA_CAPABILITY_ALLOWED_HOSTS",
                                            "",
                                        )
                                    ).split(",")
                                    if value.strip()
                                ),
                            }
                        )
                    ),
                    persistent_state_root=(
                        state_root / "capability-workers" / "tenant-state"
                    ),
                    runtime_user=f"{os.getuid()}:{os.getgid()}",
                    host_gateway_name=gateway_host,
                )
            ),
            release_loader=store.get_release,
            state_path=state_root / "capability-workers" / "workers.json",
        ),
        capability_worker_token=capability_worker_token,
    )
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=port,
        ws="none",
        log_level="info",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ActiveReleaseTransport",
    "BootstrapGateway",
    "GatewayIngressRequest",
    "GatewayUnavailableError",
    "build_standalone_gateway",
    "create_gateway_app",
    "main",
]
