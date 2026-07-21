"""Authentication boundary for the single-owner public v2 API."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hmac import compare_digest
from typing import Protocol

from fastapi import HTTPException, Request

from opentulpa.api.web_auth import bearer_token, owner_session_token
from opentulpa.capabilities.credentials import (
    CAPABILITY_CREDENTIAL_PREFIX,
    CapabilityAPIScope,
    CapabilityCredential,
)
from opentulpa.specs import AgentRunBinding


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    tenant_id: str
    actor_id: str
    interface: str = "web"
    source_id: str = "owner-web"
    channel: str = "web"
    trust_class: str = "owner"
    scopes: frozenset[str] = frozenset({"*"})
    agent_binding: AgentRunBinding | None = None
    conversation_id: str | None = None
    message_id: str | None = None


class CapabilityCredentialAuthenticator(Protocol):
    def authenticate(self, token: str) -> CapabilityCredential | None: ...


_CAPABILITY_ROUTE_SCOPES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "POST",
        re.compile(r"^/v2/agent/runs$"),
        CapabilityAPIScope.AGENT_RUN_SUBMIT.value,
    ),
    (
        "GET",
        re.compile(r"^/v2/agent/runs/[^/]+$"),
        CapabilityAPIScope.AGENT_RUN_REPLAY.value,
    ),
    (
        "GET",
        re.compile(r"^/v2/agent/runs/[^/]+/events$"),
        CapabilityAPIScope.AGENT_RUN_REPLAY.value,
    ),
    (
        "POST",
        re.compile(r"^/v2/agent/runs/[^/]+/resume$"),
        CapabilityAPIScope.AGENT_RUN_RESUME.value,
    ),
    (
        "POST",
        re.compile(r"^/v2/files$"),
        CapabilityAPIScope.FILE_UPLOAD.value,
    ),
    (
        "GET",
        re.compile(r"^/v2/notifications$"),
        CapabilityAPIScope.NOTIFICATIONS_READ.value,
    ),
    (
        "POST",
        re.compile(r"^/v2/notifications/[1-9][0-9]*/ack$"),
        CapabilityAPIScope.NOTIFICATIONS_ACK.value,
    ),
)


class OwnerPrincipalResolver:
    """Resolve a fixed deployment owner without accepting caller-owned tenant fields."""

    def __init__(
        self,
        *,
        token: str | None,
        tenant_id: str,
        actor_id: str = "web-owner",
        local_cookie_token: str | None = None,
    ) -> None:
        self._token = str(token or "").strip()
        self._tenant_id = str(tenant_id or "").strip()
        self._actor_id = str(actor_id or "").strip()
        self._local_cookie_token = str(local_cookie_token or "").strip()

    def __call__(self, request: Request) -> AuthenticatedPrincipal:
        if not self._token:
            raise HTTPException(status_code=503, detail="OPENTULPA_WEB_TOKEN is not configured")
        supplied = bearer_token(request)
        authorized = bool(supplied and compare_digest(supplied, self._token))
        if not supplied and self._local_cookie_token:
            session = owner_session_token(request, enabled=True)
            authorized = bool(
                session and compare_digest(session, self._local_cookie_token)
            )
        if not authorized:
            raise HTTPException(status_code=401, detail="unauthorized")
        if not self._tenant_id:
            raise HTTPException(status_code=503, detail="deployment owner tenant is not configured")
        return AuthenticatedPrincipal(
            tenant_id=self._tenant_id,
            actor_id=self._actor_id or "web-owner",
        )


class CapabilityPrincipalResolver:
    """Resolve a capability bearer and enforce its route scope before dispatch."""

    def __init__(self, credentials: CapabilityCredentialAuthenticator) -> None:
        self._credentials = credentials

    def __call__(self, request: Request) -> AuthenticatedPrincipal:
        supplied = bearer_token(request)
        credential = self._credentials.authenticate(supplied)
        if credential is None:
            raise HTTPException(status_code=401, detail="unauthorized")
        required_scope = _required_capability_scope(request.method, request.url.path)
        if required_scope is None or required_scope not in credential.scopes:
            raise HTTPException(
                status_code=403,
                detail="credential scope does not allow this operation",
            )
        return AuthenticatedPrincipal(
            tenant_id=credential.tenant_id,
            actor_id=credential.actor_id,
            interface=credential.interface,
            source_id=credential.source_id,
            channel=credential.channel,
            trust_class=credential.trust_class,
            scopes=credential.scopes,
            agent_binding=credential.agent_binding,
            conversation_id=_origin_header(request, "x-opentulpa-origin-conversation-id"),
            message_id=_origin_header(request, "x-opentulpa-origin-message-id"),
        )


class OwnerOrCapabilityPrincipalResolver:
    """Keep the owner bearer while routing capability-prefixed tokens to scoped auth."""

    def __init__(
        self,
        *,
        owner: Callable[[Request], AuthenticatedPrincipal | Awaitable[AuthenticatedPrincipal]],
        capability: CapabilityPrincipalResolver,
    ) -> None:
        self._owner = owner
        self._capability = capability

    async def __call__(self, request: Request) -> AuthenticatedPrincipal:
        supplied = bearer_token(request)
        if supplied.startswith(CAPABILITY_CREDENTIAL_PREFIX):
            return self._capability(request)
        resolved = self._owner(request)
        return await resolved if inspect.isawaitable(resolved) else resolved


def _required_capability_scope(method: str, path: str) -> str | None:
    normalized_method = str(method or "").upper()
    normalized_path = str(path or "")
    return next(
        (
            scope
            for expected_method, pattern, scope in _CAPABILITY_ROUTE_SCOPES
            if normalized_method == expected_method and pattern.fullmatch(normalized_path)
        ),
        None,
    )


def _origin_header(request: Request, name: str) -> str | None:
    value = str(request.headers.get(name, "") or "").strip()
    if not value:
        return None
    if len(value) > 200 or any(ord(char) < 32 for char in value):
        raise HTTPException(status_code=400, detail="invalid interface origin metadata")
    return value


__all__ = [
    "AuthenticatedPrincipal",
    "CapabilityCredentialAuthenticator",
    "CapabilityPrincipalResolver",
    "OwnerOrCapabilityPrincipalResolver",
    "OwnerPrincipalResolver",
]
