"""Shared authenticated principal boundary for tenant-scoped v2 routes."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from fastapi import HTTPException, Request
from pydantic import ValidationError

from opentulpa.specs import AgentRunBinding


class V2Principal(Protocol):
    tenant_id: str
    actor_id: str
    interface: str
    source_id: str
    channel: str
    trust_class: str
    scopes: frozenset[str]
    agent_binding: AgentRunBinding | None
    conversation_id: str | None
    message_id: str | None


@dataclass(frozen=True, slots=True)
class ResolvedV2Principal:
    """Validated identity, routing, and authorization metadata for one request."""

    tenant_id: str
    actor_id: str
    interface: str
    source_id: str
    channel: str
    trust_class: str
    scopes: frozenset[str]
    agent_binding: AgentRunBinding | None = None
    conversation_id: str | None = None
    message_id: str | None = None


async def resolve_v2_principal(
    request: Request,
    resolver: Callable[[Request], V2Principal | Awaitable[V2Principal]],
) -> ResolvedV2Principal:
    resolved = resolver(request)
    principal = await resolved if inspect.isawaitable(resolved) else resolved
    tenant_id = _identity(getattr(principal, "tenant_id", ""))
    actor_id = _identity(getattr(principal, "actor_id", ""))
    if not tenant_id:
        raise HTTPException(status_code=401, detail="authenticated tenant is required")
    if not actor_id:
        raise HTTPException(status_code=401, detail="authenticated actor is required")
    trust_class = str(getattr(principal, "trust_class", "owner") or "owner").strip()
    if trust_class not in {"owner", "background", "external"}:
        raise HTTPException(status_code=401, detail="invalid authenticated trust class")
    interface = _slug(getattr(principal, "interface", "web"), default="web")
    source_id = _identity(getattr(principal, "source_id", "owner-web")) or "owner-web"
    channel = _slug(getattr(principal, "channel", interface), default=interface)
    raw_scopes = getattr(principal, "scopes", None)
    scopes = (
        frozenset({"*"})
        if raw_scopes is None and trust_class == "owner"
        else frozenset(str(value or "").strip() for value in (raw_scopes or ()))
    )
    if "" in scopes:
        raise HTTPException(status_code=401, detail="invalid authenticated scopes")
    raw_binding = getattr(principal, "agent_binding", None)
    try:
        agent_binding = (
            AgentRunBinding.model_validate(raw_binding) if raw_binding is not None else None
        )
    except ValidationError as exc:
        raise HTTPException(status_code=401, detail="invalid authenticated agent binding") from exc
    if agent_binding is not None and (
        agent_binding.agent_spec.tenant_id != tenant_id
        or agent_binding.trust_class != trust_class
    ):
        raise HTTPException(status_code=401, detail="invalid authenticated agent binding")
    return ResolvedV2Principal(
        tenant_id=tenant_id,
        actor_id=actor_id,
        interface=interface,
        source_id=source_id,
        channel=channel,
        trust_class=trust_class,
        scopes=scopes,
        agent_binding=agent_binding,
        conversation_id=_optional_identity(getattr(principal, "conversation_id", None)),
        message_id=_optional_identity(getattr(principal, "message_id", None)),
    )


def require_v2_scope(principal: ResolvedV2Principal, scope: str) -> None:
    """Fail closed unless the authenticated principal received this operation."""

    required = str(scope or "").strip()
    if not required or ("*" not in principal.scopes and required not in principal.scopes):
        raise HTTPException(status_code=403, detail="credential scope does not allow this operation")


def _identity(value: object) -> str:
    safe = str(value or "").strip()
    if len(safe) > 200 or any(ord(char) < 32 for char in safe):
        raise HTTPException(status_code=401, detail="invalid authenticated identity")
    return safe


def _optional_identity(value: object) -> str | None:
    safe = _identity(value)
    return safe or None


def _slug(value: object, *, default: str) -> str:
    safe = str(value or default).strip()
    if (
        not safe
        or len(safe) > 64
        or not safe[0].islower()
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in safe)
    ):
        raise HTTPException(status_code=401, detail="invalid authenticated interface")
    return safe


__all__ = [
    "ResolvedV2Principal",
    "V2Principal",
    "require_v2_scope",
    "resolve_v2_principal",
]
