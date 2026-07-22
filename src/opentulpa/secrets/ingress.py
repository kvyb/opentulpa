"""Credential ingress that replaces pasted plaintext before agent persistence."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from pydantic import SecretStr

from opentulpa.secrets.models import SecretHandle, SecretState
from opentulpa.secrets.service import SecretVaultService


class SecretIngressHook(Protocol):
    """Optional pre-checkpoint hook for authenticated message interfaces."""

    def __call__(self, *, tenant_id: str, actor_id: str, text: str) -> str: ...


@dataclass(frozen=True, slots=True)
class SecretIngressResult:
    """Sanitized message text plus public-safe handles created or rotated."""

    text: str
    handles: tuple[SecretHandle, ...]


@dataclass(frozen=True, slots=True)
class _CredentialPattern:
    kind: str
    handle_id: str
    name: str
    scopes: tuple[str, ...]
    expression: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class _Match:
    start: int
    end: int
    plaintext: str
    pattern: _CredentialPattern


_PATTERNS: tuple[_CredentialPattern, ...] = (
    _CredentialPattern(
        kind="telegram",
        handle_id="telegram_bot_token",
        name="telegram_bot_token",
        scopes=("telegram.receive", "telegram.send"),
        expression=re.compile(
            r"(?<![A-Za-z0-9_-])(?P<value>[1-9][0-9]{5,14}:[A-Za-z0-9_-]{30,64})"
            r"(?![A-Za-z0-9_-])"
        ),
    ),
    _CredentialPattern(
        kind="api",
        handle_id="api_token",
        name="api_token",
        scopes=("api.invoke",),
        expression=re.compile(
            r"(?<![A-Za-z0-9_-])(?P<value>sk-[A-Za-z0-9][A-Za-z0-9_-]{15,199})"
            r"(?![A-Za-z0-9_-])"
        ),
    ),
)


class SecretIngressService:
    """Encrypt detected credentials and expose only ``secret://`` references."""

    def __init__(
        self,
        vault: SecretVaultService,
        *,
        patterns: Sequence[_CredentialPattern] = _PATTERNS,
    ) -> None:
        self._vault = vault
        self._patterns = tuple(patterns)
        self._lock = Lock()

    def __call__(self, *, tenant_id: str, actor_id: str, text: str) -> str:
        """Implement ``SecretIngressHook`` for direct V2 route composition."""

        return self.ingest(tenant_id=tenant_id, actor_id=actor_id, text=text).text

    def ingest(self, *, tenant_id: str, actor_id: str, text: str) -> SecretIngressResult:
        """Store all recognized values before returning sanitized message text."""

        matches = self._matches(text)
        if not matches:
            return SecretIngressResult(text=text, handles=())

        replacements: list[tuple[int, int, str]] = []
        handles: list[SecretHandle] = []
        ordinals: defaultdict[str, int] = defaultdict(int)
        with self._lock:
            for match in matches:
                ordinals[match.pattern.kind] += 1
                ordinal = ordinals[match.pattern.kind]
                base_id = match.pattern.handle_id
                preferred_id = base_id if ordinal == 1 else f"{base_id}_{ordinal}"
                handle = self._store_value(
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    preferred_id=preferred_id,
                    name=match.pattern.name,
                    scopes=match.pattern.scopes,
                    plaintext=match.plaintext,
                )
                handles.append(handle)
                replacements.append((match.start, match.end, f"secret://{handle.id}"))

        sanitized = text
        for start, end, reference in reversed(replacements):
            sanitized = f"{sanitized[:start]}{reference}{sanitized[end:]}"
        return SecretIngressResult(text=sanitized, handles=tuple(handles))

    def _store_value(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        preferred_id: str,
        name: str,
        scopes: tuple[str, ...],
        plaintext: str,
    ) -> SecretHandle:
        secret_id = preferred_id
        suffix = 2
        handle = self._vault.get(tenant_id=tenant_id, secret_id=secret_id)
        while handle is not None and handle.state is SecretState.REVOKED:
            secret_id = f"{preferred_id}_replacement_{suffix}"
            suffix += 1
            handle = self._vault.get(tenant_id=tenant_id, secret_id=secret_id)
        if handle is None:
            handle = self._vault.create_pending(
                tenant_id=tenant_id,
                actor_id=actor_id,
                secret_id=secret_id,
                name=name,
                scopes=scopes,
            )
        return self._vault.store(
            tenant_id=tenant_id,
            actor_id=actor_id,
            secret_id=handle.id,
            expected_revision=handle.revision,
            value=SecretStr(plaintext),
        )

    def _matches(self, text: str) -> list[_Match]:
        matches = [
            _Match(
                start=match.start("value"),
                end=match.end("value"),
                plaintext=match.group("value"),
                pattern=pattern,
            )
            for pattern in self._patterns
            for match in pattern.expression.finditer(text)
        ]
        matches.sort(key=lambda item: (item.start, item.end))
        accepted: list[_Match] = []
        previous_end = -1
        for match in matches:
            if match.start < previous_end:
                continue
            accepted.append(match)
            previous_end = match.end
        return accepted


__all__ = [
    "SecretIngressHook",
    "SecretIngressResult",
    "SecretIngressService",
]
