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


_SECRET_TAG_BLOCK_RE = re.compile(
    r"""<secret(?P<attrs>(?:\s[^>]*)?)>"""
    r"(?P<value>.*?)"
    r"</secret\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_SECRET_LINE_BLOCK_RE = re.compile(
    r"""^[ \t]*<secret(?P<attrs>[^\n>]*)\r?\n"""
    r"(?P<value>.*?)"
    r"\r?\n[ \t]*</secret\s*>",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_SECRET_NAME_ATTR_RE = re.compile(
    r"""(?:^|\s)name\s*=\s*(?:"(?P<double>[A-Za-z][A-Za-z0-9_-]{0,63})"|"""
    r"""'(?P<single>[A-Za-z][A-Za-z0-9_-]{0,63})'|"""
    r"""(?P<bare>[A-Za-z][A-Za-z0-9_-]{0,63}))(?=\s|$)""",
    flags=re.IGNORECASE,
)
_SECRET_BARE_NAME_RE = re.compile(r"^\s*(?P<name>[A-Za-z][A-Za-z0-9_-]{0,63})\s*$")
_OPENSSH_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN OPENSSH PRIVATE KEY-----.*-----END OPENSSH PRIVATE KEY-----",
    flags=re.DOTALL,
)
_NAMED_SECRET_ASSIGNMENT_RE = re.compile(
    r"""(?<![A-Za-z0-9_])"""
    r"(?P<name>[A-Za-z][A-Za-z0-9_]{0,63}"
    r"(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL))"
    r"""\s*[:=]\s*"""
    r"""(?P<quote>["']?)(?P<value>[^\s"'`]+)(?P=quote)""",
    flags=re.IGNORECASE,
)
_NAMED_SECRET_SCOPES: dict[str, tuple[str, ...]] = {
    "composio_api_key": ("composio.manage", "composio.invoke"),
    "daytona_api_key": ("daytona.manage",),
    "github_token": ("github.read", "github.write"),
    "gh_token": ("github.read", "github.write"),
    "browser_use_api_key": ("browser.manage",),
    "ssh_key": ("ssh.connect",),
    "ssh_password": ("ssh.connect",),
    "ssh_private_key": ("ssh.connect",),
}
_PLACEHOLDER_VALUES = frozenset(
    {
        "changeme",
        "example",
        "none",
        "null",
        "replace-me",
        "replace_me",
        "secret",
        "token",
        "your-key",
        "your_key",
    }
)


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
        named_matches = self._named_matches(text)
        pattern_matches = [
            _Match(
                start=match.start("value"),
                end=match.end("value"),
                plaintext=match.group("value"),
                pattern=pattern,
            )
            for pattern in self._patterns
            for match in pattern.expression.finditer(text)
        ]
        # Explicitly named credentials win over generic token-shape detection.
        matches = named_matches + pattern_matches
        matches.sort(
            key=lambda item: (
                item.start,
                0 if item in named_matches else 1,
                -(item.end - item.start),
            )
        )
        accepted: list[_Match] = []
        previous_end = -1
        for match in matches:
            if match.start < previous_end:
                continue
            accepted.append(match)
            previous_end = match.end
        return accepted

    @staticmethod
    def _named_matches(text: str) -> list[_Match]:
        matches: list[_Match] = []
        occupied: list[tuple[int, int]] = []
        for expression in (_SECRET_TAG_BLOCK_RE, _SECRET_LINE_BLOCK_RE):
            for match in expression.finditer(text):
                start, end = match.span()
                if any(start < used_end and end > used_start for used_start, used_end in occupied):
                    continue
                plaintext = match.group("value").strip("\r\n")
                normalized_name = SecretIngressService._secret_block_name(
                    attrs=match.group("attrs"),
                    plaintext=plaintext,
                )
                if not normalized_name:
                    continue
                if not SecretIngressService._is_secret_value(
                    plaintext,
                    min_bytes=1 if normalized_name == "ssh_password" else 8,
                ):
                    continue
                occupied.append((start, end))
                matches.append(
                    SecretIngressService._match_for_secret_block(
                        expression=expression,
                        start=start,
                        end=end,
                        plaintext=plaintext,
                        normalized_name=normalized_name,
                    )
                )
        for match in _NAMED_SECRET_ASSIGNMENT_RE.finditer(text):
            start, end = match.span("value")
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            plaintext = match.group("value")
            normalized_name = SecretIngressService._normalize_name(match.group("name"))
            if not SecretIngressService._is_secret_value(
                plaintext,
                min_bytes=1 if normalized_name == "ssh_password" else 8,
            ):
                continue
            occupied.append((start, end))
            matches.append(
                SecretIngressService._match_for_secret_block(
                    expression=_NAMED_SECRET_ASSIGNMENT_RE,
                    start=start,
                    end=end,
                    plaintext=plaintext,
                    normalized_name=normalized_name,
                )
            )
        return matches

    @staticmethod
    def _match_for_secret_block(
        *,
        expression: re.Pattern[str],
        start: int,
        end: int,
        plaintext: str,
        normalized_name: str,
    ) -> _Match:
        return _Match(
            start=start,
            end=end,
            plaintext=plaintext,
            pattern=_CredentialPattern(
                kind=f"named:{normalized_name}",
                handle_id=normalized_name,
                name=normalized_name,
                scopes=_NAMED_SECRET_SCOPES.get(
                    normalized_name,
                    ("credential.use",),
                ),
                expression=expression,
            ),
        )

    @staticmethod
    def _secret_block_name(*, attrs: str, plaintext: str) -> str:
        match = _SECRET_NAME_ATTR_RE.search(attrs or "")
        if match is not None:
            return SecretIngressService._normalize_name(
                match.group("double") or match.group("single") or match.group("bare") or ""
            )
        bare = _SECRET_BARE_NAME_RE.fullmatch(attrs or "")
        if bare is not None:
            return SecretIngressService._normalize_name(bare.group("name"))
        return SecretIngressService._infer_secret_name(plaintext)

    @staticmethod
    def _infer_secret_name(plaintext: str) -> str:
        if _OPENSSH_PRIVATE_KEY_RE.search(str(plaintext or "")):
            return "ssh_private_key"
        return ""

    @staticmethod
    def _normalize_name(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9_-]+", "_", value.strip().lower())
        normalized = re.sub(r"[_-]{2,}", "_", normalized).strip("_-")
        if not normalized or not normalized[0].isalpha():
            raise ValueError("secret name is invalid")
        return normalized[:64]

    @staticmethod
    def _is_secret_value(value: str, *, min_bytes: int = 8) -> bool:
        clean = str(value or "").strip()
        return (
            min_bytes <= len(clean.encode("utf-8")) <= 1_048_576
            and clean.casefold() not in _PLACEHOLDER_VALUES
            and not clean.startswith("secret://")
        )


__all__ = [
    "SecretIngressHook",
    "SecretIngressResult",
    "SecretIngressService",
]
