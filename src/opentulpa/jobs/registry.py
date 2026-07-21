"""Closed registry of typed product background handlers."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from opentulpa.jobs.models import JobArguments, JobHandlerResult

_HANDLER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_FORBIDDEN_HANDLER_TOKENS = {
    "command",
    "relaunch",
    "shell",
    "sourceedit",
    "sourceediting",
    "terminal",
}
_FORBIDDEN_ARGUMENT_FIELDS = {
    "command",
    "relaunch",
    "repository_path",
    "shell",
    "source_path",
    "working_dir",
}


class JobHandlerRegistrationError(ValueError):
    """A handler violates the closed deterministic job contract."""


class JobHandlerNotFoundError(KeyError):
    """The persisted handler name and version are not registered."""


@dataclass(frozen=True, slots=True)
class JobExecutionContext:
    """Host-owned execution metadata hidden from persisted handler arguments."""

    tenant_id: str
    job_id: str
    idempotency_key: str
    attempt: int
    _emit_progress: Callable[[dict[str, Any]], Awaitable[None]]

    async def progress(self, payload: dict[str, Any]) -> None:
        await self._emit_progress(payload)


type JobHandler[ArgumentsT: JobArguments] = Callable[
    [ArgumentsT, JobExecutionContext], Awaitable[JobHandlerResult]
]


@dataclass(frozen=True, slots=True)
class RegisteredJobHandler[ArgumentsT: JobArguments]:
    name: str
    version: int
    arguments_model: type[ArgumentsT]
    handler: JobHandler[ArgumentsT]
    timeout_seconds: float

    def parse_arguments(self, raw: dict[str, Any]) -> ArgumentsT:
        return self.arguments_model.model_validate(raw)


class JobHandlerRegistry:
    """Resolve only explicitly registered, versioned application handlers."""

    def __init__(self) -> None:
        self._handlers: dict[tuple[str, int], RegisteredJobHandler[Any]] = {}
        self._latest_versions: dict[str, int] = {}

    def register[ArgumentsT: JobArguments](
        self,
        *,
        name: str,
        arguments_model: type[ArgumentsT],
        handler: JobHandler[ArgumentsT],
        version: int = 1,
        timeout_seconds: float = 300,
    ) -> RegisteredJobHandler[ArgumentsT]:
        safe_name = str(name or "").strip().lower()
        if not _HANDLER_NAME_RE.fullmatch(safe_name):
            raise JobHandlerRegistrationError("handler name must match ^[a-z][a-z0-9_]{0,99}$")
        collapsed_name = safe_name.replace("_", "")
        if any(token in collapsed_name for token in _FORBIDDEN_HANDLER_TOKENS):
            raise JobHandlerRegistrationError(
                "shell, terminal, source editing, command, and relaunch handlers are forbidden"
            )
        if not isinstance(arguments_model, type) or not issubclass(
            arguments_model,
            JobArguments,
        ):
            raise JobHandlerRegistrationError("arguments_model must extend JobArguments")
        forbidden_fields = sorted(
            set(arguments_model.model_fields).intersection(_FORBIDDEN_ARGUMENT_FIELDS)
        )
        if forbidden_fields:
            raise JobHandlerRegistrationError(
                f"handler arguments expose forbidden fields: {', '.join(forbidden_fields)}"
            )
        if not inspect.iscoroutinefunction(handler):
            raise JobHandlerRegistrationError("registered job handler must be async")
        if version < 1:
            raise JobHandlerRegistrationError("handler version must be at least 1")
        if not 0 < float(timeout_seconds) <= 86_400:
            raise JobHandlerRegistrationError("handler timeout_seconds must be between 0 and 86400")
        key = (safe_name, int(version))
        if key in self._handlers:
            raise JobHandlerRegistrationError(
                f"handler {safe_name!r} version {version} is already registered"
            )
        registered = RegisteredJobHandler(
            name=safe_name,
            version=int(version),
            arguments_model=arguments_model,
            handler=handler,
            timeout_seconds=float(timeout_seconds),
        )
        self._handlers[key] = cast(RegisteredJobHandler[Any], registered)
        self._latest_versions[safe_name] = max(
            int(version),
            self._latest_versions.get(safe_name, 0),
        )
        return registered

    def get(
        self,
        name: str,
        version: int | None = None,
    ) -> RegisteredJobHandler[Any]:
        safe_name = str(name or "").strip().lower()
        resolved_version = version or self._latest_versions.get(safe_name)
        if resolved_version is None:
            raise JobHandlerNotFoundError(safe_name)
        try:
            return self._handlers[(safe_name, int(resolved_version))]
        except KeyError as exc:
            raise JobHandlerNotFoundError(f"{safe_name}@{resolved_version}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._latest_versions))


__all__ = [
    "JobExecutionContext",
    "JobHandler",
    "JobHandlerNotFoundError",
    "JobHandlerRegistrationError",
    "JobHandlerRegistry",
    "RegisteredJobHandler",
]
