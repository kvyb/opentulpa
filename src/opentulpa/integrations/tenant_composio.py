"""Tenant-owned Composio application adapter and background action handler."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar
from urllib.parse import urlsplit

from pydantic import Field, JsonValue

from opentulpa.jobs import (
    JobArguments,
    JobExecutionContext,
    JobHandlerRegistry,
    JobHandlerResult,
)
from opentulpa.persistence.idempotency import IdempotencyStore

_T = TypeVar("_T")
logger = logging.getLogger(__name__)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,299}$")
_SECRET_KEY_RE = re.compile(
    r"authorization|api[_-]?key|secret|token|password|passwd|cookie|credential",
    re.I,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(authorization|api[_-]?key|secret|token|password|passwd|cookie|credential)"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_INTAKE_ALLOWED_ACTIONS = frozenset({"UPSERT", "UPDATE"})
_INTAKE_FORBIDDEN_ACTIONS = frozenset(
    {
        "ADD",
        "APPEND",
        "APPROVE",
        "AUTHORIZE",
        "BUY",
        "CHARGE",
        "CREATE",
        "DELETE",
        "DISCONNECT",
        "EMAIL",
        "EXECUTE",
        "GRANT",
        "INVITE",
        "MESSAGE",
        "NOTIFY",
        "PAY",
        "POST",
        "PUBLISH",
        "PURCHASE",
        "REFUND",
        "REMOVE",
        "REVOKE",
        "RUN",
        "SEND",
        "SHARE",
        "TRANSFER",
        "UPLOAD",
    }
)


class _ComposioProvider(Protocol):
    @property
    def enabled(self) -> bool: ...

    def list_toolkits(
        self,
        *,
        customer_id: str,
        toolkits: list[str] | None = None,
        is_connected: bool | None = None,
        limit: int = 50,
        search: str | None = None,
    ) -> dict[str, Any]: ...

    def authorize_toolkit(
        self,
        *,
        customer_id: str,
        toolkit: str,
        callback_url: str | None = None,
    ) -> dict[str, Any]: ...

    def list_connected_accounts(
        self,
        *,
        customer_id: str,
        toolkits: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]: ...

    def delete_connected_account(self, *, connected_account_id: str) -> dict[str, Any]: ...

    def search_tools(
        self,
        *,
        query: str = "",
        toolkits: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]: ...

    def get_tool_schema(self, *, tool_slug: str) -> dict[str, Any]: ...

    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]: ...

    def list_google_sheets_tab_names(
        self,
        *,
        customer_id: str,
        spreadsheet_id: str,
        connected_account_id: str | None = None,
    ) -> dict[str, Any]: ...


class ComposioProviderError(RuntimeError):
    """A sanitized provider failure safe to expose to application error mapping."""


class IntegrationConnectionNotFoundError(LookupError):
    """The requested connection is absent or belongs to another tenant."""


@dataclass(frozen=True, slots=True)
class IntakeComposioBinding:
    """Exact tenant-owned connection and write action approved for an intake sink."""

    tenant_id: str
    toolkit: str
    connected_account_id: str
    tool_slug: str


class TenantComposioIntakePort:
    """Synchronous deterministic boundary used by intake activation and application."""

    def __init__(self, *, provider: _ComposioProvider) -> None:
        self._provider = provider

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._provider, "enabled", False))

    def resolve_sink_binding(
        self,
        *,
        tenant_id: str,
        toolkit: str,
        connected_account_id: str | None,
        tool_slug: str | None,
        operation_hint: str,
        required_arguments: set[str],
        allow_discovery: bool,
    ) -> IntakeComposioBinding:
        """Resolve once during activation, or revalidate an exact pin during execution."""

        tenant = self._required(tenant_id, "tenant_id")
        safe_toolkit = self._required(toolkit, "toolkit").casefold()
        if not self.enabled:
            raise ComposioProviderError("integration service is unavailable")
        connection = self._owned_active_connection(
            tenant_id=tenant,
            toolkit=safe_toolkit,
            connected_account_id=connected_account_id,
        )
        safe_slug = str(tool_slug or "").strip()
        if not safe_slug:
            if not allow_discovery:
                raise ComposioProviderError("intake sink requires a pinned tool")
            safe_slug = self._discover_single_tool(
                toolkit=safe_toolkit,
                operation_hint=operation_hint,
                required_arguments=required_arguments,
            )
        self._validate_tool(
            toolkit=safe_toolkit,
            tool_slug=safe_slug,
            required_arguments=required_arguments,
        )
        return IntakeComposioBinding(
            tenant_id=tenant,
            toolkit=safe_toolkit,
            connected_account_id=str(connection["id"]),
            tool_slug=safe_slug,
        )

    def execute_sink(
        self,
        *,
        tenant_id: str,
        toolkit: str,
        connected_account_id: str,
        tool_slug: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Revalidate ownership and the exact action immediately before execution."""

        binding = self.resolve_sink_binding(
            tenant_id=tenant_id,
            toolkit=toolkit,
            connected_account_id=connected_account_id,
            tool_slug=tool_slug,
            operation_hint="",
            required_arguments=set(arguments),
            allow_discovery=False,
        )
        result = self._provider_call(
            lambda: self._provider.execute_tool(
                customer_id=binding.tenant_id,
                tool_slug=binding.tool_slug,
                arguments=dict(arguments),
                connected_account_id=binding.connected_account_id,
            )
        )
        if not isinstance(result, dict):
            raise ComposioProviderError("integration provider returned an invalid response")
        return _safe_mapping(result)

    def list_google_sheets_tab_names(
        self,
        *,
        customer_id: str,
        spreadsheet_id: str,
        connected_account_id: str | None = None,
    ) -> dict[str, Any]:
        """Inspect a tenant-owned spreadsheet without exposing the raw provider."""

        tenant = self._required(customer_id, "customer_id")
        spreadsheet = self._required(spreadsheet_id, "spreadsheet_id")
        connection = self._owned_active_connection(
            tenant_id=tenant,
            toolkit="googlesheets",
            connected_account_id=connected_account_id,
        )
        result = self._provider_call(
            lambda: self._provider.list_google_sheets_tab_names(
                customer_id=tenant,
                spreadsheet_id=spreadsheet,
                connected_account_id=str(connection["id"]),
            )
        )
        if not isinstance(result, dict):
            raise ComposioProviderError("integration provider returned an invalid response")
        return _safe_mapping(result)

    def _owned_active_connection(
        self,
        *,
        tenant_id: str,
        toolkit: str,
        connected_account_id: str | None,
    ) -> dict[str, Any]:
        raw = self._provider_call(
            lambda: self._provider.list_connected_accounts(
                customer_id=tenant_id,
                toolkits=[toolkit],
                statuses=["ACTIVE"],
                limit=100,
            )
        )
        items = raw.get("items") if isinstance(raw, dict) else None
        owned = [
            dict(item)
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, Mapping)
            if str(item.get("user_id") or "").strip() == tenant_id
            and str(item.get("toolkit_slug") or "").strip().casefold() == toolkit
            and str(item.get("status") or "").strip().upper() == "ACTIVE"
            and str(item.get("id") or "").strip()
        ]
        requested = str(connected_account_id or "").strip()
        if requested:
            for item in owned:
                if str(item["id"]) == requested:
                    return item
            raise IntegrationConnectionNotFoundError("connection not found")
        if len(owned) != 1:
            raise IntegrationConnectionNotFoundError(
                "intake sink requires exactly one active toolkit connection"
            )
        return owned[0]

    def _discover_single_tool(
        self,
        *,
        toolkit: str,
        operation_hint: str,
        required_arguments: set[str],
    ) -> str:
        raw = self._provider_call(
            lambda: self._provider.search_tools(
                query=str(operation_hint or "").strip(),
                toolkits=[toolkit],
                limit=50,
            )
        )
        items = raw.get("items") if isinstance(raw, dict) else None
        candidates: list[str] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, Mapping):
                continue
            slug = str(item.get("slug") or "").strip()
            try:
                self._validate_tool_payload(
                    toolkit=toolkit,
                    tool_slug=slug,
                    tool=item,
                    required_arguments=required_arguments,
                )
            except ComposioProviderError:
                continue
            candidates.append(slug)
        unique = sorted(set(candidates))
        if len(unique) != 1:
            raise ComposioProviderError(
                "intake sink tool discovery must resolve exactly one approved action"
            )
        return unique[0]

    def _validate_tool(
        self,
        *,
        toolkit: str,
        tool_slug: str,
        required_arguments: set[str],
    ) -> None:
        raw = self._provider_call(lambda: self._provider.get_tool_schema(tool_slug=tool_slug))
        tool = raw.get("tool") if isinstance(raw, dict) else None
        if not isinstance(tool, Mapping):
            raise ComposioProviderError("integration action is unavailable")
        self._validate_tool_payload(
            toolkit=toolkit,
            tool_slug=tool_slug,
            tool=tool,
            required_arguments=required_arguments,
        )

    @staticmethod
    def _validate_tool_payload(
        *,
        toolkit: str,
        tool_slug: str,
        tool: Mapping[str, Any],
        required_arguments: set[str],
    ) -> None:
        returned_slug = str(tool.get("slug") or "").strip()
        returned_toolkit = str(tool.get("toolkit_slug") or "").strip().casefold()
        if not tool_slug or returned_slug != tool_slug or returned_toolkit != toolkit:
            raise ComposioProviderError("integration action does not match the approved toolkit")
        tokens = {
            token
            for token in re.split(r"[^A-Z0-9]+", returned_slug.upper())
            if token
        }
        if not tokens.intersection(_INTAKE_ALLOWED_ACTIONS) or tokens.intersection(
            _INTAKE_FORBIDDEN_ACTIONS
        ):
            raise ComposioProviderError("intake sink action is not an approved upsert or update")
        schema = tool.get("input_schema")
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        available = {str(key) for key in properties} if isinstance(properties, Mapping) else set()
        missing = sorted(required_arguments - available)
        if missing:
            raise ComposioProviderError(
                "intake sink action does not accept required arguments: " + ", ".join(missing)
            )

    @staticmethod
    def _required(value: str, field: str) -> str:
        safe = str(value or "").strip()
        if not safe or len(safe) > 300:
            raise ValueError(f"{field} is required")
        return safe

    @staticmethod
    def _provider_call(invoke: Callable[[], _T]) -> _T:
        try:
            return invoke()
        except Exception:
            raise ComposioProviderError("integration provider request failed") from None


class IntegrationInvokeJobArguments(JobArguments):
    connection_id: str = Field(min_length=1, max_length=300)
    action_name: str = Field(min_length=1, max_length=300)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


def _request_hash(operation: str, arguments: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"operation": operation, "arguments": dict(arguments)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _provider_effect_key(operation: str, idempotency_key: str) -> str:
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"composio:{operation}:{digest}"


def _safe_value(value: Any, *, key: str | None = None, depth: int = 0) -> JsonValue:
    if key and _SECRET_KEY_RE.search(key):
        return "[redacted]"
    if depth > 8:
        return "[max_depth]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        text = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", value)
        text = _BEARER_RE.sub("Bearer [redacted]", text)
        return text if len(text) <= 12_000 else f"{text[:12_000]}...[truncated]"
    if isinstance(value, bytes | bytearray | memoryview):
        return f"[binary:{len(value)} bytes]"
    if isinstance(value, Mapping):
        output: dict[str, JsonValue] = {}
        for index, (raw_key, nested) in enumerate(value.items()):
            if index >= 100:
                output["_truncated"] = True
                break
            safe_key = str(raw_key)[:300]
            output[safe_key] = _safe_value(nested, key=safe_key, depth=depth + 1)
        return output
    if isinstance(value, list | tuple | set):
        items = list(value)
        sequence: list[JsonValue] = [
            _safe_value(item, depth=depth + 1) for item in items[:100]
        ]
        if len(items) > 100:
            sequence.append(f"[truncated {len(items) - 100} items]")
        return sequence
    return _safe_value(str(value), depth=depth + 1)


def _safe_mapping(value: Any) -> dict[str, JsonValue]:
    sanitized = _safe_value(value)
    if not isinstance(sanitized, dict):
        raise ComposioProviderError("integration provider returned an invalid response")
    return sanitized


class TenantComposioService:
    """Expose Composio through tenant-scoped product operations only."""

    def __init__(self, *, provider: _ComposioProvider, idempotency: IdempotencyStore) -> None:
        self._provider = provider
        self._idempotency = idempotency

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._provider, "enabled", False))

    def register_handlers(self, registry: JobHandlerRegistry) -> None:
        registry.register(
            name="integration_invoke",
            arguments_model=IntegrationInvokeJobArguments,
            handler=self._invoke_job,
            timeout_seconds=300,
        )

    async def list_integrations(
        self,
        *,
        tenant_id: str,
        query: str | None,
    ) -> dict[str, JsonValue]:
        tenant = self._required(tenant_id, "tenant_id")
        search = str(query or "").strip()[:500] or None
        connections = await self.list_connections(tenant_id=tenant, integration_id=None)
        raw_connections = connections.get("items")
        owned = raw_connections if isinstance(raw_connections, list) else []
        by_integration: dict[str, dict[str, JsonValue]] = {}
        for item in owned:
            if not isinstance(item, dict):
                continue
            integration_id = str(item.get("integration_id", "") or "").casefold()
            if integration_id:
                by_integration[integration_id] = item

        try:
            payload = await self._provider_call(
                lambda: self._provider.list_toolkits(
                    customer_id=tenant,
                    limit=100,
                    search=search,
                )
            )
            mapping = self._provider_mapping(payload)
        except ComposioProviderError:
            fallback_items: list[JsonValue] = []
            normalized_search = str(search or "").casefold()
            for fallback_connection in by_integration.values():
                integration_id = str(
                    fallback_connection.get("integration_id", "") or ""
                ).strip()
                name = str(
                    fallback_connection.get("integration_name", "") or integration_id
                ).strip()
                if normalized_search and normalized_search not in (
                    f"{integration_id} {name}".casefold()
                ):
                    continue
                fallback_items.append(
                    {
                        "id": integration_id,
                        "name": name[:500],
                        "connected": True,
                        "connection_id": str(fallback_connection.get("id", "") or ""),
                        "connection_status": str(
                            fallback_connection.get("status", "") or ""
                        ),
                        "requires_authentication": True,
                    }
                )
            return {
                "tenant_id": tenant,
                "items": fallback_items,
                "catalog_available": False,
                "warning": "The integration catalog is temporarily unavailable.",
            }

        raw_items = mapping.get("items")
        catalog_items: list[JsonValue] = []
        for raw in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(raw, Mapping):
                continue
            integration_id = str(raw.get("slug", "") or "").strip()
            if not integration_id:
                continue
            connection = by_integration.get(integration_id.casefold())
            no_auth = bool(raw.get("is_no_auth", False))
            catalog_items.append(
                {
                    "id": integration_id,
                    "name": str(raw.get("name", "") or integration_id)[:500],
                    "connected": no_auth or connection is not None,
                    "connection_id": (
                        str(connection.get("id", "") or "") if connection is not None else None
                    ),
                    "connection_status": (
                        str(connection.get("status", "") or "")
                        if connection is not None
                        else None
                    ),
                    "requires_authentication": not no_auth,
                }
            )
        return {
            "tenant_id": tenant,
            "items": catalog_items,
            "catalog_available": True,
        }

    async def connect(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        integration_id: str,
        redirect_url: str | None,
        idempotency_key: str,
    ) -> dict[str, JsonValue]:
        tenant = self._required(tenant_id, "tenant_id")
        actor = self._required(actor_id, "actor_id")
        integration = self._identifier(integration_id, "integration_id")
        key = self._required(idempotency_key, "idempotency_key")
        callback_url = self._callback_url(redirect_url)
        arguments = {
            "actor_id": actor,
            "integration_id": integration,
            "callback_url": callback_url,
        }

        async def authorize() -> dict[str, JsonValue]:
            raw = await self._provider_call(
                lambda: self._provider.authorize_toolkit(
                    customer_id=tenant,
                    toolkit=integration,
                    callback_url=callback_url,
                )
            )
            payload = self._provider_mapping(raw)
            owner = str(payload.get("user_id") or payload.get("customer_id") or "").strip()
            if owner and owner != tenant:
                raise IntegrationConnectionNotFoundError("connection not found")
            connection_id = self._identifier(
                str(payload.get("connection_id") or payload.get("id") or ""),
                "connection_id",
            )
            authorization_url = self._authorization_url(payload.get("redirect_url"))
            return {
                "tenant_id": tenant,
                "user_id": tenant,
                "id": connection_id,
                "connection_id": connection_id,
                "integration_id": integration,
                "authorization_url": authorization_url,
                "oauth_url": authorization_url,
                "status": "authorization_required" if authorization_url else "pending",
            }

        result = await self._idempotency.execute(
            tenant_id=tenant,
            operation="integration_connect",
            idempotency_key=_provider_effect_key("integration_connect", key),
            request_hash=_request_hash("integration_connect", arguments),
            invoke=authorize,
        )
        sanitized = _safe_mapping(result)
        if isinstance(result, Mapping):
            sanitized["oauth_url"] = self._authorization_url(result.get("oauth_url"))
        return sanitized

    async def list_connections(
        self,
        *,
        tenant_id: str,
        integration_id: str | None,
    ) -> dict[str, JsonValue]:
        tenant = self._required(tenant_id, "tenant_id")
        integration = (
            self._identifier(integration_id, "integration_id") if integration_id else None
        )
        raw = await self._provider_call(
            lambda: self._provider.list_connected_accounts(
                customer_id=tenant,
                toolkits=[integration] if integration else None,
                limit=100,
            )
        )
        payload = self._provider_mapping(raw)
        raw_items = payload.get("items")
        items: list[JsonValue] = []
        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("user_id", "") or "").strip() != tenant:
                continue
            connection_id = str(item.get("id", "") or "").strip()
            toolkit_slug = str(
                item.get("toolkit_slug") or item.get("integration_id") or ""
            ).strip()
            if not connection_id or not toolkit_slug:
                continue
            if integration and toolkit_slug.casefold() != integration.casefold():
                continue
            items.append(
                {
                    "tenant_id": tenant,
                    "user_id": tenant,
                    "id": connection_id[:300],
                    "connection_id": connection_id[:300],
                    "status": str(item.get("status", "") or "")[:100],
                    "integration_id": toolkit_slug[:300],
                    "integration_name": str(
                        item.get("toolkit_name") or item.get("integration_name") or toolkit_slug
                    )[:500],
                }
            )
        return {"tenant_id": tenant, "items": items}

    async def get_connection(
        self,
        *,
        tenant_id: str,
        connection_id: str,
    ) -> dict[str, JsonValue]:
        tenant = self._required(tenant_id, "tenant_id")
        connection = self._identifier(connection_id, "connection_id")
        payload = await self.list_connections(tenant_id=tenant, integration_id=None)
        items = payload.get("items")
        for item in items if isinstance(items, list) else []:
            if (
                isinstance(item, dict)
                and str(item.get("id", "") or "") == connection
                and str(item.get("user_id", "") or "") == tenant
            ):
                return item
        raise IntegrationConnectionNotFoundError("connection not found")

    async def disconnect(
        self,
        *,
        tenant_id: str,
        connection_id: str,
        idempotency_key: str,
    ) -> dict[str, JsonValue]:
        tenant = self._required(tenant_id, "tenant_id")
        connection = self._identifier(connection_id, "connection_id")
        key = self._required(idempotency_key, "idempotency_key")

        async def remove() -> dict[str, JsonValue]:
            owned = await self.get_connection(tenant_id=tenant, connection_id=connection)
            if str(owned.get("user_id", "") or "") != tenant:
                raise IntegrationConnectionNotFoundError("connection not found")
            raw = await self._provider_call(
                lambda: self._provider.delete_connected_account(
                    connected_account_id=connection
                )
            )
            payload = self._provider_mapping(raw)
            provider_connection = payload.get("connected_account")
            if isinstance(provider_connection, Mapping):
                deleted_id = str(provider_connection.get("id", "") or "").strip()
                if deleted_id and deleted_id != connection:
                    raise ComposioProviderError("integration provider returned an invalid response")
            return {
                "tenant_id": tenant,
                "user_id": tenant,
                "id": connection,
                "connection_id": connection,
                "deleted": True,
            }

        result = await self._idempotency.execute(
            tenant_id=tenant,
            operation="connection_disconnect",
            idempotency_key=_provider_effect_key("connection_disconnect", key),
            request_hash=_request_hash(
                "connection_disconnect",
                {"connection_id": connection},
            ),
            invoke=remove,
        )
        return _safe_mapping(result)

    async def search_actions(
        self,
        *,
        tenant_id: str,
        query: str,
        integration_id: str | None,
        limit: int,
    ) -> dict[str, JsonValue]:
        tenant = self._required(tenant_id, "tenant_id")
        safe_query = str(query or "").strip()[:1_000]
        integration = (
            self._identifier(integration_id, "integration_id") if integration_id else None
        )
        if not safe_query and not integration:
            raise ValueError("query or integration_id is required")
        raw = await self._provider_call(
            lambda: self._provider.search_tools(
                query=safe_query,
                toolkits=[integration] if integration else None,
                limit=max(1, min(int(limit), 50)),
            )
        )
        payload = self._provider_mapping(raw)
        raw_items = payload.get("items")
        items: list[JsonValue] = []
        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, Mapping):
                continue
            action = self._action(item)
            if action is not None:
                items.append(action)
        return {"tenant_id": tenant, "items": items}

    async def get_action(
        self,
        *,
        tenant_id: str,
        action_name: str,
    ) -> dict[str, JsonValue]:
        tenant = self._required(tenant_id, "tenant_id")
        action = self._identifier(action_name, "action_name")
        raw = await self._provider_call(
            lambda: self._provider.get_tool_schema(tool_slug=action)
        )
        payload = self._provider_mapping(raw)
        tool = payload.get("tool")
        normalized = self._action(tool) if isinstance(tool, Mapping) else None
        if normalized is None:
            raise LookupError("integration action not found")
        return {"tenant_id": tenant, "action": normalized}

    async def _invoke_job(
        self,
        arguments: IntegrationInvokeJobArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        await context.progress({"stage": "validating_connection"})

        async def execute() -> dict[str, JsonValue]:
            connection = await self.get_connection(
                tenant_id=context.tenant_id,
                connection_id=arguments.connection_id,
            )
            if str(connection.get("user_id", "") or "") != context.tenant_id:
                raise IntegrationConnectionNotFoundError("connection not found")
            if str(connection.get("status", "") or "").upper() != "ACTIVE":
                raise ComposioProviderError("integration connection is not active")
            action_payload = await self.get_action(
                tenant_id=context.tenant_id,
                action_name=arguments.action_name,
            )
            action = action_payload.get("action")
            if not isinstance(action, dict):
                raise ComposioProviderError("integration action is unavailable")
            connection_integration = str(connection.get("integration_id", "") or "")
            action_integration = str(action.get("integration_id", "") or "")
            if (
                not action_integration
                or action_integration.casefold() != connection_integration.casefold()
            ):
                raise IntegrationConnectionNotFoundError("connection not found")
            raw = await self._provider_call(
                lambda: self._provider.execute_tool(
                    customer_id=context.tenant_id,
                    tool_slug=arguments.action_name,
                    arguments=dict(arguments.parameters),
                    connected_account_id=arguments.connection_id,
                )
            )
            result = self._provider_mapping(raw)
            if not bool(result.get("successful", False)):
                raise ComposioProviderError("integration action failed")
            return {
                "connection_id": arguments.connection_id,
                "action_name": arguments.action_name,
                "successful": True,
                "result": _safe_value(result.get("data")),
            }

        result = await self._idempotency.execute(
            tenant_id=context.tenant_id,
            operation="integration_invoke",
            idempotency_key=context.idempotency_key,
            request_hash=_request_hash(
                "integration_invoke",
                arguments.model_dump(mode="json"),
            ),
            invoke=execute,
        )
        sanitized = _safe_mapping(result)
        await context.progress({"stage": "completed"})
        return JobHandlerResult(
            summary="Integration action completed",
            data=sanitized,
        )

    async def _provider_call(self, callback: Callable[[], _T]) -> _T:
        if not self.enabled:
            raise ComposioProviderError("integration service is unavailable")
        try:
            return await asyncio.to_thread(callback)
        except (ComposioProviderError, IntegrationConnectionNotFoundError):
            raise
        except Exception as exc:
            logger.warning(
                "integration provider request failed (%s)",
                type(exc).__name__,
            )
            raise ComposioProviderError("integration provider request failed") from None

    @staticmethod
    def _provider_mapping(value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ComposioProviderError("integration provider returned an invalid response")
        return value

    @staticmethod
    def _action(value: Mapping[str, Any]) -> dict[str, JsonValue] | None:
        name = str(value.get("slug") or value.get("name") or "").strip()
        integration_id = str(
            value.get("toolkit_slug") or value.get("integration_id") or ""
        ).strip()
        if not name or not integration_id:
            return None
        schema = _safe_value(value.get("input_schema") or value.get("input_parameters") or {})
        return {
            "name": name[:300],
            "title": str(value.get("title") or value.get("name") or name)[:500],
            "description": str(value.get("description", "") or "")[:4_000],
            "integration_id": integration_id[:300],
            "integration_name": str(
                value.get("toolkit_name") or value.get("integration_name") or integration_id
            )[:500],
            "input_schema": schema if isinstance(schema, dict) else {},
        }

    @staticmethod
    def _required(value: str, field: str) -> str:
        safe = str(value or "").strip()
        if not safe:
            raise ValueError(f"{field} is required")
        if len(safe) > 300:
            raise ValueError(f"{field} is too long")
        return safe

    @classmethod
    def _identifier(cls, value: str, field: str) -> str:
        safe = cls._required(value, field)
        if not _IDENTIFIER_RE.fullmatch(safe):
            raise ValueError(f"{field} is invalid")
        return safe

    @staticmethod
    def _callback_url(value: str | None) -> str | None:
        candidate = str(value or "").strip()
        if not candidate:
            return None
        if len(candidate) > 8_192:
            raise ValueError("redirect_url is too long")
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("redirect_url must be an HTTP URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("redirect_url cannot contain credentials")
        return candidate

    @staticmethod
    def _authorization_url(value: Any) -> str | None:
        candidate = str(value or "").strip()
        if not candidate or len(candidate) > 8_192:
            return None
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            return None
        if parsed.scheme != "https" or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        return candidate


__all__ = [
    "ComposioProviderError",
    "IntakeComposioBinding",
    "IntegrationConnectionNotFoundError",
    "IntegrationInvokeJobArguments",
    "TenantComposioIntakePort",
    "TenantComposioService",
]
