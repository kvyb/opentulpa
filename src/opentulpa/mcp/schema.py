"""Canonical MCP JSON Schema validation and digesting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opentulpa.capabilities.models import canonical_json_digest


class MCPToolSchemaError(ValueError):
    """A discovered MCP schema is not safe to expose to a model."""


def normalize_tool_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe object schema without mutating transport data."""

    schema = dict(value)
    if schema.get("type") != "object":
        raise MCPToolSchemaError("MCP tool input schema must have type 'object'")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise MCPToolSchemaError("MCP tool input schema properties must be an object")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        raise MCPToolSchemaError("MCP tool input schema required must be a string list")
    if len(required) != len(set(required)):
        raise MCPToolSchemaError("MCP tool input schema required fields must be unique")
    unknown_required = set(required).difference(properties)
    if unknown_required:
        raise MCPToolSchemaError("MCP tool input schema requires unknown properties")
    try:
        canonical_json_digest(schema)
    except (TypeError, ValueError) as exc:
        raise MCPToolSchemaError("MCP tool input schema must contain only JSON values") from exc
    return schema


def tool_schema_digest(value: Mapping[str, Any]) -> str:
    return canonical_json_digest(normalize_tool_schema(value))


__all__ = ["MCPToolSchemaError", "normalize_tool_schema", "tool_schema_digest"]
