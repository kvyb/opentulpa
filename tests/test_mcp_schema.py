import pytest

from opentulpa.mcp import MCPToolSchemaError, normalize_tool_schema, tool_schema_digest


def test_schema_digest_is_canonical_and_requires_object_contract() -> None:
    left = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    right = {
        "required": ["query"],
        "properties": {"query": {"type": "string"}},
        "type": "object",
    }

    assert tool_schema_digest(left) == tool_schema_digest(right)
    assert normalize_tool_schema(left) == left
    with pytest.raises(MCPToolSchemaError, match="type 'object'"):
        tool_schema_digest({"type": "string"})
    with pytest.raises(MCPToolSchemaError, match="unknown properties"):
        tool_schema_digest(
            {
                "type": "object",
                "properties": {},
                "required": ["missing"],
            }
        )
