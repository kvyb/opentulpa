from __future__ import annotations

from opentulpa.agent.tool_parser import compact_tool_call_record, compact_tool_payload


def test_compact_tool_payload_flattens_jsonish_dict() -> None:
    payload = {
        "status": "ok",
        "data": {
            "id": "conv_123",
            "latest_message": {
                "text": "hello there",
                "sender": "customer",
            },
        },
        "items": ["a", "b"],
    }

    compact = compact_tool_payload(payload, value_char_limit=100)

    assert "status=ok" in compact
    assert "data.id=conv_123" in compact
    assert "data.latest_message.text=hello there" in compact
    assert "items[0]=a" in compact
    assert "items[1]=b" in compact


def test_compact_tool_payload_truncates_each_value_not_whole_payload() -> None:
    payload = {
        "status": "ok",
        "message": "x" * 160,
        "result": "y" * 140,
    }

    compact = compact_tool_payload(payload, value_char_limit=100)

    assert "status=ok" in compact
    assert "message=" in compact
    assert "result=" in compact
    assert "x" * 120 not in compact
    assert "y" * 120 not in compact


def test_compact_tool_call_record_formats_tool_args_and_result() -> None:
    line = compact_tool_call_record(
        tool_name="INSTAGRAM_GET_CONVERSATION",
        args={"conversation_id": "conv_123", "limit": 20},
        result={"status": "ok", "data": {"username": "@luna", "latest_message": "hi"}},
        args_value_char_limit=None,
        result_value_char_limit=100,
    )

    assert line.startswith("tool=INSTAGRAM_GET_CONVERSATION")
    assert "args[conversation_id=conv_123 | limit=20]" in line
    assert "status=ok" in line
    assert "data.username=@luna" in line
