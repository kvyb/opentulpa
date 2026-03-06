from opentulpa.agent.lc_messages import ToolMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


def test_tool_message_handoff_detected_from_control_metadata() -> None:
    message = ToolMessage(
        content="",
        tool_call_id="call_1",
        additional_kwargs={"opentulpa_control": {"status": "approval_pending", "approval_id": "apr_x"}},
    )
    assert OpenTulpaLangGraphRuntime._tool_message_indicates_approval_handoff(message) is True


def test_tool_message_handoff_detected_from_json_content() -> None:
    message = ToolMessage(
        content='{"status":"approval_pending","approval_id":"apr_x"}',
        tool_call_id="call_1",
    )
    assert OpenTulpaLangGraphRuntime._tool_message_indicates_approval_handoff(message) is True


def test_tool_message_handoff_ignores_non_json_or_non_pending_status() -> None:
    non_json = ToolMessage(content="approval pending", tool_call_id="call_1")
    non_pending = ToolMessage(content='{"status":"ok"}', tool_call_id="call_1")
    assert OpenTulpaLangGraphRuntime._tool_message_indicates_approval_handoff(non_json) is False
    assert OpenTulpaLangGraphRuntime._tool_message_indicates_approval_handoff(non_pending) is False


def test_build_approval_handoff_text_from_tool_outcome() -> None:
    result = {
        "tool_outcomes": [
            {"status": "ok"},
            {"status": "approval_pending", "approval_id": "apr_123", "tool_name": "routine_create"},
        ]
    }
    text = OpenTulpaLangGraphRuntime._build_approval_handoff_text(result)
    assert "Approval required before execution." in text
    assert "approval_id=apr_123" in text


def test_pending_context_summary_redacts_raw_payload_text() -> None:
    events = [
        {
            "source": "approval",
            "event_type": "executed",
            "payload": {
                "approval_id": "apr_123",
                "status": "approved",
                "raw_prompt": "I want you to scan my telegram Work folder",
            },
        }
    ]
    formatted = OpenTulpaLangGraphRuntime._format_pending_context(events)
    assert "approval_id=apr_123" in formatted
    assert "status=approved" in formatted
    assert "raw_prompt" not in formatted
    assert "scan my telegram" not in formatted
