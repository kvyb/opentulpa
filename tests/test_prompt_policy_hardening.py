from __future__ import annotations

from opentulpa.agent.graph_builder import (
    _build_skill_glossary_context,
    _build_system_prompt_message,
    _enforce_tool_message_protocol,
    _normalize_approval_id,
    _sanitize_history_messages_for_model,
    _validate_model_tool_call,
)
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, ToolMessage


def test_system_prompt_uses_structured_sections_and_rule_ids() -> None:
    message = _build_system_prompt_message()
    text = str(message.content or "")
    assert "[SECTION A] Core Behavior" in text
    assert "[SECTION B] Scheduling And Routines" in text
    assert "[SECTION C] Tool Selection" in text
    assert "[SECTION D] Claim Discipline And Approvals" in text
    # Critical rule IDs should be present in output for integrity checks.
    for rid in ("A06", "A08", "B03", "B04", "B06", "D01", "D07"):
        assert f"- {rid}:" in text
    assert "skill glossary as high-level discovery only" in text
    assert "call skill_get(name)" in text


def test_build_skill_glossary_context_is_high_level_and_points_to_skill_get() -> None:
    text = _build_skill_glossary_context(
        [
            {"name": "routine-schedule-composer", "description": "Compose robust routine instructions", "scope": "global"},
            {"name": "browser-ops", "description": "Use browser steps for dynamic websites", "scope": "user"},
        ]
    )
    assert "Skill glossary (high-level, non-prioritized):" in text
    assert "Call skill_get(name) to fetch full skill instructions before execution." in text
    assert "- routine-schedule-composer (global): Compose robust routine instructions" in text


def test_validate_model_tool_call_rejects_runtime_managed_args() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Daily Digest",
            "schedule": "0 9 * * *",
            "instruction": "You must run the digest script and report output.",
            "implementation_command": "python3 scripts/digest.py",
            "customer_id": "telegram_1",
        },
        latest_user_text="set recurring digest",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is not None
    assert "must not include argument(s): customer_id" in err


def test_validate_model_tool_call_rejects_legacy_routine_message_field() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Daily Digest",
            "schedule": "0 9 * * *",
            "instruction": "You must run the digest script and report output.",
            "implementation_command": "python3 scripts/digest.py",
            "message": "legacy",
        },
        latest_user_text="set recurring digest",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is not None
    assert "must not include argument(s): message" in err


def test_validate_model_tool_call_rejects_invalid_schedule_shape() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Daily Digest",
            "schedule": "every day at nine",
            "instruction": "You must run the digest script and report output.",
            "implementation_command": "python3 scripts/digest.py",
        },
        latest_user_text="set recurring digest",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is not None
    assert "schedule must be either cron" in err


def test_validate_model_tool_call_accepts_valid_routine_create() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Daily Digest",
            "schedule": "0 9 * * *",
            "instruction": "You must run the digest script and report output.",
            "implementation_command": "python3 scripts/digest.py",
        },
        latest_user_text="set recurring digest",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is None


def test_validate_model_tool_call_rejects_redundant_tulpa_prefix_for_terminal() -> None:
    err = _validate_model_tool_call(
        call_name="tulpa_run_terminal",
        args={
            "command": "python3 tulpa_stuff/tg_login.py",
            "working_dir": "tulpa_stuff",
        },
        latest_user_text="run login",
        required_args={"tulpa_run_terminal": ("command",)},
        forbidden_tool_args={},
    )
    assert err is not None
    assert "redundant working-dir path prefix" in err


def test_validate_model_tool_call_rejects_redundant_tulpa_prefix_for_routine_command() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Login refresh",
            "schedule": "0 */6 * * *",
            "instruction": "You must run scripts/tg_login.py and report output.",
            "implementation_command": "python3 tulpa_stuff/tg_login.py",
        },
        latest_user_text="set recurring login refresh",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is not None
    assert "should be relative to working_dir=tulpa_stuff" in err


def test_sanitize_history_keeps_tool_response_shape_for_approval_handoff() -> None:
    messages = [
        HumanMessage(content="run login"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "tulpa_run_terminal", "args": {"command": "python3 x.py"}}],
        ),
        ToolMessage(content='{"status":"approval_pending","approval_id":"apr_x"}', tool_call_id="call_1"),
    ]
    sanitized = _sanitize_history_messages_for_model(messages)
    assert len(sanitized) == 3
    assert isinstance(sanitized[2], ToolMessage)
    assert str(sanitized[2].content) == '{"status":"approval_pending","approval_id":"apr_x"}'
    assert str(getattr(sanitized[2], "tool_call_id", "")) == "call_1"


def test_enforce_tool_message_protocol_drops_incomplete_tool_call_segment() -> None:
    messages = [
        HumanMessage(content="run login"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "tulpa_run_terminal", "args": {"command": "python3 x.py"}}],
        ),
        HumanMessage(content="next user message"),
    ]
    repaired = _enforce_tool_message_protocol(messages)
    assert len(repaired) == 2
    assert isinstance(repaired[0], HumanMessage)
    assert isinstance(repaired[1], HumanMessage)


def test_normalize_approval_id_rejects_none_and_null_strings() -> None:
    assert _normalize_approval_id(None) == ""
    assert _normalize_approval_id("None") == ""
    assert _normalize_approval_id("null") == ""
    assert _normalize_approval_id("apr_123") == "apr_123"
