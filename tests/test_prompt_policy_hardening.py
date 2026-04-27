from __future__ import annotations

from opentulpa.agent.graph_builder import (
    _build_relevant_skill_discovery_context,
    _build_tool_validation_repair_message,
    _enforce_tool_message_protocol,
    _extract_invoked_skill_snapshot,
    _normalize_approval_id,
    _sanitize_history_messages_for_model,
    _summarize_tool_validation_errors,
    _validate_model_tool_call,
)
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.prompt_classifier import classify_prompt_mode
from opentulpa.agent.prompt_policy import (
    build_system_prompt_message as _build_system_prompt_message,
)
from opentulpa.agent.prompt_sections import (
    build_prompt_mode_message,
)
from opentulpa.agent.turn_policy import (
    build_turn_mode_system_message,
    execution_origin_for_turn_mode,
    normalize_turn_mode,
)
from opentulpa.agent.utils import message_to_text


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
    assert "ask one concise clarifying question" in text
    assert "stay in chat mode" in text
    assert "Do not write placeholder progress text as assistant prose next to a tool call" in text
    assert "Telegram may not deliver assistant prose attached to a tool-call step" in text
    assert "call send_owner_update as the first tool call" in text
    assert "live owner/support turns" in text
    assert "use send_owner_update for intentional interim progress messages" in text
    assert "For long-running live owner/support work" in text
    assert "Do not use send_owner_update for inbound lead/intake processing" in text
    assert "concrete result or a plain blocker/failure report" in text
    assert "Do not give timing promises" in text
    assert "answer that status question directly" in text
    assert "Prefer dedicated Tulpa file tools over tulpa_run_terminal" in text
    assert "restate the needed facts in the reply" in text


def test_build_relevant_skill_discovery_context_is_discovery_only() -> None:
    text = _build_relevant_skill_discovery_context(
        available_skills=[
            {"name": "browser-use-operator", "description": "Use browser steps for dynamic websites.", "scope": "global"},
            {"name": "routine-schedule-composer", "description": "Compose robust routine instructions", "scope": "global"},
        ],
        selected_names=["browser-use-operator"],
    )
    assert "Skills relevant to this task:" in text
    assert "Use skill_get(name) before relying on a skill's actual instructions." in text
    assert "browser-use-operator" in text
    assert "routine-schedule-composer" not in text


def test_extract_invoked_skill_snapshot_prefers_skill_markdown() -> None:
    result = _extract_invoked_skill_snapshot(
        {
            "name": "browser-use-operator",
            "scope": "global",
            "description": "Use browser steps for dynamic websites.",
            "skill_markdown": "# Steps\nReuse browser sessions before starting a new one.",
        },
        requested_name="browser-use-operator",
    )
    assert result is not None
    name, text = result
    assert name == "browser-use-operator"
    assert "SKILL.md:" in text
    assert "Reuse browser sessions before starting a new one." in text


def test_turn_mode_policy_messages_are_mode_specific() -> None:
    interactive = str(build_turn_mode_system_message("interactive").content)
    workflow_setup = str(build_turn_mode_system_message("workflow_setup").content)
    routine_wake = str(build_turn_mode_system_message("routine_wake").content)
    approval_recovery = str(build_turn_mode_system_message("approval_recovery").content)
    event_notification = str(build_turn_mode_system_message("event_notification").content)

    assert "live user-guided turn" in interactive
    assert "call send_owner_update as the first tool call" in interactive
    assert "collaborating on an intake workflow draft" in workflow_setup
    assert "call send_owner_update as the first tool call" in workflow_setup
    assert "track source_file_ids and prepared_knowledge_file_ids" in workflow_setup
    assert "do not ask for polling, scanning, or schedule intervals" in workflow_setup
    assert "propose it with explicit assumptions" in workflow_setup
    assert "Do not persist the workflow until the user has seen a proposal and explicitly confirmed it." in workflow_setup
    assert "scheduled routine execution" in routine_wake
    assert "execute autonomously using tools and skills as needed" in routine_wake.lower()
    assert "previously approved action" in approval_recovery
    assert "continuation of the approved execution" in approval_recovery
    assert "background event/status notification" in event_notification
    assert normalize_turn_mode("unexpected") == "interactive"
    assert execution_origin_for_turn_mode("routine_wake") == "scheduled"
    assert execution_origin_for_turn_mode("approval_recovery") == "scheduled"
    assert execution_origin_for_turn_mode("interactive", thread_id="wake_legacy") == "scheduled"
    assert execution_origin_for_turn_mode("event_notification", thread_id="wake_legacy") == "interactive"


def test_literal_chat_prompt_mode_discourages_random_follow_up_questions() -> None:
    literal_chat = str(build_prompt_mode_message("literal_chat").content)
    workflow_setup = str(build_prompt_mode_message("workflow_setup").content)

    assert "Answer the visible user question directly." in literal_chat
    assert "If the user asks a greeting or how-you-are question" in literal_chat
    assert "Do not pivot into a new topic" in literal_chat
    assert "follow-up question" in literal_chat
    assert "collaborative intake workflow setup session" in workflow_setup
    assert "Do not ask for Telegram Business DM polling/schedule intervals" in workflow_setup
    assert "propose the workflow with stated assumptions" in workflow_setup
    assert "bind only prepared knowledge files" in workflow_setup
    assert "Only commit the workflow after explicit user confirmation." in workflow_setup


def test_classify_prompt_mode_returns_workflow_setup_for_workflow_setup_turns() -> None:
    assert classify_prompt_mode("show me the current draft", turn_mode="workflow_setup") == "workflow_setup"


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
        turn_mode="interactive",
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
        turn_mode="interactive",
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
        turn_mode="interactive",
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
        turn_mode="interactive",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is None


def test_validate_model_tool_call_rejects_ambiguous_routine_create_request() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Post Draft",
            "schedule": "0 9 * * *",
            "instruction": "Post the saved draft and report status.",
            "implementation_command": "python3 post_draft.py",
        },
        latest_user_text="Make that post today. Use the one we drafted.",
        turn_mode="interactive",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is not None
    assert "ACTION_CLARIFICATION_REQUIRED" in err


def test_summarize_tool_validation_errors_keeps_distinct_error_text() -> None:
    summary = _summarize_tool_validation_errors(
        [
            ToolMessage(content="ACTION_CLARIFICATION_REQUIRED: ask one concise question.", tool_call_id="a"),
            ToolMessage(content="ACTION_CLARIFICATION_REQUIRED: ask one concise question.", tool_call_id="b"),
            ToolMessage(content="ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: provide command.", tool_call_id="c"),
        ]
    )
    assert "ACTION_CLARIFICATION_REQUIRED" in summary
    assert "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED" in summary
    assert summary.count("ACTION_CLARIFICATION_REQUIRED") == 1


def test_build_tool_validation_repair_message_blocks_false_schedule_claims() -> None:
    message = _build_tool_validation_repair_message(
        [
            ToolMessage(
                content="ACTION_CLARIFICATION_REQUIRED: routine_create is only for explicit reminders.",
                tool_call_id="a",
            )
        ]
    )
    assert "schedule was not created" in message
    assert "Do not say it was scheduled" in message
    assert "clarifying question" in message


def test_build_tool_validation_repair_message_requests_exact_argument_repair() -> None:
    message = _build_tool_validation_repair_message(
        [
            ToolMessage(
                content="ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create needs implementation_command.",
                tool_call_id="a",
            )
        ]
    )
    assert "schedule was not created yet" in message
    assert "Do not claim success" in message
    assert "Repair the tool call arguments and retry" in message


def test_validate_model_tool_call_rejects_routine_create_when_user_wants_chat_only() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Draft Post",
            "schedule": "0 9 * * *",
            "instruction": "Prepare the post and report back.",
            "implementation_command": "python3 prepare_post.py",
        },
        latest_user_text="Think it through with me here first. Do not create a routine yet.",
        turn_mode="interactive",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is not None
    assert "CHAT_MODE_LOCKED" in err


def test_validate_model_tool_call_accepts_explicit_one_time_reminder_request() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Report Reminder",
            "schedule": "2026-03-19T17:00:00+08:00",
            "instruction": "Remind the user to send the report and confirm delivery.",
            "implementation_command": "python3 remind_report.py",
        },
        latest_user_text="Remind me in 3 hours to send the report.",
        turn_mode="interactive",
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
        turn_mode="interactive",
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
        turn_mode="interactive",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is not None
    assert "should be relative to working_dir=tulpa_stuff" in err


def test_validate_model_tool_call_rejects_duplicate_tulpa_root_prefix_for_read_file() -> None:
    err = _validate_model_tool_call(
        call_name="tulpa_read_file",
        args={
            "path": "tulpa_stuff/tulpa_stuff/solana_trading_wallet.json",
        },
        latest_user_text="read the wallet file",
        turn_mode="interactive",
        required_args={"tulpa_read_file": ("path",)},
        forbidden_tool_args={},
    )
    assert err is not None
    assert "duplicated allowed-root prefix" in err
    assert "tulpa_stuff/tulpa_stuff" in err


def test_validate_model_tool_call_allows_routine_create_during_routine_wake() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Follow-up Brief",
            "schedule": "0 9 * * *",
            "instruction": "Create a daily follow-up brief routine.",
            "implementation_command": "python3 build_brief.py",
        },
        latest_user_text="System update: a scheduled routine fired. Create a follow-up routine.",
        turn_mode="routine_wake",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is None


def test_validate_model_tool_call_rejects_routine_create_during_event_notification() -> None:
    err = _validate_model_tool_call(
        call_name="routine_create",
        args={
            "name": "Follow-up Brief",
            "schedule": "0 9 * * *",
            "instruction": "Create a daily follow-up brief routine.",
            "implementation_command": "python3 build_brief.py",
        },
        latest_user_text="System update: a background event occurred.",
        turn_mode="event_notification",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is not None
    assert "TURN_MODE_MISMATCH" in err


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


def test_sanitize_history_drops_internal_system_messages() -> None:
    messages = [
        HumanMessage(content="run login"),
        SystemMessage(content="SELF_CHECK_FAILED: internal repair note."),
        AIMessage(content="done"),
    ]
    sanitized = _sanitize_history_messages_for_model(messages)
    assert len(sanitized) == 2
    assert isinstance(sanitized[0], HumanMessage)
    assert isinstance(sanitized[1], AIMessage)


def test_sanitize_history_strips_unparsed_provider_tool_calls() -> None:
    message = AIMessage(content="Let me update the draft.").model_copy(
        update={
            "additional_kwargs": {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "intake_workflow_setup_update", "arguments": "{}"},
                }
            ],
            "refusal": None,
        },
        }
    )

    sanitized = _sanitize_history_messages_for_model([HumanMessage(content="setup"), message])

    assert len(sanitized) == 2
    assert isinstance(sanitized[1], AIMessage)
    assert not getattr(sanitized[1], "tool_calls", [])
    assert "tool_calls" not in getattr(sanitized[1], "additional_kwargs", {})
    assert "refusal" in getattr(sanitized[1], "additional_kwargs", {})


def test_sanitize_history_keeps_tool_calls_and_results_verbatim() -> None:
    huge_command = "python3 -c \"" + ("print('x')\\n" * 200) + "\""
    huge_stdout = "result line " * 400
    messages = [
        HumanMessage(content="run the solar math"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "tulpa_run_terminal",
                    "args": {
                        "command": huge_command,
                        "working_dir": "tulpa_stuff",
                        "path": "tulpa_stuff/solar_antarctica.py",
                    },
                }
            ],
        ),
        ToolMessage(
            content=(
                '{"ok":true,"returncode":0,"cwd":"tulpa_stuff","stdout":"'
                + huge_stdout
                + '","stderr":"","execution_origin":"interactive"}'
            ),
            tool_call_id="call_1",
        ),
    ]

    sanitized = _sanitize_history_messages_for_model(messages)

    assert len(sanitized) == 3
    assert isinstance(sanitized[1], AIMessage)
    assert isinstance(sanitized[2], ToolMessage)
    sanitized_call = sanitized[1].tool_calls[0]
    assert sanitized_call["id"] == "call_1"
    assert sanitized_call["name"] == "tulpa_run_terminal"
    assert sanitized_call["args"]["working_dir"] == "tulpa_stuff"
    assert sanitized_call["args"]["path"] == "tulpa_stuff/solar_antarctica.py"
    assert sanitized_call["args"]["command"] == huge_command
    sanitized_tool_text = str(sanitized[2].content or "")
    assert sanitized_tool_text == (
        '{"ok":true,"returncode":0,"cwd":"tulpa_stuff","stdout":"'
        + huge_stdout
        + '","stderr":"","execution_origin":"interactive"}'
    )


def test_message_to_text_uses_compact_json_for_tool_calls() -> None:
    script = "\n".join(f"print({idx})" for idx in range(120))
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "tulpa_write_file",
                "args": {
                    "path": "tulpa_stuff/antarctica_solar.py",
                    "content": script,
                },
            }
        ],
    )

    text = message_to_text(message)

    assert "tool_calls=" in text
    assert "tulpa_write_file" in text
    assert "tulpa_stuff/antarctica_solar.py" in text
    assert '": "' not in text
    assert '", "' not in text


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


def test_enforce_tool_message_protocol_keeps_complete_tool_segment_after_system_drop() -> None:
    messages = [
        HumanMessage(content="read the file"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "tulpa_read_file", "args": {"path": "tulpa_stuff/a.txt"}}],
        ),
        ToolMessage(content="hello", tool_call_id="call_1"),
        SystemMessage(content="VALIDATION_REPAIR_REQUIRED: internal note."),
        AIMessage(content="The file says hello."),
    ]
    sanitized = _sanitize_history_messages_for_model(messages)
    repaired = _enforce_tool_message_protocol(sanitized)
    assert len(repaired) == 4
    assert isinstance(repaired[0], HumanMessage)
    assert isinstance(repaired[1], AIMessage)
    assert bool(getattr(repaired[1], "tool_calls", []))
    assert isinstance(repaired[2], ToolMessage)
    assert isinstance(repaired[3], AIMessage)


def test_normalize_approval_id_rejects_none_and_null_strings() -> None:
    assert _normalize_approval_id(None) == ""
    assert _normalize_approval_id("None") == ""
    assert _normalize_approval_id("null") == ""
    assert _normalize_approval_id("apr_123") == "apr_123"


def test_prompt_mode_classifier_prefers_literal_chat_for_short_definition_question() -> None:
    assert (
        classify_prompt_mode("what does remote fte mean?", turn_mode="interactive")
        == "literal_chat"
    )


def test_prompt_mode_classifier_prefers_execution_for_action_request() -> None:
    assert (
        classify_prompt_mode("search the web and check the latest pricing", turn_mode="interactive")
        == "execution"
    )

def test_prompt_mode_message_blocks_hidden_context_for_literal_chat() -> None:
    text = str(build_prompt_mode_message("literal_chat").content or "")
    assert "Answer the visible user question directly." in text
    assert "Do not pull in hidden project context" in text
