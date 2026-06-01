from __future__ import annotations

import json

import pytest

from opentulpa.agent.graph_builder import (
    _extract_invoked_skill_snapshot,
)
from opentulpa.agent.graph_nodes.tool_validation import build_validate_tool_calls_node
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.prompt_classifier import classify_prompt_mode
from opentulpa.agent.prompt_policy import (
    build_system_prompt_message as _build_system_prompt_message,
)
from opentulpa.agent.prompt_policy import (
    build_web_search_backend_prompt_message as _build_web_search_backend_prompt_message,
)
from opentulpa.agent.prompt_sections import (
    build_prompt_mode_message,
)
from opentulpa.agent.tool_message_protocol import (
    collapse_completed_tool_call_segments_for_model,
)
from opentulpa.agent.tool_message_protocol import (
    enforce_tool_message_protocol as _enforce_tool_message_protocol,
)
from opentulpa.agent.tool_message_protocol import (
    sanitize_history_messages_for_model as _sanitize_history_messages_for_model,
)
from opentulpa.agent.tool_validation import (
    _build_tool_validation_repair_message,
    _routine_create_intent_validation_error,
    _summarize_tool_validation_errors,
    _validate_model_tool_call,
)
from opentulpa.agent.turn_policy import (
    build_turn_mode_system_message,
    execution_origin_for_turn_mode,
    normalize_turn_mode,
)
from opentulpa.agent.turn_prompt_builder.frozen_context import (
    build_relevant_skill_discovery_context,
)
from opentulpa.agent.utils import message_to_text


class _RoutineIntentRuntime:
    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    async def classify_routine_create_intent(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return self.decision


class _RoutineWorkflowRuntime(_RoutineIntentRuntime):
    def __init__(self, workflow: dict[str, object]) -> None:
        super().__init__({"ok": True, "allow_create": True, "reason": "authorized"})
        self.workflow = workflow
        self.request_calls: list[tuple[str, str, dict[str, object]]] = []
        self._active_customer_id = "telegram_83969136"

    async def _request_with_backoff(self, method: str, path: str, **kwargs: object) -> object:
        self.request_calls.append((method, path, dict(kwargs)))

        class _Response:
            status_code = 200

            def __init__(self, workflow: dict[str, object]) -> None:
                self._workflow = workflow

            def json(self) -> dict[str, object]:
                return {"workflow": self._workflow}

        return _Response(self.workflow)


def _disable_exa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)


def test_system_prompt_uses_structured_sections_and_rule_ids() -> None:
    message = _build_system_prompt_message()
    text = str(message.content or "")
    assert "[SECTION A] Core Behavior" in text
    assert "[SECTION B] Scheduling And Routines" in text
    assert "[SECTION C] Tool Selection" in text
    assert "[SECTION D] Claim Discipline And Execution" in text
    # Critical rule IDs should be present in output for integrity checks.
    for rid in ("A06", "A08", "B03", "B04", "B06", "D01", "D03"):
        assert f"- {rid}:" in text
    assert "skill glossary as high-level discovery only" in text
    assert "call skill_get(name)" in text
    assert "ask one concise clarifying question" in text
    assert "stay in chat mode" in text
    assert "Assistant prose attached to a tool-call step may be surfaced as a live chat update" in text
    assert "do not attach private reasoning or large drafts to tool calls" in text
    assert "call send_owner_update once early" in text
    assert "owner/support turns" in text
    assert "use concise attached tool-call prose or send_owner_update" in text
    assert "For long-running owner/support work" in text
    assert "inbound lead/intake workflow execution" in text
    assert "concrete result or a plain blocker/failure report" in text
    assert "In live owner/support turns" in text
    assert "use turn_plan for longer-horizon work" in text
    assert "research, discovery, comparison, analysis" in text
    assert "clear goal and stop condition" in text
    assert "realistic current-turn plan with a clear goal" in text
    assert "routine wakes, background event notifications, or inbound/customer-message execution" in text
    assert "Do not use turn_plan for simple chat" in text
    assert "Do not give timing promises" in text
    assert "answer that status question directly" in text
    assert "Prefer dedicated Tulpa file tools over tulpa_run_terminal" in text
    assert "restate the needed facts in the reply" in text
    assert "Do not call the same successful tool with the same arguments twice in a row" in text
    assert "Runtime blocks exact consecutive repeats" in text
    assert "verify current docs/schema for the exact model" in text
    assert "change only the failing parameter or step" in text
    assert "Never embed secrets in generated files or tool arguments" in text
    assert "never execute API calls, filesystem writes, network calls, or long-running work at module import time" in text
    assert "Put executable work inside a function such as main() or run()" in text
    assert 'if __name__ == "__main__"' in text
    assert "Keep scheduled routines distinct from intake workflows" in text
    assert "Telegram Business intake workflows use Telegram webhook events" in text
    assert "empty routine_id/schedule is expected" in text
    assert "Instagram DM intake workflows use scheduled Composio polling" in text
    assert "Do not promise webhook-like handling" in text
    assert "Follow the WEB_SEARCH_BACKEND prompt note" in text
    assert "WEB_SEARCH_BACKEND: exa" not in text


def test_web_search_backend_prompt_is_provider_specific() -> None:
    exa_text = str(_build_web_search_backend_prompt_message("exa").content)
    pplx_text = str(_build_web_search_backend_prompt_message("pplx").content)
    none_text = str(_build_web_search_backend_prompt_message(None).content)

    assert "WEB_SEARCH_BACKEND: exa" in exa_text
    assert "search_type" in exa_text
    assert "category='news'" in exa_text
    assert "20 raw results" in exa_text
    assert "hard cap of 2 calls per turn" in exa_text
    assert "browser_use_run" in exa_text
    assert "num_results" in exa_text
    assert "start_published_date" not in exa_text
    assert "end_published_date" not in exa_text
    assert "WEB_SEARCH_BACKEND: pplx" in pplx_text
    assert "Pass only query" in pplx_text
    assert "at most 5 web_search calls per turn" in pplx_text
    assert "Do not pass Exa-only args" in pplx_text
    assert "WEB_SEARCH_BACKEND: none" in none_text
    assert "web_search is not configured" in none_text


def test_build_relevant_skill_discovery_context_is_discovery_only() -> None:
    text = build_relevant_skill_discovery_context(
        available_skills=[
            {"name": "browser-use-operator", "description": "Use browser steps for dynamic websites.", "scope": "global"},
            {"name": "routine-schedule-composer", "description": "Compose robust routine instructions", "scope": "global"},
        ],
        selected_names=["browser-use-operator"],
    )
    assert "Available skills registry:" in text
    assert "call skill_get(name)" in text
    assert "browser-use-operator" in text
    assert "routine-schedule-composer" not in text


def test_build_relevant_skill_discovery_context_lists_registry_without_selector() -> None:
    text = build_relevant_skill_discovery_context(
        available_skills=[
            {
                "name": "browser-use-operator",
                "description": "Use browser steps for dynamic websites with authenticated pages, JavaScript rendering, screenshots, forms, navigation, session reuse, retries, and extra words that should be trimmed down hard.",
                "scope": "global",
            },
            {"name": "routine-schedule-composer", "description": "Compose robust routine instructions", "scope": "global"},
        ],
        selected_names=[],
    )
    assert "Available skills registry:" in text
    assert "browser-use-operator" in text
    assert "routine-schedule-composer" in text
    assert "extra words that should be trimmed down hard" not in text


def test_tool_validation_helpers_live_in_dedicated_module() -> None:
    assert _validate_model_tool_call.__module__ == "opentulpa.agent.tool_validation"
    assert _build_tool_validation_repair_message.__module__ == "opentulpa.agent.tool_validation"
    assert (
        build_validate_tool_calls_node.__module__
        == "opentulpa.agent.graph_nodes.tool_validation"
    )


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
    event_notification = str(build_turn_mode_system_message("event_notification").content)

    assert "live user-guided turn" in interactive
    assert "attach one concise visible progress sentence" in interactive
    assert "call send_owner_update as the first tool call" in interactive
    assert "use turn_plan as a private current-turn checklist" in interactive
    assert "realistic for this turn's runtime with a clear goal" in interactive
    assert "Apply retrieved user preferences, directive facts, and style facts" in interactive
    assert "store a concise preference with tool_group_exec" in interactive
    assert "stop making Telegram answers look like Markdown documents" in interactive
    assert "Lists are fine when they fit the answer naturally." in interactive
    assert "collaborating on an intake workflow draft" in workflow_setup
    assert "attach one concise visible progress sentence" in workflow_setup
    assert "call send_owner_update as the first tool call" in workflow_setup
    assert "track original source_file_ids" in workflow_setup
    assert "do not ask for polling, scanning, or schedule intervals" in workflow_setup
    assert "required_fields are stable ASCII snake_case ids" in workflow_setup
    assert "field_guidance keys must match required_fields ids" in workflow_setup
    assert "propose it with explicit assumptions" in workflow_setup
    assert "Do not persist the workflow until the user has seen a proposal and explicitly confirmed it." in workflow_setup
    base_policy = str(_build_system_prompt_message().content)
    assert "choose among three first-class knowledge paths by inferred user intent" in base_policy
    assert "one-off file analysis" in base_policy
    assert "reusable user/chat context" in base_policy
    assert "workflow/business knowledge" in base_policy
    assert "If the recent message or conversation clearly implies a path, take it" in base_policy
    assert "if intent is unclear, ask one concise question" in base_policy
    assert "remember it for future chat, use it for a workflow/business bot, or answer about it once" in base_policy
    assert "For intake workflows over source docs" in base_policy
    assert "business_knowledge_index and query them with business_knowledge_query" in base_policy
    assert "reuse existing user/chat context during workflow setup" in base_policy
    assert "business_knowledge_index those file_ids into the current setup scope" in base_policy
    assert "after a workflow exists, user_context_promote_to_intake" in base_policy
    assert "For knowledge-base capabilities" in base_policy
    assert "user/chat context knowledge, workflow/business knowledge, and one-off file analysis" in base_policy
    assert "instead of talking only about business knowledge" in base_policy
    assert "do not refuse merely because login, CAPTCHA, MFA, or session persistence may be involved" in base_policy
    assert "unless the owner explicitly asks to use or persist those credentials" in base_policy
    assert "you may persist them to local files, memory, or directives" in base_policy
    assert "Do not echo or summarize secret values back to chat" in base_policy
    assert "scheduled routine execution" in routine_wake
    assert "execute autonomously using tools and skills as needed" in routine_wake.lower()
    assert "Return the user-visible routine notification" in routine_wake
    assert "background event/status notification" in event_notification
    assert normalize_turn_mode("unexpected") == "interactive"
    assert execution_origin_for_turn_mode("routine_wake") == "scheduled"
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
    assert "required_fields are stable machine field ids" in workflow_setup
    assert "sink_config.field_mapping" in workflow_setup
    assert "prepare them with business_knowledge_index" in workflow_setup
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


@pytest.mark.asyncio
async def test_routine_intent_classifier_accepts_confirmation_after_routine_question() -> None:
    runtime = _RoutineIntentRuntime(
        {
            "ok": True,
            "allow_create": True,
            "confidence": 0.94,
            "reason": "User positively confirmed the assistant's routine creation question.",
        }
    )
    args = {
        "name": "daily-ai-oss-briefing",
        "schedule": "0 10 * * *",
        "instruction": "Read briefing_last_sent.md, find new AI and OSS news, send only fresh items.",
        "implementation_command": "python3 daily_briefing.py",
    }
    err = _validate_model_tool_call(
        call_name="routine_create",
        args=args,
        latest_user_text="да, мне каждый раз нужны новые новости от тебя а не одно и то же. Создай плиз",
        turn_mode="interactive",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is None
    intent_err = await _routine_create_intent_validation_error(
        runtime,
        args=args,
        latest_user_text="да, мне каждый раз нужны новые новости от тебя а не одно и то же. Создай плиз",
        prior_assistant_text=(
            "Подтверди: хочешь, чтобы я пересоздал daily-ai-oss-briefing "
            "на 10:00 AM с антидубликатной логикой?"
        ),
        turn_mode="interactive",
    )
    assert intent_err is None
    assert runtime.calls[0]["latest_user_text"] == (
        "да, мне каждый раз нужны новые новости от тебя а не одно и то же. Создай плиз"
    )


@pytest.mark.asyncio
async def test_routine_intent_classifier_rejects_ambiguous_routine_create_request() -> None:
    runtime = _RoutineIntentRuntime(
        {
            "ok": True,
            "allow_create": False,
            "confidence": 0.89,
            "reason": "User did not ask for a schedule or automation.",
        }
    )
    args = {
        "name": "Post Draft",
        "schedule": "0 9 * * *",
        "instruction": "Post the saved draft and report status.",
        "implementation_command": "python3 post_draft.py",
    }
    err = _validate_model_tool_call(
        call_name="routine_create",
        args=args,
        latest_user_text="Make that post today. Use the one we drafted.",
        turn_mode="interactive",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is None
    intent_err = await _routine_create_intent_validation_error(
        runtime,
        args=args,
        latest_user_text="Make that post today. Use the one we drafted.",
        prior_assistant_text="",
        turn_mode="interactive",
    )
    assert intent_err is not None
    assert "ACTION_CLARIFICATION_REQUIRED" in intent_err


@pytest.mark.asyncio
async def test_routine_create_rejects_telegram_business_intake_workflow_schedule() -> None:
    runtime = _RoutineWorkflowRuntime(
        {
            "workflow_id": "iwf_fa2uqd",
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "routine_id": "",
            "schedule": "",
        }
    )
    args = {
        "name": "OpenTulpa Sales — Вика (intake каждые 2 мин)",
        "schedule": "*/2 * * * *",
        "implementation_command": "python3 -m opentulpa.intake_runner --workflow-id iwf_fa2uqd",
        "instruction": (
            "Запустить intake workflow iwf_fa2uqd. Проверить входящие сообщения "
            "в Telegram Business DM."
        ),
    }

    err = _validate_model_tool_call(
        call_name="routine_create",
        args=args,
        latest_user_text="да",
        turn_mode="interactive",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is None

    intent_err = await _routine_create_intent_validation_error(
        runtime,
        args=args,
        latest_user_text="да",
        prior_assistant_text="Создать рутину — да или нет?",
        turn_mode="interactive",
    )

    assert intent_err is not None
    assert "EVENT_DRIVEN_INTAKE_WORKFLOW" in intent_err
    assert "telegram_business_dm" in intent_err
    assert "empty routine_id/schedule is expected" in intent_err
    assert runtime.calls == []
    assert runtime.request_calls[0][1] == "/internal/intake/workflows/get"


@pytest.mark.asyncio
async def test_routine_create_allows_scheduled_instagram_intake_workflow() -> None:
    runtime = _RoutineWorkflowRuntime(
        {
            "workflow_id": "iwf_insta",
            "channel": "instagram_dm",
            "provider": "composio",
            "routine_id": "",
            "schedule": "*/2 * * * *",
        }
    )
    args = {
        "name": "Instagram intake poller",
        "schedule": "*/2 * * * *",
        "implementation_command": "python3 -m opentulpa.intake_runner --workflow-id iwf_insta",
        "instruction": "Run intake workflow iwf_insta.",
    }

    intent_err = await _routine_create_intent_validation_error(
        runtime,
        args=args,
        latest_user_text="да",
        prior_assistant_text="Создать рутину — да или нет?",
        turn_mode="interactive",
    )

    assert intent_err is None
    assert runtime.calls


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
    assert "scheduled action was not created" in message
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
    assert "scheduled action was not created yet" in message
    assert "Do not claim success" in message
    assert "Repair the tool call arguments and retry" in message


def test_build_tool_validation_repair_message_for_duplicate_success_does_not_request_repair() -> None:
    message = _build_tool_validation_repair_message(
        [
            ToolMessage(
                content=(
                    "DUPLICATE_TOOL_CALL_PREVIOUS_SUCCESS: "
                    'tool_group_exec(command="telegram_business_status", args_json={}) '
                    "already just succeeded."
                ),
                tool_call_id="a",
            )
        ]
    )

    assert "DUPLICATE_TOOL_CALL_PREVIOUS_SUCCESS" in message
    assert "Do not repair arguments" in message
    assert "previous successful tool result" in message
    assert "requested tool action was not completed yet" not in message


def test_build_tool_validation_repair_message_is_generic_for_intake_setup_errors() -> None:
    err = _validate_model_tool_call(
        call_name="intake_workflow_setup_update",
        args=None,
        latest_user_text="update the workflow setup",
        turn_mode="interactive",
        required_args={"intake_workflow_setup_update": ()},
        forbidden_tool_args={},
    )
    assert err is not None

    message = _build_tool_validation_repair_message([ToolMessage(content=err, tool_call_id="a")])

    assert "requested tool action was not completed yet" in message
    assert "Do not claim success" in message
    assert "Repair the tool call arguments and retry" in message
    assert "schedule" not in message.lower()


def test_build_tool_validation_repair_message_is_generic_for_tooling_errors() -> None:
    err = _validate_model_tool_call(
        call_name="tulpa_run_terminal",
        args={},
        latest_user_text="run the checks",
        turn_mode="interactive",
        required_args={"tulpa_run_terminal": ("command",)},
        forbidden_tool_args={},
    )
    assert err is not None

    message = _build_tool_validation_repair_message([ToolMessage(content=err, tool_call_id="a")])

    assert "requested tool action was not completed yet" in message
    assert "Do not claim success" in message
    assert "tulpa_run_terminal" in message
    assert "schedule" not in message.lower()


def test_validate_web_search_rejects_unsupported_exa_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")

    direct_error = _validate_model_tool_call(
        call_name="web_search",
        args={"query": "creators", "num_results": 20},
        latest_user_text="find creators",
        turn_mode="interactive",
        required_args={},
        forbidden_tool_args={},
    )
    nested_error = _validate_model_tool_call(
        call_name="tool_group_exec",
        args={
            "group": "web",
            "command": "web_search",
            "args_json": {"query": "creators", "num_results": 20},
        },
        latest_user_text="find creators",
        turn_mode="interactive",
        required_args={},
        forbidden_tool_args={},
    )

    assert direct_error is not None
    assert "Remove unsupported argument(s): num_results" in direct_error
    assert nested_error is not None
    assert "Nested tool_group_exec web_search command must be repaired" in nested_error


def test_validate_web_search_rejects_exa_args_when_pplx_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_exa(monkeypatch)

    error = _validate_model_tool_call(
        call_name="web_search",
        args={"query": "creators", "search_type": "deep"},
        latest_user_text="find creators",
        turn_mode="interactive",
        required_args={},
        forbidden_tool_args={},
    )

    assert error is not None
    assert "accepts only query" in error
    assert "search_type" in error


@pytest.mark.asyncio
async def test_validate_tool_calls_blocks_web_search_after_five_successes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_exa(monkeypatch)
    events: list[tuple[str, dict[str, object]]] = []

    def _log(state: dict[str, object], event: str, **kwargs: object) -> None:
        del state
        events.append((event, kwargs))

    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=_log,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    result = await node(
        {
            "messages": [
                HumanMessage(content="find reddit threads"),
                AIMessage(
                    content="searching",
                    tool_calls=[{"id": "call_1", "name": "web_search", "args": {"query": "one"}}],
                ),
                ToolMessage(
                    content='{"status":"ok"}',
                    tool_call_id="call_1",
                    additional_kwargs={"opentulpa_control": {"status": "ok"}},
                ),
                AIMessage(
                    content="searching",
                    tool_calls=[{"id": "call_2", "name": "web_search", "args": {"query": "two"}}],
                ),
                ToolMessage(
                    content='{"status":"ok"}',
                    tool_call_id="call_2",
                    additional_kwargs={"opentulpa_control": {"status": "ok"}},
                ),
                AIMessage(
                    content="searching",
                    tool_calls=[{"id": "call_3", "name": "web_search", "args": {"query": "three"}}],
                ),
                ToolMessage(
                    content='{"status":"ok"}',
                    tool_call_id="call_3",
                    additional_kwargs={"opentulpa_control": {"status": "ok"}},
                ),
                AIMessage(
                    content="searching",
                    tool_calls=[{"id": "call_4", "name": "web_search", "args": {"query": "four"}}],
                ),
                ToolMessage(
                    content='{"status":"ok"}',
                    tool_call_id="call_4",
                    additional_kwargs={"opentulpa_control": {"status": "ok"}},
                ),
                AIMessage(
                    content="searching",
                    tool_calls=[{"id": "call_5", "name": "web_search", "args": {"query": "five"}}],
                ),
                ToolMessage(
                    content='{"status":"ok"}',
                    tool_call_id="call_5",
                    additional_kwargs={"opentulpa_control": {"status": "ok"}},
                ),
                AIMessage(
                    content="searching again",
                    tool_calls=[{"id": "call_6", "name": "web_search", "args": {"query": "six"}}],
                ),
            ],
            "turn_mode": "interactive",
            "turn_budget": {
                "max_search_calls": 5,
                "used_search_calls": 5,
            },
        }
    )

    assert result.goto == "agent"
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert "WEB_SEARCH_BUDGET_EXCEEDED" in str(update_messages[0].content)
    assert "browser_use_run" in str(update_messages[0].content)
    assert "maximum web_search cap was reached" in str(update_messages[0].content)
    assert isinstance(update_messages[1], SystemMessage)
    assert any(event == "graph.validate_tools.failed" for event, _ in events)


@pytest.mark.asyncio
async def test_validate_tool_calls_blocks_exa_web_search_after_two_successes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    result = await node(
        {
            "messages": [
                HumanMessage(content="find reddit threads"),
                AIMessage(
                    content="searching",
                    tool_calls=[{"id": "call_1", "name": "web_search", "args": {"query": "one"}}],
                ),
                ToolMessage(
                    content='{"status":"ok"}',
                    tool_call_id="call_1",
                    additional_kwargs={"opentulpa_control": {"status": "ok"}},
                ),
                AIMessage(
                    content="searching again",
                    tool_calls=[{"id": "call_2", "name": "web_search", "args": {"query": "two"}}],
                ),
                ToolMessage(
                    content='{"status":"ok"}',
                    tool_call_id="call_2",
                    additional_kwargs={"opentulpa_control": {"status": "ok"}},
                ),
                AIMessage(
                    content="searching third time",
                    tool_calls=[{"id": "call_3", "name": "web_search", "args": {"query": "three"}}],
                ),
            ],
            "turn_mode": "interactive",
            "turn_budget": {
                "max_search_calls": 2,
                "used_search_calls": 2,
            },
        }
    )

    assert result.goto == "agent"
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert "EXA_SEARCH_BUDGET_EXCEEDED" in str(update_messages[0].content)
    assert "limited to 2 calls per turn" in str(update_messages[0].content)
    assert 'group="browser", command="browser_use_run"' in str(update_messages[0].content)


@pytest.mark.asyncio
async def test_validate_tool_calls_rejects_third_exa_web_search_call_in_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    assistant = AIMessage(
        content="too many exa searches",
        tool_calls=[
            {"id": "call_1", "name": "web_search", "args": {"query": "one"}},
            {"id": "call_2", "name": "web_search", "args": {"query": "two"}},
            {"id": "call_3", "name": "web_search", "args": {"query": "three"}},
        ],
    )
    result = await node(
        {
            "messages": [HumanMessage(content="find reddit threads"), assistant],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert update_messages[0].tool_call_id == "call_3"
    assert "EXA_SEARCH_BUDGET_EXCEEDED" in str(update_messages[0].content)
    assert len(assistant.tool_calls) == 3


@pytest.mark.asyncio
async def test_validate_tool_calls_rejects_sixth_web_search_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_exa(monkeypatch)
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    assistant = AIMessage(
        content="too many searches",
        tool_calls=[
            {"id": "call_1", "name": "web_search", "args": {"query": "one"}},
            {"id": "call_2", "name": "web_search", "args": {"query": "two"}},
            {"id": "call_3", "name": "web_search", "args": {"query": "three"}},
            {"id": "call_4", "name": "web_search", "args": {"query": "four"}},
            {"id": "call_5", "name": "web_search", "args": {"query": "five"}},
            {"id": "call_6", "name": "web_search", "args": {"query": "six"}},
        ],
    )
    result = await node(
        {
            "messages": [HumanMessage(content="find reddit threads"), assistant],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert update_messages[0].tool_call_id == "call_6"
    assert "WEB_SEARCH_BUDGET_EXCEEDED" in str(update_messages[0].content)
    assert len(assistant.tool_calls) == 6


@pytest.mark.asyncio
async def test_validate_tool_calls_counts_nested_tool_group_web_search_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_exa(monkeypatch)
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    assistant = AIMessage(
        content="too many nested searches",
        tool_calls=[
            {
                "id": "call_1",
                "name": "tool_group_exec",
                "args": {
                    "calls": [
                        {"group": "web", "command": "web_search", "args_json": {"query": "one"}},
                        {"group": "web", "command": "web_search", "args_json": {"query": "two"}},
                        {"group": "web", "command": "web_search", "args_json": {"query": "three"}},
                        {"group": "web", "command": "web_search", "args_json": {"query": "four"}},
                        {"group": "web", "command": "web_search", "args_json": {"query": "five"}},
                        {"group": "web", "command": "web_search", "args_json": {"query": "six"}},
                    ]
                },
            }
        ],
    )
    result = await node(
        {
            "messages": [HumanMessage(content="find leads"), assistant],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert update_messages[0].tool_call_id == "call_1"
    assert "WEB_SEARCH_BATCH_TOO_LARGE" in str(update_messages[0].content)
    assert len(assistant.tool_calls[0]["args"]["calls"]) == 6


@pytest.mark.asyncio
async def test_validate_tool_calls_counts_json_string_nested_tool_group_web_search_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_exa(monkeypatch)
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    calls = [
        {"group": "web", "command": "web_search", "args_json": {"query": str(index)}}
        for index in range(6)
    ]
    assistant = AIMessage(
        content="too many nested searches",
        tool_calls=[
            {
                "id": "call_1",
                "name": "tool_group_exec",
                "args": {"calls": json.dumps(calls)},
            }
        ],
    )
    result = await node(
        {
            "messages": [HumanMessage(content="find leads"), assistant],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert update_messages[0].tool_call_id == "call_1"
    assert "WEB_SEARCH_BATCH_TOO_LARGE" in str(update_messages[0].content)
    kept_calls = json.loads(assistant.tool_calls[0]["args"]["calls"])
    assert len(kept_calls) == 6


@pytest.mark.asyncio
async def test_validate_tool_calls_reports_over_cap_web_search_in_mixed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    assistant = AIMessage(
        content="mixed batch",
        tool_calls=[
            {"id": "call_1", "name": "turn_plan", "args": {"items": []}},
            {"id": "call_2", "name": "web_search", "args": {"query": "one"}},
            {"id": "call_3", "name": "web_search", "args": {"query": "two"}},
            {"id": "call_4", "name": "web_search", "args": {"query": "three"}},
        ],
    )
    result = await node(
        {
            "messages": [HumanMessage(content="find leads"), assistant],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert update_messages[0].tool_call_id == "call_4"
    assert "EXA_SEARCH_BUDGET_EXCEEDED" in str(update_messages[0].content)
    assert [call["id"] for call in assistant.tool_calls] == [
        "call_1",
        "call_2",
        "call_3",
        "call_4",
    ]


@pytest.mark.asyncio
async def test_validate_tool_calls_does_not_count_blocked_web_search_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_exa(monkeypatch)
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    result = await node(
        {
            "messages": [
                HumanMessage(content="find reddit threads"),
                AIMessage(
                    content="too many searches",
                    tool_calls=[
                        {"id": "call_1", "name": "web_search", "args": {"query": "one"}},
                        {"id": "call_2", "name": "web_search", "args": {"query": "two"}},
                        {"id": "call_3", "name": "web_search", "args": {"query": "three"}},
                        {"id": "call_4", "name": "web_search", "args": {"query": "four"}},
                        {"id": "call_5", "name": "web_search", "args": {"query": "five"}},
                        {"id": "call_6", "name": "web_search", "args": {"query": "six"}},
                    ],
                ),
                ToolMessage(content="WEB_SEARCH_BUDGET_EXCEEDED", tool_call_id="call_6"),
                SystemMessage(content="repair"),
                AIMessage(
                    content="try five searches",
                    tool_calls=[
                        {"id": "call_7", "name": "web_search", "args": {"query": "one"}},
                        {"id": "call_8", "name": "web_search", "args": {"query": "two"}},
                        {"id": "call_9", "name": "web_search", "args": {"query": "three"}},
                        {"id": "call_10", "name": "web_search", "args": {"query": "four"}},
                        {"id": "call_11", "name": "web_search", "args": {"query": "five"}},
                    ],
                ),
            ],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "tools"
    assert result.update["tool_validation_passed"] is True


@pytest.mark.asyncio
async def test_validate_tool_calls_repairs_sendable_file_written_to_source_root() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def _log(state: dict[str, object], event: str, **kwargs: object) -> None:
        del state
        events.append((event, kwargs))

    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={"tulpa_write_file": ("path", "content")},
        forbidden_tool_args={},
        log=_log,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    result = await node(
        {
            "messages": [
                HumanMessage(content="Write the chipmunk URL to a file and send it to me."),
                AIMessage(
                    content="I'll save it.",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "tulpa_write_file",
                            "args": {
                                "path": "src/opentulpa/skills/chipmunk_url.txt",
                                "content": (
                                    "Chipmunk photo URL: "
                                    "https://images.unsplash.com/photo-1425082661507-d6d2f66e4044?w=800"
                                ),
                            },
                        }
                    ],
                ),
            ],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    assert result.update["tool_validation_passed"] is False
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert "non-Python deliverables and artifacts" in str(update_messages[0].content)
    assert "tulpa_stuff" in str(update_messages[0].content)
    assert isinstance(update_messages[1], SystemMessage)
    assert "requested tool action was not completed yet" in str(update_messages[1].content)
    assert "Do not claim success" in str(update_messages[1].content)
    assert any(event == "graph.validate_tools.failed" for event, _ in events)


@pytest.mark.asyncio
async def test_validate_tool_calls_repairs_file_send_outside_tulpa_stuff() -> None:
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={"tulpa_file_send": ("path",)},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    result = await node(
        {
            "messages": [
                HumanMessage(content="Send me the chipmunk URL file."),
                AIMessage(
                    content="Sending.",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "tulpa_file_send",
                            "args": {"path": "src/opentulpa/skills/chipmunk_url.txt"},
                        }
                    ],
                ),
            ],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert "tulpa_file_send can only send files under" in str(update_messages[0].content)
    assert isinstance(update_messages[1], SystemMessage)
    assert "Do not claim success" in str(update_messages[1].content)


@pytest.mark.asyncio
async def test_validate_tool_calls_repairs_grouped_write_to_source_root() -> None:
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    result = await node(
        {
            "messages": [
                HumanMessage(content="Write the chipmunk URL to a file and send it to me."),
                AIMessage(
                    content="I'll save it.",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "tool_group_exec",
                            "args": {
                                "group": "workspace",
                                "command": "tulpa_write_file",
                                "args_json": {
                                    "path": "src/opentulpa/skills/chipmunk_url.txt",
                                    "content": (
                                        "Chipmunk photo URL: "
                                        "https://images.unsplash.com/photo-1425082661507-d6d2f66e4044?w=800"
                                    ),
                                },
                            },
                        }
                    ],
                ),
            ],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    assert result.update["tool_validation_passed"] is False
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert "non-Python deliverables and artifacts" in str(update_messages[0].content)
    assert "Nested tool_group_exec command `tulpa_write_file`" in str(update_messages[0].content)
    assert isinstance(update_messages[1], SystemMessage)
    assert "Do not claim success" in str(update_messages[1].content)


@pytest.mark.asyncio
async def test_validate_tool_calls_repairs_grouped_file_send_outside_tulpa_stuff() -> None:
    node = build_validate_tool_calls_node(
        runtime=object(),
        required_args={},
        forbidden_tool_args={},
        log=lambda state, event, **kwargs: None,
        loop_limit_near=lambda state: False,
        remaining_steps=lambda state: 10,
    )

    result = await node(
        {
            "messages": [
                HumanMessage(content="Send me the chipmunk URL file."),
                AIMessage(
                    content="Sending.",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "tool_group_exec",
                            "args": {
                                "group": "files",
                                "command": "tulpa_file_send",
                                "args_json": '{"path": "src/opentulpa/skills/chipmunk_url.txt"}',
                            },
                        }
                    ],
                ),
            ],
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert "tulpa_file_send can only send files under" in str(update_messages[0].content)
    assert "Nested tool_group_exec command `tulpa_file_send`" in str(update_messages[0].content)
    assert isinstance(update_messages[1], SystemMessage)
    assert "Do not claim success" in str(update_messages[1].content)


@pytest.mark.asyncio
async def test_routine_intent_classifier_rejects_chat_only_request() -> None:
    runtime = _RoutineIntentRuntime(
        {
            "ok": True,
            "allow_create": False,
            "confidence": 0.96,
            "reason": "User explicitly asked to keep planning in chat and not create a routine.",
        }
    )
    args = {
        "name": "Draft Post",
        "schedule": "0 9 * * *",
        "instruction": "Prepare the post and report back.",
        "implementation_command": "python3 prepare_post.py",
    }
    err = _validate_model_tool_call(
        call_name="routine_create",
        args=args,
        latest_user_text="Think it through with me here first. Do not create a routine yet.",
        turn_mode="interactive",
        required_args={"routine_create": ("name", "schedule", "instruction", "implementation_command")},
        forbidden_tool_args={"routine_create": {"customer_id", "message"}},
    )
    assert err is None
    intent_err = await _routine_create_intent_validation_error(
        runtime,
        args=args,
        latest_user_text="Think it through with me here first. Do not create a routine yet.",
        prior_assistant_text="",
        turn_mode="interactive",
    )
    assert intent_err is not None
    assert "ACTION_CLARIFICATION_REQUIRED" in intent_err


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


def test_validate_model_tool_call_rejects_file_send_outside_tulpa_stuff() -> None:
    err = _validate_model_tool_call(
        call_name="tulpa_file_send",
        args={
            "path": "src/opentulpa/skills/chipmunk_url.txt",
        },
        latest_user_text="send me the file",
        turn_mode="interactive",
        required_args={"tulpa_file_send": ("path",)},
        forbidden_tool_args={},
    )
    assert err is not None
    assert "tulpa_file_send can only send files under" in err
    assert "tulpa_stuff" in err


def test_validate_model_tool_call_rejects_deliverable_write_under_source_root() -> None:
    err = _validate_model_tool_call(
        call_name="tulpa_write_file",
        args={
            "path": "src/opentulpa/skills/chipmunk_url.txt",
            "content": "Chipmunk photo URL: https://images.unsplash.com/photo-1425082661507-d6d2f66e4044?w=800",
        },
        latest_user_text="write the chipmunk url to a file and send it",
        turn_mode="interactive",
        required_args={"tulpa_write_file": ("path", "content")},
        forbidden_tool_args={},
    )
    assert err is not None
    assert "non-Python deliverables and artifacts" in err
    assert "src/opentulpa/skills" in err


def test_validate_model_tool_call_rejects_traversal_deliverable_write_under_source_root() -> None:
    err = _validate_model_tool_call(
        call_name="tulpa_write_file",
        args={
            "path": "tulpa_stuff/../src/opentulpa/skills/chipmunk_url.txt",
            "content": "Chipmunk photo URL: https://example.com/chipmunk.jpg",
        },
        latest_user_text="write the chipmunk url to a file and send it",
        turn_mode="interactive",
        required_args={"tulpa_write_file": ("path", "content")},
        forbidden_tool_args={},
    )
    assert err is not None
    assert "non-Python deliverables and artifacts" in err
    assert "src/opentulpa/skills" in err


def test_validate_model_tool_call_allows_python_source_write_under_source_root() -> None:
    err = _validate_model_tool_call(
        call_name="tulpa_write_file",
        args={
            "path": "src/opentulpa/skills/example_skill.py",
            "content": "def run() -> None:\n    return None\n",
        },
        latest_user_text="patch the skill implementation",
        turn_mode="interactive",
        required_args={"tulpa_write_file": ("path", "content")},
        forbidden_tool_args={},
    )
    assert err is None


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


def test_validate_model_tool_call_rejects_owner_update_during_routine_wake() -> None:
    err = _validate_model_tool_call(
        call_name="send_owner_update",
        args={"message": "Still working."},
        latest_user_text="System update: a scheduled routine fired.",
        turn_mode="routine_wake",
        required_args={"send_owner_update": ("message",)},
        forbidden_tool_args={},
    )
    assert err is not None
    assert "send_owner_update is only for live owner/support turns" in err
    assert "routine_wake" in err


def test_validate_model_tool_call_rejects_browser_owner_input_during_routine_wake() -> None:
    err = _validate_model_tool_call(
        call_name="browser_use_owner_input_submit",
        args={"task_id": "task_1", "owner_input": "123456"},
        latest_user_text="System update: a scheduled routine fired.",
        turn_mode="routine_wake",
        required_args={"browser_use_owner_input_submit": ("task_id", "owner_input")},
        forbidden_tool_args={},
    )
    assert err is not None
    assert "browser_use_owner_input_submit is only for live owner/support chat turns" in err
    assert "routine_wake" in err


def test_validate_model_tool_call_allows_browser_owner_input_during_interactive_turn() -> None:
    err = _validate_model_tool_call(
        call_name="browser_use_owner_input_submit",
        args={"task_id": "task_1", "owner_input": "123456"},
        latest_user_text="123456",
        turn_mode="interactive",
        required_args={"browser_use_owner_input_submit": ("task_id", "owner_input")},
        forbidden_tool_args={},
    )
    assert err is None


def test_validate_model_tool_call_allows_owner_update_during_workflow_setup() -> None:
    err = _validate_model_tool_call(
        call_name="send_owner_update",
        args={"message": "Still working."},
        latest_user_text="Let's build the intake workflow.",
        turn_mode="workflow_setup",
        required_args={"send_owner_update": ("message",)},
        forbidden_tool_args={},
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


def test_collapse_completed_tool_segments_keeps_provider_prompt_valid() -> None:
    messages = [
        HumanMessage(content="set up workflow"),
        AIMessage(
            content="Checking setup.",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "tool_group_exec",
                    "args": {"group": "intake", "command": "intake_workflow_setup_begin"},
                },
                {
                    "id": "call_2",
                    "name": "tool_group_exec",
                    "args": {"group": "knowledge", "command": "business_knowledge_index"},
                },
            ],
        ),
        ToolMessage(content='{"ok": true, "session_id": "iwsetup_1"}', tool_call_id="call_1"),
        ToolMessage(content='{"ok": true, "sources": 1}', tool_call_id="call_2"),
        HumanMessage(content="confirmed, save it"),
    ]

    collapsed = collapse_completed_tool_call_segments_for_model(messages)

    assert len(collapsed) == 3
    assert isinstance(collapsed[0], HumanMessage)
    assert isinstance(collapsed[1], SystemMessage)
    assert isinstance(collapsed[2], HumanMessage)
    summary = str(collapsed[1].content)
    assert "VERIFIED_TOOL_RESULTS" in summary
    assert "Checking setup." in summary
    assert "intake_workflow_setup_begin" in summary
    assert "business_knowledge_index" in summary
    assert not any(isinstance(message, ToolMessage) for message in collapsed)
    assert not any(
        isinstance(message, AIMessage) and getattr(message, "tool_calls", None)
        for message in collapsed
    )


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
