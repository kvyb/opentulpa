"""Graph construction for OpenTulpa runtime."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from opentulpa.agent.context_engine import ContextEngine
from opentulpa.agent.graph_control_tools import (
    execute_graph_control_tool as _execute_graph_control_tool,
)
from opentulpa.agent.graph_control_tools import (
    is_graph_control_tool as _is_graph_control_tool,
)
from opentulpa.agent.graph_nodes.tool_validation import build_validate_tool_calls_node
from opentulpa.agent.lc_messages import (
    AIMessage,
    ToolMessage,
)
from opentulpa.agent.models import AgentState
from opentulpa.agent.tool_execution_policy import ToolExecutionPolicy
from opentulpa.agent.tool_loop_guardrails import tool_action_signatures
from opentulpa.agent.tool_outcome_context import (
    compact_tool_result_for_model as _compact_tool_result_for_model,
)
from opentulpa.agent.tool_outcome_context import (
    next_tool_round_id as _next_tool_round_id,
)
from opentulpa.agent.tool_outcome_finalizers import (
    final_response_hint_from_tool_outcomes as _final_response_hint_from_tool_outcomes,
)
from opentulpa.agent.turn_control import (
    consume_model_budget_for_turn as _consume_model_budget_for_turn,
)
from opentulpa.agent.turn_control import loop_limit_near as _loop_limit_near
from opentulpa.agent.turn_control import record_tool_round_for_turn as _record_tool_round_for_turn
from opentulpa.agent.turn_control import remaining_graph_steps as _remaining_graph_steps
from opentulpa.agent.turn_finalizer import (
    finalize_turn_response as _finalize_turn_response,
)
from opentulpa.agent.turn_policy import (
    normalize_turn_mode as _normalize_turn_mode,
)
from opentulpa.agent.turn_prompt_builder import build_turn_prompt as _build_turn_prompt
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    latest_user_text as _latest_user_text,
)
from opentulpa.agent.utils import (
    safe_json as _safe_json,
)

logger = logging.getLogger(__name__)

def _graph_retry_budget(runtime: Any) -> int:
    try:
        recursion_limit = int(getattr(runtime, "recursion_limit", 30))
    except Exception:
        recursion_limit = 30
    return max(3, min(24, recursion_limit - 6))


def _workflow_setup_no_progress_retry_limit(runtime: Any) -> int:
    return min(2, _graph_retry_budget(runtime))


WORKFLOW_SETUP_TOOL_PROGRESS_TEXT = (
    "I’m setting up the workflow now. I’ll send the proposal or exact blocker when "
    "validation finishes."
)
def _thread_has_active_workflow_setup(
    runtime: Any,
    *,
    customer_id: str,
    thread_id: str,
) -> bool:
    service = getattr(runtime, "workflow_setup_service", None)
    if service is None or not hasattr(service, "get_thread_session"):
        return False
    try:
        session = service.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            include_paused=False,
        )
    except Exception:
        logger.exception(
            "Failed to check workflow setup status (customer_id=%s, thread_id=%s)",
            customer_id,
            thread_id,
        )
        return False
    return str((session or {}).get("status", "") or "").strip().lower() == "active"


def _extract_invoked_skill_snapshot(result: Any, *, requested_name: str) -> tuple[str, str] | None:
    if not isinstance(result, dict):
        return None
    name = str(result.get("name", "")).strip() or str(requested_name or "").strip()
    if not name:
        return None
    description = str(result.get("description", "")).strip()
    scope = str(result.get("scope", "")).strip() or "user"
    skill_markdown = str(result.get("skill_markdown", "")).strip()
    if not skill_markdown:
        instructions = str(result.get("instructions", "")).strip()
        supporting = result.get("supporting_files")
        if instructions:
            skill_markdown = instructions
        elif isinstance(supporting, dict) and supporting:
            skill_markdown = "\n\n".join(
                f"[{key}]\n{str(value).strip()}"
                for key, value in supporting.items()
                if str(key).strip() and str(value).strip()
            ).strip()
    if not skill_markdown:
        return None
    header = [f"Skill: {name}", f"Scope: {scope}"]
    if description:
        header.append(f"Description: {description}")
    content = "\n".join(header) + f"\n\nSKILL.md:\n{skill_markdown[:3500]}"
    return name, content


def build_runtime_graph(runtime: Any):
    assert runtime._model_with_tools is not None
    assert runtime._checkpointer is not None

    required_args: dict[str, tuple[str, ...]] = {
        "send_owner_update": ("message",),
        "tulpa_write_file": ("path", "content"),
        "tulpa_validate_file": ("path",),
        "tulpa_reload": (),
        "tulpa_read_file": ("path",),
        "tulpa_run_terminal": ("command",),
        "fetch_url_content": ("url",),
        "fetch_file_content": ("url",),
        "uploaded_file_search": ("query",),
        "uploaded_file_get": ("file_id",),
        "uploaded_file_send": ("file_id",),
        "tulpa_file_send": ("path",),
        "web_image_send": ("url",),
        "uploaded_file_analyze": ("file_id",),
        "uploaded_file_inspect_structure": ("file_id",),
        "business_knowledge_index": ("file_ids",),
        "business_knowledge_query": ("query",),
        "user_context_add_files": ("file_ids",),
        "user_context_query": ("query",),
        "user_context_list_sources": (),
        "user_context_find_sources": ("query",),
        "user_context_reindex": (),
        "user_context_archive_sources": ("file_ids",),
        "user_context_promote_to_intake": ("workflow_id", "file_ids"),
        "skill_get": ("name",),
        "skill_upsert": ("name", "description", "instructions"),
        "skill_delete": ("name",),
        "intake_workflow_upsert": (
            "name",
            "intent_description",
            "required_fields",
            "sink_type",
            "sink_config",
        ),
        "intake_workflow_list": (),
        "intake_workflow_get": ("workflow_id",),
        "intake_workflow_delete": ("workflow_id",),
        "intake_workflow_setup_begin": ("mode",),
        "intake_workflow_setup_get": (),
        "intake_workflow_setup_update": (),
        "intake_workflow_setup_preflight": (),
        "intake_workflow_setup_propose_current": (),
        "intake_workflow_setup_mark_proposed": (),
        "intake_workflow_setup_confirm_current": (),
        "intake_workflow_setup_commit": (),
        "intake_workflow_setup_finalize_confirmation": (),
        "intake_workflow_setup_pause": (),
        "intake_workflow_setup_cancel": (),
        "intake_workflow_run": ("workflow_id",),
        "telegram_business_status": (),
        "composio_status": (),
        "composio_authorize_toolkit": ("toolkit",),
        "composio_wait_for_connection": ("connection_id",),
        "composio_toolkits": (),
        "composio_connected_accounts": (),
        "composio_disable_connected_account": ("connected_account_id",),
        "composio_delete_connected_account": ("connected_account_id",),
        "composio_tool_search": (),
        "composio_tool_schema": ("tool_slug",),
        "composio_instagram_reply_precheck": (),
        "composio_tool_execute": ("tool_slug",),
        "directive_set": ("directive",),
        "time_profile_set": ("utc_offset",),
        "browser_use_session_list": (),
        "browser_use_run": ("task",),
        "browser_use_task_get": ("task_id",),
        "browser_use_task_screenshot": ("task_id",),
        "browser_use_task_control": ("task_id",),
        "browser_use_owner_input_submit": ("task_id", "owner_input"),
        "routine_list": (),
        "routine_create": (
            "name",
            "schedule",
            "instruction",
            "implementation_command",
        ),
        "routine_delete": ("routine_id",),
    }
    customer_scoped_tools: set[str] = {
        "send_owner_update",
        "memory_search",
        "memory_add",
        "uploaded_file_search",
        "uploaded_file_get",
        "uploaded_file_send",
        "tulpa_file_send",
        "web_image_send",
        "uploaded_file_analyze",
        "uploaded_file_inspect_structure",
        "business_knowledge_index",
        "business_knowledge_query",
        "user_context_add_files",
        "user_context_query",
        "user_context_list_sources",
        "user_context_find_sources",
        "user_context_reindex",
        "user_context_archive_sources",
        "user_context_promote_to_intake",
        "skill_list",
        "skill_get",
        "skill_upsert",
        "skill_delete",
        "intake_workflow_upsert",
        "intake_workflow_list",
        "intake_workflow_get",
        "intake_workflow_delete",
        "intake_workflow_setup_begin",
        "intake_workflow_setup_get",
        "intake_workflow_setup_update",
        "intake_workflow_setup_preflight",
        "intake_workflow_setup_propose_current",
        "intake_workflow_setup_mark_proposed",
        "intake_workflow_setup_confirm_current",
        "intake_workflow_setup_commit",
        "intake_workflow_setup_finalize_confirmation",
        "intake_workflow_setup_pause",
        "intake_workflow_setup_cancel",
        "intake_workflow_run",
        "telegram_business_status",
        "composio_authorize_toolkit",
        "composio_toolkits",
        "composio_connected_accounts",
        "composio_disable_connected_account",
        "composio_delete_connected_account",
        "composio_tool_search",
        "composio_tool_schema",
        "composio_instagram_reply_precheck",
        "composio_tool_execute",
        "directive_get",
        "directive_set",
        "directive_clear",
        "time_profile_get",
        "time_profile_set",
        "tulpa_run_terminal",
        "routine_list",
        "routine_create",
        "routine_delete",
        "browser_use_run",
        "browser_use_task_get",
        "browser_use_task_screenshot",
        "browser_use_task_control",
        "browser_use_owner_input_submit",
    }
    forbidden_tool_args: dict[str, set[str]] = {
        name: {"customer_id"} for name in customer_scoped_tools
    }
    forbidden_tool_args["routine_create"] = {"customer_id", "message"}

    context_engine = getattr(runtime, "_context_engine", None)
    if not isinstance(context_engine, ContextEngine):
        context_engine = ContextEngine()

    def _log(state: AgentState | None, event: str, **fields: Any) -> None:
        log_event = getattr(runtime, "log_behavior_event", None)
        if not callable(log_event):
            return
        payload: dict[str, Any] = {}
        if isinstance(state, dict):
            trace_id = str(state.get("agent_trace_id", "")).strip()
            thread_id = str(state.get("thread_id", "")).strip()
            customer_id = str(state.get("customer_id", "")).strip()
            if trace_id:
                payload["trace_id"] = trace_id
            if thread_id:
                payload["thread_id"] = thread_id
            if customer_id:
                payload["customer_id"] = customer_id
        payload.update(fields)
        log_event(event=event, **payload)

    async def _emit_tool_call_preamble_update(
        state: AgentState,
        *,
        message: AIMessage,
        turn_mode: str,
    ) -> bool:
        if turn_mode not in {"interactive", "workflow_setup"}:
            return False
        text = _content_to_text(getattr(message, "content", "")).strip()
        tool_calls = getattr(message, "tool_calls", []) or []
        tool_names = [
            str(call.get("name", "")).strip()
            for call in tool_calls
            if isinstance(call, dict) and str(call.get("name", "")).strip()
        ]
        if "send_owner_update" in tool_names:
            _log(
                state,
                "graph.tools.preamble_update",
                sent=False,
                reason="send_owner_update_tool_present",
                chars=len(text),
                turn_mode=turn_mode,
            )
            return False
        if not text:
            intake_setup_call = any(
                name.startswith("intake_workflow_setup_")
                or (
                    name == "tool_group_exec"
                    and isinstance(call, dict)
                    and isinstance(call.get("args"), dict)
                    and str(call["args"].get("group", "")).strip() == "intake"
                )
                for call in tool_calls
                for name in [str(call.get("name", "")).strip()]
                if isinstance(call, dict)
            )
            if turn_mode != "workflow_setup" or not intake_setup_call:
                return False
            text = WORKFLOW_SETUP_TOOL_PROGRESS_TEXT
        emitter = getattr(runtime, "emit_interactive_update", None)
        if not callable(emitter):
            _log(
                state,
                "graph.tools.preamble_update",
                sent=False,
                reason="missing_interactive_emitter",
                chars=len(text),
                turn_mode=turn_mode,
            )
            return False
        max_chars = 1200
        visible_text = text if len(text) <= max_chars else f"{text[: max_chars - 3].rstrip()}..."
        tool_call_ids = [
            str(call.get("id", "")).strip()
            for call in tool_calls
            if isinstance(call, dict) and str(call.get("id", "")).strip()
        ]
        dedupe_source = "|".join(
            [
                str(state.get("agent_trace_id", "")).strip(),
                str(state.get("thread_id", "")).strip(),
                ",".join(tool_call_ids),
                visible_text,
            ]
        )
        dedupe_key = (
            "tool_call_preamble:" + hashlib.sha256(dedupe_source.encode("utf-8")).hexdigest()[:32]
        )
        try:
            result = await emitter(
                text=visible_text,
                dedupe_key=dedupe_key,
                thread_id=str(state.get("thread_id", "")).strip() or None,
            )
            _log(
                state,
                "graph.tools.preamble_update",
                sent=bool(isinstance(result, dict) and result.get("sent")),
                duplicate=bool(isinstance(result, dict) and result.get("duplicate")),
                chars=len(visible_text),
                turn_mode=turn_mode,
                tool_names=tool_names[:5],
            )
            return bool(isinstance(result, dict) and result.get("sent"))
        except Exception as exc:
            _log(
                state,
                "graph.tools.preamble_update",
                sent=False,
                reason="emit_failed",
                error=str(exc)[:500],
                chars=len(visible_text),
                turn_mode=turn_mode,
            )
            return False

    async def agent_node(
        state: AgentState,
        config=None,
    ) -> Command[Literal["agent", "validate_tools", "finalize_turn"]]:
        customer_id = state.get("customer_id", "")
        thread_id = state.get("thread_id", "")
        turn_mode = _normalize_turn_mode(state.get("turn_mode"))
        prompt_mode = str(state.get("prompt_mode", "task_chat")).strip().lower() or "task_chat"
        prompt_context_update: dict[str, Any] = {}
        if turn_mode == "interactive" and _thread_has_active_workflow_setup(
            runtime,
            customer_id=customer_id,
            thread_id=thread_id,
        ):
            turn_mode = "workflow_setup"
            prompt_mode = "workflow_setup"
            prompt_context_update["turn_mode"] = turn_mode
            prompt_context_update["prompt_mode"] = prompt_mode
            _log(
                state,
                "graph.workflow_setup.promoted_turn_mode",
                turn_mode=turn_mode,
            )
        messages = state.get("messages", [])
        budget_result = _consume_model_budget_for_turn(
            runtime=runtime,
            config=config,
            state=state,
            turn_mode=turn_mode,
        )
        budget_decision = budget_result.decision
        prompt_context_update.update(budget_result.update)
        if not budget_decision.allowed:
            _log(
                state,
                "graph.turn_budget.exhausted",
                reason=budget_decision.reason,
                turn_mode=turn_mode,
                used_model_calls=budget_decision.state.get("used_model_calls"),
                max_model_calls=budget_decision.state.get("max_model_calls"),
            )
            return Command(
                update={
                    **prompt_context_update,
                    "turn_status": "running",
                    "turn_finalization_reason": budget_decision.reason,
                },
                goto="finalize_turn",
            )
        live_user_steering = [
            str(item).strip()
            for item in (state.get("live_user_steering") or [])
            if str(item).strip()
        ]
        new_steering: list[str] = []
        if turn_mode == "interactive":
            drain_fragments = getattr(runtime, "drain_interactive_fragments", None)
            if callable(drain_fragments):
                drained = await drain_fragments(thread_id=thread_id)
                new_steering = [
                    str(fragment).strip() for fragment in drained if str(fragment).strip()
                ]
                if new_steering:
                    live_user_steering = [*live_user_steering, *new_steering][-8:]
        latest_user = _latest_user_text(messages)
        _log(
            state,
            "graph.agent.start",
            message_count=len(messages),
            latest_user_chars=len(latest_user),
            turn_mode=turn_mode,
            injected_user_messages=len(new_steering),
        )
        turn_prompt = await _build_turn_prompt(
            runtime=runtime,
            state=state,
            customer_id=customer_id,
            thread_id=thread_id,
            turn_mode=turn_mode,
            prompt_mode=prompt_mode,
            base_prompt_context_update=prompt_context_update,
            live_user_steering=live_user_steering,
            context_engine=context_engine,
            context_provider=runtime.context_source_provider,
            loop_limit_near=_loop_limit_near,
        )
        prompt_context_update = turn_prompt.prompt_context_update
        _log(state, "graph.agent.prompt_ready", **turn_prompt.prompt_ready_log_fields)
        model_messages = turn_prompt.model_messages
        stable_prefix_count = turn_prompt.stable_prefix_count
        cacheable_prefix_count = turn_prompt.cacheable_prefix_count
        call_context = turn_prompt.call_context
        model_with_tools = runtime.model_with_tools_for_turn_mode(turn_mode)
        assert model_with_tools is not None
        stream_model_calls = bool(state.get("stream_model_calls"))
        astream_fn = getattr(runtime, "astream_model", None)
        ainvoke_fn = getattr(runtime, "ainvoke_model", None)
        if stream_model_calls and callable(astream_fn):
            response = await astream_fn(
                model_with_tools,
                model_messages,
                stable_prefix_count=stable_prefix_count,
                cacheable_prefix_count=cacheable_prefix_count,
                call_context=call_context,
                stream_config=config,
            )
        elif callable(ainvoke_fn):
            response = await ainvoke_fn(
                model_with_tools,
                model_messages,
                stable_prefix_count=stable_prefix_count,
                cacheable_prefix_count=cacheable_prefix_count,
                call_context=call_context,
            )
        else:
            response = await model_with_tools.ainvoke(model_messages)
        response_text = _content_to_text(getattr(response, "content", ""))
        usage_fields: dict[str, Any] = {}
        usage_fields_fn = getattr(runtime, "extract_response_usage_fields", None)
        if callable(usage_fields_fn):
            try:
                usage_fields = dict(usage_fields_fn(response))
            except Exception:
                usage_fields = {}
        _log(
            state,
            "graph.agent.response",
            response_chars=len(response_text.strip()),
            tool_call_count=len(getattr(response, "tool_calls", []) or []),
            turn_mode=turn_mode,
            **usage_fields,
        )
        update: dict[str, Any] = {
            "messages": [response],
            "turn_status": "running",
            "workflow_setup_repair_instruction": "",
            "live_user_steering": turn_prompt.live_user_steering,
            **prompt_context_update,
            **turn_prompt.skill_state_update,
        }
        has_tool_calls = isinstance(response, AIMessage) and bool(getattr(response, "tool_calls", []))
        goto: Literal["validate_tools", "finalize_turn"] = (
            "validate_tools" if has_tool_calls else "finalize_turn"
        )
        if (
            turn_mode == "workflow_setup"
            and isinstance(response, AIMessage)
            and not bool(getattr(response, "tool_calls", []))
            and not response_text.strip()
        ):
            retry_count = int(state.get("workflow_setup_no_progress_retry_count", 0))
            retry_limit = _workflow_setup_no_progress_retry_limit(runtime)
            if retry_count < retry_limit:
                _log(
                    state,
                    "graph.workflow_setup.no_progress_retry",
                    retry_count=retry_count,
                    retry_limit=retry_limit,
                    turn_mode=turn_mode,
                )
                update["messages"] = [
                    response,
                ]
                update["workflow_setup_repair_instruction"] = (
                    "WORKFLOW_SETUP_NO_PROGRESS: Your previous workflow setup response "
                    "had no visible answer and no setup tool calls. Continue this same owner turn now.\n"
                    "- If the latest owner message supplied workflow facts, sink details, files, fields, "
                    "or behavior rules: call intake_workflow_setup_get if needed, then "
                    "intake_workflow_setup_update to persist the new facts.\n"
                    "- If the latest owner message supplied a local CSV path, persist it as "
                    "draft_patch.sink_type='local_csv' and draft_patch.sink_config.file_path; "
                    "do not ask for sink details already present.\n"
                    "- If the draft is complete after the update: call "
                    "intake_workflow_setup_propose_current once, then summarize the returned "
                    "draft/preflight as the proposal.\n"
                    "- If the latest owner message explicitly confirms a shown proposal: call "
                    "intake_workflow_setup_finalize_confirmation. Pass any small final behavior-rule "
                    "edits in that same tool call when needed instead of doing a separate "
                    "update/preflight loop.\n"
                    "- Do not repeat an older proposal or ask for details already present in the latest "
                    "owner message. If blocked, give the one concrete setup-tool error or follow-up."
                )
                update["workflow_setup_no_progress_retry_count"] = retry_count + 1
                return Command(update=update, goto="agent")
        return Command(update=update, goto=goto)

    validate_tool_calls_node = build_validate_tool_calls_node(
        runtime=runtime,
        required_args=required_args,
        forbidden_tool_args=forbidden_tool_args,
        log=_log,
        loop_limit_near=_loop_limit_near,
        remaining_steps=_remaining_graph_steps,
    )

    async def tools_node(
        state: AgentState,
    ) -> Command[Literal["agent", "finalize_turn", "__end__"]]:
        messages = state.get("messages", [])
        if not messages:
            return Command(update={"turn_status": "running"}, goto="agent")
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return Command(update={"turn_status": "running"}, goto="agent")

        tool_policy = ToolExecutionPolicy.from_runtime_state(runtime=runtime, state=state)
        customer_id = tool_policy.customer_id
        thread_id = tool_policy.thread_id
        turn_mode = tool_policy.turn_mode
        execution_origin = tool_policy.execution_origin
        _log(
            state,
            "graph.tools.start",
            requested_tool_calls=len(last.tool_calls),
            execution_origin=execution_origin,
            turn_mode=turn_mode,
        )
        await _emit_tool_call_preamble_update(state, message=last, turn_mode=turn_mode)

        tool_messages: list[ToolMessage] = []
        tool_outcomes: list[dict[str, Any]] = []
        tool_round_id = _next_tool_round_id(state.get("tool_outcomes"))
        had_error = False
        failed_tool_names: list[str] = []
        failed_tool_errors: list[str] = []
        graph_control_state: dict[str, Any] = {"turn_plan": state.get("turn_plan")}
        graph_control_updates: dict[str, Any] = {}
        invoked_skill_names = state.get("active_invoked_skill_names", []) or []
        invoked_skill_list = (
            [str(n).strip() for n in invoked_skill_names if str(n).strip()]
            if isinstance(invoked_skill_names, list)
            else []
        )
        invoked_skill_context = (
            str(state.get("active_invoked_skill_context", "")).strip()
            or str(state.get("active_skill_context", "")).strip()
        )
        for call in last.tool_calls:
            call_name = str(call.get("name", ""))
            call_id = str(call.get("id", ""))
            args = call.get("args", {}) or {}
            original_args = args
            tool_signatures = tool_action_signatures(call_name, original_args)
            primary_tool_signature = tool_signatures[0].key if tool_signatures else ""
            try:
                tool_fn = runtime._tools.get(call_name)
                if tool_fn is None:
                    raise ValueError(f"Unknown tool: {call_name}")
                tool_policy.validate_call(
                    call_name=call_name,
                    customer_scoped_tools=customer_scoped_tools,
                )
                args = tool_policy.prepare_args(
                    call_name=call_name,
                    args=args,
                    messages=messages,
                )
                if _is_graph_control_tool(call_name):
                    control_result = _execute_graph_control_tool(
                        tool_name=call_name,
                        args=args,
                        state={**state, **graph_control_state},
                    )
                    graph_control_state.update(control_result.state_update)
                    graph_control_updates.update(control_result.state_update)
                    result = control_result.result
                else:
                    args = runtime.resolve_link_aliases_in_args(customer_id=customer_id, args=args)
                    scope_token = None
                    thread_scope_token = None
                    turn_mode_scope_token = None
                    set_customer_scope = getattr(runtime, "set_active_customer_id", None)
                    if callable(set_customer_scope):
                        scope_token = set_customer_scope(customer_id)
                    set_thread_scope = getattr(runtime, "set_active_thread_id", None)
                    if callable(set_thread_scope):
                        thread_scope_token = set_thread_scope(thread_id)
                    set_turn_mode_scope = getattr(runtime, "set_active_turn_mode", None)
                    if callable(set_turn_mode_scope):
                        turn_mode_scope_token = set_turn_mode_scope(turn_mode)
                    tool_span = None
                    span_factory = getattr(
                        getattr(runtime, "_langfuse_tracer", None), "tool_span", None
                    )
                    if callable(span_factory) and not bool(
                        state.get("langfuse_graph_callback_attached")
                    ):
                        tool_span = span_factory(
                            trace_id=str(state.get("agent_trace_id", "")).strip() or None,
                            tool_name=call_name,
                            tool_call_id=call_id,
                            args=args,
                            metadata={
                                "thread_id": thread_id,
                                "customer_id": customer_id,
                                "turn_mode": turn_mode,
                                "execution_origin": execution_origin,
                            },
                        )
                    try:
                        if tool_span is None:
                            result = await tool_fn.ainvoke(args)
                        else:
                            with tool_span:
                                result = await tool_fn.ainvoke(args)
                                tool_span.set_result(result, status="ok")
                    finally:
                        reset_turn_mode_scope = getattr(runtime, "reset_active_turn_mode", None)
                        if turn_mode_scope_token is not None and callable(reset_turn_mode_scope):
                            reset_turn_mode_scope(turn_mode_scope_token)
                        reset_thread_scope = getattr(runtime, "reset_active_thread_id", None)
                        if thread_scope_token is not None and callable(reset_thread_scope):
                            reset_thread_scope(thread_scope_token)
                        reset_customer_scope = getattr(runtime, "reset_active_customer_id", None)
                        if scope_token is not None and callable(reset_customer_scope):
                            reset_customer_scope(scope_token)
                runtime.register_links_from_text(
                    customer_id=customer_id,
                    text=_safe_json(result),
                    source=f"tool:{call_name}",
                    limit=40,
                )
                result_text = _safe_json(result)
                model_visible_result_text = _compact_tool_result_for_model(
                    tool_name=call_name,
                    result=result,
                )
                final_response_hint = _final_response_hint_from_tool_outcomes(
                    [{"status": "ok", "result_text": result_text}]
                )
                _log(
                    state,
                    "graph.tools.success",
                    tool_round_id=tool_round_id,
                    tool_name=call_name,
                    tool_call_id=call_id,
                    result_chars=len(result_text),
                    model_visible_result_chars=len(model_visible_result_text),
                    tool_result_compressed=model_visible_result_text != result_text,
                )
                tool_messages.append(
                    ToolMessage(
                        content=model_visible_result_text,
                        tool_call_id=call_id,
                        additional_kwargs={"opentulpa_control": {"status": "ok"}},
                    )
                )
                tool_outcomes.append(
                    {
                        "round_id": tool_round_id,
                        "tool_name": call_name,
                        "tool_call_id": call_id,
                        "status": "ok",
                        "result_text": model_visible_result_text,
                        "final_response_hint": final_response_hint,
                        "tool_signature": primary_tool_signature,
                        "tool_signatures": [signature.key for signature in tool_signatures],
                        "trace_id": str(state.get("agent_trace_id", "") or "").strip(),
                    }
                )
                if call_name == "skill_get":
                    requested_name = str(args.get("name", "")).strip()
                    snapshot = _extract_invoked_skill_snapshot(
                        result, requested_name=requested_name
                    )
                    if snapshot is not None:
                        skill_name, skill_text = snapshot
                        merged_names = [*invoked_skill_list]
                        if skill_name not in merged_names:
                            merged_names.append(skill_name)
                        invoked_skill_list = merged_names[-3:]
                        if invoked_skill_context:
                            invoked_skill_context = (
                                f"{invoked_skill_context}\n\n---\n\n{skill_text}"
                            )
                        else:
                            invoked_skill_context = skill_text
            except Exception as exc:
                had_error = True
                error_text = f"TOOL_ERROR: {call_name} failed: {exc}"
                failed_tool_names.append(call_name)
                failed_tool_errors.append(str(exc).strip())
                _log(
                    state,
                    "graph.tools.error",
                    tool_round_id=tool_round_id,
                    tool_name=call_name,
                    tool_call_id=call_id,
                    error=str(exc)[:500],
                )
                tool_messages.append(
                    ToolMessage(
                        content=error_text,
                        tool_call_id=call_id,
                        additional_kwargs={
                            "opentulpa_control": {
                                "status": "error",
                                "error": str(exc)[:500],
                            }
                        },
                    )
                )
                tool_outcomes.append(
                    {
                        "round_id": tool_round_id,
                        "tool_name": call_name,
                        "tool_call_id": call_id,
                        "status": "error",
                        "error": str(exc)[:500],
                        "result_text": error_text,
                    }
                )
        update: dict[str, Any] = {
            "messages": tool_messages,
            "tool_outcomes": tool_outcomes,
            "turn_status": "running",
            "turn_budget": _record_tool_round_for_turn(
                runtime=runtime,
                state=state,
                turn_mode=turn_mode,
                tool_calls=list(last.tool_calls),
            ),
            "active_invoked_skill_names": invoked_skill_list,
            "active_invoked_skill_context": invoked_skill_context,
            "active_skill_context": invoked_skill_context,
        }
        if had_error:
            next_tool_error_count = int(state.get("tool_error_count", 0)) + 1
            last_tool_error = next(
                (item for item in reversed(failed_tool_errors) if item),
                "tool execution failed",
            )
            update["tool_error_count"] = next_tool_error_count
            update["last_tool_error"] = last_tool_error
            if (
                turn_mode == "routine_wake"
                and next_tool_error_count >= 2
                and "composio_tool_execute" in failed_tool_names
            ):
                failure_summary = (
                    "AUTOMATION_EXECUTION_FAILED: repeated composio_tool_execute errors during "
                    f"wake execution. Latest error: {last_tool_error[:500]}"
                )
                _log(
                    state,
                    "graph.tools.abort_after_repeated_error",
                    tool_name="composio_tool_execute",
                    tool_error_count=next_tool_error_count,
                    error=last_tool_error[:500],
                    turn_mode=turn_mode,
                )
                update["messages"] = [
                    *tool_messages,
                    AIMessage(content=failure_summary),
                ]
                update["turn_status"] = "failed"
                return Command(update=update, goto="finalize_turn")
        update.update(graph_control_updates)
        _log(
            state,
            "graph.tools.complete",
            emitted_messages=len(tool_messages),
            had_error=had_error,
        )
        return Command(update=update, goto="agent")

    async def finalize_turn_node(state: AgentState) -> dict[str, Any]:
        return await _finalize_turn_response(runtime=runtime, state=state)

    builder = StateGraph(AgentState)
    builder.add_node(
        "agent",
        agent_node,
        retry_policy=RetryPolicy(max_attempts=3),
        destinations=("agent", "validate_tools", "finalize_turn"),
    )
    builder.add_node(
        "validate_tools",
        cast(Any, validate_tool_calls_node),
        retry_policy=RetryPolicy(max_attempts=2),
        destinations=("tools", "agent"),
    )
    builder.add_node(
        "tools",
        tools_node,
        retry_policy=RetryPolicy(max_attempts=3),
        destinations=("agent", END),
    )
    builder.add_node("finalize_turn", finalize_turn_node, retry_policy=RetryPolicy(max_attempts=1))
    builder.add_edge(START, "agent")
    return builder.compile(checkpointer=runtime._checkpointer)
