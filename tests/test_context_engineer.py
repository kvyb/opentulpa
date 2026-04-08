from __future__ import annotations

from opentulpa.agent.context_engineer import ContextEngineer
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, ToolMessage


def test_context_engineer_keeps_recent_chat_and_tool_windows() -> None:
    engineer = ContextEngineer(raw_chat_limit=20, raw_tool_limit=10)
    messages = []
    for idx in range(24):
        messages.append(HumanMessage(content=f"user {idx}"))
        messages.append(AIMessage(content=f"assistant {idx}"))
    for idx in range(12):
        messages.append(ToolMessage(content=f'{{"status":"ok","result":"tool {idx}"}}', tool_call_id=f"call_{idx}"))

    result = engineer.build_history_working_set(messages, token_budget=10000)

    assert result.raw_chat_count == 20
    assert result.raw_tool_count == 10
    assert len(result.raw_messages) == 30
    assert result.summary_text
    assert "user 0" in result.summary_text or "assistant 0" in result.summary_text


def test_context_engineer_summarizes_stale_tool_arguments_and_errors() -> None:
    engineer = ContextEngineer(raw_chat_limit=2, raw_tool_limit=2)
    messages = [
        HumanMessage(content="book 4pm"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_old",
                    "name": "composio_tool_execute",
                    "args": {"tool_slug": "googlesheets", "slot": "4pm", "day": "tomorrow"},
                }
            ],
        ),
        ToolMessage(content='{"status":"error","error":"slot conflict"}', tool_call_id="call_old"),
        HumanMessage(content="latest question"),
        AIMessage(content="latest answer"),
        ToolMessage(content='{"status":"ok","result":"recent"}', tool_call_id="call_new"),
        ToolMessage(content='{"status":"ok","result":"recent2"}', tool_call_id="call_new2"),
    ]

    result = engineer.build_history_working_set(messages, token_budget=4000)

    assert result.summary_text
    assert "composio_tool_execute" in result.summary_text
    assert "slot" in result.summary_text
    assert "slot conflict" in result.summary_text


def test_context_engineer_preserves_active_tool_dependency_suffix() -> None:
    engineer = ContextEngineer(raw_chat_limit=2, raw_tool_limit=2)
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="old answer"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_live",
                    "name": "intake_workflow_list",
                    "args": {"customer_id": "telegram_1"},
                }
            ],
        ),
        ToolMessage(content='{"status":"ok","items":[]}', tool_call_id="call_live"),
    ]

    result = engineer.build_history_working_set(messages, token_budget=1000)

    assert result.protected_count == 2
    assert any(isinstance(msg, AIMessage) and bool(getattr(msg, "tool_calls", [])) for msg in result.raw_messages)
    assert any(isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", "") == "call_live" for msg in result.raw_messages)


def test_context_engineer_optional_context_rules() -> None:
    engineer = ContextEngineer()

    assert engineer.should_include_optional_context(kind="thread_rollup", prompt_mode="literal_chat", should_retrieve=True) is False
    assert engineer.should_include_optional_context(kind="thread_rollup", prompt_mode="task_chat", should_retrieve=True) is True
    assert engineer.should_include_optional_context(kind="link_aliases", prompt_mode="task_chat", should_retrieve=False) is False
    assert engineer.should_include_optional_context(kind="pending_context", prompt_mode="execution", should_retrieve=False) is True
