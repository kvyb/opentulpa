from __future__ import annotations

import pytest

from opentulpa.agent.graph_nodes.tool_validation import build_validate_tool_calls_node
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.tool_loop_guardrails import (
    duplicate_tool_error,
    find_duplicate_tool_calls,
    tool_action_signature,
    tool_action_signatures,
)


def test_tool_group_exec_signature_normalizes_args_json_string() -> None:
    first = tool_action_signature(
        "tool_group_exec",
        {
            "group": "Knowledge",
            "command": "user_context_add_files",
            "args_json": '{"file_ids":["file_1"],"scope":"chat"}',
        },
    )
    second = tool_action_signature(
        "tool_group_exec",
        {
            "group": "knowledge",
            "command": "user_context_add_files",
            "args_json": {"scope": "chat", "file_ids": ["file_1"]},
        },
    )

    assert first is not None
    assert second is not None
    assert first.key == second.key
    assert "user_context_add_files" in first.label


def test_duplicate_guardrail_blocks_same_request_duplicate() -> None:
    duplicates = find_duplicate_tool_calls(
        requested_calls=[
            {
                "id": "call_1",
                "name": "tool_group_exec",
                "args": {
                    "group": "memory",
                    "command": "memory_add",
                    "args_json": {"summary": "User prefers concise replies"},
                },
            },
            {
                "id": "call_2",
                "name": "tool_group_exec",
                "args": {
                    "group": "memory",
                    "command": "memory_add",
                    "args_json": {"summary": "User prefers concise replies"},
                },
            },
        ],
        prior_tool_outcomes=[],
        trace_id="turn_1",
    )

    assert len(duplicates) == 1
    assert duplicates[0].tool_call_id == "call_2"
    assert "DUPLICATE_TOOL_CALL_PREVIOUS_SUCCESS" in duplicates[0].error


def test_tool_group_exec_batch_exposes_nested_action_signatures() -> None:
    signatures = tool_action_signatures(
        "tool_group_exec",
        {
            "calls": [
                {
                    "group": "knowledge",
                    "command": "user_context_add_files",
                    "args_json": {"file_ids": ["file_1"]},
                },
                {
                    "group": "memory",
                    "command": "memory_add",
                    "args_json": {"summary": "User prefers concise replies"},
                },
            ]
        },
    )

    assert len(signatures) == 2
    assert "user_context_add_files" in signatures[0].label
    assert "memory_add" in signatures[1].label


def test_duplicate_guardrail_blocks_nested_batch_repeat_after_prior_batch_success() -> None:
    prior_signatures = tool_action_signatures(
        "tool_group_exec",
        {
            "calls": [
                {
                    "group": "knowledge",
                    "command": "user_context_add_files",
                    "args_json": {"file_ids": ["file_1"]},
                },
                {
                    "group": "memory",
                    "command": "memory_add",
                    "args_json": {"summary": "User prefers concise replies"},
                },
            ]
        },
    )
    assert len(prior_signatures) == 2

    duplicates = find_duplicate_tool_calls(
        requested_calls=[
            {
                "id": "call_2",
                "name": "tool_group_exec",
                "args": {
                    "group": "knowledge",
                    "command": "user_context_add_files",
                    "args_json": {"file_ids": ["file_1"]},
                },
            }
        ],
        prior_tool_outcomes=[
            {
                "status": "ok",
                "tool_signatures": [signature.key for signature in prior_signatures],
                "trace_id": "turn_a",
            }
        ],
        trace_id="turn_a",
    )

    assert len(duplicates) == 1
    assert duplicates[0].tool_call_id == "call_2"
    assert "user_context_add_files" in duplicates[0].error


def test_duplicate_guardrail_ignores_prior_success_from_other_trace() -> None:
    signature = tool_action_signature(
        "tool_group_exec",
        {
            "group": "knowledge",
            "command": "user_context_add_files",
            "args_json": {"file_ids": ["file_1"]},
        },
    )
    assert signature is not None

    duplicates = find_duplicate_tool_calls(
        requested_calls=[
            {
                "id": "call_2",
                "name": "tool_group_exec",
                "args": {
                    "group": "knowledge",
                    "command": "user_context_add_files",
                    "args_json": {"file_ids": ["file_1"]},
                },
            }
        ],
        prior_tool_outcomes=[
            {
                "status": "ok",
                "tool_signature": signature.key,
                "trace_id": "previous_turn",
            }
        ],
        trace_id="current_turn",
    )

    assert duplicates == []


def test_duplicate_guardrail_ignores_non_consecutive_prior_success() -> None:
    repeated = tool_action_signature(
        "tool_group_exec",
        {
            "group": "knowledge",
            "command": "user_context_add_files",
            "args_json": {"file_ids": ["file_1"]},
        },
    )
    intervening = tool_action_signature(
        "tool_group_exec",
        {
            "group": "memory",
            "command": "memory_add",
            "args_json": {"summary": "User prefers concise replies"},
        },
    )
    assert repeated is not None
    assert intervening is not None

    duplicates = find_duplicate_tool_calls(
        requested_calls=[
            {
                "id": "call_3",
                "name": "tool_group_exec",
                "args": {
                    "group": "knowledge",
                    "command": "user_context_add_files",
                    "args_json": {"file_ids": ["file_1"]},
                },
            }
        ],
        prior_tool_outcomes=[
            {
                "status": "ok",
                "tool_signature": repeated.key,
                "trace_id": "turn_a",
            },
            {
                "status": "ok",
                "tool_signature": intervening.key,
                "trace_id": "turn_a",
            },
        ],
        trace_id="turn_a",
    )

    assert duplicates == []


@pytest.mark.asyncio
async def test_validate_tool_calls_blocks_prior_successful_idempotent_repeat() -> None:
    signature = tool_action_signature(
        "tool_group_exec",
        {
            "group": "knowledge",
            "command": "user_context_add_files",
            "args_json": {"file_ids": ["file_1"]},
        },
    )
    assert signature is not None

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
                HumanMessage(content="Remember this file for this chat."),
                AIMessage(
                    content="Adding file.",
                    tool_calls=[
                        {
                            "id": "call_2",
                            "name": "tool_group_exec",
                            "args": {
                                "group": "knowledge",
                                "command": "user_context_add_files",
                                "args_json": {"file_ids": ["file_1"]},
                            },
                        }
                    ],
                ),
            ],
            "tool_outcomes": [
                {
                    "round_id": 1,
                    "tool_name": "tool_group_exec",
                    "tool_call_id": "call_1",
                    "status": "ok",
                    "result_text": '{"ok":true}',
                    "tool_signature": signature.key,
                    "trace_id": "turn_a",
                }
            ],
            "agent_trace_id": "turn_a",
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    assert result.update["tool_validation_passed"] is False
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert update_messages[0].tool_call_id == "call_2"
    assert "DUPLICATE_TOOL_CALL_PREVIOUS_SUCCESS" in str(update_messages[0].content)
    assert isinstance(update_messages[1], SystemMessage)
    assert "Do not repair arguments or retry that same call" in str(update_messages[1].content)


@pytest.mark.asyncio
async def test_validate_tool_calls_blocks_prior_successful_query_repeat() -> None:
    signature = tool_action_signature(
        "tool_group_exec",
        {
            "group": "knowledge",
            "command": "user_context_query",
            "args_json": {
                "query": "Launch package price and retainer work offered",
                "max_extract_chars": 3000,
            },
        },
    )
    assert signature is not None

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
                HumanMessage(content="Use user_context_query now."),
                AIMessage(
                    content="Querying again.",
                    tool_calls=[
                        {
                            "id": "call_2",
                            "name": "tool_group_exec",
                            "args": {
                                "group": "knowledge",
                                "command": "user_context_query",
                                "args_json": {
                                    "query": "Launch package price and retainer work offered",
                                    "max_extract_chars": 3000,
                                },
                            },
                        }
                    ],
                ),
            ],
            "tool_outcomes": [
                {
                    "round_id": 1,
                    "tool_name": "tool_group_exec",
                    "tool_call_id": "call_1",
                    "status": "ok",
                    "result_text": '{"answer_extract":"Launch package is $900"}',
                    "tool_signature": signature.key,
                    "trace_id": "turn_b",
                }
            ],
            "agent_trace_id": "turn_b",
            "turn_mode": "interactive",
        }
    )

    assert result.goto == "agent"
    assert result.update["tool_validation_passed"] is False
    update_messages = result.update["messages"]
    assert isinstance(update_messages[0], ToolMessage)
    assert update_messages[0].tool_call_id == "call_2"
    assert "DUPLICATE_TOOL_CALL_PREVIOUS_SUCCESS" in str(update_messages[0].content)


def test_duplicate_tool_error_tells_model_to_use_previous_success_not_repair() -> None:
    error = duplicate_tool_error('tool_group_exec(command="telegram_business_status", args_json={})')

    assert "DUPLICATE_TOOL_CALL_PREVIOUS_SUCCESS" in error
    assert "already just succeeded" in error
    assert "Do not repair arguments or retry" in error
    assert "Use the previous result" in error
