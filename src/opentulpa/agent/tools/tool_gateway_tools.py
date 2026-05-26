"""Compact gateway tools for grouped OpenTulpa commands."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

TOOL_GATEWAY_TOOL_NAMES: set[str] = {
    "tool_group_list",
    "tool_group_describe",
    "tool_group_exec",
}

TOOL_GROUP_EXEC_MAX_BATCH_SIZE = 5

TOOL_GROUP_EXEC_BATCH_COMMANDS: set[str] = {
    "memory_search",
    "directive_get",
    "time_profile_get",
    "server_time",
    "web_search",
    "fetch_url_content",
    "fetch_file_content",
    "uploaded_file_search",
    "uploaded_file_get",
    "uploaded_file_analyze",
    "uploaded_file_inspect_structure",
    "business_knowledge_query",
    "user_context_query",
    "user_context_list_sources",
    "user_context_find_sources",
    "tulpa_read_file",
    "tulpa_catalog",
    "task_status",
    "task_events",
    "task_artifacts",
    "telegram_business_status",
    "intake_workflow_list",
    "intake_workflow_get",
    "intake_workflow_setup_get",
    "composio_status",
    "composio_toolkits",
    "composio_connected_accounts",
    "composio_tool_search",
    "composio_tool_schema",
    "routine_list",
    "skill_list",
    "skill_get",
}


TOOL_GROUP_DEFINITIONS: dict[str, dict[str, Any]] = {
    "memory": {
        "summary": "Long-lived memory, directives, and user time profile.",
        "use_for": "preferences, remembered facts, current directive, timezone/UTC offset.",
        "commands": {
            "memory_search",
            "memory_add",
            "directive_get",
            "directive_set",
            "directive_clear",
            "time_profile_get",
            "time_profile_set",
            "server_time",
        },
    },
    "web": {
        "summary": "Current web search and URL/file fetching.",
        "use_for": "fresh facts, direct URLs, page/PDF/DOCX/image URL extraction.",
        "commands": {"web_search", "fetch_url_content", "fetch_file_content", "web_image_send"},
    },
    "browser": {
        "summary": "Real browser automation for dynamic pages, login, CAPTCHA/MFA, and multi-step websites.",
        "use_for": "JS-heavy pages, account flows, browser sessions, screenshots, owner input.",
        "commands": {
            "browser_use_session_list",
            "browser_use_run",
            "browser_use_task_get",
            "browser_use_task_screenshot",
            "browser_use_task_control",
            "browser_use_owner_input_submit",
        },
    },
    "files": {
        "summary": "Uploaded files and one-off file/image send operations.",
        "use_for": "searching, reading, analyzing, sending uploaded files and generated images.",
        "commands": {
            "uploaded_file_search",
            "uploaded_file_get",
            "uploaded_file_send",
            "tulpa_file_send",
            "uploaded_file_analyze",
            "uploaded_file_inspect_structure",
        },
    },
    "knowledge": {
        "summary": "Reusable user context and business knowledge indexing/querying.",
        "use_for": "durable context sources, workflow knowledge files, business document Q&A.",
        "commands": {
            "business_knowledge_index",
            "business_knowledge_query",
            "user_context_add_files",
            "user_context_query",
            "user_context_list_sources",
            "user_context_find_sources",
            "user_context_reindex",
            "user_context_archive_sources",
            "user_context_promote_to_intake",
        },
    },
    "workspace": {
        "summary": "Tulpa workspace files, terminal, validation, and task artifacts.",
        "use_for": "writing/running code in tulpa_stuff, validating files, task status/artifacts.",
        "commands": {
            "tulpa_write_file",
            "tulpa_validate_file",
            "tulpa_reload",
            "tulpa_run_terminal",
            "tulpa_read_file",
            "tulpa_catalog",
            "task_status",
            "task_events",
            "task_artifacts",
            "task_relaunch",
            "task_cancel",
        },
    },
    "intake": {
        "summary": "Intake workflow CRUD, setup wizard, Telegram Business status, and manual runs.",
        "use_for": "creating/editing/deleting workflows, setup drafts, preflight, commit, run/debug.",
        "commands": {
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
        },
    },
    "composio": {
        "summary": "External account connections and Composio tool execution.",
        "use_for": "OAuth, connected accounts, Google Sheets/Drive/Instagram/other external tools.",
        "commands": {
            "composio_status",
            "composio_authorize_toolkit",
            "composio_wait_for_connection",
            "composio_toolkits",
            "composio_connected_accounts",
            "composio_disable_connected_account",
            "composio_delete_connected_account",
            "composio_tool_search",
            "composio_tool_schema",
            "composio_instagram_reply_precheck",
            "composio_tool_execute",
        },
    },
    "routine": {
        "summary": "Scheduled routines and wake/follow-up automation.",
        "use_for": "reminders, repeated checks, routine listing/deletion.",
        "commands": {"routine_list", "routine_create", "routine_delete"},
    },
    "skills": {
        "summary": "Skill discovery and skill CRUD.",
        "use_for": "finding user skills, loading full skill instructions, creating/updating reusable skills.",
        "commands": {"skill_list", "skill_get", "skill_upsert", "skill_delete"},
    },
}


def _safe_tool_description(tool_obj: Any) -> str:
    return " ".join(str(getattr(tool_obj, "description", "") or "").split()).strip()


def _coerce_json_like(value: Any) -> Any:
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith(("{", "[")):
            try:
                return _coerce_json_like(json.loads(raw))
            except Exception:
                return value
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_json_like(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_json_like(item) for item in value]
    return value


def _group_commands(group: str, source_tools: dict[str, Any]) -> list[str]:
    definition = TOOL_GROUP_DEFINITIONS.get(group)
    if not isinstance(definition, dict):
        return []
    configured = definition.get("commands", set())
    names = configured if isinstance(configured, set) else set()
    return [name for name in source_tools if name in names and name not in TOOL_GATEWAY_TOOL_NAMES]


def _command_group(command: str, source_tools: dict[str, Any]) -> str:
    for group in TOOL_GROUP_DEFINITIONS:
        if command in _group_commands(group, source_tools):
            return group
    return ""


def _command_schema(tool_obj: Any) -> dict[str, Any]:
    try:
        schema = convert_to_openai_tool(tool_obj)
    except Exception:
        return {
            "name": str(getattr(tool_obj, "name", "") or "").strip(),
            "description": _safe_tool_description(tool_obj),
        }
    if isinstance(schema, dict):
        return schema
    return {"schema": str(schema)}


def _function_schema(tool_obj: Any) -> dict[str, Any]:
    schema = _command_schema(tool_obj)
    function = schema.get("function")
    if isinstance(function, dict):
        return function
    return schema


def _compact_arg_spec(tool_obj: Any) -> dict[str, Any]:
    function = _function_schema(tool_obj)
    parameters = function.get("parameters")
    params = parameters if isinstance(parameters, dict) else {}
    properties = params.get("properties")
    props = properties if isinstance(properties, dict) else {}
    required_raw = params.get("required")
    required = [str(item) for item in required_raw] if isinstance(required_raw, list) else []
    args: dict[str, Any] = {}
    for name, spec in props.items():
        if not isinstance(spec, dict):
            args[str(name)] = {"required": str(name) in required}
            continue
        entry: dict[str, Any] = {"required": str(name) in required}
        arg_type = spec.get("type")
        if arg_type:
            entry["type"] = arg_type
        description = " ".join(str(spec.get("description", "") or "").split()).strip()
        if description:
            entry["description"] = description[:160]
        enum = spec.get("enum")
        if isinstance(enum, list) and enum:
            entry["enum"] = enum[:12]
        args[str(name)] = entry
    return {"required": required, "args": args}


def _compact_repair_hint(group: str, command: str, tool_obj: Any) -> dict[str, Any]:
    return {
        "expected_args": _compact_arg_spec(tool_obj),
        "example_call": {
            "tool": "tool_group_exec",
            "group": group,
            "command": command,
            "args_json": "JSON object with the expected args above",
        },
        "next_step": "Fix args_json and retry tool_group_exec directly; do not call tool_group_describe unless this hint is still insufficient.",
    }


def _missing_required_args(tool_obj: Any, args: dict[str, Any]) -> list[str]:
    required = _compact_arg_spec(tool_obj).get("required")
    if not isinstance(required, list):
        return []
    return [str(name) for name in required if str(name) not in args]


def _looks_like_arg_error(value: Any) -> bool:
    text = " ".join(str(value or "").lower().split())
    if not text:
        return False
    markers = ("required", "missing", "invalid", "must be", "args", "argument", "validation")
    return any(marker in text for marker in markers)


def _repair_command_args(command: str, args: dict[str, Any]) -> dict[str, Any]:
    safe_command = str(command or "").strip()
    repaired = dict(args)
    if safe_command == "web_image_send" and "url" not in repaired and "image_url" in repaired:
        repaired["url"] = repaired.pop("image_url")
    if safe_command in {
        "intake_workflow_setup_update",
        "intake_workflow_setup_finalize_confirmation",
    } and not any(key in repaired for key in ("draft_patch", "scratchpad_patch")):
        draft_patch = repaired.get("draft_upsert") if isinstance(repaired.get("draft_upsert"), dict) else repaired
        repaired = {"draft_patch": draft_patch}
    draft_patch = repaired.get("draft_patch")
    if isinstance(draft_patch, dict):
        if isinstance(draft_patch.get("draft"), dict):
            normalized_draft = dict(draft_patch["draft"])
        elif isinstance(draft_patch.get("draft_upsert"), dict):
            normalized_draft = dict(draft_patch["draft_upsert"])
        else:
            normalized_draft = dict(draft_patch)
        if normalized_draft.get("provider") in {"telegram", "telegram_business"}:
            normalized_draft["provider"] = "telegram_bot_api"
        if (
            normalized_draft.get("channel") == "telegram_business_dm"
            and not normalized_draft.get("provider")
        ):
            normalized_draft["provider"] = "telegram_bot_api"
        repaired["draft_patch"] = normalized_draft
    return repaired


def _batch_disallowed_commands(calls: list[dict[str, Any]]) -> list[str]:
    disallowed: list[str] = []
    for call in calls:
        command = str(call.get("command", "")).strip()
        if command not in TOOL_GROUP_EXEC_BATCH_COMMANDS and command not in disallowed:
            disallowed.append(command)
    return disallowed


def register_tool_gateway_tools(runtime: Any, source_tools: dict[str, Any]) -> dict[str, Any]:
    assert isinstance(source_tools, dict)

    @tool
    async def tool_group_list() -> Any:
        """List compact OpenTulpa tool groups and when to use each group."""
        groups: list[dict[str, Any]] = []
        for group, definition in TOOL_GROUP_DEFINITIONS.items():
            commands = _group_commands(group, source_tools)
            if not commands:
                continue
            groups.append(
                {
                    "group": group,
                    "summary": definition["summary"],
                    "use_for": definition["use_for"],
                    "command_count": len(commands),
                    "common_commands": commands[:8],
                }
            )
        return {"groups": groups}

    @tool
    async def tool_group_describe(group: str, command: str = "") -> Any:
        """Describe a tool group or one exact command schema before execution."""
        safe_group = str(group or "").strip().lower()
        safe_command = str(command or "").strip()
        if safe_group not in TOOL_GROUP_DEFINITIONS:
            return {
                "error": "unknown tool group",
                "available_groups": [
                    name for name in TOOL_GROUP_DEFINITIONS if _group_commands(name, source_tools)
                ],
            }
        commands = _group_commands(safe_group, source_tools)
        if safe_command:
            if safe_command not in commands:
                return {
                    "error": "unknown command for group",
                    "group": safe_group,
                    "available_commands": commands,
                }
            tool_obj = source_tools[safe_command]
            return {
                "group": safe_group,
                "command": safe_command,
                "description": _safe_tool_description(tool_obj),
                "schema": _command_schema(tool_obj),
                "call_pattern": {
                    "tool": "tool_group_exec",
                    "group": safe_group,
                    "command": safe_command,
                    "args_json": "object matching schema.parameters.properties",
                },
            }
        return {
            "group": safe_group,
            "summary": TOOL_GROUP_DEFINITIONS[safe_group]["summary"],
            "use_for": TOOL_GROUP_DEFINITIONS[safe_group]["use_for"],
            "commands": [
                {
                    "command": name,
                    "description": _safe_tool_description(source_tools[name])[:280],
                }
                for name in commands
            ],
            "next_step": (
                "If you know the args, call tool_group_exec directly. If args are unclear, "
                "call tool_group_describe for one exact command."
            ),
        }

    async def _execute_one_tool_call(group: str, command: str, args_json: Any) -> dict[str, Any]:
        assert isinstance(source_tools, dict)
        safe_group = str(group or "").strip().lower()
        safe_command = str(command or "").strip()
        commands = _group_commands(safe_group, source_tools)
        if not commands:
            return {
                "error": "unknown or empty tool group",
                "available_groups": [
                    name for name in TOOL_GROUP_DEFINITIONS if _group_commands(name, source_tools)
                ],
            }
        if safe_command not in commands:
            actual_group = _command_group(safe_command, source_tools)
            return {
                "error": "unknown command for group",
                "group": safe_group,
                "command": safe_command,
                "actual_group": actual_group,
                "available_commands": commands,
            }
        raw_args = {} if args_json is None else _coerce_json_like(args_json)
        if not isinstance(raw_args, dict):
            tool_obj = source_tools.get(safe_command)
            repair_hint = (
                _compact_repair_hint(safe_group, safe_command, tool_obj)
                if tool_obj is not None
                else None
            )
            return {
                "error": "args_json must be a JSON object",
                "received_type": type(raw_args).__name__,
                "repair_hint": repair_hint,
            }
        raw_args = _repair_command_args(safe_command, raw_args)
        tool_obj = source_tools.get(safe_command)
        if tool_obj is None:
            return {"error": "command not registered", "command": safe_command}
        missing_args = _missing_required_args(tool_obj, raw_args)
        if missing_args:
            return {
                "error": "missing required args",
                "group": safe_group,
                "command": safe_command,
                "missing_args": missing_args,
                "repair_hint": _compact_repair_hint(safe_group, safe_command, tool_obj),
            }
        assert safe_command
        assert isinstance(raw_args, dict)
        try:
            result = await tool_obj.ainvoke(raw_args)
        except Exception as exc:
            return {
                "error": f"{safe_command} failed: {exc}",
                "group": safe_group,
                "command": safe_command,
                "args_json": raw_args,
                "repair_hint": _compact_repair_hint(safe_group, safe_command, tool_obj),
            }
        if isinstance(result, dict) and result.get("error"):
            response: dict[str, Any] = {
                "group": safe_group,
                "command": safe_command,
                "ok": False,
                "error": str(result.get("error") or ""),
                "result": result,
            }
            if _looks_like_arg_error(result.get("error")):
                response["repair_hint"] = _compact_repair_hint(safe_group, safe_command, tool_obj)
            return response
        return {
            "group": safe_group,
            "command": safe_command,
            "ok": True,
            "result": result,
        }

    def _normalize_batch_calls(calls: Any) -> list[dict[str, Any]] | dict[str, Any]:
        raw_calls = _coerce_json_like(calls)
        if not isinstance(raw_calls, list):
            return {
                "error": "calls must be a JSON array",
                "received_type": type(raw_calls).__name__,
            }
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(raw_calls):
            if not isinstance(item, dict):
                return {
                    "error": "each batch call must be a JSON object",
                    "index": index,
                    "received_type": type(item).__name__,
                }
            normalized.append(
                {
                    "group": str(item.get("group", "")).strip(),
                    "command": str(item.get("command", "")).strip(),
                    "args_json": item.get("args_json"),
                }
            )
        if not normalized:
            return {"error": "calls must include at least one invocation"}
        if len(normalized) > TOOL_GROUP_EXEC_MAX_BATCH_SIZE:
            return {
                "error": "too many batch calls",
                "max_batch_size": TOOL_GROUP_EXEC_MAX_BATCH_SIZE,
                "received_count": len(normalized),
            }
        return normalized

    @tool
    async def tool_group_exec(
        group: str = "",
        command: str = "",
        args_json: dict[str, Any] | str | None = None,
        calls: list[dict[str, Any]] | str | None = None,
    ) -> Any:
        """
        Execute one command, or batch independent read/search/status commands using calls.

        Single call shape: group, command, args_json.
        Batch shape: calls=[{"group": "...", "command": "...", "args_json": {...}}, ...].
        """
        batch_input = calls
        if batch_input is None and not str(group or "").strip() and not str(command or "").strip():
            maybe_calls = _coerce_json_like(args_json)
            if isinstance(maybe_calls, list):
                batch_input = maybe_calls
        if batch_input is None:
            return await _execute_one_tool_call(group, command, args_json)
        normalized_calls = _normalize_batch_calls(batch_input)
        if isinstance(normalized_calls, dict):
            return normalized_calls
        assert normalized_calls
        assert len(normalized_calls) <= TOOL_GROUP_EXEC_MAX_BATCH_SIZE
        if len(normalized_calls) == 1:
            call = normalized_calls[0]
            return await _execute_one_tool_call(
                call["group"],
                call["command"],
                call["args_json"],
            )
        disallowed = _batch_disallowed_commands(normalized_calls)
        if disallowed:
            return {
                "error": "unsupported batch commands",
                "unsupported_commands": disallowed,
                "allowed_batch_commands": sorted(TOOL_GROUP_EXEC_BATCH_COMMANDS),
                "repair_hint": (
                    "Use one tool_group_exec call for side-effecting, browser, terminal, "
                    "send, write, account-change, or workflow-mutation commands. Batch only "
                    "independent read/search/status/fetch/inspect calls."
                ),
            }
        results = await asyncio.gather(
            *[
                _execute_one_tool_call(call["group"], call["command"], call["args_json"])
                for call in normalized_calls
            ]
        )
        return {
            "ok": all(bool(result.get("ok", False)) for result in results),
            "batched": True,
            "parallel": True,
            "results": results,
        }

    return {
        "tool_group_list": tool_group_list,
        "tool_group_describe": tool_group_describe,
        "tool_group_exec": tool_group_exec,
    }
