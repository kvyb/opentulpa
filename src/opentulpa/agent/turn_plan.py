"""Current-turn plan helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from opentulpa.agent.models import TurnPlanItem, TurnPlanStatus

VALID_TURN_PLAN_STATUSES: set[TurnPlanStatus] = {
    "pending",
    "in_progress",
    "completed",
    "cancelled",
}


class TurnPlanValidationError(ValueError):
    """Raised when the model sends an invalid turn_plan payload."""


def turn_plan_enabled_for_turn_mode(turn_mode: Any) -> bool:
    return str(turn_mode or "").strip().lower() in {"interactive", "workflow_setup"}


def normalize_turn_plan_items(items: Any) -> list[TurnPlanItem]:
    if not isinstance(items, list):
        return []
    normalized: list[TurnPlanItem] = []
    seen: dict[str, int] = {}
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("id", "") or "").strip()
        if not item_id:
            item_id = f"step-{len(normalized) + 1}"
        content = str(raw.get("content", "") or "").strip()
        if not content:
            content = "(no description)"
        status = str(raw.get("status", "pending") or "pending").strip().lower()
        if status not in VALID_TURN_PLAN_STATUSES:
            status = "pending"
        item: TurnPlanItem = {
            "id": item_id,
            "content": content,
            "status": cast(TurnPlanStatus, status),
        }
        if item_id in seen:
            normalized[seen[item_id]] = item
        else:
            seen[item_id] = len(normalized)
            normalized.append(item)
    return _with_single_in_progress(normalized)


def validate_turn_plan_items(items: Any) -> list[TurnPlanItem]:
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError as exc:
            raise TurnPlanValidationError(
                "items must be a list of turn plan items or a JSON-encoded list"
            ) from exc
    if not isinstance(items, list):
        raise TurnPlanValidationError("items must be a list of turn plan items")
    validated: list[TurnPlanItem] = []
    seen_ids: set[str] = set()
    in_progress_count = 0
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise TurnPlanValidationError(f"items[{index}] must be an object")
        item_id = str(raw.get("id", "") or "").strip()
        if not item_id:
            raise TurnPlanValidationError(f"items[{index}].id is required")
        if item_id in seen_ids:
            raise TurnPlanValidationError(f"duplicate turn plan id: {item_id}")
        content = str(raw.get("content", "") or "").strip()
        if not content:
            raise TurnPlanValidationError(f"items[{index}].content is required")
        status_text = str(raw.get("status", "") or "").strip().lower()
        if status_text not in VALID_TURN_PLAN_STATUSES:
            allowed = ", ".join(sorted(VALID_TURN_PLAN_STATUSES))
            raise TurnPlanValidationError(f"items[{index}].status must be one of: {allowed}")
        if status_text == "in_progress":
            in_progress_count += 1
        seen_ids.add(item_id)
        validated.append(
            {
                "id": item_id,
                "content": content,
                "status": cast(TurnPlanStatus, status_text),
            }
        )
    if in_progress_count > 1:
        raise TurnPlanValidationError("only one turn plan item may be in_progress")
    return validated


def _with_single_in_progress(items: list[TurnPlanItem]) -> list[TurnPlanItem]:
    in_progress_indexes = [
        index for index, item in enumerate(items) if item.get("status") == "in_progress"
    ]
    if len(in_progress_indexes) <= 1:
        return [_copy_turn_plan_item(item) for item in items]
    keep_index = in_progress_indexes[-1]
    out: list[TurnPlanItem] = []
    for index, item in enumerate(items):
        updated: TurnPlanItem = {
            "id": item["id"],
            "content": item["content"],
            "status": item["status"],
        }
        if index != keep_index and updated.get("status") == "in_progress":
            updated["status"] = "pending"
        out.append(updated)
    return out


def merge_turn_plan_items(
    existing: list[TurnPlanItem],
    updates: list[TurnPlanItem],
) -> list[TurnPlanItem]:
    if not existing:
        return [_copy_turn_plan_item(item) for item in updates]
    by_id: dict[str, TurnPlanItem] = {
        str(item.get("id", "")): _copy_turn_plan_item(item) for item in existing
    }
    order = [str(item.get("id", "")) for item in existing]
    for update in updates:
        item_id = str(update.get("id", ""))
        if not item_id:
            continue
        if item_id not in by_id:
            order.append(item_id)
            by_id[item_id] = _copy_turn_plan_item(update)
            continue
        current = by_id[item_id]
        if update.get("content"):
            current["content"] = str(update["content"])
        if update.get("status") in VALID_TURN_PLAN_STATUSES:
            current["status"] = update["status"]
    return _with_single_in_progress([by_id[item_id] for item_id in order if item_id in by_id])


def _copy_turn_plan_item(item: TurnPlanItem) -> TurnPlanItem:
    return {"id": item["id"], "content": item["content"], "status": item["status"]}


def update_turn_plan(
    existing: Any,
    *,
    items: Any,
    merge: Any = False,
) -> list[TurnPlanItem]:
    updates = validate_turn_plan_items(items)
    current = normalize_turn_plan_items(existing)
    return merge_turn_plan_items(current, updates) if bool(merge) else updates


def build_turn_plan_result(items: list[TurnPlanItem]) -> dict[str, Any]:
    next_item = next(
        (
            _copy_turn_plan_item(item)
            for item in items
            if str(item.get("status", "")).strip() in {"in_progress", "pending"}
        ),
        None,
    )
    return {
        "ok": True,
        "items": items,
        "summary": summarize_turn_plan(items),
        "next_item": next_item,
        "model_instruction": (
            "Use this plan as current-turn control state. Continue with the "
            "in_progress item or first pending item. Update statuses when work "
            "moves forward. Do not treat the plan itself as the deliverable; "
            "execute the next actionable step in this same turn. Do not mark a "
            "step completed unless this turn's context/tool results support it. "
            "When all items are completed/cancelled, answer the user with the "
            "concrete result."
        ),
    }


def summarize_turn_plan(items: list[TurnPlanItem]) -> dict[str, int]:
    return {
        "total": len(items),
        "pending": sum(1 for item in items if item.get("status") == "pending"),
        "in_progress": sum(1 for item in items if item.get("status") == "in_progress"),
        "completed": sum(1 for item in items if item.get("status") == "completed"),
        "cancelled": sum(1 for item in items if item.get("status") == "cancelled"),
    }


def format_turn_plan_context(items: list[TurnPlanItem]) -> str:
    if not items:
        return ""
    markers = {
        "pending": "[ ]",
        "in_progress": "[>]",
        "completed": "[x]",
        "cancelled": "[-]",
    }
    lines = [
        "CURRENT_TURN_PLAN",
        "Use this as the active plan for the current user request. Continue from the in_progress item or first pending item. Update it with turn_plan when steps change. Do not redo completed work. Do not mark a step completed unless this turn's context/tool results support it. The plan is not the deliverable; execute the next actionable step or give the concrete result/blocker now.",
    ]
    for item in items:
        status = str(item.get("status", "pending"))
        lines.append(
            f"- {markers.get(status, '[ ]')} {item.get('id', '')}: {item.get('content', '')} ({status})"
        )
    return "\n".join(lines)


def build_turn_plan_prompt_context(state: Mapping[str, Any]) -> str:
    return format_turn_plan_context(normalize_turn_plan_items(state.get("turn_plan")))
