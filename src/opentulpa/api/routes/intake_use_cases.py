"""Use-case helpers for intake workflow HTTP routes."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from opentulpa.api.file_helpers import sanitize_uploaded_file_record


def workflow_upsert_kwargs(
    body: dict[str, Any],
    *,
    customer_id: str,
    workflow_id: str | None = None,
    default_schedule_on_falsey: bool = False,
) -> dict[str, Any]:
    schedule_value = body.get("schedule", "*/2 * * * *")
    if default_schedule_on_falsey:
        schedule_value = body.get("schedule") or "*/2 * * * *"
    return {
        "customer_id": customer_id,
        "workflow_id": workflow_id
        if workflow_id is not None
        else str(body.get("workflow_id", "")).strip() or None,
        "name": str(body.get("name", "")).strip(),
        "channel": str(body.get("channel", "instagram_dm")).strip() or "instagram_dm",
        "provider": str(body.get("provider", "composio")).strip() or "composio",
        "source_config": body.get("source_config")
        if isinstance(body.get("source_config"), dict)
        else None,
        "intent_description": str(body.get("intent_description", "")).strip(),
        "required_fields": body.get("required_fields")
        if isinstance(body.get("required_fields"), list)
        else [],
        "field_guidance": body.get("field_guidance")
        if isinstance(body.get("field_guidance"), dict)
        else None,
        "assistant_instructions": str(body.get("assistant_instructions", "")).strip(),
        "business_facts": body.get("business_facts")
        if isinstance(body.get("business_facts"), dict)
        else None,
        "knowledge_file_ids": body.get("knowledge_file_ids")
        if isinstance(body.get("knowledge_file_ids"), list)
        else None,
        "sink_type": str(body.get("sink_type", "")).strip(),
        "sink_config": body.get("sink_config") if isinstance(body.get("sink_config"), dict) else None,
        "schedule": str(schedule_value).strip() or "*/2 * * * *",
        "notify_user": bool(body.get("notify_user", True)),
        "enabled": bool(body.get("enabled", True)),
        "reply_mode": str(body.get("reply_mode", "auto")).strip() or "auto",
    }


def workflow_with_knowledge_files(
    workflow: dict[str, Any],
    *,
    file_vault: Any | None,
) -> dict[str, Any]:
    item = dict(workflow)
    customer_id = str(item.get("customer_id", "") or "").strip()
    file_ids = [
        str(file_id or "").strip()
        for file_id in item.get("knowledge_file_ids", [])
        if str(file_id or "").strip()
    ]
    knowledge_files: list[dict[str, Any]] = []
    if file_vault is not None and customer_id and file_ids:
        for record in file_vault.get_many(customer_id, file_ids):
            clean = sanitize_uploaded_file_record(record, include_excerpt=False)
            file_id = str(clean.get("id") or "").strip()
            if not file_id:
                continue
            knowledge_files.append(
                {
                    "id": file_id,
                    "kind": clean.get("kind"),
                    "original_filename": clean.get("original_filename"),
                    "mime_type": clean.get("mime_type"),
                    "size_bytes": clean.get("size_bytes"),
                    "caption": clean.get("caption"),
                    "summary": clean.get("summary"),
                    "created_at": clean.get("created_at"),
                    "content_path": (
                        f"/web/files/{quote(file_id)}/content"
                        f"?customer_id={quote(customer_id, safe='')}"
                    ),
                    "metadata_path": (
                        f"/web/files/{quote(file_id)}/metadata"
                        f"?customer_id={quote(customer_id, safe='')}"
                    ),
                }
            )
    item["knowledge_files"] = knowledge_files
    return item
