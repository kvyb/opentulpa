"""Instagram-specific Composio adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any


class InstagramComposioAdapter:
    def __init__(self, core: Any) -> None:
        self.core = core

    def inspect_reply_target(
        self,
        *,
        customer_id: str,
        recipient_id: str | None = None,
        conversation_id: str | None = None,
        connected_account_id: str | None = None,
        scan_limit: int = 10,
    ) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        safe_recipient = str(recipient_id or "").strip()
        safe_conversation = str(conversation_id or "").strip()
        safe_account = str(connected_account_id or "").strip() or None
        if not safe_customer:
            raise ValueError("customer_id is required")
        if not safe_recipient and not safe_conversation:
            raise ValueError("recipient_id or conversation_id is required")

        conversation = self._target_conversation(
            customer_id=safe_customer,
            recipient_id=safe_recipient,
            conversation_id=safe_conversation,
            connected_account_id=safe_account,
            scan_limit=scan_limit,
        )
        if not conversation:
            return _missing_conversation_payload(
                customer_id=safe_customer,
                conversation_id=safe_conversation,
                recipient_id=safe_recipient,
            )
        summary = _summarize_instagram_conversation(
            conversation=conversation,
            requested_recipient_id=safe_recipient or None,
        )
        summary["ok"] = True
        summary["customer_id"] = safe_customer
        summary["connected_account_id"] = safe_account
        return summary

    def list_conversations(
        self,
        *,
        customer_id: str,
        connected_account_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        if not safe_customer:
            raise ValueError("customer_id is required")
        safe_account = str(connected_account_id or "").strip() or None
        response = self.core._sdk_execute_tool(
            slug="INSTAGRAM_LIST_ALL_CONVERSATIONS",
            arguments={"limit": max(1, min(int(limit), 25))},
            connected_account_id=safe_account,
            user_id=safe_customer,
        )
        if not bool(response.get("successful", False)):
            raise RuntimeError(str(response.get("error") or "failed to list Instagram conversations"))
        return self._conversation_list_payload(
            customer_id=safe_customer,
            connected_account_id=safe_account,
            items=_safe_list(_safe_dict(response.get("data")).get("data")),
        )

    def get_conversation(
        self,
        *,
        customer_id: str,
        conversation_id: str,
        connected_account_id: str | None = None,
    ) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        safe_conversation = str(conversation_id or "").strip()
        safe_account = str(connected_account_id or "").strip() or None
        if not safe_customer:
            raise ValueError("customer_id is required")
        if not safe_conversation:
            raise ValueError("conversation_id is required")
        conversation = self._fetch_conversation(
            customer_id=safe_customer,
            conversation_id=safe_conversation,
            connected_account_id=safe_account,
        )
        summary = _summarize_instagram_conversation(
            conversation=conversation,
            requested_recipient_id=None,
        )
        summary["ok"] = True
        summary["customer_id"] = safe_customer
        summary["connected_account_id"] = safe_account
        return {
            "ok": True,
            "customer_id": safe_customer,
            "connected_account_id": safe_account,
            "conversation": conversation,
            "summary": summary,
        }

    def _target_conversation(
        self,
        *,
        customer_id: str,
        recipient_id: str,
        conversation_id: str,
        connected_account_id: str | None,
        scan_limit: int,
    ) -> dict[str, Any] | None:
        if conversation_id:
            return self._fetch_conversation(
                customer_id=customer_id,
                conversation_id=conversation_id,
                connected_account_id=connected_account_id,
            )
        return self._find_conversation_for_recipient(
            customer_id=customer_id,
            recipient_id=recipient_id,
            connected_account_id=connected_account_id,
            scan_limit=scan_limit,
        )

    def _conversation_list_payload(
        self,
        *,
        customer_id: str,
        connected_account_id: str | None,
        items: list[Any],
    ) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        warnings: list[dict[str, str]] = []
        for item in items:
            conversation_id = str(_safe_dict(item).get("id", "") or "").strip()
            if not conversation_id:
                continue
            try:
                conversation = self._fetch_conversation(
                    customer_id=customer_id,
                    conversation_id=conversation_id,
                    connected_account_id=connected_account_id,
                )
            except Exception as exc:
                warnings.append({"conversation_id": conversation_id, "error": str(exc)})
                continue
            summary = _summarize_instagram_conversation(
                conversation=conversation,
                requested_recipient_id=None,
            )
            summary["ok"] = True
            summary["customer_id"] = customer_id
            summary["connected_account_id"] = connected_account_id
            summaries.append(summary)
        return {
            "ok": True,
            "customer_id": customer_id,
            "connected_account_id": connected_account_id,
            "items": summaries,
            "warnings": warnings,
        }

    def _fetch_conversation(
        self,
        *,
        customer_id: str,
        conversation_id: str,
        connected_account_id: str | None,
    ) -> dict[str, Any]:
        result = self.core._sdk_execute_tool(
            slug="INSTAGRAM_GET_CONVERSATION",
            arguments={"conversation_id": conversation_id},
            connected_account_id=connected_account_id,
            user_id=customer_id,
        )
        if not bool(result.get("successful", False)):
            raise RuntimeError(str(result.get("error") or "failed to fetch Instagram conversation"))
        return _safe_dict(result.get("data"))

    def _find_conversation_for_recipient(
        self,
        *,
        customer_id: str,
        recipient_id: str,
        connected_account_id: str | None,
        scan_limit: int,
    ) -> dict[str, Any] | None:
        if not recipient_id:
            return None
        response = self.core._sdk_execute_tool(
            slug="INSTAGRAM_LIST_ALL_CONVERSATIONS",
            arguments={"limit": max(1, min(int(scan_limit), 25))},
            connected_account_id=connected_account_id,
            user_id=customer_id,
        )
        if not bool(response.get("successful", False)):
            raise RuntimeError(str(response.get("error") or "failed to list Instagram conversations"))
        for item in _safe_list(_safe_dict(response.get("data")).get("data")):
            conversation_id = str(_safe_dict(item).get("id", "") or "").strip()
            if not conversation_id:
                continue
            conversation = self._fetch_conversation(
                customer_id=customer_id,
                conversation_id=conversation_id,
                connected_account_id=connected_account_id,
            )
            if recipient_id in _participant_ids(conversation):
                return conversation
        return None


def _missing_conversation_payload(
    *,
    customer_id: str,
    conversation_id: str,
    recipient_id: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "customer_id": customer_id,
        "conversation_id": conversation_id or None,
        "recipient_id": recipient_id or None,
        "recipient_id_verified": False,
        "matched": False,
        "reply_window_status": "unconfirmed",
        "reply_window_reason": "No Instagram conversation matching this target was found.",
    }


def _participant_ids(conversation: dict[str, Any]) -> set[str]:
    return {
        str(_safe_dict(participant).get("id", "") or "").strip()
        for participant in _safe_list(_safe_dict(conversation.get("participants")).get("data"))
    }


def _summarize_instagram_conversation(
    *,
    conversation: dict[str, Any],
    requested_recipient_id: str | None,
) -> dict[str, Any]:
    payload = _safe_dict(conversation.get("data")) if "data" in conversation else conversation
    participants = _safe_list(_safe_dict(payload.get("participants")).get("data"))
    messages = _normalized_messages(payload)
    participant_ids = [
        str(_safe_dict(item).get("id", "") or "").strip()
        for item in participants
        if str(_safe_dict(item).get("id", "") or "").strip()
    ]
    verified_recipient = _verified_recipient(
        participant_ids=participant_ids,
        requested_recipient_id=requested_recipient_id,
    )
    own_ids = [item for item in participant_ids if item != verified_recipient] if verified_recipient else []
    latest_inbound, latest_outbound = _latest_directional_messages(
        messages=messages,
        verified_recipient=verified_recipient,
        own_participant_ids=own_ids,
    )
    latest_message = messages[0] if messages else None
    return _conversation_summary_payload(
        payload=payload,
        participant_ids=participant_ids,
        participant_usernames=_participant_usernames(participants),
        requested_recipient_id=requested_recipient_id,
        verified_recipient=verified_recipient,
        latest_message=latest_message,
        latest_inbound=latest_inbound,
        latest_outbound=latest_outbound,
    )


def _normalized_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = _safe_list(_safe_dict(payload.get("messages")).get("data"))
    normalized: list[dict[str, Any]] = []
    for item in messages:
        msg = _safe_dict(item)
        sender = _safe_dict(msg.get("from"))
        recipients = _safe_list(_safe_dict(msg.get("to")).get("data"))
        normalized.append(
            {
                "id": str(msg.get("id", "") or "").strip(),
                "created_time": str(msg.get("created_time", "") or "").strip(),
                "created_at": _parse_datetime(msg.get("created_time")),
                "message": str(msg.get("message", "") or "").strip(),
                "from_id": str(sender.get("id", "") or "").strip(),
                "from_username": str(sender.get("username", "") or "").strip(),
                "to_ids": [
                    str(_safe_dict(recipient).get("id", "") or "").strip()
                    for recipient in recipients
                    if str(_safe_dict(recipient).get("id", "") or "").strip()
                ],
            }
        )
    normalized.sort(key=lambda item: item["created_at"] or datetime.min, reverse=True)
    return normalized


def _latest_directional_messages(
    *,
    messages: list[dict[str, Any]],
    verified_recipient: str | None,
    own_participant_ids: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    latest_inbound = None
    latest_outbound = None
    for item in messages:
        sender_id = item["from_id"]
        if verified_recipient and sender_id == verified_recipient and latest_inbound is None:
            latest_inbound = item
        if own_participant_ids and sender_id in own_participant_ids and latest_outbound is None:
            latest_outbound = item
        if latest_inbound is not None and (latest_outbound is not None or not own_participant_ids):
            break
    return latest_inbound, latest_outbound


def _conversation_summary_payload(
    *,
    payload: dict[str, Any],
    participant_ids: list[str],
    participant_usernames: dict[str, str],
    requested_recipient_id: str | None,
    verified_recipient: str | None,
    latest_message: dict[str, Any] | None,
    latest_inbound: dict[str, Any] | None,
    latest_outbound: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "matched": True,
        "conversation_id": str(payload.get("id", "") or "").strip() or None,
        "conversation_updated_time": str(payload.get("updated_time", "") or "").strip() or None,
        "participant_ids": participant_ids,
        "participant_usernames": participant_usernames,
        "recipient_id": verified_recipient or requested_recipient_id or None,
        "recipient_id_verified": bool(verified_recipient),
        "latest_message_id": latest_message["id"] if latest_message else None,
        "latest_message_created_time": latest_message["created_time"] if latest_message else None,
        "latest_message_sender_id": latest_message["from_id"] if latest_message else None,
        "latest_message_sender_username": latest_message["from_username"] if latest_message else None,
        "latest_message_text_preview": latest_message["message"][:280] if latest_message else None,
        "latest_inbound_message_id": latest_inbound["id"] if latest_inbound else None,
        "latest_inbound_message_created_time": latest_inbound["created_time"] if latest_inbound else None,
        "latest_inbound_sender_id": latest_inbound["from_id"] if latest_inbound else None,
        "latest_inbound_sender_username": latest_inbound["from_username"] if latest_inbound else None,
        "latest_inbound_message_text_preview": latest_inbound["message"][:280] if latest_inbound else None,
        "latest_outbound_message_id": latest_outbound["id"] if latest_outbound else None,
        "latest_outbound_message_created_time": latest_outbound["created_time"] if latest_outbound else None,
        "reply_window_status": "unconfirmed",
        "reply_window_reason": _reply_window_reason(latest_inbound),
    }


def _verified_recipient(
    *,
    participant_ids: list[str],
    requested_recipient_id: str | None,
) -> str | None:
    verified = str(requested_recipient_id or "").strip() or None
    if verified and verified not in participant_ids:
        verified = None
    if not verified and len(participant_ids) == 2:
        verified = participant_ids[1]
    return verified


def _participant_usernames(participants: list[Any]) -> dict[str, str]:
    return {
        str(_safe_dict(item).get("id", "") or "").strip(): str(_safe_dict(item).get("username", "") or "").strip()
        for item in participants
        if str(_safe_dict(item).get("id", "") or "").strip()
    }


def _reply_window_reason(latest_inbound: dict[str, Any] | None) -> str:
    if latest_inbound:
        return (
            "Exact thread verified and latest inbound timestamp captured, but Meta still decides whether the "
            "reply window is open at send time."
        )
    return "Exact thread verified, but no inbound message timestamp was found on this thread."


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
