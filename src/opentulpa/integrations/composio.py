"""Composio integration service for auth, toolkit inspection, and tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from composio import Composio
from composio_langchain import LangchainProvider

from opentulpa.core.public_urls import build_public_composio_callback_url


def _normalize_toolkit_slug(value: str) -> str:
    return str(value or "").strip().lower()


def _coerce_toolkit_list(values: list[str] | None) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _coerce_status_list(values: list[str] | None) -> list[str]:
    allowed = {"INITIALIZING", "INITIATED", "ACTIVE", "FAILED", "EXPIRED", "INACTIVE"}
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip().upper()
        if not text or text not in allowed or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


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


def _is_invalid_instagram_reply_to_error(error: Any) -> bool:
    text = str(error or "").lower()
    if not text:
        return False
    return "invalid message id" in text or "error_subcode\\\":2534002" in text or "error_subcode:2534002" in text


@dataclass(slots=True)
class ComposioService:
    """Thin wrapper around the Composio SDK for OpenTulpa."""

    api_key: str
    default_callback_url: str | None = None
    _client: Composio | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.api_key = str(self.api_key or "").strip()
        self.default_callback_url = str(self.default_callback_url or "").strip() or None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _resolved_callback_url(self, callback_url: str | None = None) -> str | None:
        explicit = str(callback_url or "").strip()
        if explicit:
            return explicit
        if self.default_callback_url:
            return self.default_callback_url
        dynamic = build_public_composio_callback_url()
        return dynamic or None

    def status(self) -> dict[str, Any]:
        resolved_callback = self._resolved_callback_url()
        return {
            "ok": True,
            "enabled": self.enabled,
            "callback_url_configured": bool(resolved_callback),
            "default_callback_url": self.default_callback_url,
            "resolved_callback_url": resolved_callback,
        }

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("Composio is not configured")

    def _sdk(self) -> Composio:
        self._require_enabled()
        if self._client is None:
            self._client = Composio(
                api_key=self.api_key,
                provider=LangchainProvider(),
            )
        return self._client

    def _session(
        self,
        *,
        customer_id: str,
        manage_connections: bool | None = None,
        connected_accounts: dict[str, str] | None = None,
        toolkits: list[str] | None = None,
    ) -> Any:
        customer = str(customer_id or "").strip()
        if not customer:
            raise ValueError("customer_id is required")
        kwargs: dict[str, Any] = {"user_id": customer}
        if manage_connections is not None:
            kwargs["manage_connections"] = bool(manage_connections)
        normalized_accounts = {
            _normalize_toolkit_slug(key): str(value or "").strip()
            for key, value in (connected_accounts or {}).items()
            if str(key or "").strip() and str(value or "").strip()
        }
        if normalized_accounts:
            kwargs["connected_accounts"] = normalized_accounts
        normalized_toolkits = [str(item or "").strip() for item in _coerce_toolkit_list(toolkits)]
        if normalized_toolkits:
            kwargs["toolkits"] = normalized_toolkits
        return self._sdk().create(**kwargs)

    def authorize_toolkit(
        self,
        *,
        customer_id: str,
        toolkit: str,
        callback_url: str | None = None,
    ) -> dict[str, Any]:
        session = self._session(customer_id=customer_id, manage_connections=False)
        safe_toolkit = str(toolkit or "").strip()
        if not safe_toolkit:
            raise ValueError("toolkit is required")
        resolved_callback = self._resolved_callback_url(callback_url)
        request = session.authorize(
            toolkit=safe_toolkit,
            callback_url=resolved_callback,
        )
        redirect_url = str(getattr(request, "redirect_url", "") or "").strip()
        connection_id = str(getattr(request, "id", "") or "").strip()
        return {
            "ok": True,
            "customer_id": str(customer_id),
            "toolkit": safe_toolkit,
            "connection_id": connection_id,
            "redirect_url": redirect_url,
            "callback_url": resolved_callback,
            "next_action": (
                "Send redirect_url to the user and ask them to finish authorization in the browser."
                if redirect_url
                else "Tell the user to authorize the toolkit in Composio."
            ),
            "message_for_user": (
                f"Connect your {safe_toolkit} account here: {redirect_url}"
                if redirect_url
                else f"Please authorize your {safe_toolkit} account in Composio."
            ),
            "instructions": (
                f"Open this URL to connect {safe_toolkit}: {redirect_url}"
                if redirect_url
                else f"Authorize {safe_toolkit} in Composio."
            ),
        }

    def wait_for_connection(
        self,
        *,
        connection_id: str,
        timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        safe_id = str(connection_id or "").strip()
        if not safe_id:
            raise ValueError("connection_id is required")
        result = self._sdk().connected_accounts.wait_for_connection(
            id=safe_id,
            timeout=max(1.0, min(float(timeout_seconds), 600.0)),
        )
        return self._serialize_connected_account(result)

    def list_toolkits(
        self,
        *,
        customer_id: str,
        toolkits: list[str] | None = None,
        is_connected: bool | None = None,
        limit: int = 50,
        search: str | None = None,
    ) -> dict[str, Any]:
        session = self._session(customer_id=customer_id, manage_connections=False)
        result = session.toolkits(
            toolkits=_coerce_toolkit_list(toolkits) or None,
            is_connected=is_connected,
            limit=max(1, min(int(limit), 100)),
            search=str(search or "").strip() or None,
        )
        items: list[dict[str, Any]] = []
        for item in list(getattr(result, "items", []) or []):
            connection = getattr(item, "connection", None)
            connected_account = getattr(connection, "connected_account", None) if connection else None
            auth_config = getattr(connection, "auth_config", None) if connection else None
            items.append(
                {
                    "slug": str(getattr(item, "slug", "") or ""),
                    "name": str(getattr(item, "name", "") or ""),
                    "is_no_auth": bool(getattr(item, "is_no_auth", False)),
                    "is_connected": bool(getattr(connection, "is_active", False)) if connection else False,
                    "connected_account_id": str(getattr(connected_account, "id", "") or "") or None,
                    "connected_account_status": str(getattr(connected_account, "status", "") or "") or None,
                    "auth_config_id": str(getattr(auth_config, "id", "") or "") or None,
                    "auth_mode": str(getattr(auth_config, "mode", "") or "") or None,
                }
            )
        return {
            "ok": True,
            "customer_id": str(customer_id),
            "items": items,
            "next_cursor": str(getattr(result, "next_cursor", "") or "") or None,
            "total_pages": int(getattr(result, "total_pages", 0) or 0),
        }

    def list_connected_accounts(
        self,
        *,
        customer_id: str,
        toolkits: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        response = self._sdk().connected_accounts.list(
            user_ids=[str(customer_id).strip()],
            toolkit_slugs=_coerce_toolkit_list(toolkits) or None,
            statuses=_coerce_status_list(statuses) or None,
            limit=max(1, min(int(limit), 100)),
        )
        items = [self._serialize_connected_account(item) for item in list(getattr(response, "items", []) or [])]
        return {
            "ok": True,
            "customer_id": str(customer_id),
            "items": items,
            "next_cursor": str(getattr(response, "next_cursor", "") or "") or None,
        }

    def disable_connected_account(
        self,
        *,
        connected_account_id: str,
    ) -> dict[str, Any]:
        safe_id = str(connected_account_id or "").strip()
        if not safe_id:
            raise ValueError("connected_account_id is required")
        result = self._sdk().connected_accounts.disable(safe_id)
        payload = self._serialize_connected_account(result) if result is not None else {"id": safe_id}
        payload["disabled"] = True
        return {
            "ok": True,
            "connected_account": payload,
        }

    def delete_connected_account(
        self,
        *,
        connected_account_id: str,
    ) -> dict[str, Any]:
        safe_id = str(connected_account_id or "").strip()
        if not safe_id:
            raise ValueError("connected_account_id is required")
        result = self._sdk().connected_accounts.delete(safe_id)
        payload = {"id": safe_id, "deleted": True}
        if result is not None:
            if isinstance(result, dict):
                payload.update(result)
            else:
                serialized = self._serialize_connected_account(result)
                payload.update({k: v for k, v in serialized.items() if v is not None and v != ""})
                payload["deleted"] = True
        return {
            "ok": True,
            "connected_account": payload,
        }

    def search_tools(
        self,
        *,
        query: str = "",
        toolkits: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        safe_query = str(query or "").strip()
        normalized_toolkits = _coerce_toolkit_list(toolkits)
        if not safe_query and not normalized_toolkits:
            raise ValueError("query or toolkits is required")
        tools = self._sdk().tools.get_raw_composio_tools(
            search=safe_query or None,
            toolkits=normalized_toolkits or None,
            limit=max(1, min(int(limit), 50)),
        )
        return {
            "ok": True,
            "items": [self._serialize_tool_schema(item) for item in tools],
        }

    def get_tool_schema(self, *, tool_slug: str) -> dict[str, Any]:
        safe_slug = str(tool_slug or "").strip()
        if not safe_slug:
            raise ValueError("tool_slug is required")
        tool = self._sdk().tools.get_raw_composio_tool_by_slug(safe_slug)
        return {
            "ok": True,
            "tool": self._serialize_tool_schema(tool),
        }

    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        safe_slug = str(tool_slug or "").strip()
        if not safe_slug:
            raise ValueError("tool_slug is required")
        safe_arguments = dict(arguments) if isinstance(arguments, dict) else {}
        preflight: dict[str, Any] | None = None
        if safe_slug.upper() == "INSTAGRAM_SEND_TEXT_MESSAGE":
            preflight = self.inspect_instagram_reply_target(
                customer_id=customer_id,
                recipient_id=str(safe_arguments.get("recipient_id", "")).strip() or None,
                conversation_id=str(safe_arguments.pop("conversation_id", "")).strip() or None,
                connected_account_id=str(connected_account_id or "").strip() or None,
            )
            if not bool(preflight.get("recipient_id_verified")):
                return {
                    "ok": True,
                    "tool_slug": safe_slug,
                    "successful": False,
                    "error": (
                        "Instagram send blocked: could not verify the exact conversation for "
                        "this recipient_id. Inspect the thread first and retry with the verified target."
                    ),
                    "data": {"blocked": True, "preflight": preflight},
                }
            if not str(preflight.get("latest_inbound_message_created_time", "") or "").strip():
                return {
                    "ok": True,
                    "tool_slug": safe_slug,
                    "successful": False,
                    "error": (
                        "Instagram send blocked: no inbound message timestamp was found on the "
                        "verified thread, so OpenTulpa cannot claim the reply window is open."
                    ),
                    "data": {"blocked": True, "preflight": preflight},
                }
        result = self._sdk_execute_tool(
            slug=safe_slug,
            arguments=safe_arguments,
            connected_account_id=str(connected_account_id or "").strip() or None,
            user_id=str(customer_id or "").strip(),
            text=str(text or "").strip() or None,
        )
        retried_without_reply_to = False
        if (
            safe_slug.upper() == "INSTAGRAM_SEND_TEXT_MESSAGE"
            and str(safe_arguments.get("reply_to_message_id", "") or "").strip()
            and not bool(result.get("successful", False))
            and _is_invalid_instagram_reply_to_error(result.get("error"))
        ):
            retry_arguments = dict(safe_arguments)
            retry_arguments.pop("reply_to_message_id", None)
            retry_result = self._sdk_execute_tool(
                slug=safe_slug,
                arguments=retry_arguments,
                connected_account_id=str(connected_account_id or "").strip() or None,
                user_id=str(customer_id or "").strip(),
                text=str(text or "").strip() or None,
            )
            retried_without_reply_to = True
            result = retry_result
        data = result.get("data")
        if preflight is not None:
            payload = data if isinstance(data, dict) else {"result": data}
            payload["preflight"] = preflight
            if retried_without_reply_to:
                payload["retried_without_reply_to_message_id"] = True
                payload["retry_reason"] = "Meta rejected reply_to_message_id as invalid, so OpenTulpa retried as a plain DM."
            if not bool(result.get("successful", False)) and "outside of allowed window" in str(
                result.get("error") or ""
            ).lower():
                preflight["reply_window_status"] = "rejected_by_meta"
                preflight["reply_window_reason"] = (
                    "Meta rejected the send on this verified thread as outside the allowed window."
                )
            data = payload
        return {
            "ok": True,
            "tool_slug": safe_slug,
            "successful": bool(result.get("successful", False)),
            "error": result.get("error"),
            "data": data,
        }

    def inspect_instagram_reply_target(
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

        conversation: dict[str, Any] | None = None
        if safe_conversation:
            conversation = self._fetch_instagram_conversation(
                customer_id=safe_customer,
                conversation_id=safe_conversation,
                connected_account_id=safe_account,
            )
        else:
            conversation = self._find_instagram_conversation_for_recipient(
                customer_id=safe_customer,
                recipient_id=safe_recipient,
                connected_account_id=safe_account,
                scan_limit=scan_limit,
            )

        if not conversation:
            return {
                "ok": True,
                "customer_id": safe_customer,
                "conversation_id": safe_conversation or None,
                "recipient_id": safe_recipient or None,
                "recipient_id_verified": False,
                "matched": False,
                "reply_window_status": "unconfirmed",
                "reply_window_reason": "No Instagram conversation matching this target was found.",
            }

        summary = self._summarize_instagram_conversation(
            conversation=conversation,
            requested_recipient_id=safe_recipient or None,
        )
        summary["ok"] = True
        summary["customer_id"] = safe_customer
        summary["connected_account_id"] = safe_account
        return summary

    def _sdk_execute_tool(
        self,
        *,
        slug: str,
        arguments: dict[str, Any],
        connected_account_id: str | None,
        user_id: str,
        text: str | None = None,
    ) -> dict[str, Any]:
        return self._sdk().tools.execute(
            slug=slug,
            arguments=arguments,
            connected_account_id=connected_account_id,
            user_id=user_id,
            text=text,
            dangerously_skip_version_check=True,
        )

    def _fetch_instagram_conversation(
        self,
        *,
        customer_id: str,
        conversation_id: str,
        connected_account_id: str | None,
    ) -> dict[str, Any]:
        result = self._sdk_execute_tool(
            slug="INSTAGRAM_GET_CONVERSATION",
            arguments={"conversation_id": conversation_id},
            connected_account_id=connected_account_id,
            user_id=customer_id,
        )
        if not bool(result.get("successful", False)):
            raise RuntimeError(str(result.get("error") or "failed to fetch Instagram conversation"))
        payload = _safe_dict(result.get("data"))
        return payload

    def _find_instagram_conversation_for_recipient(
        self,
        *,
        customer_id: str,
        recipient_id: str,
        connected_account_id: str | None,
        scan_limit: int,
    ) -> dict[str, Any] | None:
        if not recipient_id:
            return None
        response = self._sdk_execute_tool(
            slug="INSTAGRAM_LIST_ALL_CONVERSATIONS",
            arguments={"limit": max(1, min(int(scan_limit), 25))},
            connected_account_id=connected_account_id,
            user_id=customer_id,
        )
        if not bool(response.get("successful", False)):
            raise RuntimeError(str(response.get("error") or "failed to list Instagram conversations"))
        items = _safe_list(_safe_dict(response.get("data")).get("data"))
        for item in items:
            conversation_id = str(_safe_dict(item).get("id", "") or "").strip()
            if not conversation_id:
                continue
            conversation = self._fetch_instagram_conversation(
                customer_id=customer_id,
                conversation_id=conversation_id,
                connected_account_id=connected_account_id,
            )
            participant_ids = {
                str(_safe_dict(participant).get("id", "") or "").strip()
                for participant in _safe_list(_safe_dict(conversation.get("participants")).get("data"))
            }
            if recipient_id in participant_ids:
                return conversation
        return None

    @staticmethod
    def _summarize_instagram_conversation(
        *,
        conversation: dict[str, Any],
        requested_recipient_id: str | None,
    ) -> dict[str, Any]:
        payload = _safe_dict(conversation.get("data")) if "data" in conversation else conversation
        participants = _safe_list(_safe_dict(payload.get("participants")).get("data"))
        messages = _safe_list(_safe_dict(payload.get("messages")).get("data"))
        participant_ids = [
            str(_safe_dict(item).get("id", "") or "").strip()
            for item in participants
            if str(_safe_dict(item).get("id", "") or "").strip()
        ]
        participant_usernames = {
            str(_safe_dict(item).get("id", "") or "").strip(): str(_safe_dict(item).get("username", "") or "").strip()
            for item in participants
            if str(_safe_dict(item).get("id", "") or "").strip()
        }
        verified_recipient = str(requested_recipient_id or "").strip() or None
        if verified_recipient and verified_recipient not in participant_ids:
            verified_recipient = None
        if not verified_recipient and len(participant_ids) == 2:
            verified_recipient = participant_ids[1]

        own_participant_ids = [item for item in participant_ids if item != verified_recipient] if verified_recipient else []
        normalized_messages = []
        for item in messages:
            msg = _safe_dict(item)
            sender = _safe_dict(msg.get("from"))
            recipients = _safe_list(_safe_dict(msg.get("to")).get("data"))
            normalized_messages.append(
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
        normalized_messages.sort(key=lambda item: item["created_at"] or datetime.min, reverse=True)

        latest_message = normalized_messages[0] if normalized_messages else None
        latest_inbound = None
        latest_outbound = None
        for item in normalized_messages:
            sender_id = item["from_id"]
            if verified_recipient and sender_id == verified_recipient and latest_inbound is None:
                latest_inbound = item
            if own_participant_ids and sender_id in own_participant_ids and latest_outbound is None:
                latest_outbound = item
            if latest_inbound is not None and (latest_outbound is not None or not own_participant_ids):
                break

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
            "reply_window_reason": (
                "Exact thread verified and latest inbound timestamp captured, but Meta still decides whether the "
                "reply window is open at send time."
                if latest_inbound
                else "Exact thread verified, but no inbound message timestamp was found on this thread."
            ),
        }

    @staticmethod
    def _serialize_connected_account(item: Any) -> dict[str, Any]:
        auth_config = getattr(item, "auth_config", None)
        toolkit = getattr(item, "toolkit", None)
        return {
            "id": str(getattr(item, "id", "") or getattr(item, "nanoid", "") or ""),
            "status": str(getattr(item, "status", "") or ""),
            "user_id": str(getattr(item, "user_id", "") or ""),
            "toolkit_slug": str(getattr(toolkit, "slug", "") or ""),
            "toolkit_name": str(getattr(toolkit, "name", "") or ""),
            "auth_config_id": str(getattr(auth_config, "id", "") or "") or None,
            "auth_scheme": str(getattr(auth_config, "auth_scheme", "") or "") or None,
        }

    @staticmethod
    def _serialize_tool_schema(item: Any) -> dict[str, Any]:
        toolkit = getattr(item, "toolkit", None)
        return {
            "slug": str(getattr(item, "slug", "") or ""),
            "name": str(getattr(item, "name", "") or ""),
            "description": str(getattr(item, "description", "") or ""),
            "toolkit_slug": str(getattr(toolkit, "slug", "") or ""),
            "toolkit_name": str(getattr(toolkit, "name", "") or ""),
            "input_schema": getattr(item, "input_parameters", None),
        }
