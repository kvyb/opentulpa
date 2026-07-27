"""Telegram slash commands for per-thread inference and Codex authentication."""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, cast

from opentulpa.inference import InferenceModel, InferenceSelection
from opentulpa.inference.service import InferenceConflictError, InferenceUnavailableError
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.state_store import TelegramStateStore
from opentulpa.tooling import AgentChannel

logger = logging.getLogger(__name__)
_INFERENCE_COMMANDS = {"/model", "/models", "/reasoning", "/codex"}


class TelegramInferenceAgent(Protocol):
    async def ensure_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        channel: str,
    ) -> None: ...

    async def get_thread_inference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> dict[str, Any] | None: ...

    async def update_thread_inference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        expected_revision: int,
        selection: InferenceSelection | None,
    ) -> dict[str, Any] | None: ...


class TelegramInferenceService(Protocol):
    def codex_connected(self, tenant_id: str) -> bool: ...

    async def status(self, tenant_id: str) -> dict[str, Any]: ...

    async def models(
        self,
        tenant_id: str,
        provider: Literal["api", "codex"],
        *,
        query: str = "",
    ) -> tuple[InferenceModel, ...]: ...

    async def start_device_login(self, tenant_id: str) -> dict[str, Any]: ...

    async def get_device_login(
        self,
        tenant_id: str,
        login_id: str,
    ) -> dict[str, Any] | None: ...


class TelegramInferenceControls:
    def __init__(
        self,
        *,
        agent: TelegramInferenceAgent,
        inference: TelegramInferenceService | None,
        client: TelegramClient,
        state: TelegramStateStore,
    ) -> None:
        self._agent = agent
        self._inference = inference
        self._client = client
        self._state = state

    async def handle(
        self,
        *,
        chat_id: int,
        tenant_id: str,
        thread_id: str,
        text: str,
    ) -> bool:
        parts = str(text or "").split()
        if not parts:
            return False
        command = parts[0].split("@", 1)[0].casefold()
        if command not in _INFERENCE_COMMANDS:
            return False
        if self._inference is None:
            await self._send_plain(chat_id, "Inference controls are unavailable.")
            return True
        try:
            await self._agent.ensure_thread(
                tenant_id=tenant_id,
                thread_id=thread_id,
                channel=AgentChannel.TELEGRAM.value,
            )
            if command == "/model":
                await self._handle_model(
                    chat_id=chat_id,
                    tenant_id=tenant_id,
                    thread_id=thread_id,
                    arguments=parts[1:],
                )
            elif command == "/models":
                await self._handle_models(
                    chat_id=chat_id,
                    tenant_id=tenant_id,
                    thread_id=thread_id,
                    arguments=parts[1:],
                )
            elif command == "/reasoning":
                await self._handle_reasoning(
                    chat_id=chat_id,
                    tenant_id=tenant_id,
                    thread_id=thread_id,
                    arguments=parts[1:],
                )
            else:
                await self._handle_codex(
                    chat_id=chat_id,
                    tenant_id=tenant_id,
                    arguments=parts[1:],
                )
        except (InferenceConflictError, InferenceUnavailableError, ValueError) as exc:
            await self._send_plain(chat_id, str(exc))
        except Exception:
            logger.exception(
                "Telegram inference command failed",
                extra={"tenant_id": tenant_id, "thread_id": thread_id, "command": command},
            )
            await self._send_plain(chat_id, "The inference command could not be completed.")
        return True

    async def _handle_model(
        self,
        *,
        chat_id: int,
        tenant_id: str,
        thread_id: str,
        arguments: list[str],
    ) -> None:
        current = await self._thread_inference(tenant_id=tenant_id, thread_id=thread_id)
        if not arguments:
            await self._send_plain(
                chat_id,
                self._selection_text("Current model", current["effective"]),
            )
            return
        if len(arguments) not in {2, 3}:
            raise ValueError("Usage: /model <api|codex> <model> [reasoning]")
        provider = arguments[0].casefold()
        if provider not in {"api", "codex"}:
            raise ValueError("Provider must be api or codex.")
        selection = InferenceSelection(
            provider=cast(Literal["api", "codex"], provider),
            model=arguments[1],
            reasoning_effort=arguments[2] if len(arguments) > 2 else None,
        )
        updated = await self._agent.update_thread_inference(
            tenant_id=tenant_id,
            thread_id=thread_id,
            expected_revision=int(current["revision"]),
            selection=selection,
        )
        if updated is None:
            raise RuntimeError("Telegram conversation is unavailable")
        await self._send_plain(
            chat_id,
            self._selection_text("Model selected", updated["effective"]),
        )

    async def _handle_models(
        self,
        *,
        chat_id: int,
        tenant_id: str,
        thread_id: str,
        arguments: list[str],
    ) -> None:
        current = await self._thread_inference(tenant_id=tenant_id, thread_id=thread_id)
        provider = (
            arguments[0].casefold()
            if arguments and arguments[0].casefold() in {"api", "codex"}
            else str(current["effective"]["provider"])
        )
        query_parts = arguments[1:] if arguments and arguments[0].casefold() == provider else arguments
        models = await self._require_inference().models(
            tenant_id,
            cast(Literal["api", "codex"], provider),
            query=" ".join(query_parts),
        )
        lines = [f"Available {provider} models:"]
        for model in models[:20]:
            efforts = f" [{', '.join(model.reasoning_efforts)}]" if model.reasoning_efforts else ""
            lines.append(f"- {model.id}{efforts}")
        if len(models) > 20:
            lines.append(f"...and {len(models) - 20} more. Add a search term to /models.")
        lines.append(f"Select with: /model {provider} <model> [reasoning]")
        await self._send_plain(chat_id, "\n".join(lines))

    async def _handle_reasoning(
        self,
        *,
        chat_id: int,
        tenant_id: str,
        thread_id: str,
        arguments: list[str],
    ) -> None:
        if len(arguments) != 1:
            raise ValueError("Usage: /reasoning <minimal|low|medium|high|xhigh|max|ultra>")
        current = await self._thread_inference(tenant_id=tenant_id, thread_id=thread_id)
        effective = current["effective"]
        selection = InferenceSelection(
            provider=effective["provider"],
            model=effective["model"],
            reasoning_effort=arguments[0].casefold(),
            service_tier=effective.get("service_tier"),
            fallback_to_api=bool(effective.get("fallback_to_api", False)),
        )
        updated = await self._agent.update_thread_inference(
            tenant_id=tenant_id,
            thread_id=thread_id,
            expected_revision=int(current["revision"]),
            selection=selection,
        )
        if updated is None:
            raise RuntimeError("Telegram conversation is unavailable")
        await self._send_plain(
            chat_id,
            self._selection_text("Reasoning updated", updated["effective"]),
        )

    async def _handle_codex(
        self,
        *,
        chat_id: int,
        tenant_id: str,
        arguments: list[str],
    ) -> None:
        action = arguments[0].casefold() if arguments else "status"
        if action == "login":
            inference = self._require_inference()
            if inference.codex_connected(tenant_id):
                await self._send_plain(
                    chat_id,
                    "Codex is already connected. Use /models codex to select a model.",
                )
                return
            login = await inference.start_device_login(tenant_id)
            login_id = str(login["id"])
            self._remember_codex_login(chat_id, login_id)
            await self._send_plain(
                chat_id,
                (
                    "Connect Codex:\n"
                    f"1. Open {login['verification_url']}\n"
                    f"2. Enter code {login['user_code']}\n"
                    f"3. Run /codex status {login_id}"
                ),
            )
            return
        if action != "status":
            raise ValueError("Usage: /codex [login|status [login_id]]")
        login_id = arguments[1] if len(arguments) > 1 else self._codex_login(chat_id)
        if login_id:
            login_state = await self._require_inference().get_device_login(
                tenant_id,
                login_id,
            )
            if login_state is None:
                raise ValueError("Codex login was not found. Start again with /codex login.")
            login_status = str(login_state.get("status") or "unknown")
            if login_status == "authorized":
                self._remember_codex_login(chat_id, None)
                await self._send_plain(
                    chat_id,
                    "Codex is connected. Use /models codex to select a model.",
                )
                return
            await self._send_plain(
                chat_id,
                (
                    f"Codex login status: {login_status}. "
                    f"Run /codex status {login_id} to check again."
                ),
            )
            return
        service_status = await self._require_inference().status(tenant_id)
        connected = bool(service_status.get("codex", {}).get("connected"))
        await self._send_plain(
            chat_id,
            (
                "Codex is connected. Use /models codex to select a model."
                if connected
                else "Codex is not connected. Run /codex login."
            ),
        )

    async def _thread_inference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> dict[str, Any]:
        current = await self._agent.get_thread_inference(
            tenant_id=tenant_id,
            thread_id=thread_id,
        )
        if current is None:
            raise RuntimeError("Telegram conversation is unavailable")
        return current

    def _require_inference(self) -> TelegramInferenceService:
        if self._inference is None:
            raise RuntimeError("Inference controls are unavailable")
        return self._inference

    @staticmethod
    def _selection_text(title: str, selection: dict[str, Any]) -> str:
        reasoning = str(selection.get("reasoning_effort") or "provider default")
        return (
            f"{title}:\n"
            f"Provider: {selection.get('provider')}\n"
            f"Model: {selection.get('model')}\n"
            f"Reasoning: {reasoning}"
        )

    async def _send_plain(self, chat_id: int, text: str) -> None:
        await self._client.send_message(chat_id=chat_id, text=text, parse_mode=None)

    def _remember_codex_login(self, chat_id: int, login_id: str | None) -> None:
        def update(state: dict[str, Any]) -> None:
            logins = state.get("codex_logins")
            if not isinstance(logins, dict):
                logins = {}
            if login_id:
                logins[str(chat_id)] = login_id
            else:
                logins.pop(str(chat_id), None)
            state["codex_logins"] = logins

        self._state.update(update)

    def _codex_login(self, chat_id: int) -> str:
        logins = self._state.load().get("codex_logins")
        if not isinstance(logins, dict):
            return ""
        return str(logins.get(str(chat_id)) or "").strip()


__all__ = [
    "TelegramInferenceAgent",
    "TelegramInferenceControls",
    "TelegramInferenceService",
]
