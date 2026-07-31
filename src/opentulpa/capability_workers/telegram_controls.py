"""Telegram inference and Codex commands backed only by the Agent API."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal, Protocol, cast

from opentulpa.capability_workers.agent_api import AgentAPIError
from opentulpa.capability_workers.state import TelegramWorkerState

logger = logging.getLogger(__name__)
INFERENCE_COMMANDS = frozenset({"/model", "/models", "/reasoning", "/codex"})


class InferenceClient(Protocol):
    async def get_owner_inference(self) -> dict[str, Any]: ...

    async def update_owner_inference(
        self,
        *,
        expected_revision: int,
        selection: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    async def inference_status(self) -> dict[str, Any]: ...

    async def list_models(
        self,
        *,
        provider: Literal["api", "codex"],
        query: str = "",
    ) -> list[dict[str, Any]]: ...

    async def start_codex_login(self) -> dict[str, Any]: ...

    async def get_codex_login(self, login_id: str) -> dict[str, Any]: ...


class MessageSender(Protocol):
    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> list[int]: ...


class TelegramInferenceControls:
    def __init__(
        self,
        *,
        agent: InferenceClient,
        telegram: MessageSender,
        state: TelegramWorkerState,
    ) -> None:
        self._agent = agent
        self._telegram = telegram
        self._state = state

    async def handle(
        self,
        *,
        chat_id: int,
        thread_id: str,
        text: str,
    ) -> bool:
        parts = str(text or "").split()
        command = parts[0].split("@", 1)[0].casefold() if parts else ""
        if command not in INFERENCE_COMMANDS:
            return False
        try:
            del thread_id
            if command == "/model":
                await self._model(chat_id, parts[1:])
            elif command == "/models":
                await self._models(chat_id, parts[1:])
            elif command == "/reasoning":
                await self._reasoning(chat_id, parts[1:])
            else:
                await self._codex(chat_id, parts[1:])
        except (AgentAPIError, ValueError) as exc:
            await self._send(chat_id, str(exc))
        except Exception:
            logger.exception("Telegram inference command failed", extra={"command": command})
            await self._send(chat_id, "The inference command could not be completed.")
        return True

    async def _model(
        self,
        chat_id: int,
        arguments: list[str],
    ) -> None:
        current = await self._agent.get_owner_inference()
        if not arguments:
            await self._send(chat_id, self._selection_text("Global model", current))
            return
        if len(arguments) not in {2, 3}:
            raise ValueError("Usage: /model <api|codex> <model> [reasoning]")
        provider = arguments[0].casefold()
        if provider not in {"api", "codex"}:
            raise ValueError("Provider must be api or codex.")
        selection = {
            "provider": provider,
            "model": arguments[1],
            "reasoning_effort": arguments[2] if len(arguments) == 3 else None,
        }
        updated = await self._agent.update_owner_inference(
            expected_revision=int(current["revision"]),
            selection=selection,
        )
        await self._send(chat_id, self._selection_text("Global model selected", updated))

    async def _models(
        self,
        chat_id: int,
        arguments: list[str],
    ) -> None:
        current = await self._agent.get_owner_inference()
        effective = self._effective(current)
        explicit_provider = arguments and arguments[0].casefold() in {"api", "codex"}
        provider = (
            arguments[0].casefold()
            if explicit_provider
            else str(effective.get("provider") or "api")
        )
        query = " ".join(arguments[1:] if explicit_provider else arguments)
        models = await self._agent.list_models(
            provider=cast(Literal["api", "codex"], provider),
            query=query,
        )
        lines = [f"Available {provider} models:"]
        for model in models[:20]:
            efforts = model.get("reasoning_efforts")
            suffix = (
                f" [{', '.join(str(item) for item in efforts)}]"
                if isinstance(efforts, list) and efforts
                else ""
            )
            lines.append(f"- {model.get('id')}{suffix}")
        if not models:
            lines.append("- No matching models.")
        if len(models) > 20:
            lines.append(f"...and {len(models) - 20} more. Add a search term to /models.")
        lines.append(f"Select globally with: /model {provider} <model> [reasoning]")
        await self._send(chat_id, "\n".join(lines))

    async def _reasoning(
        self,
        chat_id: int,
        arguments: list[str],
    ) -> None:
        if len(arguments) != 1:
            raise ValueError("Usage: /reasoning <minimal|low|medium|high|xhigh|max|ultra>")
        current = await self._agent.get_owner_inference()
        effective = self._effective(current)
        selection = {
            "provider": effective["provider"],
            "model": effective["model"],
            "reasoning_effort": arguments[0].casefold(),
            "service_tier": effective.get("service_tier"),
            "fallback_to_api": bool(effective.get("fallback_to_api", False)),
        }
        updated = await self._agent.update_owner_inference(
            expected_revision=int(current["revision"]),
            selection=selection,
        )
        await self._send(chat_id, self._selection_text("Global reasoning updated", updated))

    async def _codex(self, chat_id: int, arguments: list[str]) -> None:
        action = arguments[0].casefold() if arguments else "status"
        if action == "login":
            status = await self._agent.inference_status()
            if bool(status.get("codex", {}).get("connected")):
                await self._send(
                    chat_id,
                    "Codex is already connected. Use /models codex to select the global model.",
                )
                return
            login = await self._agent.start_codex_login()
            login_id = str(login["login_id"])
            self._state.set_codex_login(chat_id, login_id)
            await self._send(
                chat_id,
                (
                    "Connect Codex:\n"
                    f"1. Open {login['verification_url']}\n"
                    f"2. Enter code {login['user_code']}\n"
                    f"3. Run /codex status {login_id}"
                ),
            )
            return
        if action != "status" or len(arguments) > 2:
            raise ValueError("Usage: /codex [login|status [login_id]]")
        login_id = arguments[1] if len(arguments) == 2 else self._state.codex_login(chat_id)
        if login_id:
            login = await self._agent.get_codex_login(login_id)
            login_status = str(login.get("status") or "unknown")
            if login_status == "authorized":
                self._state.set_codex_login(chat_id, None)
                await self._send(
                    chat_id,
                    "Codex is connected. Use /models codex to select the global model.",
                )
                return
            if login_status in {"expired", "failed"}:
                self._state.set_codex_login(chat_id, None)
                await self._send(
                    chat_id,
                    f"Codex login {login_status}. Run /codex login to start again.",
                )
                return
            await self._send(
                chat_id,
                (
                    f"Codex login status: {login_status}. "
                    f"Run /codex status {login_id} to check again."
                ),
            )
            return
        status = await self._agent.inference_status()
        connected = bool(status.get("codex", {}).get("connected"))
        await self._send(
            chat_id,
            (
                "Codex is connected. Use /models codex to select the global model."
                if connected
                else "Codex is not connected. Run /codex login."
            ),
        )

    @staticmethod
    def _effective(result: Mapping[str, Any]) -> Mapping[str, Any]:
        effective = result.get("effective")
        if not isinstance(effective, dict):
            raise AgentAPIError("Agent API returned invalid inference state.")
        return effective

    @classmethod
    def _selection_text(cls, title: str, result: Mapping[str, Any]) -> str:
        selection = cls._effective(result)
        reasoning = str(selection.get("reasoning_effort") or "provider default")
        return (
            f"{title}:\n"
            f"Provider: {selection.get('provider')}\n"
            f"Model: {selection.get('model')}\n"
            f"Reasoning: {reasoning}"
        )

    async def _send(self, chat_id: int, text: str) -> None:
        await self._telegram.send_message(chat_id=chat_id, text=text)


__all__ = ["INFERENCE_COMMANDS", "TelegramInferenceControls"]
