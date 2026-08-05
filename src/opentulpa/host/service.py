"""Activation transaction joining validation, runtime lifecycle, and capabilities."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import httpx

from opentulpa.host.models import HostConfig, HostConfigInput, HostConfigView
from opentulpa.host.runtime import RuntimeSupervisor
from opentulpa.host.store import HostStore


class HostActivationError(RuntimeError):
    """A candidate configuration failed without replacing the active revision."""


class HostService:
    """Keep the host stable while atomically replacing its mutable child runtime."""

    def __init__(
        self,
        *,
        store: HostStore,
        runtime: RuntimeSupervisor,
        client: httpx.AsyncClient | None = None,
        bootstrap_config: HostConfigInput | None = None,
        evolution: Any | None = None,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=30, write=30, pool=5), trust_env=False
        )
        self._owns_client = client is None
        self._bootstrap_config = bootstrap_config
        self._evolution = evolution
        self._activation_lock = asyncio.Lock()
        self._activating = False

    @property
    def activating(self) -> bool:
        return self._activating

    async def start(self) -> None:
        if self._evolution is not None:
            await self._evolution.prepare()
        active = self.store.active()
        if active is None:
            if self._bootstrap_config is not None:
                try:
                    await self.apply(self._bootstrap_config)
                except Exception:
                    return
            return
        if self._bootstrap_config is not None and self._bootstrap_config.expected_revision == active.revision:
            try:
                await self.apply(self._bootstrap_config)
                return
            except Exception:
                active = self.store.active()
                if active is None:
                    return
        try:
            await self.runtime.start(active)
        except Exception:
            # The stable setup and recovery surface stays available.
            return
        if self._evolution is not None:
            await self._evolution.start()

    async def shutdown(self) -> None:
        failures: list[BaseException] = []
        if self._evolution is not None:
            try:
                await self._evolution.shutdown()
            except BaseException as exc:
                failures.append(exc)
        try:
            await self.runtime.shutdown()
        except BaseException as exc:
            failures.append(exc)
        if self._owns_client:
            try:
                await self._client.aclose()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]

    async def apply(self, value: HostConfigInput) -> HostConfigView:
        async with self._activation_lock:
            return await self._apply(value)

    async def _apply(self, value: HostConfigInput) -> HostConfigView:
        previous = self.store.active()
        await self._validate_external(value, previous=previous)
        staged = self.store.stage(value)
        self._activating = True
        try:
            await self.runtime.replace(staged, rollback=previous)
            await self._configure_telegram(staged)
            if self._evolution is not None:
                await self._evolution.start()
            active = self.store.activate(staged.revision)
        except Exception as exc:
            self.store.fail(staged.revision, self._safe_error(exc))
            if previous is None:
                await self.runtime.stop()
            elif self.runtime.revision != previous.revision:
                with suppress(Exception):
                    await self.runtime.replace(previous, rollback=previous)
                    await self._configure_telegram(previous)
            raise HostActivationError(self._safe_error(exc)) from exc
        finally:
            self._activating = False
        return self.store.view(active)

    async def restart(self) -> None:
        async with self._activation_lock:
            self._activating = True
            try:
                await self.runtime.restart_current()
            finally:
                self._activating = False

    async def _validate_external(
        self,
        value: HostConfigInput,
        *,
        previous: HostConfig | None,
    ) -> None:
        api_key = (
            value.api_key.get_secret_value()
            if value.api_key is not None
            else previous.api_key.get_secret_value()
            if previous is not None
            else ""
        )
        if not api_key:
            raise HostActivationError("A model API key is required.")
        try:
            response = await self._client.get(
                f"{value.base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        except httpx.HTTPError as exc:
            raise HostActivationError("The model endpoint could not be reached.") from exc
        if not response.is_success:
            raise HostActivationError("The model endpoint rejected these credentials.")

        telegram_token = (
            value.telegram_bot_token.get_secret_value()
            if value.telegram_bot_token is not None
            else previous.telegram_bot_token.get_secret_value()
            if previous is not None
            and previous.telegram_bot_token is not None
            and value.telegram_user_id is not None
            else None
        )
        if telegram_token is None:
            return
        try:
            response = await self._client.get(f"https://api.telegram.org/bot{telegram_token}/getMe")
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HostActivationError("Telegram could not validate this bot token.") from exc
        if not response.is_success or not isinstance(body, dict) or body.get("ok") is not True:
            raise HostActivationError("Telegram rejected this bot token.")

    async def _configure_telegram(self, config: HostConfig) -> None:
        endpoint = self.runtime.endpoint
        if endpoint is None:
            raise HostActivationError("The runtime is unavailable.")
        headers = {"Authorization": f"Bearer {config.internal_runtime_token.get_secret_value()}"}
        capability = await self._request(
            "GET", f"{endpoint}/v2/capabilities/telegram", headers=headers
        )
        manifest = capability.get("manifest")
        activation = capability.get("activation")
        if not isinstance(manifest, dict):
            raise HostActivationError("The bundled Telegram capability is unavailable.")
        revision = int(manifest["revision"])
        generation = int(activation["generation"]) if isinstance(activation, dict) else None
        if config.telegram_bot_token is None:
            if generation is not None:
                await self._request(
                    "DELETE",
                    f"{endpoint}/v2/capabilities/telegram",
                    headers=headers,
                    params={"expected_generation": generation},
                )
            return

        secret_id = "telegram-bot-token"
        secret_response = await self._client.get(
            f"{endpoint}/v2/secrets/{secret_id}", headers=headers
        )
        if secret_response.status_code == 404:
            pending = await self._request(
                "POST",
                f"{endpoint}/v2/secrets/pending",
                headers=headers,
                json={
                    "id": secret_id,
                    "name": secret_id,
                    "scopes": ["telegram.receive", "telegram.send"],
                },
            )
            secret = pending["secret"]
        elif secret_response.is_success:
            secret = secret_response.json()["secret"]
        else:
            raise HostActivationError("The Telegram credential store is unavailable.")
        stored = await self._request(
            "PUT",
            f"{endpoint}/v2/secrets/{secret_id}",
            headers=headers,
            json={
                "expected_revision": int(secret["revision"]),
                "value": config.telegram_bot_token.get_secret_value(),
            },
        )
        if not stored.get("secret"):
            raise HostActivationError("The Telegram credential was not stored.")
        await self._request(
            "POST",
            f"{endpoint}/v2/capabilities/telegram/test",
            headers=headers,
            json={"revision": revision},
        )
        await self._request(
            "POST",
            f"{endpoint}/v2/capabilities/telegram/activate",
            headers=headers,
            json={
                "revision": revision,
                "expected_generation": generation,
                "config": {},
                "secret_handles": {"TELEGRAM_BOT_TOKEN": secret_id},
                "refresh_agent_binding": generation is not None,
            },
        )

    async def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise HostActivationError("The runtime control plane could not be reached.") from exc
        if not response.is_success:
            detail = ""
            try:
                payload = response.json()
                detail = str(payload.get("detail") or "") if isinstance(payload, dict) else ""
            except ValueError:
                pass
            raise HostActivationError(detail or "The runtime rejected its configuration.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise HostActivationError("The runtime returned an invalid control response.") from exc
        if not isinstance(payload, dict):
            raise HostActivationError("The runtime returned an invalid control response.")
        return payload

    @staticmethod
    def _safe_error(error: Exception) -> str:
        if isinstance(error, HostActivationError):
            return str(error)[:1_000]
        return "The runtime could not activate this configuration."


__all__ = ["HostActivationError", "HostService"]
