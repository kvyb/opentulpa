"""CapSolver API adapter for optional CAPTCHA solving."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx


class CapSolverError(RuntimeError):
    """Raised when CapSolver cannot create or complete a solve task."""


@dataclass(frozen=True, slots=True)
class CapSolverSolveResult:
    task_id: str
    token: str
    captcha_type: str


class CapSolverClient:
    """Small async CapSolver client with no browser-specific behavior."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.capsolver.com",
        poll_interval_seconds: float = 5.0,
        timeout_seconds: float = 120.0,
        request_timeout_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._base_url = str(base_url or "").strip().rstrip("/") or "https://api.capsolver.com"
        self._poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self._http_client = http_client

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    async def get_balance(self) -> dict[str, Any]:
        payload = await self._post_json("/getBalance", {"clientKey": self._require_api_key()})
        self._raise_for_capsolver_error(payload, operation="getBalance")
        return payload

    async def solve_recaptcha_v2(
        self,
        *,
        website_url: str,
        website_key: str,
    ) -> CapSolverSolveResult:
        task_id = await self._create_task(
            {
                "type": "ReCaptchaV2TaskProxyLess",
                "websiteURL": self._require_text(website_url, "website_url"),
                "websiteKey": self._require_text(website_key, "website_key"),
            }
        )
        token = await self._poll_for_token(
            task_id=task_id,
            solution_keys=("gRecaptchaResponse", "token"),
        )
        return CapSolverSolveResult(task_id=task_id, token=token, captcha_type="recaptcha_v2")

    async def solve_recaptcha_v3(
        self,
        *,
        website_url: str,
        website_key: str,
        page_action: str | None = None,
    ) -> CapSolverSolveResult:
        task = {
            "type": "ReCaptchaV3TaskProxyLess",
            "websiteURL": self._require_text(website_url, "website_url"),
            "websiteKey": self._require_text(website_key, "website_key"),
        }
        safe_action = str(page_action or "").strip()
        if safe_action:
            task["pageAction"] = safe_action
        task_id = await self._create_task(task)
        token = await self._poll_for_token(
            task_id=task_id,
            solution_keys=("gRecaptchaResponse", "token"),
        )
        return CapSolverSolveResult(task_id=task_id, token=token, captcha_type="recaptcha_v3")

    async def solve_turnstile(
        self,
        *,
        website_url: str,
        website_key: str,
    ) -> CapSolverSolveResult:
        task_id = await self._create_task(
            {
                "type": "AntiTurnstileTaskProxyLess",
                "websiteURL": self._require_text(website_url, "website_url"),
                "websiteKey": self._require_text(website_key, "website_key"),
            }
        )
        token = await self._poll_for_token(
            task_id=task_id,
            solution_keys=("token", "turnstileToken"),
        )
        return CapSolverSolveResult(task_id=task_id, token=token, captcha_type="turnstile")

    async def _create_task(self, task: dict[str, Any]) -> str:
        payload = await self._post_json(
            "/createTask",
            {
                "clientKey": self._require_api_key(),
                "task": task,
            },
        )
        self._raise_for_capsolver_error(payload, operation="createTask")
        task_id = str(payload.get("taskId") or "").strip()
        if not task_id:
            raise CapSolverError("CapSolver createTask failed: missing taskId")
        return task_id

    async def _poll_for_token(
        self,
        *,
        task_id: str,
        solution_keys: tuple[str, ...],
    ) -> str:
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            payload = await self._post_json(
                "/getTaskResult",
                {
                    "clientKey": self._require_api_key(),
                    "taskId": self._require_text(task_id, "task_id"),
                },
            )
            self._raise_for_capsolver_error(payload, operation="getTaskResult")
            status = str(payload.get("status") or "").strip().lower()
            if status == "ready":
                solution = payload.get("solution")
                token = self._extract_solution_token(solution, solution_keys=solution_keys)
                if token:
                    return token
                raise CapSolverError("CapSolver getTaskResult failed: ready result had no token")
            if status == "failed":
                raise CapSolverError("CapSolver getTaskResult failed: task status is failed")
            if time.monotonic() >= deadline:
                raise CapSolverError("CapSolver getTaskResult timed out waiting for a solution")
            await asyncio.sleep(self._poll_interval_seconds)

    async def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        try:
            if self._http_client is not None:
                response = await self._http_client.post(url, json=payload)
            else:
                timeout = httpx.Timeout(self._request_timeout_seconds)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CapSolverError(f"CapSolver request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise CapSolverError("CapSolver returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise CapSolverError("CapSolver returned a non-object JSON response")
        return data

    def _require_api_key(self) -> str:
        if not self._api_key:
            raise CapSolverError("CAPSOLVER_API_KEY is not configured")
        return self._api_key

    @staticmethod
    def _require_text(value: str, name: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise CapSolverError(f"CapSolver {name} is required")
        return text

    @staticmethod
    def _extract_solution_token(
        solution: Any,
        *,
        solution_keys: tuple[str, ...],
    ) -> str:
        if not isinstance(solution, dict):
            return ""
        for key in solution_keys:
            token = str(solution.get(key) or "").strip()
            if token:
                return token
        return ""

    @staticmethod
    def _raise_for_capsolver_error(payload: dict[str, Any], *, operation: str) -> None:
        error_id = int(payload.get("errorId") or 0)
        if error_id == 0:
            return
        code = str(payload.get("errorCode") or "").strip()
        description = str(payload.get("errorDescription") or "").strip()
        detail = ": ".join(part for part in (code, description) if part)
        if detail:
            raise CapSolverError(f"CapSolver {operation} failed: {detail}")
        raise CapSolverError(f"CapSolver {operation} failed")
