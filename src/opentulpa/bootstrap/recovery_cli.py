"""Host-only CLI for the immutable bootstrap recovery API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx


class RecoveryCliError(RuntimeError):
    """Sanitized host CLI failure."""


class RecoveryClient:
    """Small non-browser client that never puts recovery authority in a URL."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = _recovery_url(base_url)
        safe_token = str(token or "").strip()
        if len(safe_token) < 32 or any(character.isspace() for character in safe_token):
            raise RecoveryCliError("OPENTULPA_RECOVERY_TOKEN is missing or invalid")
        self._client = client or httpx.Client(
            headers={"Authorization": f"Bearer {safe_token}"},
            follow_redirects=False,
            timeout=httpx.Timeout(30.0, read=300.0),
            trust_env=False,
        )
        self._owns_client = client is None
        if client is not None:
            self._client.headers["Authorization"] = f"Bearer {safe_token}"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def status(self) -> dict[str, Any]:
        return self._json("GET", "/bootstrap/v1/status")

    def rollback(self, *, reason: str) -> dict[str, Any]:
        return self._json(
            "POST",
            "/bootstrap/v1/rollback",
            json={"reason": str(reason or "")[:4_000]},
        )

    def restart(self) -> dict[str, Any]:
        return self._json("POST", "/bootstrap/v1/restart")

    def safe_mode(self) -> dict[str, Any]:
        return self._json("POST", "/bootstrap/v1/safe-mode")

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, f"{self._base_url}{path}", **kwargs)
            self._raise_for_status(response)
            payload = response.json()
        except httpx.HTTPError as exc:
            raise RecoveryCliError("recovery service is unavailable") from exc
        except ValueError as exc:
            raise RecoveryCliError("recovery service returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RecoveryCliError("recovery service returned an invalid response")
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if 200 <= response.status_code < 300:
            return
        detail = "request failed"
        try:
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
                detail = payload["detail"][:500]
        except ValueError:
            pass
        raise RecoveryCliError(f"recovery request failed ({response.status_code}): {detail}")


def run(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client: httpx.Client | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    environment = os.environ if environ is None else environ
    try:
        recovery = RecoveryClient(
            base_url=environment.get("OPENTULPA_RECOVERY_URL", "http://127.0.0.1:8000"),
            token=environment.get("OPENTULPA_RECOVERY_TOKEN", ""),
            client=client,
        )
        try:
            payload = _execute(recovery, args)
        finally:
            recovery.close()
    except RecoveryCliError as exc:
        print(f"opentulpa-recovery: {exc}", file=sys.stderr)
        return 2
    if payload is not None:
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(run())


def _execute(client: RecoveryClient, args: argparse.Namespace) -> dict[str, Any] | None:
    if args.command == "status":
        return client.status()
    if args.command == "rollback":
        return client.rollback(reason=args.reason)
    if args.command == "restart":
        return client.restart()
    if args.command == "safe-mode":
        return client.safe_mode()
    raise RecoveryCliError("unknown recovery command")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opentulpa-recovery",
        description="Operate the immutable OpenTulpa bootstrap from the host shell.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show stable bootstrap and release status.")
    rollback = commands.add_parser("rollback", help="Activate the previous healthy release.")
    rollback.add_argument("--reason", default="Stable owner requested rollback from host CLI")
    commands.add_parser("restart", help="Restart the last known good release.")
    commands.add_parser("safe-mode", help="Stop release traffic and enter safe mode.")
    return parser


def _recovery_url(value: str) -> str:
    cleaned = str(value or "").strip().rstrip("/")
    parsed = urlsplit(cleaned)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RecoveryCliError("OPENTULPA_RECOVERY_URL must be an HTTP(S) origin")
    return cleaned


__all__ = ["RecoveryCliError", "RecoveryClient", "main", "run"]
