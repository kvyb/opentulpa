#!/usr/bin/env python3
"""OpenTulpa quality gate for refactors and feature work."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    return int(proc.returncode)


def _run_smoke() -> int:
    print("$ [smoke] runtime/api/telegram")
    try:
        from fastapi.testclient import TestClient

        from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
        from opentulpa.api.app import create_app
        from opentulpa.interfaces.telegram.chat_service import handle_telegram_text
        from opentulpa.interfaces.telegram.client import parse_telegram_update

        app = create_app()
        with TestClient(app) as client:
            r1 = client.get("/healthz")
            r2 = client.get("/agent/healthz")
            assert r1.status_code == 200 and r1.json().get("status") == "ok"
            assert r2.status_code == 200 and "status" in r2.json()

        body = {
            "message": {
                "chat": {"id": 123},
                "from": {"id": 456, "username": "demo"},
                "text": "/status",
            }
        }
        parsed = parse_telegram_update(body)
        assert parsed == (123, 456, "/status")
        out = asyncio.run(handle_telegram_text(body=body, bot_token=None, agent_runtime=None))
        assert isinstance(out, str) and "OpenTulpa status:" in out

        rt = OpenTulpaLangGraphRuntime(
            app_url="http://127.0.0.1:8000",
            openrouter_api_key="dummy",
            model_name="google/gemini-2.0-flash-exp:free",
            checkpoint_db_path=".opentulpa/test-checkpoints.sqlite",
        )
        assert hasattr(rt, "ainvoke_text") and hasattr(rt, "astream_text")
        assert rt._extract_uploaded_text(
            raw_bytes=b"hello", filename="a.txt", mime_type="text/plain"
        ) == "hello"
        return 0
    except Exception as exc:
        print(f"Smoke checks failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenTulpa quality gate")
    parser.add_argument(
        "--strict-tests",
        action="store_true",
        help="Fail when pytest collects no tests (exit code 5).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run import/runtime/API/Telegram smoke checks.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    os.chdir(repo_root)

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".opentulpa/.uv-cache")

    lint_targets = [
        "src/opentulpa",
        "scripts/manager.py",
        "pyproject.toml",
        "README.md",
        "tulpa_stuff/README.md",
    ]
    lint_targets = [p for p in lint_targets if Path(p).exists()]

    if _run(["uv", "run", "ruff", "check", *lint_targets], env=env) != 0:
        return 1

    if _run(["python3", "-m", "compileall", "src/opentulpa"], env=env) != 0:
        return 1

    pytest_rc = _run(["uv", "run", "pytest", "-q"], env=env)
    if pytest_rc != 0:
        if pytest_rc == 5 and not args.strict_tests:
            print("Pytest collected no tests (exit 5); continuing (non-strict mode).")
        else:
            return pytest_rc

    if args.smoke and _run_smoke() != 0:
        return 1

    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
