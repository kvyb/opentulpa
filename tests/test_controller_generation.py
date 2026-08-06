from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HELPER = Path(__file__).parents[1] / "controller_generation.py"


def _identity_environment() -> dict[str, str]:
    return {
        "PROFILE": "controller-evaluation-no-dev",
        "BOOTSTRAP_PYTHON": "/opt/python/bin/python",
        "BOOTSTRAP_PYTHON_SHA256": "a" * 64,
        "LOCK_SHA256": "b" * 64,
        "PYTHON_IDENTITY": '{"implementation":"CPython","platform":"linux-x86_64",'
        '"python":"3.12.1","soabi":"cpython-312-x86_64-linux-gnu"}',
        "REQUIREMENTS_SHA256": "c" * 64,
        "SOURCE_COMMIT": "1" * 40,
        "SOURCE_SEED_SHA256": "d" * 64,
        "SOURCE_TREE_OID": "2" * 40,
        "TUI_NAME": "",
        "TUI_SHA256": "",
        "UV_SHA256": "e" * 64,
        "WHEEL_NAME": "opentulpa.whl",
        "WHEEL_SHA256": "f" * 64,
        "WHEELHOUSE_SHA256": "0" * 64,
    }


def test_generation_id_matches_v1_golden_identity() -> None:
    environment = {**os.environ, **_identity_environment()}
    result = subprocess.run(
        [sys.executable, str(HELPER), "generation-id"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    identity = {
        "install_profile": environment["PROFILE"],
        "bootstrap_python": environment["BOOTSTRAP_PYTHON"],
        "bootstrap_python_sha256": environment["BOOTSTRAP_PYTHON_SHA256"],
        "lock_sha256": environment["LOCK_SHA256"],
        "python": json.loads(environment["PYTHON_IDENTITY"]),
        "requirements_sha256": environment["REQUIREMENTS_SHA256"],
        "source_commit": environment["SOURCE_COMMIT"],
        "source_seed_sha256": environment["SOURCE_SEED_SHA256"],
        "source_tree_oid": environment["SOURCE_TREE_OID"],
        "tui_name": None,
        "tui_sha256": None,
        "uv_sha256": environment["UV_SHA256"],
        "wheel_name": environment["WHEEL_NAME"],
        "wheel_sha256": environment["WHEEL_SHA256"],
        "wheelhouse_sha256": environment["WHEELHOUSE_SHA256"],
    }
    expected = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert result.returncode == 0
    assert result.stdout.strip() == expected


def test_manifest_verification_rejects_tampering(tmp_path: Path) -> None:
    identity = {"source_commit": "1" * 40, "wheel_sha256": "a" * 64}
    generation_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {"format_version": 1, "generation_id": generation_id, "identity": identity}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")

    valid = subprocess.run(
        [sys.executable, str(HELPER), "verify-manifest", str(path), generation_id],
        capture_output=True,
        check=False,
    )
    path.write_text(
        path.read_text().replace(
            '"wheel_sha256":"' + "a" * 64,
            '"wheel_sha256":"' + "b" * 64,
        )
    )
    tampered = subprocess.run(
        [sys.executable, str(HELPER), "verify-manifest", str(path), generation_id],
        capture_output=True,
        check=False,
    )

    assert valid.returncode == 0
    assert tampered.returncode == 1
