"""Standard-library controller-generation v1 identity and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
from typing import Any


def canonical_json_bytes(value: object) -> bytes:
    """Return the exact canonical JSON representation used by installer v1."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identity_from_environment() -> dict[str, Any]:
    return {
        "install_profile": os.environ["PROFILE"],
        "bootstrap_python": os.environ["BOOTSTRAP_PYTHON"],
        "bootstrap_python_sha256": os.environ["BOOTSTRAP_PYTHON_SHA256"],
        "lock_sha256": os.environ["LOCK_SHA256"],
        "python": json.loads(os.environ["PYTHON_IDENTITY"]),
        "requirements_sha256": os.environ["REQUIREMENTS_SHA256"],
        "source_commit": os.environ["SOURCE_COMMIT"],
        "source_seed_sha256": os.environ["SOURCE_SEED_SHA256"],
        "source_tree_oid": os.environ["SOURCE_TREE_OID"],
        "tui_name": os.environ["TUI_NAME"] or None,
        "tui_sha256": os.environ["TUI_SHA256"] or None,
        "uv_sha256": os.environ["UV_SHA256"],
        "wheel_name": os.environ["WHEEL_NAME"],
        "wheel_sha256": os.environ["WHEEL_SHA256"],
        "wheelhouse_sha256": os.environ["WHEELHOUSE_SHA256"],
    }


def _manifest_from_environment() -> dict[str, Any]:
    return {
        "format_version": 1,
        "generation_id": os.environ["GENERATION_ID"],
        "identity": json.loads(os.environ["IDENTITY_JSON"]),
        "runtime_tree_sha256": os.environ["RUNTIME_TREE_SHA256"],
        "source": {
            "actual_remote": os.environ["ACTUAL_REMOTE"] or None,
            "configured_ref": os.environ["VERIFIED_REF"] or None,
            "explicit": os.environ["EXPLICIT_SOURCE"] == "1",
            "kind": os.environ["SOURCE_KIND"],
            "oid": os.environ["VERIFIED_OID"],
            "ref": os.environ["REF"],
            "repository": os.environ["REPOSITORY"],
            "root": os.environ["SOURCE_ROOT"],
        },
        "wheelhouse": json.loads(os.environ["WHEELHOUSE_JSON"]),
    }


def _verify_manifest(path: pathlib.Path, generation_id: str) -> None:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not an object")
    if raw != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("manifest is not canonical JSON")
    identity = manifest.get("identity")
    if (
        manifest.get("format_version") != 1
        or manifest.get("generation_id") != generation_id
        or not isinstance(identity, dict)
        or hashlib.sha256(canonical_json_bytes(identity)).hexdigest() != generation_id
    ):
        raise ValueError("manifest identity is invalid")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "generation-id":
        print(hashlib.sha256(canonical_json_bytes(_identity_from_environment())).hexdigest())
        return
    if command == "identity":
        print(canonical_json_bytes(_identity_from_environment()).decode())
        return
    if command == "write-manifest" and len(sys.argv) == 3:
        pathlib.Path(sys.argv[2]).write_bytes(
            canonical_json_bytes(_manifest_from_environment()) + b"\n"
        )
        return
    if command == "verify-manifest" and len(sys.argv) == 4:
        try:
            _verify_manifest(pathlib.Path(sys.argv[2]), sys.argv[3])
        except (OSError, ValueError, json.JSONDecodeError):
            raise SystemExit(1) from None
        return
    raise SystemExit("usage: controller_generation.py {generation-id|identity|write-manifest|verify-manifest}")


if __name__ == "__main__":
    main()
