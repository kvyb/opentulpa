from __future__ import annotations

import json
from pathlib import Path

import pytest

from opentulpa.capabilities.declarative import (
    DeclarativeCapabilityError,
    load_declarative_capabilities,
)


def _manifest(*, name: str = "slack", command: list[str] | None = None) -> dict[str, object]:
    return {
        "name": name,
        "version": "1.0.0",
        "workers": [
            {
                "name": "slack_interface",
                "kind": "interface",
                "protocol": "agent-interface-v1",
                "command": command
                or ["python", "-m", "opentulpa.capability_workers.slack"],
            }
        ],
        "eval_commands": [["unused"]],
    }


def _write(root: Path, payload: dict[str, object], *, name: str = "slack.json") -> None:
    root.mkdir()
    (root / name).write_text(json.dumps(payload), encoding="utf-8")


def test_loads_reviewed_worker_manifest_without_importing_code(tmp_path: Path) -> None:
    root = tmp_path / "manifests"
    payload = _manifest()
    payload["eval_commands"] = [{"argv": ["pytest", "-q", "tests/test_slack.py"]}]
    _write(root, payload)

    manifests = load_declarative_capabilities(root)

    assert [item.name for item in manifests] == ["slack"]
    assert manifests[0].seed is True
    assert manifests[0].workers[0].command[2] == "opentulpa.capability_workers.slack"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"module": "opentulpa.bad", "entrypoint": "run"}), "in process"),
        (lambda value: value.update({"dependencies": ["arbitrary-package"]}), "dependencies"),
        (
            lambda value: value["workers"][0].update(  # type: ignore[index,union-attr]
                {"command": ["python", "/tmp/worker.py"]}
            ),
            "python -m",
        ),
        (
            lambda value: value["workers"][0].update(  # type: ignore[index,union-attr]
                {"command": ["python", "-m", "attacker.worker"]}
            ),
            "inside opentulpa.capability_workers",
        ),
        (
            lambda value: value.update(
                {
                    "secrets": [
                        {
                            "name": "OPENAI_COMPATIBLE_API_KEY",
                            "scopes": ["api.invoke"],
                            "source": "host",
                        }
                    ]
                }
            ),
            "host-owned secrets",
        ),
    ],
)
def test_rejects_manifest_escape_hatches(tmp_path: Path, mutate: object, message: str) -> None:
    root = tmp_path / "manifests"
    payload = _manifest()
    payload["eval_commands"] = [{"argv": ["pytest", "-q", "tests/test_slack.py"]}]
    mutate(payload)  # type: ignore[operator]
    _write(root, payload)

    with pytest.raises(DeclarativeCapabilityError, match=message):
        load_declarative_capabilities(root)


def test_rejects_filename_and_manifest_name_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "manifests"
    payload = _manifest(name="discord")
    payload["eval_commands"] = [{"argv": ["pytest", "-q", "tests/test_discord.py"]}]
    _write(root, payload)

    with pytest.raises(DeclarativeCapabilityError, match="filename"):
        load_declarative_capabilities(root)
