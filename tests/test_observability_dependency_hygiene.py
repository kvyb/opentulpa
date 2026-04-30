from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_opentulpa_has_no_direct_posthog_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject.get("project", {}).get("dependencies", [])

    direct_posthog_deps = [
        dependency
        for dependency in dependencies
        if str(dependency).strip().lower().startswith("posthog")
    ]

    assert direct_posthog_deps == []


def test_opentulpa_source_has_no_direct_posthog_wiring() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "opentulpa").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "posthog" in text.lower():
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
