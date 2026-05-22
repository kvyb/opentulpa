"""Sandboxed file and terminal operations for task execution."""

from __future__ import annotations

import ast
import contextlib
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TULPA_STUFF_DIR = (PROJECT_ROOT / "tulpa_stuff").resolve()
INTEGRATIONS_DIR = (PACKAGE_ROOT / "integrations").resolve()
INTERFACES_DIR = (PACKAGE_ROOT / "interfaces").resolve()
TOOLS_DIR = (PACKAGE_ROOT / "tools").resolve()
SKILLS_DIR = (PACKAGE_ROOT / "skills").resolve()
REPO_VENV_DIR = (PROJECT_ROOT / ".venv").resolve()
AGENT_VENV_DIR = (
    Path(os.environ.get("OPENTULPA_AGENT_VENV_PATH", "")).expanduser().resolve()
    if str(os.environ.get("OPENTULPA_AGENT_VENV_PATH", "")).strip()
    else (PROJECT_ROOT / ".opentulpa" / "agent_venv").resolve()
)
ARTIFACTS_ROOT = (TULPA_STUFF_DIR / "artifacts").resolve()
CATALOG_PATH = (TULPA_STUFF_DIR / ".tulpa_catalog.json").resolve()
CATALOG_README_PATH = (TULPA_STUFF_DIR / "README.md").resolve()
DEBUG_LOG_PATH = (PROJECT_ROOT / ".cursor" / "debug.log").resolve()

ALLOWED_TERMINAL_DIRS = {
    "tulpa_stuff": TULPA_STUFF_DIR,
    "integrations": INTEGRATIONS_DIR,
    "interfaces": INTERFACES_DIR,
    "tools": TOOLS_DIR,
    "skills": SKILLS_DIR,
    "opentulpa": PACKAGE_ROOT,
}
ALLOWED_READ_DIRS = {
    "tulpa_stuff": TULPA_STUFF_DIR,
    "integrations": INTEGRATIONS_DIR,
    "interfaces": INTERFACES_DIR,
    "tools": TOOLS_DIR,
    "skills": SKILLS_DIR,
}

_WORKING_DIR_PREFIXES: dict[str, str] = {
    "tulpa_stuff": "tulpa_stuff",
    "integrations": "src/opentulpa/integrations",
    "interfaces": "src/opentulpa/interfaces",
    "tools": "src/opentulpa/tools",
    "skills": "src/opentulpa/skills",
    "opentulpa": "src/opentulpa",
}

DEFAULT_TERMINAL_COMMAND_ALLOWLIST = {
    "wget",
    "curl",
    "python",
    "python3",
    "uv",
    "pip",
    "pip3",
    "ls",
    "pwd",
    "mkdir",
    "touch",
    "cp",
    "mv",
    "sed",
    "awk",
    "rg",
    "pytest",
    "sqlite3",
}
TERMINAL_COMMAND_ALLOWLIST_ENV = "OPENTULPA_TERMINAL_COMMAND_ALLOWLIST"
def get_terminal_command_allowlist() -> set[str]:
    raw = str(os.environ.get(TERMINAL_COMMAND_ALLOWLIST_ENV, "")).strip()
    if not raw:
        return set()
    if raw.lower() == "default":
        return set(DEFAULT_TERMINAL_COMMAND_ALLOWLIST)
    return {item.strip() for item in raw.split(",") if item.strip()}


def _is_tulpa_router_module(path: Path) -> bool:
    return (
        path.suffix == ".py"
        and is_within(path, TULPA_STUFF_DIR)
        and path.name != "__init__.py"
        and not path.name.startswith("_")
    )


def _debug_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "runId": "sandbox",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json_dumps(payload) + "\n")
    except Exception:
        pass


def is_within(path: Path, root: Path) -> bool:
    path_r = path.resolve()
    root_r = root.resolve()
    return path_r == root_r or root_r in path_r.parents


def get_allowed_read_roots() -> list[str]:
    roots: list[str] = []
    for key in ALLOWED_READ_DIRS:
        prefix = _WORKING_DIR_PREFIXES.get(key, key).strip("/")
        roots.append(f"{prefix}/")
    return roots


def _normalize_redundant_allowed_root_prefix(relative_path: str) -> str:
    rel = str(relative_path or "").strip()
    for key in ALLOWED_READ_DIRS:
        prefix = _WORKING_DIR_PREFIXES.get(key, key).strip("/")
        duplicate = f"{prefix}/{prefix}/"
        if rel.startswith(duplicate):
            return f"{prefix}/{rel[len(duplicate):]}"
    return rel


def resolve_allowed_write_path(relative_path: str) -> Path:
    rel = relative_path.strip()
    if not rel:
        raise ValueError("path is required")
    if Path(rel).is_absolute():
        raise ValueError("path must be relative")

    target = (PROJECT_ROOT / rel).resolve()
    if not (
        is_within(target, TULPA_STUFF_DIR)
        or is_within(target, INTEGRATIONS_DIR)
        or is_within(target, INTERFACES_DIR)
        or is_within(target, TOOLS_DIR)
        or is_within(target, SKILLS_DIR)
    ):
        raise ValueError(
            "path must be under tulpa_stuff/, src/opentulpa/integrations/, src/opentulpa/interfaces/, "
            "src/opentulpa/tools/, or src/opentulpa/skills/"
        )
    return target


def write_file(relative_path: str, content: str) -> Path:
    target = resolve_allowed_write_path(relative_path)
    previous_content: str | None = None
    had_existing = target.exists()
    if had_existing and target.is_file():
        previous_content = target.read_text(encoding="utf-8", errors="replace")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8")
    try:
        validate_generated_file(relative_path)
    except Exception:
        if had_existing and previous_content is not None:
            target.write_text(previous_content, encoding="utf-8")
        else:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
        raise
    _record_catalog_path(target)
    return target


def delete_file(relative_path: str, *, missing_ok: bool = True) -> dict[str, Any]:
    target = resolve_allowed_write_path(relative_path)
    if target.is_dir():
        raise ValueError("path is a directory")
    if not target.exists():
        if missing_ok:
            return {
                "ok": True,
                "deleted": False,
                "missing": True,
                "path": str(target.relative_to(PROJECT_ROOT)),
            }
        raise ValueError("file not found")
    target.unlink()
    return {
        "ok": True,
        "deleted": True,
        "path": str(target.relative_to(PROJECT_ROOT)),
    }


def _extract_router_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                names.add(bound)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and str(test.comparators[0].value) == "__main__"
        ):
            return True
    return False


def validate_generated_file(relative_path: str) -> dict[str, Any]:
    rel = str(relative_path or "").strip()
    if not rel:
        raise ValueError("path is required")
    target = resolve_allowed_write_path(rel)
    if not target.exists():
        raise ValueError("file not found")
    if target.is_dir():
        raise ValueError("path is a directory")

    result: dict[str, Any] = {
        "ok": True,
        "path": str(target.relative_to(PROJECT_ROOT)),
        "python_syntax_ok": None,
        "router_contract_ok": None,
    }
    if target.suffix != ".py":
        return result

    text = target.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(target))
        compile(text, str(target), "exec")
    except SyntaxError as exc:
        raise ValueError(f"Python syntax validation failed: {exc.msg} (line {exc.lineno})") from exc
    result["python_syntax_ok"] = True

    if _is_tulpa_router_module(target) and not _has_main_guard(tree):
        names = _extract_router_names(tree)
        if "router" not in names:
            raise ValueError(
                "tulpa_stuff Python file must either define a top-level 'router' for FastAPI mounting "
                "or be a standalone executable script with if __name__ == '__main__':. "
                "Use router modules only when the file is meant for tulpa_reload."
            )
        result["router_contract_ok"] = True
    return result


def read_file(relative_path: str, max_chars: int = 12000) -> str:
    rel = _normalize_redundant_allowed_root_prefix(relative_path)
    if not rel:
        raise ValueError("path is required")
    if Path(rel).is_absolute():
        raise ValueError("path must be relative")
    target = (PROJECT_ROOT / rel).resolve()
    if not any(is_within(target, root) for root in ALLOWED_READ_DIRS.values()):
        allowed_roots = ", ".join(get_allowed_read_roots())
        raise PermissionError(f"path outside allowed read roots; allowed roots: {allowed_roots}")
    if not target.exists():
        raise FileNotFoundError(f"file not found under allowed read roots: {rel}")
    if target.is_dir():
        raise IsADirectoryError(f"path is a directory: {rel}")
    return target.read_text(encoding="utf-8", errors="replace")[:max_chars]


def task_artifact_dir(task_id: str) -> Path:
    path = (ARTIFACTS_ROOT / task_id).resolve()
    if not is_within(path, ARTIFACTS_ROOT):
        raise ValueError("invalid task_id for artifact path")
    path.mkdir(parents=True, exist_ok=True)
    _record_catalog_path(path / "events.jsonl", kind="artifact_log")
    return path


def list_artifacts(task_id: str) -> list[dict[str, Any]]:
    root = task_artifact_dir(task_id)
    files: list[dict[str, Any]] = []
    for file in sorted(root.rglob("*")):
        if file.is_file():
            files.append(
                {
                    "path": str(file.relative_to(PROJECT_ROOT)),
                    "size_bytes": file.stat().st_size,
                    "name": file.name,
                }
            )
    return files


def get_tulpa_catalog() -> dict[str, Any]:
    if not CATALOG_PATH.exists():
        _write_catalog(_default_catalog())
    try:
        data = json_load(CATALOG_PATH.read_text(encoding="utf-8"))
        return cast("dict[str, Any]", data) if isinstance(data, dict) else _default_catalog()
    except Exception:
        return _default_catalog()


def append_task_event_log(task_id: str, event: dict[str, Any]) -> str:
    root = task_artifact_dir(task_id)
    log_file = (root / "events.jsonl").resolve()
    if not is_within(log_file, ARTIFACTS_ROOT):
        raise ValueError("invalid task event log path")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json_dumps(event) + "\n")
    _record_catalog_path(log_file, kind="artifact_log")
    return str(log_file.relative_to(PROJECT_ROOT))


def run_terminal(
    command: str,
    working_dir: str,
    timeout_seconds: int = 90,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    def _ensure_agent_venv() -> Path:
        if AGENT_VENV_DIR.exists():
            return AGENT_VENV_DIR
        AGENT_VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        _debug_log(
            hypothesis_id="sandbox",
            location="tasks/sandbox.py:run_terminal",
            message="agent_venv_create_start",
            data={"venv_path": str(AGENT_VENV_DIR)},
        )
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(AGENT_VENV_DIR)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        except Exception as exc:
            _debug_log(
                hypothesis_id="sandbox",
                location="tasks/sandbox.py:run_terminal",
                message="agent_venv_create_failed",
                data={"venv_path": str(AGENT_VENV_DIR), "error": str(exc)},
            )
            raise RuntimeError(
                f"Agent venv setup failed at {AGENT_VENV_DIR}. "
                "Create it manually with: python3 -m venv --system-site-packages .opentulpa/agent_venv"
            ) from exc
        _debug_log(
            hypothesis_id="sandbox",
            location="tasks/sandbox.py:run_terminal",
            message="agent_venv_create_ok",
            data={"venv_path": str(AGENT_VENV_DIR)},
        )
        return AGENT_VENV_DIR

    cmd = str(command).strip()
    if not cmd:
        raise ValueError("command is required")
    if working_dir not in ALLOWED_TERMINAL_DIRS:
        raise ValueError(
            "working_dir must be one of: " + ", ".join(sorted(ALLOWED_TERMINAL_DIRS.keys()))
        )
    try:
        parts = shlex.split(cmd)
    except ValueError as exc:
        raise ValueError("invalid command syntax") from exc
    if not parts:
        raise ValueError("command is required")
    # region agent log
    _debug_log(
        hypothesis_id="sandbox",
        location="tasks/sandbox.py:run_terminal",
        message="terminal_command_received",
        data={
            "working_dir": working_dir,
            "command_bin": parts[0],
            "timeout_seconds": timeout_seconds,
        },
    )
    # endregion
    allowed_commands = get_terminal_command_allowlist()
    if allowed_commands and parts[0] not in allowed_commands:
        # region agent log
        _debug_log(
            hypothesis_id="sandbox",
            location="tasks/sandbox.py:run_terminal",
            message="terminal_command_rejected",
            data={
                "working_dir": working_dir,
                "command_bin": parts[0],
                "reason": "not_allowlisted",
                "allowlist_env": TERMINAL_COMMAND_ALLOWLIST_ENV,
            },
        )
        # endregion
        raise PermissionError(
            f"command '{parts[0]}' is not allowed by {TERMINAL_COMMAND_ALLOWLIST_ENV}"
        )
    agent_venv_dir = _ensure_agent_venv()

    prefix = _WORKING_DIR_PREFIXES.get(working_dir)
    normalized_parts = list(parts)
    if prefix and len(parts) > 1:
        rel_markers = (f"{prefix}/", f"./{prefix}/")
        abs_prefix = str((PROJECT_ROOT / prefix).resolve()) + "/"

        def _strip_prefix(token: str) -> str:
            raw = str(token)
            for marker in rel_markers:
                if raw.startswith(marker):
                    return raw[len(marker) :]
            if raw.startswith(abs_prefix):
                return raw[len(abs_prefix) :]
            if raw.startswith("--") and "=" in raw:
                key, value = raw.split("=", 1)
                for marker in rel_markers:
                    if value.startswith(marker):
                        return f"{key}={value[len(marker):]}"
                if value.startswith(abs_prefix):
                    return f"{key}={value[len(abs_prefix):]}"
            return raw

        normalized_parts = [parts[0], *(_strip_prefix(item) for item in parts[1:])]
        if normalized_parts != parts:
            _debug_log(
                hypothesis_id="sandbox",
                location="tasks/sandbox.py:run_terminal",
                message="terminal_command_normalized",
                data={
                    "working_dir": working_dir,
                    "original": cmd[:500],
                    "normalized": shlex.join(normalized_parts)[:500],
                },
            )

    cwd = ALLOWED_TERMINAL_DIRS[working_dir]
    cwd.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    run_env["VIRTUAL_ENV"] = str(agent_venv_dir)
    run_env["PATH"] = f"{agent_venv_dir / 'bin'}:{run_env.get('PATH', '')}"
    run_env["PIP_REQUIRE_VIRTUALENV"] = "true"
    run_env["UV_PROJECT_ENVIRONMENT"] = str(agent_venv_dir)
    if extra_env:
        run_env.update(extra_env)

    try:
        proc = subprocess.run(
            normalized_parts,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            env=run_env,
            timeout=max(1, min(int(timeout_seconds), 300)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _debug_log(
            hypothesis_id="sandbox",
            location="tasks/sandbox.py:run_terminal",
            message="terminal_command_timeout",
            data={
                "working_dir": working_dir,
                "command_bin": parts[0],
                "timeout_seconds": timeout_seconds,
            },
        )
        raise TimeoutError("command timed out") from exc

    result = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-12000:],
        "stderr": (proc.stderr or "")[-12000:],
        "cwd": str(cwd.relative_to(PROJECT_ROOT)),
        "venv": str(agent_venv_dir.relative_to(PROJECT_ROOT)),
    }
    # region agent log
    _debug_log(
        hypothesis_id="sandbox",
        location="tasks/sandbox.py:run_terminal",
        message="terminal_command_finished",
        data={
            "working_dir": working_dir,
            "command_bin": parts[0],
            "ok": result["ok"],
            "returncode": result["returncode"],
        },
    )
    # endregion
    return result


def _default_catalog() -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "roots": {
            "tulpa_stuff": "tulpa_stuff",
            "artifacts": "tulpa_stuff/artifacts",
            "integrations": "src/opentulpa/integrations",
            "interfaces": "src/opentulpa/interfaces",
            "tools": "src/opentulpa/tools",
            "skills": "src/opentulpa/skills",
        },
        "entries": [],
    }


def _category_for_path(path: Path) -> str:
    if is_within(path, ARTIFACTS_ROOT):
        return "artifact"
    if is_within(path, TULPA_STUFF_DIR):
        return "tulpa_stuff"
    if is_within(path, INTEGRATIONS_DIR):
        return "integration"
    if is_within(path, INTERFACES_DIR):
        return "interface"
    if is_within(path, TOOLS_DIR):
        return "tool"
    if is_within(path, SKILLS_DIR):
        return "skill"
    return "other"


def _record_catalog_path(path: Path, kind: str | None = None) -> None:
    target = path.resolve()
    rel = str(target.relative_to(PROJECT_ROOT)) if is_within(target, PROJECT_ROOT) else str(target)
    catalog = get_tulpa_catalog()
    entries = catalog.get("entries", [])
    now = _utc_now()
    category = kind or _category_for_path(target)
    replaced = False
    for entry in entries:
        if entry.get("path") == rel:
            entry["updated_at"] = now
            entry["kind"] = category
            replaced = True
            break
    if not replaced:
        entries.append({"path": rel, "kind": category, "updated_at": now})
    entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
    catalog["entries"] = entries[:5000]
    catalog["generated_at"] = now
    _write_catalog(catalog)
    _write_catalog_readme(catalog)


def _write_catalog(catalog: dict[str, Any]) -> None:
    TULPA_STUFF_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json_dumps(catalog, indent=2) + "\n", encoding="utf-8")


def _write_catalog_readme(catalog: dict[str, Any]) -> None:
    lines = [
        "# Tulpa Stuff Catalog",
        "",
        "Auto-generated index of instruments, skills, integration files, and artifacts.",
        "",
        f"Generated: {catalog.get('generated_at')}",
        "",
        "## Roots",
        "",
    ]
    roots = catalog.get("roots", {})
    for key, value in roots.items():
        lines.append(f"- `{key}` -> `{value}`")
    lines.extend(["", "## Recent Entries", ""])
    entries = catalog.get("entries", [])[:200]
    if not entries:
        lines.append("- (no entries yet)")
    else:
        for entry in entries:
            lines.append(
                f"- `{entry.get('path')}` ({entry.get('kind')}) updated `{entry.get('updated_at')}`"
            )
    CATALOG_README_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def json_dumps(value: Any, indent: int | None = None) -> str:
    import json

    if indent is None:
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, indent=indent, ensure_ascii=False)


def json_load(value: str) -> Any:
    import json

    return json.loads(value)
