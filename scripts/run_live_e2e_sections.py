from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Section:
    name: str
    paths: tuple[str, ...]


@dataclass
class RunningSection:
    section: Section
    process: subprocess.Popen[str]
    log_path: Path
    started_at: float
    command: list[str]


@dataclass
class SectionResult:
    name: str
    returncode: int
    duration_seconds: float
    command: list[str]
    log_path: Path
    trace_summary: dict[str, Any]


SECTIONS: tuple[Section, ...] = (
    Section(
        name="intake_workflow",
        paths=("tests/e2e/scenarios/test_telegram_intake_workflow_real_chat.py",),
    ),
    Section(
        name="intake_runtime",
        paths=(
            "tests/e2e/scenarios/test_google_sheets_sink_resolution.py",
            "tests/e2e/scenarios/test_telegram_intake_debounce.py",
            "tests/e2e/scenarios/test_telegram_workflow_setup_spam.py",
            "tests/e2e/test_lead_simulator.py",
        ),
    ),
    Section(
        name="interactive",
        paths=(
            "tests/e2e/scenarios/test_interactive_chat.py",
            "tests/e2e/scenarios/test_telegram_interactive_owner_update.py",
        ),
    ),
    Section(
        name="support",
        paths=("tests/e2e/scenarios/test_telegram_support_act_as.py",),
    ),
    Section(
        name="context_files",
        paths=("tests/e2e/scenarios/test_telegram_user_context_real_files.py",),
    ),
    Section(
        name="ingress",
        paths=("tests/e2e/scenarios/test_instagram_ingress.py",),
    ),
    Section(
        name="smoke",
        paths=("tests/e2e/scenarios/test_chat_api.py",),
    ),
    Section(
        name="external_smokes",
        paths=(
            "tests/e2e/live/test_browser_use_google.py",
            "tests/e2e/live/test_chipmunk_image_search.py",
            "tests/e2e/live/test_intake_workflow_composio.py",
        ),
    ),
)

DEFAULT_PYTEST_ARGS = ("--run-e2e", "--run-live-llm", "--tb=short", "-q", "-rs")


def _section_by_name() -> dict[str, Section]:
    return {section.name: section for section in SECTIONS}


def _default_basetemp_root() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp") / f"opentulpa-live-e2e-sections-{stamp}"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    section_names = tuple(section.name for section in SECTIONS)
    parser = argparse.ArgumentParser(
        description="Run live OpenTulpa e2e tests in parallel pytest sections.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, len(SECTIONS)),
        help="maximum pytest section processes to run at once",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=section_names,
        help="section to run; repeat for multiple sections; default runs all sections",
    )
    parser.add_argument(
        "--basetemp-root",
        type=Path,
        default=_default_basetemp_root(),
        help="root directory for isolated pytest basetemp dirs, logs, and summary",
    )
    parser.add_argument(
        "--no-default-pytest-args",
        action="store_true",
        help="do not add --run-e2e --run-live-llm --tb=short -q -rs",
    )
    parser.add_argument(
        "--max-trace-cost-usd",
        type=float,
        default=None,
        help="fail after run if traced native LLM cost exceeds this value",
    )
    parser.add_argument("--list", action="store_true", help="print sections and exit")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    parser.add_argument(
        "--failure-tail-lines",
        type=int,
        default=80,
        help="number of log lines to print for failed sections",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.pytest_args and args.pytest_args[0] == "--":
        args.pytest_args = args.pytest_args[1:]
    if "--basetemp" in args.pytest_args or any(
        item.startswith("--basetemp=") for item in args.pytest_args
    ):
        parser.error("pass --basetemp-root to this runner instead of pytest --basetemp")
    if int(args.workers) < 1:
        parser.error("--workers must be at least 1")
    return args


def _selected_sections(names: list[str] | None) -> list[Section]:
    if not names:
        return list(SECTIONS)
    by_name = _section_by_name()
    seen: set[str] = set()
    selected: list[Section] = []
    for name in names:
        if name in seen:
            continue
        selected.append(by_name[name])
        seen.add(name)
    return selected


def _command_for_section(
    *,
    section: Section,
    basetemp_root: Path,
    default_pytest_args: bool,
    extra_pytest_args: list[str],
) -> list[str]:
    section_root = basetemp_root / section.name
    command = [sys.executable, "-m", "pytest", *section.paths]
    if default_pytest_args:
        command.extend(DEFAULT_PYTEST_ARGS)
    command.extend(extra_pytest_args)
    command.extend(("--basetemp", str(section_root / "basetemp")))
    return command


def _print_sections(sections: list[Section]) -> None:
    for section in sections:
        joined_paths = " ".join(section.paths)
        print(f"{section.name}: {joined_paths}", flush=True)


def _numeric(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _summarize_traces(section_root: Path) -> dict[str, Any]:
    files = sorted(section_root.rglob("llm_call_traces.jsonl"))
    records = [record for path in files for record in _read_jsonl(path)]
    total_prompt = sum(int(_numeric(item.get("native_tokens_prompt"))) for item in records)
    total_completion = sum(int(_numeric(item.get("native_tokens_completion"))) for item in records)
    total_tokens = sum(int(_numeric(item.get("native_tokens_total"))) for item in records)
    if total_tokens == 0:
        total_tokens = total_prompt + total_completion
    error_records = [
        item
        for item in records
        if item.get("error") or item.get("error_type") or item.get("provider_error_body")
    ]
    return {
        "trace_files": [str(path) for path in files],
        "calls": len(records),
        "errors": len(error_records),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "reasoning_tokens": sum(int(_numeric(item.get("native_tokens_reasoning"))) for item in records),
        "cached_tokens": sum(int(_numeric(item.get("native_tokens_cached"))) for item in records),
        "cost_usd": sum(_numeric(item.get("native_cost_usd")) for item in records),
        "provider_elapsed_seconds": sum(_numeric(item.get("provider_elapsed_ms")) for item in records)
        / 1000.0,
        "elapsed_seconds": sum(_numeric(item.get("elapsed_ms")) for item in records) / 1000.0,
    }


def _start_section(section: Section, command: list[str], basetemp_root: Path) -> RunningSection:
    section_root = basetemp_root / section.name
    section_root.mkdir(parents=True, exist_ok=True)
    log_path = section_root / "pytest.log"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENTULPA_E2E_PARALLEL_SECTION"] = section.name
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    log_file.close()
    print(f"started {section.name} pid={process.pid} log={log_path}", flush=True)
    return RunningSection(
        section=section,
        process=process,
        log_path=log_path,
        started_at=time.monotonic(),
        command=command,
    )


def _finish_section(running: RunningSection, basetemp_root: Path) -> SectionResult:
    returncode = int(running.process.returncode or 0)
    duration = time.monotonic() - running.started_at
    summary = _summarize_traces(basetemp_root / running.section.name)
    status = "passed" if returncode == 0 else "failed"
    print(
        f"{status} {running.section.name} rc={returncode} "
        f"duration={duration:.1f}s cost=${summary['cost_usd']:.6f} "
        f"tokens={summary['total_tokens']}",
        flush=True,
    )
    return SectionResult(
        name=running.section.name,
        returncode=returncode,
        duration_seconds=duration,
        command=running.command,
        log_path=running.log_path,
        trace_summary=summary,
    )


def _tail(path: Path, limit: int) -> str:
    if limit <= 0 or not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def _write_summary(basetemp_root: Path, results: list[SectionResult]) -> Path:
    totals = {
        "sections": len(results),
        "failed_sections": sum(1 for item in results if item.returncode != 0),
        "duration_seconds": sum(item.duration_seconds for item in results),
        "calls": sum(int(item.trace_summary["calls"]) for item in results),
        "trace_errors": sum(int(item.trace_summary["errors"]) for item in results),
        "prompt_tokens": sum(int(item.trace_summary["prompt_tokens"]) for item in results),
        "completion_tokens": sum(int(item.trace_summary["completion_tokens"]) for item in results),
        "total_tokens": sum(int(item.trace_summary["total_tokens"]) for item in results),
        "reasoning_tokens": sum(int(item.trace_summary["reasoning_tokens"]) for item in results),
        "cached_tokens": sum(int(item.trace_summary["cached_tokens"]) for item in results),
        "cost_usd": sum(float(item.trace_summary["cost_usd"]) for item in results),
        "provider_elapsed_seconds": sum(
            float(item.trace_summary["provider_elapsed_seconds"]) for item in results
        ),
        "elapsed_seconds": sum(float(item.trace_summary["elapsed_seconds"]) for item in results),
    }
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "basetemp_root": str(basetemp_root),
        "totals": totals,
        "sections": [
            {
                "name": item.name,
                "returncode": item.returncode,
                "duration_seconds": item.duration_seconds,
                "command": item.command,
                "log_path": str(item.log_path),
                "trace_summary": item.trace_summary,
            }
            for item in results
        ],
    }
    summary_path = basetemp_root / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def _run_sections(
    *,
    sections: list[Section],
    basetemp_root: Path,
    workers: int,
    default_pytest_args: bool,
    extra_pytest_args: list[str],
) -> list[SectionResult]:
    pending = deque(sections)
    running: list[RunningSection] = []
    results: list[SectionResult] = []
    basetemp_root.mkdir(parents=True, exist_ok=True)
    try:
        while pending or running:
            while pending and len(running) < workers:
                section = pending.popleft()
                command = _command_for_section(
                    section=section,
                    basetemp_root=basetemp_root,
                    default_pytest_args=default_pytest_args,
                    extra_pytest_args=extra_pytest_args,
                )
                running.append(_start_section(section, command, basetemp_root))
            time.sleep(0.5)
            still_running: list[RunningSection] = []
            for item in running:
                if item.process.poll() is None:
                    still_running.append(item)
                else:
                    results.append(_finish_section(item, basetemp_root))
            running = still_running
    except KeyboardInterrupt:
        for item in running:
            item.process.terminate()
        raise
    return results


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    sections = _selected_sections(args.section)
    if args.list:
        _print_sections(sections)
        return 0

    default_pytest_args = not bool(args.no_default_pytest_args)
    if args.dry_run:
        for section in sections:
            command = _command_for_section(
                section=section,
                basetemp_root=args.basetemp_root,
                default_pytest_args=default_pytest_args,
                extra_pytest_args=args.pytest_args,
            )
            print(" ".join(command), flush=True)
        return 0

    results = _run_sections(
        sections=sections,
        basetemp_root=args.basetemp_root,
        workers=min(int(args.workers), len(sections)),
        default_pytest_args=default_pytest_args,
        extra_pytest_args=args.pytest_args,
    )
    summary_path = _write_summary(args.basetemp_root, results)
    total_cost = sum(float(item.trace_summary["cost_usd"]) for item in results)
    print(f"summary={summary_path}", flush=True)
    print(f"total_cost=${total_cost:.6f}", flush=True)

    failed = [item for item in results if item.returncode != 0]
    for item in failed:
        print(f"\n--- {item.name} tail {item.log_path} ---", flush=True)
        print(_tail(item.log_path, int(args.failure_tail_lines)), flush=True)

    if args.max_trace_cost_usd is not None and total_cost > float(args.max_trace_cost_usd):
        print(
            f"trace cost ${total_cost:.6f} exceeds --max-trace-cost-usd "
            f"${float(args.max_trace_cost_usd):.6f}",
            flush=True,
        )
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
