---
name: opentulpa-engineering
description: Enforce maintainable, testable, and robust engineering practices for the OpenTulpa codebase. Use when implementing features, refactors, bug fixes, or integrations in this repository, especially for LangGraph runtime, FastAPI routes, Telegram interfaces, scheduler/task flows, and memory/directive behavior. Apply this skill to plan separation-of-concerns changes, preserve runtime behavior, and run repository-specific quality gates and regression checks.
---

# OpenTulpa Engineering

## Overview

Implement OpenTulpa changes with small modules, explicit boundaries, and repeatable verification.
Prefer behavior-preserving refactors backed by targeted tests and deterministic smoke checks.

## Workflow

1. Scope impact before editing.
2. Keep modules focused and dependency-injected.
3. Add or update tests for changed behavior.
4. Run quality gates and smoke checks before finishing.
5. Report what is proven vs what remains unproven.

## Scope First

- Read the touched component and its immediate callers before edits.
- Map change boundaries to current structure:
  - `src/opentulpa/agent/*` for runtime, graph, tools, compaction, file analysis.
  - `src/opentulpa/interfaces/telegram/*` for interface concerns (client, formatting, relay, chat orchestration).
  - `src/opentulpa/api/app.py` (or future route modules) for HTTP wiring.
  - `src/opentulpa/tasks/*` and `src/opentulpa/scheduler/*` for background execution.
- Avoid broad rewrites when only one concern changes.

## Design Rules

- Keep orchestration thin; move logic into focused modules.
- Isolate pure logic in helpers so unit tests do not require network or process startup.
- Inject services/clients (`memory`, `file_vault`, runtime deps) instead of hard-coding globals where feasible.
- Preserve external contracts unless the task explicitly changes them:
  - route paths and payload shapes,
  - tool names and required args,
  - wake/notification semantics,
  - directive/time-profile behavior.
- Remove dead compatibility code when migration is intentional and approved.

## Testing Rules

- Add tests in `tests/` for every behavior change or bug fix.
- Prefer layered coverage:
  - Unit tests for pure helpers.
  - Integration tests for FastAPI handlers with `TestClient`.
  - Runtime/graph tests with deterministic fakes/mocks.
- Mock external boundaries:
  - OpenRouter/model calls,
  - network fetches (`httpx`),
  - Telegram/Slack API calls,
  - filesystem side effects when not under test.
- Assert observable behavior, not private implementation details.

## Mandatory Verification

Run this from repository root after edits:

```bash
UV_CACHE_DIR=.opentulpa/.uv-cache \
  uv run python skills/opentulpa-engineering/scripts/quality_gate.py --smoke
```

Use `--strict-tests` when you expect real pytest coverage and want failure on "no tests collected".

## Refactor Acceptance Criteria

- Lint passes.
- Code compiles.
- Existing tests pass.
- New/updated tests validate changed behavior.
- Smoke checks pass for runtime/API/Telegram entry points.
- Summary clearly states parity confidence and any remaining risk.

## References

- Use `references/quality-gate.md` for command policy and pass/fail interpretation.
- Use `references/test-matrix.md` for what to test per subsystem.
- Use `references/external-sources.md` for authoritative upstream docs used by this skill.
