# OpenTulpa Architecture

This document describes the current runtime design, request flows, safety controls, and extension points.

## Design goals

- Keep interfaces thin and replaceable.
- Keep decision logic centralized in the agent runtime/graph.
- Keep domain/application boundaries explicit for easier testing and refactoring.
- Persist user context, directives, and artifacts across sessions.
- Enforce external-impact safety at tool-action time.

## Layered modules

- `src/opentulpa/api`: FastAPI composition and route registration.
- `src/opentulpa/api/routes`: internal route surface (`/internal/*`) + Telegram webhook.
- `src/opentulpa/application`: orchestration use-cases (`TurnOrchestrator`, `WakeOrchestrator`, `ApprovalExecutionOrchestrator`).
- `src/opentulpa/domain`: domain contracts (for example conversation turn request/result).
- `src/opentulpa/agent`: LangGraph runtime, graph nodes, compaction, tool registry.
- `src/opentulpa/interfaces/telegram`: Telegram transport, parsing, and streaming relay.
- `src/opentulpa/approvals`: broker, adapters, store, approval models.
- `src/opentulpa/policy`: approval intent/policy evaluator used by broker.
- `src/opentulpa/context`: profiles, event backlog, file vault, thread rollups, link aliases.
- `src/opentulpa/skills`: skill storage and retrieval.
- `src/opentulpa/scheduler`: routine scheduling service.
- `src/opentulpa/tasks`: task runtime, sandbox, wake queue integration.

## Primary request flows

### Telegram turn flow

1. Telegram calls `POST /webhook/telegram`.
2. `interfaces/telegram/chat_service.py` parses text/files/voice and resolves `customer_id`/`thread_id`.
3. Streaming path calls `runtime.astream_text(...)`.
4. LangGraph runs nodes: `agent -> validate_tools -> guardrail_precheck -> tools -> claim_check`.
5. Assistant reply is streamed/posted to Telegram.
6. Deferred approval prompts (if any) are flushed after the assistant reply for the same turn.

### Direct API turn flow (non-Telegram)

1. Client calls `POST /internal/chat`.
2. `TurnOrchestrator` validates/normalizes the turn.
3. Runtime executes `ainvoke_text(...)`.
4. Route returns normalized `{ok, status, customer_id, thread_id, text}`.

### Approval decision + execution flow

1. Guardrail precheck calls `POST /internal/approvals/evaluate`.
2. `ApprovalBroker` + `policy/evaluator.py` decide `allow|require_approval|deny`.
3. `require_approval` creates a pending record in approvals DB.
4. User approves/denies via Telegram callback or `/approve` token path.
5. Approved action executes once via `POST /internal/approvals/execute`.
6. `ApprovalExecutionOrchestrator` summarizes execution outcome back to user.

### Background wake flow

1. Scheduler/task events enqueue wake payloads.
2. `WakeOrchestrator` classifies notify-vs-backlog behavior.
3. Notify-worthy events are drafted through agent runtime and delivered via interface.
4. Non-notify events are persisted to context backlog for later turn injection.

## Agent graph behavior

- Tool-call validation happens before execution (`validate_tools`).
- Guardrail precheck evaluates each requested action and only allows approved tool call IDs through.
- Claim-check verifies immediate execution claims against tool evidence before turn end.
- Claim-check includes retry/backoff handling for:
  - empty assistant output,
  - unusable checker output,
  - claim/evidence mismatch.
- Streaming path has a fallback path that guarantees a visible user-facing message when no chunks are produced.

## Context window policy (current defaults)

Configured in `src/opentulpa/core/config.py`:

- `AGENT_CONTEXT_TOKEN_LIMIT` default `12000` (short-term high watermark).
- `AGENT_CONTEXT_RECENT_TOKENS` default `3500` (post-compaction target).
- `AGENT_CONTEXT_ROLLUP_TOKENS` default `2200` (older-context rollup budget).
- `AGENT_CONTEXT_COMPACTION_SOURCE_TOKENS` default `100000` (max old span compacted per pass).

Compaction is hysteresis-based: compact at high watermark, then reduce toward low watermark, while folding older history into a bounded rollup injected as system context.

## Approval model details

- Internal/read-oriented actions are deterministically allowed by policy.
- External-impact actions are gated through approval broker.
- Pending approvals are durable in SQLite (`pending_approvals.db`).
- Telegram approvals are currently delivered with deferred queue/flush so approval bubbles appear after the assistant message in that turn.
- State machine: `pending -> approved|denied|expired`, and `approved -> executed` (single-use).

## Internal API boundary

- `/webhook/*` is the public webhook ingress surface (Telegram and future integrations).
- Public internet clients are denied for all non-webhook routes except health checks
  (`/healthz`, `/agent/healthz`) for platform liveness probing.
- `/webhook/telegram` requires Telegram secret header auth
  (`x-telegram-bot-api-secret-token`).
- `/internal/*` routes are intended for server-local traffic only
  (`localhost`/private network).
- `scripts/manager.py` auto-generates `TELEGRAM_WEBHOOK_SECRET` for tunnel runs
  when not provided.
- Cloud deploy bootstrap (`python -m opentulpa`) can auto-register Telegram webhook
  when `TELEGRAM_BOT_TOKEN` and `PUBLIC_BASE_URL` (or `RAILWAY_PUBLIC_DOMAIN`) are set.

## Runtime data stores

- LangGraph checkpoints: `.opentulpa/langgraph_checkpoints.sqlite`
- Approvals: `.opentulpa/pending_approvals.db`
- Context events: `.opentulpa/context_events.db`
- Customer profiles: `.opentulpa/customer_profiles.db`
- Thread rollups: `.opentulpa/thread_rollups.db`
- Link aliases: `.opentulpa/link_aliases.db`
- Skills: `.opentulpa/skills.db`
- File vault: `.opentulpa/file_vault.db` + file storage
- Tasks/wake queue: `.opentulpa/tasks.db`, `.opentulpa/wake_events.db`

## Observability and debugging

- Structured agent behavior log (JSONL):
  - enabled by default (`AGENT_BEHAVIOR_LOG_ENABLED=true`)
  - path default `.opentulpa/logs/agent_behavior.jsonl`
- Includes turn lifecycle, graph node outcomes, guardrail decisions, claim-check retries, and tool execution outcomes.

## Separation of concerns

- Interfaces: transport + serialization only.
- Application layer: request orchestration and outcome shaping.
- Domain layer: typed request/result contracts.
- Agent runtime: reasoning, graph progression, tool orchestration.
- Policy + approvals: guardrail classification, lifecycle, authorization.
- Sandbox/tasks: constrained command/file execution for generated automation code.

## Extension points

- Add tools in `src/opentulpa/agent/tools_registry.py`.
- Add internal APIs in `src/opentulpa/api/routes/*`.
- Add interface adapters under `src/opentulpa/interfaces/*`.
- Add approval adapters under `src/opentulpa/approvals/adapters/*`.
- Add skills via `src/opentulpa/skills/*`.

For external integrations, follow `docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md`.

## Failure behavior

- Guardrail classifier uncertainty defaults to approval-required.
- If approval delivery cannot be completed, action remains non-executed.
- Tool-call failures return explicit tool error messages back into the graph.
- Wake delivery failures are persisted to context backlog for recovery.
