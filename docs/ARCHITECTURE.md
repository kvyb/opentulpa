# OpenTulpa Architecture

This document explains how OpenTulpa is put together today.

If you only need the mental model, it is this:

```text
an inbound event arrives -> the runtime reloads durable context -> the agent plans and uses tools -> risky actions are gated -> results are persisted for the next turn
```

OpenTulpa is designed around one core assumption: a useful agent should behave like a long-running worker, not a stateless chat session.

## Design goals

- Keep interfaces thin and replaceable
- Keep agent decision logic centralized in the runtime graph
- Keep domain boundaries explicit for easier testing and refactoring
- Persist user context, directives, and artifacts across sessions
- Enforce safety at tool-action time, not as an afterthought

## Main runtime pieces

- `src/opentulpa/api`: FastAPI app composition and route registration
- `src/opentulpa/api/routes`: internal API routes, Telegram webhook routes, and Composio callback/status routes
- `src/opentulpa/application`: orchestration use cases such as `TurnOrchestrator`, `WakeOrchestrator`, and `ApprovalExecutionOrchestrator`
- `src/opentulpa/domain`: typed domain contracts
- `src/opentulpa/agent`: LangGraph runtime, graph nodes, compaction, and tool registry
- `src/opentulpa/interfaces/telegram`: Telegram transport, parsing, streaming relay, and Telegram Business inbox persistence
- `src/opentulpa/approvals`: approval broker, adapters, store, and approval models
- `src/opentulpa/policy`: approval intent and policy evaluation
- `src/opentulpa/context`: profiles, event backlog, file vault, thread rollups, and link aliases
- `src/opentulpa/skills`: durable skill storage and retrieval
- `src/opentulpa/scheduler`: routine scheduling
- `src/opentulpa/tasks`: task runtime, sandbox, and wake queue integration

## The important architectural idea

The system is split so that transports and storage are replaceable, while the agent runtime stays the center of decision-making.

- Interfaces move data in and out
- The application layer shapes requests and responses
- The runtime decides what to do
- Policy and approvals decide whether it is allowed
- Context, skills, and artifacts make future turns better

## Primary request flows

### Telegram turn flow

1. Telegram calls `POST /webhook/telegram`
2. `interfaces/telegram/chat_service.py` parses text, files, and voice, then resolves `customer_id` and `thread_id`
3. The streaming path calls `runtime.astream_text(...)`
4. LangGraph runs nodes such as `agent`, `validate_tools`, `guardrail_precheck`, `tools`, and `claim_check`
5. The assistant reply is streamed back to Telegram
6. If a tool requires approval, the runtime emits the approval interrupt immediately and stops normal reply streaming for that action
7. Telegram webhook handling makes sure the approval challenge is surfaced before any optional follow-up assistant message

### External DM intake flow

This is the flow behind persistent lead handling such as Telegram Business inboxes.

1. A user configures an intake workflow through normal OpenTulpa conversation
2. OpenTulpa persists that workflow and stores a synced durable workflow skill
3. Inbound messages arrive through the configured source
4. Intake service loads the external conversation plus any active or recent booking state for that lead
5. The runtime decides whether the message matches the workflow, whether follow-up is needed, and whether the booking is ready to save
6. Intake service performs the idempotent reply or save step
7. Per-conversation cursors prevent reprocessing the same inbound message as fresh work

### Direct API turn flow

1. A client calls `POST /internal/chat`
2. `TurnOrchestrator` validates and normalizes the request
3. Runtime executes `ainvoke_text(...)`
4. The route returns normalized `{ok, status, customer_id, thread_id, text}`

### Composio integration flow

1. App startup checks whether `COMPOSIO_API_KEY` is configured
2. If present, `ComposioService` is initialized lazily and real SDK-backed routes are used
3. If absent, OpenTulpa keeps the status route available but reports `enabled: false`
4. When configured, auth and tool flows run through `/internal/composio/*` routes on behalf of the active user

### Approval decision and execution flow

1. Guardrail precheck calls `POST /internal/approvals/evaluate`
2. `ApprovalBroker` and `policy/evaluator.py` decide `allow`, `require_approval`, or `deny`
3. `require_approval` creates a durable pending record
4. User approves or denies through Telegram callback or `/approve` token path
5. Approved actions execute once through `POST /internal/approvals/execute`
6. `ApprovalExecutionOrchestrator` summarizes the outcome back to the user

### Background wake flow

1. Scheduler or task events enqueue wake payloads
2. `WakeOrchestrator` classifies notify-vs-backlog behavior
3. Notify-worthy events are drafted through the runtime and delivered through the interface
4. Non-notify events are persisted to the context backlog for later turn injection

## Agent graph behavior

- Tool-call validation runs before execution
- Guardrail precheck evaluates requested actions and only allows approved tool call IDs through
- Claim-check verifies immediate execution claims against tool evidence before the turn ends
- Claim-check has retry and backoff handling for empty assistant output, unusable checker output, and claim or evidence mismatch
- Streaming has a fallback path that guarantees a visible user-facing message when no chunks are produced

## Context policy

Configured in `src/opentulpa/core/config.py`:

- `AGENT_CONTEXT_TOKEN_LIMIT` default `12000`
- `AGENT_CONTEXT_RECENT_TOKENS` default `3500`
- `AGENT_CONTEXT_ROLLUP_TOKENS` default `2200`
- `AGENT_CONTEXT_COMPACTION_SOURCE_TOKENS` default `100000`

Compaction is hysteresis-based: the runtime compacts at the high watermark, then reduces toward a lower target while folding older history into a bounded rollup injected as system context.

## Prompt caching

- Controlled by `AGENT_PROMPT_CACHING_ENABLED`
- Stable prompt prefix content is separated from turn-volatile context before model invocation
- Anthropic models use request-level cache control
- Gemini models use per-message cache breakpoints on the stable prefix
- OpenAI-compatible models that cache automatically do not receive explicit cache markers

## Approval model

- Internal and read-oriented actions are deterministically allowed by policy
- External-impact actions are gated through the approval broker
- Pending approvals are durable in SQLite at `.opentulpa/pending_approvals.db`
- Approval prompts are surfaced immediately when handoff is detected
- State machine: `pending -> approved|denied|expired`, then `approved -> executed`

## Internal API boundary

- `/webhook/*` is the public webhook ingress surface for Telegram plus the Composio OAuth callback path
- `/webhook/telegram` handles both ordinary Telegram chat updates and Telegram Business inbox updates
- Public internet clients are denied for all non-webhook routes except health checks
- `/webhook/telegram` requires Telegram secret header auth through `x-telegram-bot-api-secret-token`
- `/webhook/composio/callback` is the public landing path for Composio auth flows
- `/internal/*` routes are intended for server-local traffic only
- `scripts/manager.py` auto-generates `TELEGRAM_WEBHOOK_SECRET` for tunnel runs when not provided
- `python -m opentulpa` can auto-register the Telegram webhook when public URL settings are available

## Runtime data stores

- LangGraph checkpoints: `.opentulpa/langgraph_checkpoints.sqlite`
- Approvals: `.opentulpa/pending_approvals.db`
- Context events: `.opentulpa/context_events.db`
- Customer profiles: `.opentulpa/customer_profiles.db`
- Thread rollups: `.opentulpa/thread_rollups.db`
- Link aliases: `.opentulpa/link_aliases.db`
- Skills: `.opentulpa/skills.db`
- File vault: `.opentulpa/file_vault.db` plus file storage
- Intake workflows and bookings: `.opentulpa/intake.db`
- Telegram Business inbox state: `.opentulpa/telegram_business.db`
- Tasks and wake queue: `.opentulpa/tasks.db`, `.opentulpa/wake_events.db`

## Observability

- Structured agent behavior log is enabled by default through `AGENT_BEHAVIOR_LOG_ENABLED=true`
- Default path: `.opentulpa/logs/agent_behavior.jsonl`
- Logs include turn lifecycle, graph node outcomes, guardrail decisions, claim-check retries, and tool execution outcomes
- Optional PostHog analytics can be enabled with `POSTHOG_API_KEY` and `POSTHOG_HOST`

## Extension points

- Add tools in `src/opentulpa/agent/tools_registry.py`
- Add internal APIs in `src/opentulpa/api/routes/*`
- Add interface adapters under `src/opentulpa/interfaces/*`
- Add approval adapters under `src/opentulpa/approvals/adapters/*`
- Add skills via `src/opentulpa/skills/*`

For external integrations, also read `docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md`.

## Failure behavior

- Guardrail classifier uncertainty defaults to approval-required
- If approval delivery fails, the action remains non-executed
- Tool-call failures return explicit tool error messages back into the graph
- Wake delivery failures are persisted to the context backlog for later recovery
