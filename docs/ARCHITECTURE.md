# OpenTulpa Architecture

This document explains how OpenTulpa is put together today.

If you only need the mental model, it is this:

```text
an inbound event arrives -> the runtime reloads durable context -> the agent plans and uses tools -> results are persisted for the next turn
```

OpenTulpa is designed around one core assumption: a useful agent should behave like a long-running worker, not a stateless chat session.

## Design goals

- Keep interfaces thin and replaceable
- Keep agent decision logic centralized in the runtime graph
- Keep domain boundaries explicit for easier testing and refactoring
- Persist user context, directives, and artifacts across sessions
- Prepare durable operating context from source material instead of injecting broad raw files forever
- Enforce safety at tool-action time, not as an afterthought

## Main runtime pieces

- `src/opentulpa/api`: FastAPI app composition and route registration
- `src/opentulpa/api/routes`: internal API routes, Telegram webhook routes, and Composio callback/status routes
- `src/opentulpa/application`: orchestration use cases such as `TurnOrchestrator` and `WakeOrchestrator`
- `src/opentulpa/domain`: typed domain contracts
- `src/opentulpa/agent`: LangGraph runtime, graph nodes, compaction, and tool registry
- `src/opentulpa/interfaces/telegram`: Telegram transport, parsing, streaming relay, and Telegram Business inbox persistence
- `src/opentulpa/context`: profiles, event backlog, file vault, thread rollups, and link aliases
- `src/opentulpa/skills`: durable skill storage and retrieval
- `src/opentulpa/scheduler`: routine scheduling
- `src/opentulpa/tasks`: task runtime, sandbox, and wake queue integration

## The important architectural idea

The system is split so that transports and storage are replaceable, while the agent runtime stays the center of decision-making.

- Interfaces move data in and out
- The application layer shapes requests and responses
- The runtime decides what to do
- Tool validation and execution determine what can run this turn
- Context, skills, and artifacts make future turns better

## Primary request flows

### Telegram turn flow

1. Telegram calls `POST /webhook/telegram`
2. `interfaces/telegram/chat_service.py` parses text, files, and voice, then resolves `customer_id` and `thread_id`
3. Owner/support chats resolve the active turn mode; active workflow setup threads use workflow setup prompting
4. LangGraph runs nodes such as `agent`, `validate_tools`, `tools`, and `finalize_turn`
5. The assistant reply is streamed back to Telegram
6. Tool-call preambles and interactive progress updates can be surfaced before the final reply when the runtime emits them
7. Telegram webhook handling returns quickly while the tracked turn continues in the background when needed

### External DM intake flow

This is the flow behind persistent lead handling such as Telegram Business inboxes.

1. A user configures an intake workflow through normal OpenTulpa conversation
2. OpenTulpa persists that workflow and stores a synced durable workflow skill
3. Inbound messages arrive through the configured source
4. Intake service loads the external conversation plus any active or recent booking state for that lead
5. The runtime decides whether the message matches the workflow, whether follow-up is needed, and whether the booking is ready to save
6. Intake service performs the idempotent reply or save step
7. Per-conversation cursors prevent reprocessing the same inbound message as fresh work

### Workflow setup and prepared knowledge flow

This is the flow behind "brief and equip the employee."

1. Owner or bound support operator describes the job in chat
2. Runtime enters workflow setup mode when the setup tools/session are active
3. Uploaded source files are stored in the file vault
4. For broad source material, the agent inspects structure first, then selects relevant sections
5. The selected material is compiled into a smaller workflow knowledge file
6. The draft workflow stores that prepared file id in `knowledge_file_ids`
7. Owner/support confirms the proposed workflow before activation
8. Future intake decisions load the workflow, bound knowledge files, active booking state, and recent conversation state

The runtime should not treat a large spreadsheet, PDF, or policy dump as permanent raw prompt context. The setup phase prepares the operational subset the worker needs, and the intake phase uses that durable prepared knowledge.

### Telegram Business identity and scoping

Telegram Business has three separate identity layers:

1. The deployed bot and webhook are runtime configuration. `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`, and `PUBLIC_BASE_URL` identify the bot process and public webhook endpoint. A connected Telegram Business account is not configured in env vars.
2. The OpenTulpa tenant is the durable owner/customer scope, represented by `customer_id`. Normal owner chats default to `telegram_<owner_user_id>` unless support act-as binds an operator to an existing customer tenant.
3. The connected Telegram Business inbox is runtime state discovered from Telegram. Telegram sends a `business_connection` update, OpenTulpa stores its `business_connection_id` under the active `customer_id`, and future `business_message` updates refer back to that connection id.

An intake workflow is scoped by `customer_id` and, for Telegram Business sources, by `source_config.business_connection_id`. If the owner has exactly one connected Business account, setup can resolve that connection automatically. If one tenant has multiple connected Business accounts, the workflow must bind the intended `business_connection_id` explicitly.

Lead/customer conversations inside the connected Business inbox are scoped below the Business connection by Telegram chat id. That chat id is not the owner chat and is not an env var; it identifies the external conversation being handled by the workflow. Intake state, booking cursors, and idempotency are stored per workflow and per external conversation.

This keeps these concerns separate:

- owner/operator chat history: the private OpenTulpa setup/debug conversation
- support chat history: a support-specific thread that may act on a bound customer tenant
- customer tenant state: workflows, files, skills, memory, and Business connections keyed by `customer_id`
- Telegram Business inbox state: stored `business_connection_id` plus external lead chat/message state
- bot runtime config: token, webhook secret, and public URL used to receive Telegram updates

If a Business account is not visible after setup, inspect webhook delivery and stored connection state instead of adding the Business account to env vars. The useful checks are Telegram `getWebhookInfo`, `/debug_logs` webhook diagnostics, `.opentulpa/telegram_business.db`, and the workflow `source_config.business_connection_id`.

### Profile identity aliases

OpenTulpa identity is not required to be Telegram identity. The durable account id can be a generic `user_id` such as `usr_default`, and a Telegram user id can be bound later to add Telegram chat capability. Generic identity alone does not enable Telegram: Telegram chat and Telegram Business require a real Telegram user id plus a configured bot token/client.

Profile binding is alias resolution, not data copying. Before storage reads or writes, API and Telegram entrypoints resolve the inbound id to one storage scope:

```text
request customer_id: usr_default  -> storage_user_id: usr_default
request customer_id: telegram_123 -> storage_user_id: usr_default
```

Use the generic id when creating a non-Telegram OpenTulpa account. All normal OpenTulpa state can be created under that id: profiles, workflows, files, skills, memory, web chat events, scheduler rows, wake/search state, artifacts, and sandbox files. Example workflow payload:

```json
{
  "customer_id": "usr_default",
  "name": "Car Wash Intake"
}
```

Bind Telegram only after the real Telegram user id is known:

```http
POST /profiles/bind-telegram
Content-Type: application/json

{"user_id":"usr_default","telegram_user_id":"123"}
```

After that bind, requests using either `usr_default` or `telegram_123` operate on the same canonical storage scope. The id `telegram_123` is the Telegram alias format; callers pass the raw Telegram id only to the bind route.

First-created storage wins. If `usr_default` exists first, binding maps `telegram_123` to `usr_default`. If `telegram_123` already existed first, binding maps `usr_default` to the existing Telegram storage so legacy Telegram users keep their data. If both ids already have separate data, OpenTulpa rejects automatic binding because a silent merge can corrupt conversation history, artifacts, workflow state, or sandbox references.

The current product model supports one generic OpenTulpa user and at most one main Telegram user binding for it. Existing Telegram-canonical users keep working because unbound Telegram chats still fall back to `telegram_<telegram_user_id>`.

Manage ids with:

- `GET /profiles`: list known profile ids and bindings.
- `POST /profiles/bind-telegram`: bind a generic `user_id` to a Telegram user id.
- Any customer-scoped route: pass either the generic id or its Telegram alias; the route resolves to canonical storage before reading or writing.

Do not copy history, artifacts, or workflow rows between ids. If a user needs Telegram capability, create or keep the generic account, bind the Telegram user id, and let alias resolution share the same storage scope.

### Support act-as flow

Support operators are trusted operators configured by `TELEGRAM_SUPPORT_USER_IDS` or `TELEGRAM_SUPPORT_USERNAMES`.

Normal Telegram access is controlled separately by `TELEGRAM_ALLOWED_USER_IDS` or `TELEGRAM_ALLOWED_USERNAMES`. Those users are allowed to use the bot as owners/operators, but they do not automatically share one owner tenant. A normal allowed chat creates or reuses its own owner session and defaults to `customer_id=telegram_<user_id>` when no existing mapping is present.

Generic-first deployments can set `OPENTULPA_OWNER_CUSTOMER_ID=usr_default` and one `TELEGRAM_ALLOWED_USERNAMES` value before the owner's numeric Telegram id is known. When the first allowed username message arrives, including from a group mention, OpenTulpa binds the observed numeric Telegram id to the generic owner id. This bootstrap does not run when the owner id is already Telegram-derived, such as `telegram_123`, and it does not run for multiple allowed usernames.

1. Support chat sends `/support_customers`
2. Support binds to a customer with `/support_bind <number-or-customer_id>`
3. Normal support messages run with `customer_id=<bound_customer_id>` and a support-specific `thread_id`
4. `/fresh` resets only the support thread for that bound customer
5. Owner chat history remains separate from support setup/debug history
6. Customer-facing proactive events still route to the owner by default
7. Support binding/unbinding and support-originated actions are recorded in internal support audit state

This keeps tenant/customer state, owner chat history, and support-operator chat history separate.

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

### Background wake flow

1. Scheduler or task events enqueue wake payloads
2. `WakeOrchestrator` classifies notify-vs-backlog behavior
3. Notify-worthy events are drafted through the runtime and delivered through the interface
4. Non-notify events are persisted to the context backlog for later turn injection

## Agent graph behavior

- Tool-call validation runs before execution
- Workflow setup has a no-progress repair path that injects the current draft state when the model stalls
- Blank or unusable streamed output falls back to a visible user-facing message
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
- Context events: `.opentulpa/context_events.db`
- Customer profiles: `.opentulpa/customer_profiles.db`
- Thread rollups: `.opentulpa/thread_rollups.db`
- Link aliases: `.opentulpa/link_aliases.db`
- Skills: `.opentulpa/skills.db`
- File vault: `.opentulpa/file_vault.db` plus file storage
- Intake workflows and bookings: `.opentulpa/intake.db`
- Telegram Business inbox state: `.opentulpa/telegram_business.db`
- Telegram owner/support sessions and support audit: `.opentulpa/telegram_state.json`
- Tasks and wake queue: `.opentulpa/tasks.db`, `.opentulpa/wake_events.db`

## Observability

- Structured agent behavior log is enabled by default through `AGENT_BEHAVIOR_LOG_ENABLED=true`
- Default path: `.opentulpa/logs/agent_behavior.jsonl`
- Logs include turn lifecycle, graph node outcomes, workflow setup retries, and tool execution outcomes
- Optional Langfuse observability can be enabled with `LANGFUSE_PUBLIC_KEY`
  and `LANGFUSE_SECRET_KEY`; `LANGFUSE_BASE_URL` defaults to
  `https://us.cloud.langfuse.com`
- Langfuse environment filtering uses `LANGFUSE_TRACING_ENVIRONMENT` when set;
  otherwise OpenTulpa derives it from `LANGFUSE_DEPLOYMENT_TAG`, Railway service
  or environment metadata, then `local`

## Extension points

- Add tools in `src/opentulpa/agent/tools_registry.py`
- Add internal APIs in `src/opentulpa/api/routes/*`
- Add interface adapters under `src/opentulpa/interfaces/*`
- Add skills via `src/opentulpa/skills/*`

For external integrations, also read `docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md`.

## Failure behavior

- Tool-call failures return explicit tool error messages back into the graph
- Wake delivery failures are persisted to the context backlog for later recovery
