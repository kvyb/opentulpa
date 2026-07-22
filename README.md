<p align="center">
  <img src="docs/assets/opentulpa-logo.png" alt="OpenTulpa" width="180"/>
</p>

<h1 align="center">OpenTulpa</h1>

<p align="center">
  <strong>A small, self-hosted Deep Agents operator that can inspect, edit, test, and safely replace its own code.</strong><br/>
  One Agent API, explicit tools, durable triggers, and a fixed rollback boundary.
</p>

## What It Is

OpenTulpa is one persistent agent with multiple replaceable ways to reach it. The local
terminal client, Telegram worker, schedules, intake, browser automation, and future
integrations all submit work through the same tenant-scoped Agent API. They do not own
their own model loops or conversation stores.

[`deepagents==0.6.12`](https://github.com/langchain-ai/deepagents) owns the agent loop,
planning, delegation, checkpointing, summarization, memory, skills, and filesystem
middleware. OpenTulpa supplies the smaller product and safety boundary around it:

- authenticated tenant and actor identity;
- a generated, typed tool contract with persisted approvals;
- deterministic intake, scheduling, delivery, and external side effects;
- tenant memory, skills, and persistent workspaces;
- capability workers that use the universal Agent API;
- an immutable bootstrap that evaluates, activates, and rolls back source releases.

There is no custom LangGraph harness, model loop, tool gateway, prompt compactor, or
legacy runtime facade.

## Fixed And Mutable

OpenTulpa keeps deployment machinery outside the application image it replaces.

| Fixed host boundary | Mutable application runtime |
|---|---|
| Setup, owner authentication, encrypted configuration, process health, logs, proxy, recovery | Deep Agent service, prompts, tools, and approval policy |
| Source-worktree sandbox limits and trusted evaluation commands | Agent API, product services, integrations, and interfaces |
| Source sandbox, fixed evaluator, activation health, and rollback | Capability workers, manifests, local TUI, tests, and documentation |
| Host credentials and deployment authority | Any other normal, secret-free repository source path |

The owner can ask the main OpenTulpa agent to change any source. That agent edits a
detached Git worktree through an unprivileged sandbox, never the serving checkout. It can run
tests and experiments over multiple chat turns, inspect its own traces, and ask the owner
for feedback. `source_release` creates one persisted chat approval; after approval the
fixed supervisor commits and evaluates the exact bytes, binds a release artifact, and
queues health-checked activation. Railway and Docker use a persistent source overlay
inside the stable host; managed VM mode builds a rootless OCI image. No second model or
host-shell approval participates.

The dependency lock and trusted image recipe remain fixed during this path. A dependency
change requires an administrator to rebuild the immutable runtime base. Host recovery
commands remain available if both the mutable release and its normal approval interface
are unavailable.

The chat approval is a policy enforced by the currently trusted application release. It
is not cryptographic protection against an already malicious release: that release holds
a scoped evolution credential so it can drive the source workflow. The stable bootstrap
proves source lineage, fixed checks, image identity, staged health, and rollback; it does
not independently prove human intent. Requiring protection from a hostile active release
would need a second authority outside the application, which this deliberately simple
one-approval design does not include.

This gives instances room to specialize without silently fragmenting the project. An
instance keeps a Git lineage, and an evaluated candidate can be exported as a sanitized,
digest-checked patch for normal upstream review.

## Interfaces And Capabilities

FastAPI exposes the headless public API. The bundled `opentulpa` command opens a native
OpenTUI client with streamed Markdown, a visible activity spinner, compact expandable
tool calls, drag-and-drop attachments, server-backed sessions, and clickable
**Approve**, **Edit**, and **Reject** controls. Telegram uses the same Agent API with a
scoped credential. Use `ctrl+p` or `/sessions` to reopen the same durable Deep Agents
threads from another client. Type `/` in the composer to search and select commands.

The API key remains the zero-friction default. Inside any owner conversation, `/model`,
`/reasoning`, and `/speed` select that thread's next-run inference settings without
restarting OpenTulpa.
`/login codex` optionally connects a ChatGPT Codex subscription through device login;
the rotating OAuth credential is encrypted on the server and never shared with an
existing Codex CLI login. Codex has no implicit cross-provider fallback. Use
`/model codex MODEL fallback` only when you explicitly want transient Codex failures to
fall back to the configured Kimi-to-GLM API chain.

For example, from the terminal client you can write:

> Enable Telegram for this OpenTulpa. Here is my BotFather token: `<token>`.

Recognized credentials are encrypted before the message reaches a checkpoint; the
model sees an opaque `secret://` handle. OpenTulpa can test the bundled Telegram
capability, ask for activation approval, and start its worker. Pair the first Telegram
account with `/start <code>`; by default the one-time code is the last eight characters
of the bot token. A later request to change Telegram behavior becomes a source
candidate and managed release rather than an in-place edit.

In managed OCI mode the stable bootstrap derives the active release image and runs that
worker rootless with only private capability `/state`; product `/workspace`, source,
databases, host credentials, and the container socket are never mounted. Generation
handover stops the old poller first and restores it if the new generation fails. Direct
development mode keeps reviewed subprocess workers. The bundled Docker and Railway host
can replace and roll back its Deep Agents child without a container socket.

The stable setup host imports `TELEGRAM_BOT_TOKEN` into the encrypted capability vault
and starts only the long-poll worker. Explicit direct `server` mode retains the webhook
adapter for development and blocks dynamic Telegram so two consumers cannot use one
bot.

Browser automation, Composio, document parsers, and Crawl4AI are optional adapters,
not agent-runtime dependencies. The core installs and starts without them:

```bash
uv sync --no-dev                         # lean core and API
uv sync --no-dev --extra browser         # Browser Use Cloud SDK and Playwright CDP client
uv sync --no-dev --extra integrations    # Composio
uv sync --no-dev --extra documents       # PDF, workbook, and encoding helpers
uv sync --no-dev --extra research        # Crawl4AI extraction
uv sync --no-dev --extra bundled         # all optional bundled adapters
```

`start.sh` keeps the lean core by default. Set `OPENTULPA_EXTRAS=bundled` (or a
comma-separated subset such as `integrations,documents`) to retain those adapters
across installs and bake the same extras into a managed runtime image.

With `BROWSER_USE_API_KEY`, explicit browser tools use a tenant-scoped Browser Use
Cloud session. The Playwright package is only the CDP control client; Chromium and its
target network run in Browser Use Cloud isolation. OpenTulpa has no host-browser
fallback. It requires explicit destination domains and rejects direct private or
link-local targets, but it cannot DNS-pin Chromium inside the vendor environment.
`content_fetch` has a bounded built-in HTML
extractor and uses Crawl4AI only when the research extra is present.
When the configured model endpoint is OpenRouter, `web_search` uses OpenRouter's
model-agnostic web plugin with the same API key and returns only grounded URL-cited
results; no separate search-provider key is required. `EXA_API_KEY` remains an
optional direct-provider override.

## Runs, Notifications, And Approvals

All owner interfaces use the same V2 surfaces:

| Endpoint | Purpose |
|---|---|
| `POST /v2/agent/threads` | Create a durable server-backed conversation |
| `GET /v2/agent/threads` | List tenant-owned conversations |
| `GET /v2/agent/threads/{thread_id}/timeline` | Replay messages, tools, artifacts, and approvals |
| `PATCH /v2/agent/threads/{thread_id}` | Rename or archive a conversation |
| `GET/PATCH /v2/agent/threads/{thread_id}/inference` | Read or change that conversation's model |
| `/v2/inference` | Discover models and manage optional Codex device authentication |
| `POST /v2/agent/runs` | Start an owner run and stream normalized SSE events |
| `GET /v2/agent/runs/{run_id}` | Read tenant-scoped status and pending approvals |
| `POST /v2/agent/runs/{run_id}/resume` | Approve, edit, or reject an interrupted run |
| `GET /v2/notifications` | Long-poll durable background and approval notifications |
| `POST /v2/notifications/{id}/ack` | Acknowledge one notification for this interface |
| `/v2/agent-specs` | Revisioned model, tools, memory, workspace, and instruction policy |
| `/v2/trigger-specs` | Revisioned time, interval, or authenticated-event triggers |
| `/v2/schedules` | Simple reminder and agent-job projection over trigger specs |
| `/v2/capabilities` | Test and activate release-bundled capability revisions |
| `/v2/evolution` | Inspect release lineage and export evaluated contribution patches |

Run streaming is normalized to `run.started`, `message.delta`, `tool.started`,
`tool.completed`, `approval.required`, `artifact.ready`, `run.completed`, and
`run.failed`. Each event has a run ID, monotonic sequence, and timestamp. The durable
notification stream carries schedule results, pending approvals, candidate results,
activation failures, and rollback outcomes back to the TUI and Telegram after restarts.

Tool arguments never contain tenant IDs, credentials, filesystem roots, or ownership
identifiers. The registry in `opentulpa.tooling` generates LangChain tools, approval
policy, audit metadata, JSON Schema, and the committed
[tool contract](docs/tool-contract.md).

## AgentSpecs And Triggers

An `AgentSpec` is an immutable behavioral revision: model alias, instructions, tool
allowlist, memory scope, workspace scope, delegation, and runtime budgets. Secrets
remain behind explicit tools and capability manifests rather than entering an agent
configuration. Owner, routine, and intake are seed specs, not separate runtimes.

A `TriggerSpec` selects an exact `AgentSpec` revision and adds a source, instruction,
and delivery rule. Sources can be one-off time, cron with an IANA timezone, interval,
or an authenticated event. APScheduler and the durable dispatcher only submit runs;
they never auto-approve an interrupted action.

The simpler `/v2/schedules` and schedule tools project:

```text
At | Cron -> Reminder | AgentJob -> routine AgentSpec -> owner notification
```

Use full AgentSpec and TriggerSpec revisions when a background process needs another
model, a narrower tool set, isolated memory, different instructions, or an external
event source.

## Self-Improvement

On Docker, Railway, and managed installations, the owner agent controls one isolated
source checkout backed by persistent Git lineage:

- `source_shell` creates or resumes it and can inspect, edit, test, and experiment with
  any OpenTulpa source using ordinary shell commands;
- `source_status` shows the current checkout and bounded diff without changing it;
- `trace_list` and `trace_get` expose the agent's own redacted durable execution traces;
- `source_release` requests one native chat approval, runs fixed checks, builds the exact
  source commit, and queues safe activation;
- `source_rollback` restores the previous healthy release with owner approval.

The source shell cannot see production data, credentials, the serving checkout,
bootstrap state, deployment controls, or a container socket. It is unprivileged,
resource-bounded, and has outbound network access without gaining access to host
secrets. The agent may change core runtime, API, integrations, interfaces, tools,
prompts, schedules, or add new code. The stable bootstrap and its release recipe remain
outside the mutable release.

Evaluation and release building are bound to the source commit, dependency lock hash,
evaluator fingerprint, and artifact digest. The candidate's Dockerfile is not used.
Dependency-lock changes still require an administrator-built runtime base.

Failures remain in the lineage with a sanitized cause. The originating owner thread
and notification stream receive completion or failure. After a successful cutover,
the new release uses the same persistent checkpoints, memory, skills, notification
store, and workspace, so it can explain what happened. If the new release fails during
activation or probation, the immutable bootstrap automatically restores the previous
release and reports the rollback. Only release-coupled capability worker state is
restored; messages, checkpoints, files, memories, bookings, schedules, and other
product data written during probation are preserved. Self-updates therefore must not
perform irreversible product-data migrations.

Rollback restores code, the serving process, and release-coupled capability state. It
cannot retract an external message, purchase, authorization change, or other provider
effect already emitted while a candidate was serving probation traffic. Changes near
external effects must be rehearsed with fake sinks and continue to rely on tool approval
and idempotency; image rollback is not a transaction over the outside world.

## Start

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
./install.sh
opentulpa
```

Choose **Run here**, enter the model API key once, and the CLI starts the private server
and opens the native TUI. A source checkout builds and caches the platform client on the
first run; CI also produces macOS and Linux platform archives. Later, just run
`opentulpa` again.

In the TUI, use `/model`, `/reasoning`, `/speed`, or `/login codex`. Remote
connections change the remote thread; no Codex environment variable, callback server,
or extra process is used.

```bash
# Remote server
opentulpa server --public-url https://tulpa.example

# Local machine
opentulpa connect https://tulpa.example
```

Paste the one-time pairing code printed by the server. See
[Deployment](docs/DEPLOYMENT.md) for Docker, Railway, managed self-improvement, and
non-interactive configuration.

### Managed self-improving mode

Set at least:

```env
OPENAI_COMPATIBLE_API_KEY=...
OPENTULPA_OWNER_TOKEN=...
EVOLUTION_ENABLED=true
OPENTULPA_RECOVERY_TOKEN=<32-or-more-random-characters>
OPENTULPA_INGRESS_TOKEN=<32-or-more-random-characters>
OPENTULPA_RELEASE_BASE_IMAGE=opentulpa-runtime-base:0.1.0
OPENTULPA_RELEASE_EGRESS_NETWORK=opentulpa-release-egress
OPENTULPA_RELEASE_WORKSPACE=/absolute/persistent/opentulpa-release-data
```

The egress network must be created and restricted by the administrator. Then:

```bash
./start.sh install managed   # builds runtime, evaluator, and tenant sandbox images
./start.sh doctor managed    # checks Git, OCI, images, network, and writable state
./start.sh run managed       # starts only, without reinstalling
# or: ./start.sh managed     # install, then start
```

The immutable gateway remains on the public host/port and proxies to the active
release. The release gets a persistent `/workspace`; it never receives the source
checkout, bootstrap database, `.env`, container socket, or sandbox image authority.
Tenant commands cross a private, lease-bound endpoint and the stable host derives the
tenant root and launches the exact locally resolved reviewed image. See
[Deployment](docs/DEPLOYMENT.md) for the complete host contract.

Use `opentulpa-recovery status`, `rollback`, `restart`, or `safe-mode` from the host if
the mutable release or its normal approval interface is unavailable. `/recovery` is
reserved and always returns `404`; recovery credentials are never accepted from a
browser control page.

Health checks are `/healthz` and `/agent/healthz` in host, direct, and managed modes.

## Persistence And Migration

The default is one active process with SQLite:

- fresh Deep Agents checkpoints and persisted approval interrupts;
- a tenant-namespaced store for `/memories/` and `/skills/`;
- persistent tenant `/workspace` directories;
- separate product stores for runs, notifications, jobs, triggers, intake, files,
  knowledge, profiles, connections, secrets, capabilities, and evolution lineage.

Multiple active replicas require shared saver/store and product persistence. Historical
legacy chat checkpoints are intentionally not imported.

Dry-run legacy data migration first:

```bash
uv run --extra migration opentulpa-migrate-deepagents \
  --data-root /path/to/copied-data --dry-run
```

The migration preserves product records, translates valid routines to AgentSpec and
TriggerSpec-backed schedules, translates setup sessions to drafts, exports tenant
memories, and converts user-authored skills. Invalid legacy rows are reported and
disabled rather than guessed. Cutover verification fails if a preserved product
database is absent, which prevents a wrong `--data-root` from looking successful.
`--allow-missing` is reserved for a verified new installation with no legacy data.
It also transactionally rebases persisted uploaded-file paths to the selected data
root. Destination content conflicts block cutover without disabling their legacy
source rows, so they can be resolved and the migration safely rerun.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src
```

| Document | Contents |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Fixed/mutable boundary, universal protocol, and flows |
| [Tool Contract](docs/tool-contract.md) | Complete model-visible tool surface |
| [Deployment](docs/DEPLOYMENT.md) | Direct and managed host setup |
| [E2E Testing](docs/E2E_TESTING.md) | Migration, interface, improvement, and rollback rehearsal |
| [Prompt Cookbook](docs/CHAT_COOKBOOK.md) | Accurate requests for capabilities, triggers, and evolution |

MIT licensed.
