# Architecture

OpenTulpa is a stable deployment host around a mutable Deep Agents application. The
host defines how a complete candidate source snapshot is evaluated, activated, and
rolled back. Inside that boundary the main agent may change the runtime, APIs, product
services, integrations, interfaces, tests, and presentation.

## System Boundary

```text
local owner TUI    Telegram worker       time/event trigger       intake ingress
     |                   |                       |                       |
     +-------------------+-----------------------+-----------------------+
                                 |
                    stable host proxy and auth
             setup + encrypted config + logs + /_host
                                 |
                    universal Agent API protocol
                identity + AgentSpec + origin + files
                                 |
                         DeepAgentService
                   create_deep_agent (0.6.12)
                                 |
                explicit tools + tenant backends
                                 |
           deterministic product services and adapters
```

Deep Agents owns:

- the model/tool loop;
- planning and subagent delegation;
- checkpointing, interrupts, and context summarization;
- native memory, skills, scratch files, and filesystem middleware;
- native streaming and human-in-the-loop control.

OpenTulpa owns:

- authentication and trusted tenant, actor, thread, and origin identity;
- AgentSpec and TriggerSpec revision control;
- the model-visible tool registry and approval classification;
- profiles, files, knowledge, connections, jobs, artifacts, and notifications;
- deterministic intake, schedules, delivery, idempotency, and audit records;
- tenant sandbox, network, path, and external credential policy;
- capability worker lifecycle and scoped Agent API credentials;
- candidate isolation, trusted evaluation, source lineage, release activation, and rollback.

`DeepAgentService` owns application lifecycle, persistence, and stream normalization
around `create_deep_agent`. It builds the Deep Agents profiles, backends, tools, and
callbacks, but it does not implement another loop, graph, planner, retry controller,
or context compactor.

## Context Compaction

Deep Agents 0.6.12 installs its summarization and filesystem middleware on the main
agent and its general-purpose subagent. OpenTulpa does not run a second compactor.

The configured OpenRouter models do not expose a LangChain `max_input_tokens` profile,
so Deep Agents uses its conservative fallback policy: summarize at approximately
170,000 tokens and keep the six most recent messages. A standard
`ContextOverflowError` triggers immediate summarization and one retry. Before full
summarization, arguments on older tool calls are clipped after the history grows past
20 messages. Tool results above 20,000 tokens and human messages above 50,000 tokens
are offloaded to backend files and replaced by bounded previews and paths.

Summarization generates a structured working summary and preserves the evicted history
under `/conversation_history/<thread-id>.md` in the thread backend. Deep Agents keeps
the raw message log in checkpoint state while using the summary plus recent messages
for subsequent model calls. OpenTulpa filters stream chunks marked
`lc_source=summarization`, so internal summary text is traced and checkpointed but is
never rendered as the assistant's answer.

## Inference Selection

Owner threads may override the configured API model with a revisioned API or Codex
selection. The service resolves and persists an immutable inference plan before a run
starts; approval resume and restart recovery reuse that exact plan. Graphs are compiled
and bounded-cached by AgentSpec, dynamic tools, provider, model, reasoning effort,
service tier, and Codex credential revision, so Deep Agents delegation and summarization
use the same model as the main turn. Routine, intake, and explicit non-default AgentSpec
model aliases ignore owner-thread preferences.

Codex uses the pinned private LangChain adapter directly, not a Codex or Hermes process.
OAuth credentials and device sessions are encrypted with the host key. A 401 forces one
serialized refresh and retry. Cross-provider fallback is off unless the thread explicitly
enables it, and is then limited to transient failures before output or tool activity.

## Fixed Kernel

Normal `serve` deployments have a small stable host process. It owns first-run claim,
owner authentication, AES-GCM encrypted runtime configuration, child readiness,
redacted process logs, restart, and HTTP/SSE proxying. The child gets a separate
internal owner credential. Telegram credentials never enter child process arguments or
model-visible configuration; the host stores them in the child's capability vault and
activates the reviewed polling worker through the Agent API.

Configuration replacement is revisioned. The host validates external credentials,
starts the candidate child, activates its interfaces, and only then advances the active
revision. If any step fails, it marks that revision failed and restores the prior child.
The host setup and recovery API remains available throughout.

Managed self-release adds a stronger immutable deployment layer:

The immutable bootstrap is a separate, long-lived host process. It owns:

- the public gateway and active-release lease;
- durable ingress and outbox records during release replacement;
- content-addressed OCI release preparation;
- lease-fenced OCI authority for mutable capability workers;
- staging, health checks, drain, cutover, probation, and rollback;
- safe-mode recovery if neither the candidate nor previous release is healthy;
- the source archive, evaluator, release builder, and contribution sanitizer.

The owner agent may edit any source in a detached checkout. Managed releases receive
none of the canonical serving checkout, Git metadata, bootstrap state, host environment
file, container socket, or tenant sandbox image authority.

The fixed Agent API is intentional. An interface can be rewritten without changing
how identity, streaming, approvals, files, or notifications work.

## Mutable Layer

The mutable layer is the complete secret-free OpenTulpa repository snapshot. The owner
agent edits it through a dedicated source shell, never through the ordinary tenant
`/workspace`. The stable builder exports the exact Git commit, rejects secret paths and
dependency-lock changes, and copies the full snapshot over a reviewed dependency base.
The candidate's Dockerfile and Git configuration are ignored.

The trusted builder and host both require the OCI label
`org.opentulpa.release.runtime-overlay=full-source-v1`. An image without that exact
boundary marker cannot be staged or activated.

In managed production, source-overlay workers never execute as subprocesses inside the
mutable release. The release sends only a reviewed module name, manifest, tenant config,
and declared ephemeral grants to a lease- and control-authenticated bootstrap API. The
stable host derives the active release image and runs that module rootless with CPU,
memory, PID, output, wall-time, and network policy. It mounts only a hashed
tenant/capability `/state`; it never mounts product `/workspace`, source, databases,
host credentials, an environment file, or the container socket. Direct development may
run reviewed bundled workers as subprocesses, but direct mode cannot provide safe
self-replacement or stable-host rollback.

## Universal Agent Protocol

`AgentRunContext` is the trusted envelope shared by web, interfaces, and triggers:

```text
tenant_id, actor_id, thread_id, channel, run_kind, correlation_id,
origin, agent_spec, trust_class
```

`RunSubmission` adds text, file IDs, a submission ID, timestamp, and idempotency key.
Authentication constructs the context; an interface cannot override its tenant,
channel, run kind, trust class, or AgentSpec. Owner API authentication resolves the
active owner spec in trusted application code. Every capability credential instead
persists one exact `AgentSpecRef` revision plus its reviewed run kind and trust class.
That binding comes from the approved interface worker manifest, not request text or
caller fields. The bundled Telegram worker explicitly receives the owner binding;
external interfaces must declare and receive an external, non-owner binding.

The public owner API is:

```text
POST /v2/agent/runs
GET  /v2/agent/runs/{run_id}
POST /v2/agent/runs/{run_id}/resume
GET  /v2/notifications
POST /v2/notifications/{notification_id}/ack
```

The stable bootstrap has a narrow authenticated control protocol for durable ingress,
release events, health, and drain. Mutable release handlers translate those envelopes
back into the same Agent API. There is no second model loop in the gateway.

Native LangGraph streaming is normalized to:

```text
run.started
message.delta
tool.started
tool.completed
approval.required
artifact.ready
run.completed
run.failed
```

Every event contains a run ID, monotonic sequence, and timestamp. Public tool arguments
and results are redacted. Runs and approval interrupts remain queryable after restart.

Checkpoint thread IDs include the exact AgentSpec ID and revision. Different specs can
therefore use the same logical thread name without sharing history or blocking each
other's unresolved approvals. Only authenticated uses of the exact owner spec and
revision share owner continuity across interfaces such as the local TUI and Telegram. Restricted
origins are additionally scoped by their immutable credential authority.

The notification store is a separate durable delivery stream. Each interface has its
own acknowledgement cursor, so the local TUI and Telegram can independently receive a
scheduled result or approval. Trigger, evolution, activation, and rollback outcomes
are deduplicated before delivery.

## AgentSpecs

An `AgentSpec` is an immutable tenant-owned behavioral revision. It selects:

- runtime profile and model alias;
- instructions and optional output schema;
- profile-default tools or an explicit allowlist;
- owner, spec-local, or disabled memory;
- read-write, read-only, or disabled workspace access;
- delegation policy;
- model-call and wall-time budgets;
- private or external isolation.

The seed specs are:

| Spec | Boundary |
|---|---|
| Owner | Full product tools, native planning/delegation, owner memory, read-write workspace |
| Routine | Restricted automation profile, owner memory, read-only workspace |
| Intake | External isolation, knowledge reads only, no memory, workspace, delegation, or secrets |

They are configurations compiled by the same factory, not custom runtimes. An owner
can create another spec with a different model alias or tool policy and bind a trigger
to that exact revision.

## Triggers And Schedules

A `TriggerSpec` contains one source, one exact AgentSpec reference, one instruction,
and one delivery rule. Sources are:

- `At` with an aware timestamp and IANA timezone;
- five-field `Cron` with an IANA timezone;
- fixed `Interval`;
- authenticated or trusted-internal `Event`.

The durable dispatcher and APScheduler run outside Deep Agents. They serialize and
deduplicate executions, skip stale one-offs and cron misfires, submit a restricted run,
and publish the result or pending approval. They never auto-approve a scheduled run.

The simple `Schedule` API is a projection onto this control plane:

```text
At | Cron -> Reminder(message) | AgentJob(instruction)
```

It uses the active routine AgentSpec. Advanced background processes use AgentSpec and
TriggerSpec directly, including another model, a narrower tool set, isolated memory,
or an authenticated event source. Internal intake polling remains an intake trigger,
not a user schedule.

## Tools And Trusted Context

The registry in `opentulpa.tooling` is the single source of truth for:

- tool names and versions;
- effect, approval, idempotency, execution, and timeout policy;
- Pydantic and LangChain schemas;
- Deep Agents `interrupt_on` configuration;
- machine-readable JSON Schema and [the committed tool contract](tool-contract.md).

Trusted `AgentRunContext` is injected through `ToolRuntime`; tenant IDs, credentials,
filesystem roots, and ownership identifiers are absent from model-visible arguments.
Application services repeat ownership checks at the data boundary. External writes
require idempotency. Deletes, authorization changes, sends, workflow activation,
browser submission, capability activation, and source promotion require persisted
owner approval.

Tools call application ports directly. There is no loopback model-tool HTTP gateway or
generic group-dispatch surface.

## Persistence And Filesystems

The single-active-process release uses:

- `AsyncSqliteSaver` for new Deep Agents checkpoints and interrupts;
- tenant-namespaced `AsyncSqliteStore` for native memory and skills;
- a run database for ordered events and approval state;
- product SQLite stores for specs, triggers, notifications, intake, jobs, files,
  knowledge, profiles, connections, capabilities, secrets, and idempotency.

Deep Agents routes:

- `/memories/` and `/skills/` to `StoreBackend`;
- `/workspace/` to `TenantContainerBackend`;
- ephemeral scratch paths to `StateBackend`.

Each tenant workspace persists across releases. Tenant commands execute as non-root
processes in disposable OCI containers with bounded CPU, memory, PIDs, wall time,
paths, output, and outbound bridge networking so the agent can fetch public code and
dependencies and exercise networked integrations. In direct mode the app uses a locally
resolved immutable image ID. In managed mode only the stable bootstrap owns the OCI
engine and image ID: the active lease holder may submit a bounded command and hidden
tenant ID to a private authenticated endpoint, while the host derives the workspace
path. The repository, host credentials, environment files, and container socket are
not mounted or exposed through that endpoint. Operators that need destination-level
egress restrictions enforce them at the OCI host or outbound proxy.

Persistent commands are transactional. The sandbox mounts a private copy, validates
the complete result, and atomically promotes it under a cross-process tenant lock.
Forbidden paths, links, special files, or oversized results discard only that command's
copy, leaving the prior valid workspace available to the next command. Every execution
has a host-generated container name. Timeout, cancellation, or output overflow forces
that container to stop and proves it absent before the private copy can be discarded;
such a copy is never promoted. A fsynced transaction journal and deletion tombstone let
the next direct or stable-host operation restore the previous workspace after a crash at
any rename or cleanup phase instead of silently creating an empty workspace.

The managed release has its own persistent `/workspace`, which contains product
databases and tenant workspace directories. Bootstrap state and canonical Git source
remain separate host paths.

## Capability Model

`CapabilityManifest` is an import-free, revisioned contract. It declares exported
tools and services, workers, config schema, dependencies, permissions, secrets,
network policy, and deterministic evaluation commands. Activation is bound to a
passing manifest digest and requires owner approval.

The mutable release accepts no tenant-supplied capability manifests or executables.
Its V2 capability API can only seed, test, activate, roll back, and deactivate exact
templates bundled with the current reviewed source release. In managed production the
composed worker client has no OCI engine or socket: the stable host verifies that the
command is exactly `python -m opentulpa.capability_workers[.<module>]`, derives the
active release image, and supplies the fixed mount/runtime policy. A new or modified
capability must enter through an isolated, evaluated source candidate; after promotion
it is part of the next reviewed release and can use the same lifecycle.

The seed distribution contains:

| Capability | Form | Purpose |
|---|---|---|
| Web | source-bundled | Fixed FastAPI surface plus mutable owner presentation |
| Browser | source-bundled, optional dependencies | Browser Use Cloud sessions controlled through a Playwright CDP client; no host browser |
| Telegram | installable interface worker | Polls Telegram and uses scoped Agent API/file/notification endpoints |

Capability workers do not receive an owner bearer token. They receive a scoped,
revocable credential plus ephemeral values resolved only from declared, revision-bound
secret handles. Rotating a bound secret reconciles and restarts the active capability.
Agent API credentials also persist the exact AgentSpec revision, run kind, and trust
class declared by the reviewed interface manifest. Requests cannot elevate that
binding. Telegram is explicitly owner-bound; an external interface must name a
non-owner spec and external trust class.

An interface worker calls the Agent API. A bundled tool worker exposes only its
manifest-declared MCP methods; `MCPToolRuntime` publishes those exact schemas into the
tenant's dynamic Deep Agents tool registry and removes them when the generation stops.
Replacing one capability revision swaps its full MCP bundle atomically. Two different
capabilities still cannot export the same name. A reviewed alternative to a fixed
product operation is deliberately exposed as `<capability>__<tool>` rather than
silently shadowing the kernel operation; this is the safe indirection for optional
Browser, Composio, and future adapter replacements.
This is the runtime path for reviewed model tools without a generic name/argument
dispatch tool. MCP audit events and completed idempotent results are persisted in
SQLite, so an activated tool's audit trail and replay protection survive a process
restart. Caller-provided idempotency keys are hashed with the trusted tenant,
capability revision, worker, and tool scope before use.

The stable host accepts a Telegram token in `/_host` or during non-interactive first
boot, verifies it with Telegram, writes it to an encrypted tenant secret handle, and
activates the long-poll worker. A supplied numeric owner ID pre-binds the private chat;
otherwise the worker uses its one-time `/start <code>` pairing contract. Explicit
legacy `server` development mode may still compose the webhook adapter, but `serve`
never starts both consumers for one bot.

Interface generation changes are exclusive: the old worker is stopped before the new
worker starts, both generations use the same capability-owned `/state`, and a failed
start restarts the prior generation without advancing the durable activation pointer.
The bootstrap persists non-secret container handles so this fencing survives bootstrap
process restarts.

Deactivation is a durable three-state transition: `active -> deactivating -> inactive`.
The compare-and-swap to `deactivating` hides the exact generation before tools, workers,
or credentials are stopped. A failed request either restores that same generation before
making it active again or leaves the hidden transition for startup reconciliation. The
inactive row is an idempotency tombstone, and a later activation advances the generation
instead of reusing it. This prevents a crash from leaving an active pointer aimed at a
stopped or revoked generation and prevents duplicate worker or tool exposure on retry.

## Intake

`IntakeWorkflow` is the only active intake configuration. Drafts use optimistic
revisions:

```text
save -> prepare -> exact proposal + hash-bound token -> atomic activate
```

A failed edit leaves the previous active workflow untouched. The intake Deep Agent can
only return a typed `IntakeDecision`. A deterministic applier validates required fields
and grounding, owns booking transitions and the completed-booking edit window, writes
sinks using booking idempotency, sends at most once, and advances the cursor only after
successful or intentional handling.

Webhook authentication, fast acknowledgement, durable ingestion, source/customer
binding, debounce/coalescing, per-conversation serialization, stale-decision
suppression, and restart recovery remain outside the agent.

## Source Evolution

OpenTulpa keeps source editing simple and deployment authority separate:

```text
owner asks main agent for a change
  -> source_shell creates or resumes its detached worktree
  -> main agent edits, tests, experiments, and discusses results in chat
  -> source_release pauses once for native owner approval
  -> stable host reruns fixed checks and commits the exact bytes
  -> trusted full-source OCI image bound to commit and evaluator
  -> durable promotion attempt starts automatically
  -> bootstrap staging, drain, cutover, health, probation
  -> active release or automatic rollback
  -> durable outcome returns to the originating conversation
```

There is no second improvement agent and no weakness-mining runtime. The same persistent
owner agent that receives the request controls the checkout and may use its normal planning,
memory, delegation, trace reads, and conversation context. Failed tests and builds leave the
workspace available for another edit-and-test turn.

The source sandbox hides Git metadata and cannot see production data, credentials, deployment
controls, bootstrap state, or the container socket. It runs rootless with CPU, memory, PID,
output, and wall-time limits and has outbound bridge networking for dependency installation,
public repository access, and experiments. The separate trusted evaluator and release builder
remain offline. The trusted host owns commit creation, fixed evaluation, the release recipe,
activation, and rollback.

The candidate's Dockerfile and ignore rules are not executed by the trusted builder. The builder
exports the exact commit and replaces the application snapshot on an administrator-reviewed,
lock-hash-bound dependency image. Emergency recovery remains host-only, but ordinary promotion
needs only the persisted Deep Agents approval requested by `source_release`.

That approval is enforced by the currently trusted mutable release. The bootstrap validates
what is built and whether it can serve safely; it cannot independently prove that a request from
an already malicious release reflects the owner's intent. Adding that adversarial guarantee
would require a separate host-side signing or approval authority and is intentionally outside
this one-approval design.

Dependency-lock changes fail closed and require a new administrator-reviewed runtime
base. Evaluation and release building verify that neither source files nor the commit
change during the process.

Passing candidates and failures are both archive nodes. A failed candidate records a
sanitized terminal cause and notifies its origin. A promotion failure before cutover
keeps the current release. A failure after cutover starts the previous content-addressed
image and restores its lease. Bootstrap snapshots only release-coupled state under
`.opentulpa/deepagents/capability_state`; rollback never rewinds conversations,
checkpoints, files, bookings, schedules, or other product data written during
probation. That scoped snapshot includes a manifest-bound record of active seed config
and secret-handle revisions. Persisted seed capability pointers and those activation
values are reconciled to exact manifests from the restored release, while candidate-only
seeds are deactivated without deleting their revision history. If restoration also
fails, the bootstrap enters safe mode rather than forwarding to an unknown release.

Probation is live traffic with production credentials. Rollback cannot retract messages,
purchases, authorization changes, or other provider writes already emitted during that
window. External-effect changes therefore require fake-sink rehearsal, approval, and
idempotency; the rollback guarantee covers the image, process, lease, and release-coupled
capability state rather than the outside world.

The new or restored release opens the same persistent checkpoints and notification
store. Bootstrap events are reintroduced as trusted, non-instruction data in the
originating thread, allowing the agent to tell the owner whether the improvement,
activation, or rollback failed without relying on the dead process's memory.

## Trace-Guided Improvement And Contribution

Completed, failed, cancelled, and interrupted runs remain in the tenant-scoped trace
store. `trace_list` and `trace_get` expose bounded redacted tool arguments, results,
failure fingerprints, and optional message events to the owner agent. The agent can use
that evidence directly in its normal conversation, edit the persistent source session,
run a regression test, and ask the owner to evaluate the result. There is no separate
weakness miner, proposal model, or improvement loop.

This is interactive, rollback-protected self-development, not proof of autonomous
recursive improvement. OpenTulpa does not yet generate held-out tasks, score competing
versions, or prove that a change improved behavior. The fixed evaluator establishes
release fitness, not scientific improvement.

An instance may legitimately diverge in its private Git lineage. The archive can still
produce a digest-bound, sanitized text patch for normal upstream review. Upstream
credentials stay outside OpenTulpa, so deploying locally and contributing to the
canonical repository remain separate decisions.

## External Adapters

Telegram, Browser Use Cloud, Composio, search providers, Crawl4AI, file parsers,
APScheduler, and Langfuse are injected adapters, not agent-runtime components.
FastAPI is the core HTTP protocol surface; Browser Use Cloud (controlled through a
Playwright CDP client), Composio, parsers, and Crawl4AI are optional dependency groups.

`content_fetch` rejects credentials in URLs, private and link-local targets, unsafe
redirects, DNS rebinding, unsupported content types, and oversized or slow responses.
Browser and Composio actions that cannot be classified safely fail closed into owner
approval.

## Migration And Cutover

The migration command is idempotent and reports dry-run counts and checksums. It
preserves product data, converts valid legacy routines to AgentSpec/TriggerSpec-backed
schedules, converts valid setup sessions to drafts, exports tenant memory, and converts
user-authored skills. Historical agent checkpoints and generated workflow skills are
not imported. Missing preserved databases fail verification by default; an explicit
`--allow-missing` is only for a verified new installation.
Preservation checks cover disabled workflows, intake cursors and pending runs, Telegram
Business messages, and knowledge preflight state. Legacy absolute file paths are
transactionally rebased to the current vault root after their bytes are verified.
Destination content conflicts block cutover without disabling the source row, and the
routine dry run evaluates the actual AgentSpec/TriggerSpec destination snapshots.

Production cutover is single-runtime: rehearse against copied data and fake sinks,
pause consumers and triggers, back up, migrate, deploy coordinated V2 clients, smoke
test, then resume. There is no legacy runtime-selection flag. Rollback of the data
migration restores the previous image and pre-cutover snapshot; managed source release
rollback uses the immutable bootstrap and does not mutate product data.
