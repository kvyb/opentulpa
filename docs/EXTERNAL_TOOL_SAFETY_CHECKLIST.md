# External Tool Safety Checklist

Use this checklist for every product tool or external adapter.

## Contract

- Add one versioned `ToolSpec` with explicit provider, effect, approval, idempotency, execution mode, and timeout.
- Expose only the minimum model-visible arguments; inject tenant, actor, thread, channel, credentials, and roots through trusted runtime context.
- Generate the LangChain schema, approval policy, audit metadata, JSON Schema, and `tool-contract.md` from the registry.
- Reject unregistered operations and unknown action classifications.
- If the feature is an interface, submit work through `RunSubmission`, resume through the run API, and consume the durable notification stream; do not add another agent loop or checkpoint store.
- Keep fixed-kernel changes separate from mutable capability work. A capability cannot replace trusted identity, the Agent API, tool policy, sandbox, evaluator, or bootstrap.

## Authorization

- Validate tenant ownership in the application service immediately before every read and write.
- Revalidate provider accounts, browser sessions, files, jobs, and artifacts at execution time.
- Never infer ownership from a model argument or provider display name.
- Add explicit cross-tenant tests for every resource identifier.

## Side Effects

- Execute authorized product effects without per-call approval; only recursive forced removal through owner shell tools uses a persisted approval interrupt.
- Require an idempotency key for every external write and reject key reuse with different arguments.
- Distinguish accepted background work from completed side effects.
- Return concrete success/failure evidence without leaking raw provider payloads.

## Data And Network

- Keep secrets in host-side adapters; never place them in prompts, tool arguments/results, traces, or sandbox mounts.
- Redact tokens, cookies, headers, credentials, internal paths, and media payloads from events and audit output.
- Bound response bytes, execution time, redirects, and content types.
- Reject private, loopback, link-local, metadata, rebinding, traversal, symlink, and special-file targets.

## Failure And Recovery

- Sanitize provider exceptions and accurately mark retryable errors.
- Persist destructive-shell approval interrupts and job state before returning to a client.
- Test destructive-shell approval recovery, rejection and editing, duplicate requests, partial failures, and indeterminate provider outcomes.
- Fail closed when ownership, classification, idempotency, or enforcement cannot be proven.

## Mutable Capabilities

- Bind activation to an immutable manifest revision, content digest, and passing deterministic test result.
- In managed production, launch every source-overlay worker through the stable lease-fenced rootless OCI authority; mount only capability `/state`, never product `/workspace`, source, databases, credentials, or a container socket.
- Permit reviewed subprocess workers only in direct development, and state that direct mode has no safe self-replacement or stable rollback.
- Give a worker only scoped, revocable Agent API credentials and declared secret handles; never an owner token.
- Record the exact config, secret-handle revisions, worker protocol, permissions, and network policy in the activation generation.
- Reconcile or disable the worker when a bound secret rotates; never continue with an untracked stale credential.
- Block duplicate transports, such as webhook and polling consumers for the same Telegram bot.
- Stop an old interface generation before starting its replacement; persist non-secret lifecycle state and restart the old generation without advancing activation if handover fails.
- Atomically replace same-capability MCP generations, reject collisions across capabilities, and namespace intentional alternatives to fixed tools instead of shadowing them.
- Require exact tested revisions, tenant authorization, and idempotency for activation, rollback, and deactivation.

## Source Evolution

- Give the owner agent only a context-owned detached source worktree; never edit the serving checkout.
- Permit normal repository source paths, but reject secrets, traversal, `.git`, `.venv`, symlinks, special files, and oversized trees before commit.
- Run fixed public, security, and kernel-contract checks in a secret-less evaluator over a disposable copy outside candidate control.
- Bind every release to the exact inspected source commit, lock hash, evaluator fingerprint, and release artifact digest before committing, evaluating, building, or queuing activation.
- State the trust model explicitly: the active release remains trusted deployment authority; hostile-release authorization requires a separate external authority.
- Stage and health-check before cutover; retain a content-addressed previous release for automatic rollback.
- Deliver sanitized failure and rollback context through the durable owner notification stream.
- Sanitize contribution patches and keep upstream credentials outside OpenTulpa.
