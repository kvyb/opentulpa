# End-To-End Testing

OpenTulpa does not maintain a second agent harness. Tests exercise the Deep Agents
service, universal protocol, deterministic product services, interface workers, and
immutable bootstrap directly.

## Automated Gates

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src
git diff --check
```

The suite covers:

- Deep Agent compilation, new checkpoints, memory/skills, ordered streaming, and
  approval approve/edit/reject after restart;
- generated tool schemas, hidden context, ownership, effect policy, idempotency,
  direct-service invocation, and sanitized errors;
- tenant sandbox mounts, paths, network, limits, and secret absence;
- AgentSpec and TriggerSpec revisions, model/tool policy, event authentication,
  schedules, timezones, misfires, duplicate dispatch, and paused approvals;
- intake draft revision/token conflicts, atomic activation, deterministic decisions,
  send-once delivery, sink retry, cursor handling, and recovery;
- jobs, file ownership, integration ownership, SSRF protection, browser sessions,
  capability workers, secret rotation, and Telegram;
- persistent source worktrees, secret/path rejection, fixed kernel-contract evaluation,
  exact full-source OCI construction, promotion persistence, bootstrap fencing,
  staging, automatic rollback, and restart recovery;
- migration dry-run, idempotency, counts, and checksums.

Default pytest uses fake adapters or deterministic local fixtures. Live model, Telegram,
Composio, Browser Use, and external sink calls are separate smoke tests.

## Data Migration Rehearsal

Use a copy of representative legacy data:

1. Stop every process that can mutate the copied databases.
2. Configure fake messaging, booking, and integration sinks.
3. Run `opentulpa-migrate-deepagents --dry-run` and archive the count/checksum report.
4. Run the migration without `--dry-run`.
5. Verify profiles, files, knowledge, workflows, setup drafts, bookings, connections,
   routines, memories, and user skills against the report.
6. Confirm invalid routines and setup rows are reported and disabled, not guessed.
7. Start the V2 application against the migrated copy.
8. Restart it and verify approvals, jobs, specs, triggers, memory, skills, and workspaces.
9. Confirm no fake sink received duplicate writes or sends.

Any product-data loss, unexpected checksum change, cross-tenant access, duplicate
external effect, or non-resumable approval fails the rehearsal.

## Universal Interface Rehearsal

Exercise web first, then prove another interface uses the same state:

1. Start a web-only deployment with no host `TELEGRAM_BOT_TOKEN`.
2. Open `/`, authenticate, submit text and an attachment, and observe ordered SSE.
3. Trigger a risky reversible test tool and verify the web approval can be approved,
   edited, and rejected.
4. Paste a dedicated BotFather test token and ask OpenTulpa to enable Telegram.
5. Verify the stored run text and traces contain only `secret://telegram_bot_token`, not
   plaintext.
6. Verify the Telegram manifest test is digest-bound and activation interrupts for
   owner approval.
7. Approve it, message the bot, and verify Telegram continues the same tenant's Deep
   Agents context. Pair the first account with `/start <last-eight-token-characters>`
   or the configured host override.
8. Trigger a background run and verify both web and Telegram receive their own durable
   notification and acknowledgement state.
9. Rotate the Telegram secret and verify the worker restarts on the new vault revision
   without replaying old updates.

Repeat with a host-configured webhook bot. Confirm the dynamic Telegram capability is
blocked so only one consumer owns the bot.

## AgentSpec And Trigger Rehearsal

1. Create an AgentSpec with a non-default configured model alias, a small tool
   allowlist, spec-local memory, no workspace, and bounded calls/time.
2. Activate that exact revision.
3. Create an `At` TriggerSpec referring to it and verify one execution.
4. Create a cron trigger in an explicit IANA timezone and exercise both DST directions.
5. Restart before a fire and prove it still executes once.
6. Replay the same source event ID and prove it does not execute twice.
7. Miss a one-off and a stale cron fire and prove both are skipped.
8. Make the run request approval and prove the dispatcher pauses rather than
   auto-approving.
9. Resume through web and Telegram and verify one terminal notification.

Also exercise `/v2/schedules` to prove its reminder and agent-job records are projections
over the same TriggerSpec store rather than a second scheduler database.

## Managed Self-Improvement Rehearsal

Run this against a copied release workspace and fake external sinks. Keep production
credentials out of the evaluator and candidate environment.

1. Build reviewed runtime/evaluator images with `./start.sh install managed`.
2. Create and restrict a dedicated `OPENTULPA_RELEASE_EGRESS_NETWORK`.
3. Run `./start.sh doctor managed`.
4. Start managed mode with empty bootstrap state and verify the canonical checkout is
   built, started with ingress disabled, health-checked, and installed as the first
   release.
5. Verify the gateway serves `/`, `/healthz`, `/agent/healthz`, and V2 streaming while
   the mutable container has no source mount, bootstrap state, `.env`, secrets file, or
   OCI socket.
6. Ask for a harmless core or UI improvement. Verify `source_shell` creates a detached
   worktree, can change the requested source, and resumes it on the next chat turn.
7. Run tests in the source shell, inspect a redacted `trace_get`, and iterate after owner
   feedback without creating another agent run loop.
8. Call `source_release` and approve its one persisted Deep Agents interrupt.
9. Verify fixed public, security, and kernel-contract evaluation run without network and
   are bound to the same source commit, lock hash, evaluator fingerprint, and OCI digest.
10. Observe staging, old-release drain, cutover, health checks, and probation. Verify the
    web change and the same conversation/memory after restart.
11. Verify the origin receives the release outcome through the
    notification stream and trusted thread event.
12. Queue rollback and verify the previous content-addressed image and lease return.

### Failure And Rollback Cases

Run separate candidates or fake-host injections for:

- evaluator failure before an artifact exists;
- staging health failure before cutover;
- drain timeout with in-flight work;
- production health failure after cutover;
- probation failure;
- bootstrap restart at every persisted activation phase;
- previous image missing during rollback;
- duplicate approval and promotion requests;
- candidate dependency-lock change;
- candidate attempt to commit a secret, symlink, traversal path, or special file;
- binary, secret-bearing, private-path, or oversized contribution patch.

Expected outcomes:

- pre-cutover failure leaves the old release active;
- post-cutover failure automatically restores the old release;
- failed restoration enters safe mode and stops forwarding;
- activation records and sanitized failure context survive restart;
- the originating owner receives failure/rollback notifications;
- the restored agent can explain the failure in the original thread using the
  persisted trusted event, without raw logs or secrets.

## Interactive Self-Improvement Rehearsal

1. Ask the owner agent to make a small core or interface change.
2. Verify `source_shell` creates one detached workspace and that later chat turns resume
   the same workspace.
3. Have the agent edit a file, run a failing test, inspect its own `trace_get` result,
   repair the code, and rerun the test successfully.
4. Verify the source shell cannot read production data, credentials, Git metadata,
   bootstrap state, host paths, or the container socket and has no network by default.
5. Approve `source_release` once in the originating web or Telegram conversation.
6. Verify fixed evaluation commits and builds the exact tested bytes and that bootstrap
   stages, drains, cuts over, and starts probation without a second CLI approval.
7. Inject a startup or probation failure and verify automatic previous-image rollback.
8. Verify the original conversation receives the release or rollback outcome and the
   agent can explain it using durable traces after restart.
9. Verify a dependency-lock change is rejected and the editable workspace remains
   available for repair.
10. Do not include irreversible product-data migrations: image rollback deliberately
    does not rewind product databases.

This is the convergence path for independently evolved instances. Promotion of a local
candidate and contribution to the canonical repository remain separate decisions.

## Live Smoke

Before resuming production consumers, verify with dedicated accounts and reversible
actions:

- `/healthz` and `/agent/healthz` return success;
- web streams ordered events and restores pending approvals after refresh;
- Telegram owner chat and attachments use the expected tenant and thread;
- an approval can be approved, edited, and rejected from web and Telegram;
- one intake reply and one booking reach the intended sink exactly once;
- one reminder and one restricted AgentSpec trigger execute and notify the owner;
- Composio connection ownership and a risky invocation require approval;
- a risky browser submission requires approval;
- tenant workspace data persists after release replacement and is invisible to another
  tenant;
- Langfuse contains corresponding run/model/tool traces without secrets;
- a benign managed candidate can activate and roll back without losing context.

Do not point rehearsal traffic at production customer inboxes, payment actions, or
irreversible sinks.
