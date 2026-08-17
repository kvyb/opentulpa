# Architecture

OpenTulpa separates an immutable host controller, a mutable application child, and
persistent state. The child can replace its application source; it cannot replace the host
process currently supervising it.

## Boundaries

```text
owner/interfaces -> stable host -> active application child
                         |                 |
                         |                 +-- product state and capabilities
                         +-- trusted source repository
                         +-- activation journal
                         +-- process identity, health, probation, rollback
```

The host owns authentication, encrypted configuration, the evolution control token, Git
commit selection, dependency environments, child process identity, strict readiness,
probation, and rollback. The child owns Deep Agents and product behavior. Product databases,
memories, skills, and workspaces live outside both source checkouts.

## Source Model

The host maintains two repositories:

- The live repository is the exact checkout used to launch the child. Only the host imports
  release commits and changes its detached `HEAD`.
- `bootstrap/evolution/source` is one independent repository on persistent storage. Trusted
  owner tools edit it directly across conversations and host restarts.

`source_read`, `source_write`, and `source_edit` provide bounded text operations.
`source_bash` runs a bounded shell in the trusted repository, so ordinary Git handles
remotes, branches, fetches, merges, and conflict state. There is no candidate database,
parallel lineage model, patch-export protocol, or custom dependency proposal workflow.

Source bash is trusted owner execution, not an adversarial sandbox. It receives a small
environment without model/provider credentials. File operations reject path escapes and
symbolic links. Activation refuses credential-file paths such as `.env`, private keys, and
PEM files. Runtime `.env` changes remain a separate host-owned operation.

## Activation Journal

`bootstrap/evolution/activations.db` is the sole source-evolution journal. It stores:

- immutable release IDs bound to exact Git commits;
- one active, previous, and last-known-good release decision;
- one in-progress activation at a time; and
- tenant-scoped idempotency keys, request hashes, terminal results, and notification state.

An activation is recorded before dependency preparation or runtime replacement. A host
restart launches the journal's active commit and resumes any `preparing` operation. A commit
created before a crash remains in the persistent repository even if no activation row was
written yet.

## Dependency Environments

The host prepares immutable runtime environments with `uv sync --frozen --no-dev
--no-install-project` plus the deployment's configured optional bundles. Their identity is
derived from `pyproject.toml`, `uv.lock`, Python, the install profile, and those bundles, so
source-only commits reuse the existing environment. The mutable project itself is loaded from
the exact live checkout through `PYTHONPATH`.

Before replacing the child, the host runs bounded source compilation with its isolated trusted
interpreter; it never imports editable source. Full Ruff, mypy, and pytest runs belong in the
trusted worktree or CI, not in the production activation path. Child startup then exercises
application and tool-contract imports, real database opening, identity checks, and health endpoints.

## Serving Lifecycle

```text
commit -> journal preparing -> dependency environment -> safe compile
       -> stop old -> start exact commit -> strict readiness -> live probation
       -> stable Deep Agent review -> active or exact rollback + owner repair handoff
```

Cutover is sequential and has an availability gap. The current implementation is not a
standby or zero-downtime A/B system because both child versions share one live checkout,
one product state root, and one ownership record. On startup or probation failure, the
runtime supervisor restores the exact previous `RuntimeLiveSourceSpec`. The journal advances
only after that call succeeds.

The reviewer runs in the stable host with the initiating owner's pinned inference plan. Its prompt
comes from the previous release, so a candidate cannot weaken its own review. It inspects disposable
copies of both generations and may probe the running candidate, redacted logs, tests, processes,
ports, networking, Docker, and services. A rejection is delivered as a trusted evolution event to
the restored owner runtime; that agent repairs the persistent source worktree and retries activation.

Rollback activates the recorded previous healthy release through the same runtime path.
Rollback does not rewind product database migrations, messages, purchases, authorization
changes, or other external effects. Those paths retain their own idempotency and recovery
contracts.

## Host Evolution

Edits to child-visible modules become active through `source_activate`. Edits to the host
controller code are present in Git but cannot replace the already-running controller.
Activating those changes fully requires the outer supervisor: a new Docker/Railway deploy or
local controller installation. This boundary keeps rollback authority outside mutable code.

## Persistent State

Back up the whole configured data root. In particular, preserve product databases,
`bootstrap/evolution/source`, `bootstrap/evolution/activations.db`, host configuration, the
runtime `.env`, and runtime dependency environments. OpenTulpa is a single-controller,
single-active-child SQLite deployment; multiple replicas require shared-store replacements
and explicit leader fencing first.
