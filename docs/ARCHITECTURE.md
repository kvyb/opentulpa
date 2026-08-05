# Architecture

OpenTulpa has a stable host/controller, immutable Git-native wheel/venv generations, and
external persistent state. The controller is not replaced by the runtime it serves.

## Boundaries

```text
owner/interfaces -> stable host/controller -> active runtime generation
                         |                         |
                         |                         +-- product state/workspaces
                         +-- Git lineage, evaluation, activation, rollback
                         +-- sandbox/evaluator authority
```

The stable controller owns authentication, configuration, process supervision, source
mutation policy, fixed evaluation, generation publication, strict readiness, probation,
cutover, rollback, and recovery. The active runtime owns the Deep Agents application and
product behavior. Persistent product state is external to the generation so replacing a
runtime does not replace conversations, memories, skills, databases, or tenant workspaces.

## Immutable Generations

Each generation is identified by content and environment inputs including the exact Git
commit/tree, wheel, lockfile, evaluator fingerprint, Python runtime, state contract, and
runtime tree hash. The generation contract owns canonical serialization, identity, and
manifest digesting. `GenerationStore` alone seals the assembled tree, computes its runtime
hash, writes and verifies the manifest, and transitions `BUILDING` to `COMPLETE`; builders
cannot publish generations directly. Python venvs are not portable;
their path and interpreter layout are part of the target installation. See the
[Python venv documentation](https://docs.python.org/3/library/venv.html).

The stable launcher/controller generation remains present while runtime generations
change. `controller/current` names the active controller generation and
`controller/previous` retains the exact prior generation for operator recovery. Release
records similarly retain current and previous serving releases plus the HostConfig needed
to recreate them. Evolution releases, bootstrap release aliases, and host activation share
one typed immutable artifact-provenance contract rather than selecting trusted metadata
through independent key lists.

## Dependency Bases

Dependency changes use a separate stable-controller authority. The candidate may change
only `[project].dependencies` and `[project.optional-dependencies]`; it cannot select an
index, direct URL, local path, build hook, or workspace source. The controller binds the
candidate and its current diff before starting dependency resolution.

The resolver starts from the immutable trusted baseline lock and runs fixed `uv` and `pip`
argument vectors in an immutable OCI image with no production credentials. Network access
is limited to lock metadata resolution and hash-checked binary wheel download from the
configured HTTPS simple index. Source-only metadata for unselected optional groups remains
inside this worker. The controller rejects direct references, custom source configuration,
mutable output, untrusted artifact hosts, and any exported requirement that cannot be
downloaded and installed as a hash-pinned binary wheel.

The result is a sealed, content-addressed dependency base containing `uv.lock`, exported
hash-locked requirements, a wheelhouse, package inventory, and an offline-installed
dependency site. Candidate evaluation mounts that exact site read-only and includes it in
the evaluator fingerprint. Generation construction selects the same base by lock hash,
rebuilds the final-path venv offline, and records dependency-base provenance in the
release. System-package and arbitrary build-system evolution are not provided by this
resolver.

A native controller gives the worker a bind mount containing only one resolution staging
directory. A containerized controller instead uses a dedicated named volume and mounts
only that resolution's volume subpath into the sibling worker. Neither transport exposes
the candidate workspace, product data, controller state, or other resolver staging work.

## Git Lineage And Candidates

Source mutation starts in a detached Git candidate worktree. The candidate workspace is
not the serving source tree and is not reused as the source for a later serving process.
The controller commits the exact candidate bytes, evaluates that commit, and builds a new
wheel/venv generation from it. The runtime supervisor accepts only a verified generation
identity; it has no source-directory launch target or source-overlay activation path.

Instance, upstream, and accepted-upstream lineage are native Git refs. Upstream changes
are merged with Git's native merge base, index stages, `MERGE_HEAD`, and conflict paths.
Conflicts remain durable in Git state across restart until explicitly resolved; there is
no parallel conflict database that can disagree with Git.

## Mutation And Trust Boundaries

Strong source mutation/evaluation isolation is supported only on rootful Linux where
trusted `bwrap`, `setpriv`, and `prlimit` are available and namespace probing succeeds.
Docker Compose must provide the required `SYS_ADMIN` and `NET_ADMIN` capabilities and
the configured relaxed seccomp/AppArmor settings. Unsupported namespace environments,
non-root Linux, macOS, and Railway fail closed for source mutation while immutable
serving continues.

The served runtime and candidate are distinct identities and trust domains. The served
runtime defaults to UID/GID `65532`; candidate processes default to UID/GID `65533`.
The evaluator has its own bounded process identity and no controller authority. Namespace,
mount, network, PID, CPU, memory, file-size, and capability limits are applied before
candidate commands run. Candidates cannot access controller credentials, product state,
the resolver engine/socket, or the host environment. Hash verification detects tampering;
the UID and capability separation is the strong mutation boundary.

When the boundary is unavailable, the controller does not silently use a weaker source
mutation path. Serving an already-built immutable generation remains allowed.

## Serving Lifecycle

Promotion is deliberately a stop/start-fenced cutover:

```text
prepare -> staging -> strict readiness -> drain -> stop old -> start new
        -> strict readiness -> live probation -> active
```

There is an availability gap between stopping the old child and starting the new child.
This is not standby and not zero-downtime. The active child does not survive normal
promotion. During probation the candidate serves live traffic with production credentials.

If a pre-cutover check fails, the current release remains active. If activation or live
probation fails, the controller stops the candidate and restores the exact previous
generation and HostConfig. It also restores release-coupled capability state when a
snapshot exists. Product-state mutations and external side effects are not rewound.
If exact restoration is impossible, safe mode stops forwarding rather than serving an
unverified child.

The controller persists activation phases and lease fences. On controller death or
restart, it reconciles those records, retains or restores the previous release as
appropriate, and never assumes that a partially activated child is safe merely because a
process still exists.

## Persistent State

Product databases, checkpoints, memories, skills, files, capabilities, secrets, tenant
workspaces, Git refs, generation metadata, activation records, and rollback state are
external to immutable runtime bytes. Release rollback can restore release-coupled state
that was explicitly snapshotted, but it cannot undo already committed product effects or
external provider effects.

The installer lock is a private authoritative `mkdir` directory. It has no PID metadata,
is never automatically reclaimed, and may be removed by an operator only after verifying
that no installer or descendant remains. This protects generation publication from
concurrent installers without pretending that a stale PID can establish ownership.

## Interfaces And Workers

The stable host authenticates owner/interface requests and proxies the universal Agent API
to the active runtime. Capability workers receive scoped, revision-bound credentials and
are lease-fenced to the active release. They do not receive the owner bearer token,
controller credentials, or generic OCI authority.

Local automatic host restart is separately guarded by pidfd support. There is no numeric
PID fallback, because a reused PID is not a safe process identity.
