# End-To-End Testing

Tests exercise the real Deep Agents service, stable controller, Git lineage, immutable
generation store, sandbox/evaluator boundary, and persistent product services. Do not
use production credentials or irreversible external sinks in rehearsal.

## Automated Gates

The default pytest configuration excludes slow tests:

```bash
uv sync --extra dev
uv run --no-sync pytest -q
```

Run the excluded slow tests explicitly with:

```bash
uv run --no-sync pytest -q -o addopts='' -m slow
```

Other checks:

```bash
uv run --no-sync ruff check .
uv run --no-sync mypy src
git diff --check
```

Build the dependency resolver and run its real package-index integration separately:

```bash
docker build -f docker/dependency-resolver.Dockerfile \
  -t opentulpa-dependency-resolver-e2e .
resolver_id=$(docker image inspect opentulpa-dependency-resolver-e2e --format '{{.Id}}')
OPENTULPA_TEST_DEPENDENCY_RESOLVER_IMAGE="$resolver_id" \
  uv run --no-sync pytest -q -o addopts='' \
  tests/test_dependency_resolver.py::test_real_oci_resolver_builds_offline_dependency_base
```

To prove the same resolved dependency is evaluated, built into a generation, activated by
the rootful runtime, and removed by rollback, use the E2E-only image that adds a Docker CLI.
The controller and sibling resolver exchange sealed artifacts through a dedicated named
volume. This command assumes a local Unix engine:

```bash
docker build -t opentulpa-rootful-e2e .
docker build -f docker/rootful-dependency-e2e.Dockerfile \
  -t opentulpa-rootful-dependency-e2e .
docker_socket=$(docker context inspect --format '{{(index .Endpoints "docker").Host}}')
docker_socket=${docker_socket#unix://}
resolver_volume=opentulpa-dependency-e2e
docker volume create "$resolver_volume"
docker run --rm \
  --cap-add SYS_ADMIN \
  --cap-add NET_ADMIN \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --mount "type=bind,src=$docker_socket,dst=/var/run/docker.sock" \
  --mount \
    "type=volume,src=$resolver_volume,dst=/var/lib/opentulpa-dependency-resolver" \
  --env OPENTULPA_CONTAINER_CLI=/usr/local/bin/docker \
  --env OPENTULPA_DEPENDENCY_RESOLVER_IMAGE_DIGEST="$resolver_id" \
  --env OPENTULPA_DEPENDENCY_RESOLVER_VOLUME="$resolver_volume" \
  opentulpa-rootful-dependency-e2e \
  /opt/opentulpa-install/controller/generations/image/bin/python \
  /opt/opentulpa-source/tests/rootful_self_evolution.py
```

The socket is exposed only to this dedicated controller rehearsal image. It is not present
in the production image and is not mounted into candidate or served-runtime namespaces.

Run the real rootful-Linux self-edit, cutover, failed-activation restoration, explicit
rollback, conversation-notification, and restart rehearsal against the built image:

```bash
docker build -t opentulpa-rootful-e2e .
docker run --rm \
  --cap-add SYS_ADMIN \
  --cap-add NET_ADMIN \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  opentulpa-rootful-e2e \
  /opt/opentulpa-install/controller/generations/image/bin/python \
  /opt/opentulpa-source/tests/rootful_self_evolution.py
```

The suite covers protocol/authentication, product ownership and idempotency, checkpoints,
approvals, interfaces, capability workers, tenant workspaces, Git candidate worktrees,
native lineage/conflicts, wheel/venv generation integrity, staging, cutover, probation,
rollback, and restart recovery.

## Isolation Preconditions

Source mutation/evaluation tests must distinguish supported and unsupported environments.
The strong path requires rootful Linux, trusted `bwrap`/`setpriv`/`prlimit`, permitted
namespaces, and the required Docker Compose capabilities. Non-root Linux, macOS, blocked
namespaces, and Railway must report source mutation unavailable while immutable serving
tests continue to pass. Never “fix” those tests by weakening the boundary.

Verify that candidate and served runtime identities are distinct, that candidate commands
cannot read controller credentials or product state, and that no candidate workspace is
treated as serving source. Verify the exact source commit produces the exact wheel and
final-path venv generation.

## Managed Release Rehearsal

Use a copied persistent data root and fake external sinks.

1. Install a reviewed controller generation and verify `current` and `previous` pointers.
2. Initialize Git instance/upstream refs and verify native merge-base and conflict behavior.
3. Start an immutable release and verify strict readiness before serving traffic.
4. Create a detached candidate, make a harmless change, commit it, and verify fixed
   evaluation and generation hashes are bound to that commit.
5. Verify the candidate worktree is not the serving source and is not reused for runtime.
6. Promote the generation and observe drain, old-process stop, candidate start, strict
   readiness, live probation, and the intentional availability gap.
7. Verify the active child does not survive normal promotion and the controller remains
   available throughout recovery operations.
8. Verify product state persists across promotion and that current/previous generation
   metadata identifies exact rollback targets.
9. Inject a probation failure and verify exact previous generation plus HostConfig rollback.
10. Verify no product database mutation or external side effect is removed by rollback.

## Crash Matrix

Inject controller death at each point and restart from copied persistent state:

| Crash point | Required result |
|---|---|
| Staging | Staging child is discarded; current release remains serving; activation is terminally failed or safely reconciled. |
| Activation/cutover | A partially started candidate is not trusted; previous release is restored with its exact HostConfig or safe mode is entered. |
| Live probation | Candidate is stopped; exact previous generation and HostConfig are restored; product/external effects remain. |
| Rollback | Recovery resumes only from durable activation state; if the previous release cannot be verified, safe mode stops forwarding. |
| Source projection | Native Git refs and merge state are re-read; stale compare-and-swap or unresolved native conflicts fail closed without corrupting lineage. |
| Lock recovery | The private `mkdir` install lock remains; no PID metadata or stale-lock automation is consulted. Remove it only after verifying no installer or descendant survives. |

Also test missing previous artifacts, duplicate activation/rollback requests, dependency
lock changes, secret/symlink/traversal/special-file candidates, invalid manifests, and
tampered generation trees.

## Rollback Assertions

- Pre-cutover failure leaves the old release active.
- Post-cutover failure stops the candidate before restoring the exact previous release.
- Readiness is strict both before activation and during probation.
- Rollback restores release-coupled capability state only when a valid snapshot exists.
- Product state, messages, purchases, authorization changes, and provider writes are not
  rewound.
- Failed restoration enters safe mode and does not forward to an unverified process.
- Activation records, sanitized errors, lineage, and owner notifications survive restart.

## Local Restart And Platform Matrix

On a platform with pidfd support, verify automatic local restart signals the exact
remembered process through pidfd and rejects identity changes. On a platform without
pidfd, verify automatic restart fails closed and offers no numeric-PID fallback.

Verify immutable serving on macOS, non-root Linux, unsupported namespace environments,
and Railway. Verify source mutation is disabled on each of them rather than silently
falling back to a weaker isolation mode. Verify Docker Compose source mutation only with
the documented rootful capabilities and security options.
