# Deployment

OpenTulpa separates a stable launcher/controller from immutable runtime generations and
external persistent state. The controller owns installation, source evolution, release
activation, readiness, probation, rollback, and recovery. Product databases, memories,
skills, tenant workspaces, Git lineage, and controller metadata live outside a runtime
generation and must be backed up separately.

## Runtime Model

The installer:

- verifies a Git source commit and archives that exact commit;
- builds the controller wheel and dependency wheelhouse;
- creates the Python virtualenv at the final generation path, not in a temporary path;
- records hashes and provenance in the generation manifest; and
- atomically updates `controller/current`, retaining the prior target as `controller/previous`.

The launcher validates the selected generation before execution. A venv is inherently
machine- and path-specific, not a portable artifact. OpenTulpa therefore builds each
venv at its final path, as described by the
[Python documentation](https://docs.python.org/3/library/venv.html), rather than copying
one between hosts.

The installer lock is a private authoritative directory created with `mkdir` under the
controller root. It has no PID metadata and is never automatically reclaimed. If an
install is interrupted, an operator may remove the lock only after verifying that no
installer process or installer descendants are still running. There is no stale-lock
automation.

## Supported Source Mutation

Source mutation is enabled by default on the installed-host path. The stable controller
opens a disposable full Git worktree, runs trusted-local source commands inside that
worktree, runs fixed supervisor-owned evaluation gates, builds a sealed Python
generation, and owns activation/rollback. The candidate worktree is never the serving
source tree and cannot directly publish a release.

This default is a trusted single-tenant mode, not an adversarial sandbox. Candidate shell
commands run as the controller user inside the deployment container/host. Use it for
personal, owner-controlled deployments. For untrusted tenants or adversarial code, disable
source evolution or use the optional hardened backend on infrastructure that supports it.
The private product sandbox worker follows the same default: it uses strong process
isolation when available and otherwise falls back to trusted-local command execution so
single-tenant Railway/VPC deployments can boot without Docker-in-Docker.

The installed generation must include these trusted inputs:

- `OPENTULPA_SOURCE_SEED_ROOT`: full source seed used to seed persistent Git lineage;
- `OPENTULPA_TRUSTED_WHEELHOUSE`: offline wheelhouse used for generation builds;
- `OPENTULPA_UV_BIN`: trusted `uv` executable used by the release builder;
- `OPENTULPA_DATA_ROOT`: persistent external state root for product data, Git lineage,
  release records, and runtime generations.

The Docker image and `./install.sh` both produce this installed layout. `Settings` keeps
`evolution_enabled=True` by default; source mutation is disabled only when explicitly
configured off or when the required installed inputs are missing.

The default package includes what core source evolution needs. Optional bundles such as
browser automation, integrations, document parsing, research tooling, hosted sandboxes,
or the OCI dependency resolver must be selected and configured separately.

Rootful Linux process isolation remains available as an optional hardened backend when
all of these are true:

- the supervisor is rootful Linux;
- trusted, root-owned `bwrap`, `setpriv`, and `prlimit` executables are available;
- bubblewrap mount, PID, IPC, UTS, and network namespaces pass the startup probe; and
- the container deployment supplies the required Docker Compose capabilities.

That optional backend uses distinct runtime identities and capability boundaries. The
served runtime uses UID/GID `65532`; hardened source candidate processes use a distinct
candidate identity (default UID/GID `65533`). The default trusted-local path does not
require these namespace capabilities.

## Dependency Resolver

Dependency resolution is disabled unless `OPENTULPA_DEPENDENCY_RESOLVER_IMAGE_DIGEST`
names an exact local OCI image ID. Build and inspect the dedicated fixed-command image:

```bash
docker build -f docker/dependency-resolver.Dockerfile \
  -t opentulpa-dependency-resolver .
docker image inspect opentulpa-dependency-resolver --format '{{.Id}}'
```

Set the resulting `sha256:...` value and, if needed, set
`OPENTULPA_CONTAINER_CLI` to an exact Docker or Podman executable. The controller verifies
the image ID and rejects credential-bearing index environment variables before each
resolution. The worker receives only a copied `pyproject.toml` and a private output
directory; it receives no product state, controller credentials, source Git metadata, or
arbitrary candidate command.

The selected engine must be reachable by the stable controller. Do not expose its socket
to the served runtime or candidate sandbox. A Docker socket is host-level authority, so
prefer a dedicated rootless engine or isolated resolver host and restrict which controller
account can access it. Railway does not provide this local resolver path.

When the stable controller itself runs in a container, create a dedicated named volume,
mount it at `/var/lib/opentulpa-dependency-resolver`, and set
`OPENTULPA_DEPENDENCY_RESOLVER_VOLUME` to its exact engine-level name. The controller uses
the mounted filesystem while each sibling worker receives only its current resolution
subpath through an OCI volume mount. Do not reuse the product-data or controller-state
volume for this purpose. Native controllers leave this variable unset and use the default
private bind-mount staging directory.

## Local Install

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
./install.sh
opentulpa
```

Local automatic restart is deliberately strict: it requires both `pidfd_open` and
`pidfd_send_signal`. There is no numeric-PID fallback. If pidfd support is unavailable,
stop the remembered host manually before starting the new controller generation.

## Docker Compose

```bash
cp .env.example .env
```

The image starts the installed immutable host controller directly and includes the source
seed, wheelhouse, and `uv` toolchain needed by default source evolution.

The included Compose service may run the stable host with `SYS_ADMIN` and `NET_ADMIN`,
`seccomp=unconfined`, and `apparmor=unconfined` for the optional rootful Linux namespace
sandbox. Those capabilities are not required for the default trusted-local source
evolution path.

If you enable the optional rootful sandbox, this is a significant hardened-production
implication: the service is not equivalent to a least-privilege container. Operators
should isolate the host, restrict access to the Docker daemon, constrain network egress,
and review the Compose security options.

Persist the `opentulpa_data` volume and set `OPENTULPA_DATA_ROOT=/app/opentulpa_data`.
It contains product state, Git lineage, release records, and the controller's external
persistent state. Do not treat a runtime generation or candidate worktree as a backup.

The root image starts the immutable `opentulpa-host` controller directly. Host releases
execute only from sealed Python generations. Managed OCI releases are complete immutable
images selected by digest; neither path executes a source directory, mutable `/app/.venv`,
or candidate-controlled `start.sh`.

## Railway

Railway should use the Dockerfile/installed-host path. That path supports default
trusted-local source evolution without a Docker socket or rootful namespace privileges.

Required Railway configuration:

```env
OPENTULPA_DATA_ROOT=/app/opentulpa_data
```

Attach a persistent volume at that path and run a single replica for the controller.
Configure unattended owner/model credentials as required by the deployment. Do not mount
the Docker socket into the service; it is not needed for the default self-evolution loop.

The Dockerfile already sets the installed-input defaults:

```env
OPENTULPA_INSTALL_ROOT=/opt/opentulpa-install
OPENTULPA_SOURCE_SEED_ROOT=/opt/opentulpa-source
OPENTULPA_TRUSTED_WHEELHOUSE=/opt/opentulpa-install/controller/generations/image/wheelhouse
OPENTULPA_UV_BIN=/usr/local/bin/uv
```

Railway does not support the optional local OCI dependency resolver path unless you add a
separate resolver service. Ordinary source edits, fixed evaluation, generation builds,
promotion, and rollback do not require that resolver.

## Private VPC Or Single-Tenant Host

For a personal VPC/VPS-style deployment, use the same installed-host model:

- build or pull the OpenTulpa Docker image;
- persist `OPENTULPA_DATA_ROOT` on durable storage;
- run one controller replica;
- configure the owner/model/Telegram credentials;
- expose only the public app endpoint and keep controller data private;
- rely on default trusted-local evolution only for trusted owner-controlled code;
- add the optional rootful namespace backend or dependency resolver only when you need
  those stronger or broader capabilities.

## Release Cutover And Rollback

The release controller follows this sequence:

```text
prepare -> isolated staging -> strict readiness -> drain -> stop old
        -> start candidate -> strict readiness -> live probation -> active
```

Cutover is stop/start fenced and has an availability gap. It is explicitly not standby
and not zero-downtime. The old active child does not survive normal promotion. If staging
fails, the current release remains active. After cutover, a readiness or probation
failure stops the candidate and restores the exact previous release generation and its
recorded HostConfig. If that restoration fails, the controller enters safe mode rather
than serving an unverified process.

Rollback restores release-coupled capability state where the implementation has a saved
snapshot. It does not rewind product-state mutations or external side effects such as
messages, purchases, authorization changes, or provider writes. Use idempotency and fake
sinks when rehearsing releases near external effects.

## Git Lineage

Source sessions use detached Git worktrees. Instance, upstream, and accepted-upstream
lineage are native Git refs, and upstream integration uses Git's native merge index and
conflict stages. A candidate worktree is an editing/evaluation workspace, not a serving
source directory and not a reusable source projection. Promotion exports the exact
committed source bytes into a new wheel/venv generation.

## Backups And Recovery

Back up together:

- the persistent product data root;
- controller/bootstrap state and release records;
- the canonical Git repository, including private `refs/opentulpa/*` refs; and
- every generation artifact still referenced by `current`, `previous`, or rollback metadata.

Recovery is controller-owned. Verify the installer lock and all installer descendants
before manually removing an abandoned lock. Never infer lock ownership from a PID file:
the installer intentionally creates no PID lock metadata.
