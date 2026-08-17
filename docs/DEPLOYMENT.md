# Deployment

OpenTulpa runs one stable host controller and one mutable application child. Persistent
product and evolution state lives outside both processes' installed code.

## Installed Host

`./install.sh` and the Docker image build an immutable controller wheel and virtualenv at
its final path. The launcher verifies the selected controller generation before execution.
Python virtualenvs are machine- and path-specific, so they are built on the target rather
than copied between hosts.

The installer lock is a private directory under the controller root. It is never reclaimed
automatically. Remove it only after proving no installer process or descendant remains.

The installed host requires:

- a Git source checkout selected by `OPENTULPA_SOURCE_ROOT`;
- persistent `OPENTULPA_DATA_ROOT` storage;
- an executable `uv`, optionally selected by `OPENTULPA_UV_BIN`; and
- one active controller replica.

If `OPENTULPA_SOURCE_ROOT` is absent, the host can clone
`EVOLUTION_SOURCE_REPOSITORY` at `OPENTULPA_INSTALL_REF` into that path.

## Trusted Source Evolution

Source evolution is enabled by default. The host creates an independent persistent Git
repository at:

```text
$OPENTULPA_DATA_ROOT/bootstrap/evolution/source
```

Owner source tools edit that repository directly. This is trusted controller-user
execution, not an untrusted-code sandbox. Normal Git commands manage upstream remotes and
merges. The live checkout is separate and changes only when the host imports and activates
an exact commit.

Activation state is stored at:

```text
$OPENTULPA_DATA_ROOT/bootstrap/evolution/activations.db
```

The journal retains active, previous, and last-known-good release IDs plus idempotent
activation results. Runtime dependency environments live under
`$OPENTULPA_DATA_ROOT/runtime-source-envs` and are reused while `pyproject.toml` and
`uv.lock` remain unchanged.

Production activation runs a bounded compile gate with the isolated host interpreter before
child startup exercises imports and the tool contract during real readiness and probation.
Run Ruff, mypy, and the relevant pytest suite with `source_bash` before activation or in CI.

## Local Install

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
./install.sh
opentulpa
```

Local automatic controller restart requires `pidfd_open` and `pidfd_send_signal`; there is
no numeric-PID fallback. If unavailable, stop the remembered host before starting a newly
installed controller generation.

## VPS With systemd

Run the installed dispatcher as a root-owned service and set `OPENTULPA_SYSTEMD_UNIT` to
that service's exact name. The dispatcher must remain stable; do not pin `ExecStart` to one
generation.

```ini
[Unit]
Description=OpenTulpa host
After=network-online.target

[Service]
Type=simple
User=root
Environment=OPENTULPA_SYSTEMD_UNIT=opentulpa.service
Environment=OPENTULPA_DATA_ROOT=/var/lib/opentulpa
Environment=PORT=8000
ExecStart=/root/.local/share/opentulpa/install/bin/opentulpa-host
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

After a reviewer-approved source activation, the stable host starts a transient root-owned
systemd unit. It runs the existing verified installer against the exact clean commit,
restarts `opentulpa.service`, and requires both the new generation's process and
`/agent/healthz` to become healthy. Failure atomically restores `controller/previous`,
restarts the old generation, and reports the result to the owner thread. No Docker socket,
SSH fallback, or unrestricted sudo command is exposed to the mutable child.

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

Persist the `opentulpa_data` volume. The included Compose path may enable strong process
isolation for product/repository sandboxes on rootful Linux, but source evolution itself is
trusted and does not require Docker-in-Docker or a Docker socket.

## Railway

Railway uses the repository Dockerfile and should attach one persistent volume at:

```env
OPENTULPA_DATA_ROOT=/app/opentulpa_data
```

The image already configures:

```env
OPENTULPA_SOURCE_ROOT=/app/opentulpa_data/source
EVOLUTION_SOURCE_REPOSITORY=https://github.com/kvyb/opentulpa.git
OPENTULPA_INSTALL_REF=main
OPENTULPA_UV_BIN=/usr/local/bin/uv
```

Use a single replica and `overlapSeconds = 0`; the host and child use local SQLite and one
shared persistent source checkout. Do not mount a Docker socket for core self-evolution.

## Activation And Rollback

```text
commit -> persist preparing -> prepare dependencies -> safe compile
       -> stop old child -> start new child -> readiness -> probation -> active
```

This path has a short availability gap and is not zero-downtime. A failed start or
probation restores the exact prior source and dependency environment. If restoration cannot
be proven, the host remains unavailable rather than claiming an unsafe release is active.

`source_rollback` selects the journal's previous healthy release. It does not reverse
database migrations or external side effects. Database changes must remain backward
compatible with the previous application commit. It intentionally keeps the newest healthy
stable controller; only the mutable child source rolls back. A controller that fails its own
systemd health validation is restored separately by the privileged updater.

On a configured systemd VPS, reviewer approval also updates the stable controller through
the root-owned handoff above. Docker and Railway still require a new outer deployment to
replace the stable host process.

## Backups

Back up the complete data root, including:

- product databases, memories, skills, and tenant workspaces;
- host configuration and secret-vault material;
- `bootstrap/evolution/source` and `activations.db`;
- the runtime `.env`; and
- runtime dependency environments still referenced by active or rollback releases.

Restore these together. Do not restore only a source checkout and assume product or
activation state can be reconstructed from it.
