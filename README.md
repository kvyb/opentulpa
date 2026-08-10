<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./assets/readme/hero-mobile.svg">
    <img src="./assets/readme/hero.svg" width="100%" alt="OpenTulpa, the self-evolving agent">
  </picture>
</p>

OpenTulpa is a self-hosted Deep Agents application that can edit, activate, and roll back
its own source without replacing the stable host controller.

## Get started

OpenTulpa needs Git, curl, and a model API key. The installer builds an immutable,
content-addressed controller wheel and virtualenv generation, then points the stable
launcher at it.

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
./install.sh
opentulpa
```

Choose **Run here** and enter the model API key. Persistent product state is kept outside
the installed generation. Python virtual environments are not portable, so each venv is
built at its final path for the target machine; see the
[Python venv documentation](https://docs.python.org/3/library/venv.html).

## Deployment

OpenTulpa runs locally, in Docker Compose, or on Railway. The Docker image and
`./install.sh` package an immutable host controller, a Git source seed, and `uv`. The host
keeps one independent source worktree and its activation journal on persistent storage.

Source editing is intentionally trusted and intended for personal or single-tenant
deployments. Direct source commands run as the controller user; they are not an adversarial
code sandbox. Product and repository sandboxes remain separate boundaries.

Optional capability bundles such as browser automation, integrations, document tooling,
research tooling, and hosted sandboxes are separate deployment choices.
See [Deployment](docs/DEPLOYMENT.md).
The image starts the immutable host controller directly. Mutable application children run
from exact Git commits selected by that controller.

## How self-evolution works

1. `source_read`, `source_write`, `source_edit`, and `source_bash` operate on one persistent,
   independent Git repository. Normal Git commands handle remotes, branches, and merges.
2. `source_activate` commits the current worktree and records an idempotent activation in
   SQLite before any runtime change.
3. The host prepares a dependency environment keyed by `pyproject.toml`, `uv.lock`, Python,
   and install profile without running dependency build scripts, then safely compiles the source.
4. The runtime supervisor selects the exact commit; child startup exercises imports and the tool
   contract before strict readiness and live probation record it as active.
5. A failed activation restores the exact prior commit. `source_rollback` activates the
   journal's previous healthy release through the same runtime path.

Cutover is stop/start fenced and has a short availability gap; it is not standby or
zero-downtime. Product database mutations and external side effects are not rewound.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Tool contract](docs/tool-contract.md)
- [E2E testing](docs/E2E_TESTING.md)

MIT licensed.
