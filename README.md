<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./assets/readme/hero-mobile.svg">
    <img src="./assets/readme/hero.svg" width="100%" alt="OpenTulpa, the self-evolving agent">
  </picture>
</p>

OpenTulpa is a self-hosted Deep Agents application that can evaluate and publish changes
to its own source without replacing the stable host controller.

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

OpenTulpa can serve immutable releases locally, in Docker Compose, or on Railway.
The default production path is the installed host controller: the Docker image and
`./install.sh` both package the controller generation, source seed, trusted evaluation
wheelhouse, and `uv` toolchain that core self-evolution needs. Source evolution is
enabled by default and uses a trusted-local candidate worktree on hosts such as Railway
where rootful namespace isolation is unavailable.

Trusted-local evolution is intended for single-tenant/personal deployments. It is not a
security sandbox for adversarial tenants: candidate commands run as the controller user in
a disposable worktree, while the stable controller keeps release and rollback authority.

Rootful Linux `bwrap`/process isolation and OCI-backed dependency resolution remain
optional hardened backends. They are not required for the default self-evolution loop.
Optional capability bundles such as browser automation, integrations, document tooling,
research tooling, and hosted sandboxes are separate deployment choices.
See [Deployment](docs/DEPLOYMENT.md).
The image starts the immutable host controller directly. Host releases execute only from
sealed Python generations; managed OCI releases execute only from immutable image digests.

## How self-evolution works

1. The stable controller opens a detached Git candidate worktree.
2. Native Git refs retain instance and upstream lineage; native merge state records and
   exposes conflicts instead of inventing a parallel conflict format.
3. By default, the candidate uses a trusted-local full Git worktree and normal local
   shell while the stable controller owns evaluation, release building, activation, and
   rollback. The candidate workspace is never the serving source tree.
4. On supported rootful Linux, optional hardened candidate/evaluator isolation can run
   with separate UIDs and namespace boundaries.
5. Optional dependency proposals go through a credential-free, content-addressed OCI
   resolver; fixed evaluation and generation building use its exact lock and wheelhouse.
6. Promotion is a stop/start-fenced cutover with an availability gap, strict readiness,
   and live probation. It is not standby or zero-downtime promotion.
7. A failure restores the exact previous generation and its recorded HostConfig. Product
   database mutations and external side effects are not rewound.

The active child cannot survive normal promotion: it is drained and stopped before the
new child starts. The stable host/controller and its `current` and `previous` generation
references remain in place throughout.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Tool contract](docs/tool-contract.md)
- [E2E testing](docs/E2E_TESTING.md)

MIT licensed.
