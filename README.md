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
Railway serves immutable installed releases but does not self-mutate their source.
Strong source mutation and evaluation isolation is supported only on rootful Linux with
working `bwrap` namespaces and the required Docker Compose capabilities. Unsupported
namespace environments, non-root Linux, macOS, and Railway fail closed for source
mutation while immutable serving continues.

Docker Compose requires the documented rootful capabilities and relaxed confinement for
the namespace sandbox. Treat that as a hardened-production consideration, not as a
general-purpose container hardening profile. See [Deployment](docs/DEPLOYMENT.md).
The image starts the immutable host controller directly; managed OCI candidates use its
trusted interpreter and a reviewed source overlay rather than a mutable `/app/.venv`.

## How self-evolution works

1. The stable controller opens a detached Git candidate worktree.
2. Native Git refs retain instance and upstream lineage; native merge state records and
   exposes conflicts instead of inventing a parallel conflict format.
3. On supported rootful Linux, the candidate and evaluator run with separate UIDs and
   capability boundaries. The candidate workspace is never the serving source tree.
4. Optional dependency proposals go through a credential-free, content-addressed OCI
   resolver; fixed evaluation and generation building use its exact lock and wheelhouse.
5. Promotion is a stop/start-fenced cutover with an availability gap, strict readiness,
   and live probation. It is not standby or zero-downtime promotion.
6. A failure restores the exact previous generation and its recorded HostConfig. Product
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
