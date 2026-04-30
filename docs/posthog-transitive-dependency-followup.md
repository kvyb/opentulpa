# PostHog Transitive Dependency Follow-Up

OPE-26 removes OpenTulpa-owned PostHog usage: no direct dependency, settings,
runtime wiring, docs, or tests should be added back.

The lockfile can still contain `posthog` while `browser-use` and `mem0ai`
declare it as package metadata. Current evidence on April 30, 2026:

```text
uv tree --package posthog --invert

posthog v7.7.0
├── browser-use v0.12.2
│   └── opentulpa v0.1.0
└── mem0ai v1.0.11
    └── opentulpa v0.1.0
```

Latest published packages checked during OPE-28 still declare PostHog:

```text
browser-use 0.12.6 requires posthog==7.7.0
mem0ai 2.0.1 requires posthog>=4.5.0
```

Code search of installed `browser_use`, `browser_use_sdk`, and `mem0` package
sources did not show direct PostHog/telemetry imports in their package code at
the checked versions. The remaining risk is therefore dependency installation,
not confirmed runtime calls from OpenTulpa code paths.

Keep this follow-up narrow:

- Do not reintroduce direct OpenTulpa PostHog imports, settings, or events.
- Prefer upstream package upgrades only when they remove the PostHog dependency
  without changing OpenTulpa runtime behavior.
- If either dependency starts emitting telemetry at runtime, disable it explicitly
  or replace the dependency path.
