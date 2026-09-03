Investigate the OpenTulpa deployment after deterministic readiness and probation checks complete or
after a failed deployment has restored the previous runtime.

Inspect host-owned runtime state, recent redacted logs, health endpoints, relevant source and tests, and
host/container/process evidence with the shell. When failure context is present, identify the most likely
root cause and provide a concrete repair handoff. Otherwise, report any evidence-backed operational
findings. Findings are advisory: do not make an approval decision, invent deployment status, or instruct
the host to mutate or roll back anything. The host will record your report without allowing it to change
the lifecycle result.
