# End-To-End Testing

Use fake external providers and disposable data roots. Never rehearse source activation with
production credentials or irreversible sinks.

## Automated Gates

```bash
uv sync --extra dev
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync mypy src
git diff --check
```

The default pytest configuration excludes tests marked `slow`. Run them explicitly with:

```bash
uv run --no-sync pytest -q -o addopts='' -m slow
```

The normal suite covers the persistent trusted worktree, exact Git object import,
credential-file rejection, activation-journal replay, runtime dependency identity,
activation, explicit rollback, failed-activation restoration, owner-only tools, and the
authenticated host API.

## Rootful Rehearsal

Build the production image and run its black-box source rehearsal:

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

The script starts a real stable host against a fake model endpoint, then verifies:

1. The initial child is healthy and bound to the journal's active commit.
2. A direct source write is committed and activated through the authenticated host API.
3. The serving process ownership record names the exact activated commit.
4. The runtime uses UID/GID `65532`, no effective capabilities, `no_new_privs`, and its
   recorded process group.
5. Explicit rollback restores the initial release and process commit.
6. A full host restart preserves the rollback decision and trusted worktree history.

## Manual Failure Injection

For a release intended for deployment, rehearse these boundaries on a copied data root:

- syntax or import failure before child replacement;
- dependency-lock mismatch or failed `uv sync --frozen`;
- child exit before strict readiness;
- health failure during live probation;
- host termination while an activation row is `preparing`;
- repeated activation and rollback requests with the same idempotency key;
- changed request data with a reused idempotency key;
- missing active or previous Git objects;
- dirty live checkout or credential files in the trusted worktree; and
- failed exact rollback, which must remain unavailable rather than advance the journal.

## Expected Recovery

| Failure point | Expected result |
| --- | --- |
| Before journal insert | Worktree commit remains available; active release is unchanged. |
| Dependency or fixed check | Activation becomes failed; current child keeps serving. |
| Startup or probation | Runtime restores the exact previous spec; activation records `rolled_back`. |
| Host death during activation | Restart serves journal active, then resumes the `preparing` operation. |
| Explicit rollback | Previous healthy release becomes active; prior active becomes the new previous. |
| Idempotent replay | Exact terminal result is returned without another runtime replacement. |
| Rollback cannot be proven | Host does not mark the target active. |

Product database mutations and external effects are not reversed by source rollback. Test
backward-compatible migrations separately with realistic copied databases and fake provider
sinks.
