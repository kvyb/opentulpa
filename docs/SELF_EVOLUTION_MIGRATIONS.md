# Self-Evolution Migration Notes

## Legacy deterministic release lineage

Deployments that can still boot the release-lineage implementation removed by `c8fab45` need a
bootstrap-seeding regression fix before automatic updates are resumed. A persisted release can
retain the same deterministic artifact ID while evaluator provenance changes; seeding must reuse
that release or assign a distinct ID, never create a release whose predecessor is itself.

The current trusted-source journal keys releases only by exact Git commit and does not include
evaluator fingerprints in release identity. Keep that invariant. A prior deployment rollback can
otherwise be overwritten by its updater until the legacy fix is deployed, so updater holds must
remain in place for affected records during migration.
