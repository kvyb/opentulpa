# Declarative Capability Manifests

Each `<capability>.json` file describes one reviewed source-bundled capability. The
fixed loader never imports these files as Python. A manifest may only launch a worker
module below `opentulpa.capability_workers` and may use dependencies already present in
the reviewed runtime base.

Source evolution may add or revise manifests and worker modules. The fixed evaluator
tests the exact candidate commit, and the stable bootstrap binds owner approval to
that commit before the overlay becomes a trusted release.
