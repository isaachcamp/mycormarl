# Core source and scientific evidence versions

Core owns biological/algorithm contracts. Downstream research commits its exact
core Git revision and independent uv.lock. Source package version0.1.0 alone is
not a scientific identity. Changes to biological defaults, numerical kernels,
actor/state interfaces or RNG require the relevant contract/scientific checks
before a downstream pin upgrade.

Use development branches (default codex/), and fixed tags for report/submission
snapshots. Keep original report source refs/archives and record original-to-squash-
to-extraction mappings before branch cleanup. Update downstream pins only after
the final core source is accessible and independent-package checks pass. Earlier
results retain their original core/runtime identities and are never relabelled.

Detailed exploration data and selected publication archives have separate
retention policies. Policy network exports require configuration, actor context,
interfaces, original runtime, training/evaluation/design seeds and stream derivation.
Publications initially use private GitHub release assets, with a later Zenodo
deposit. Exporting never authorizes pruning. Core ADR0014/0015 remain authoritative;
retain original IDs and supersession status in downstream indexes.
