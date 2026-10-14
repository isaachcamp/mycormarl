# P0 — Preserve report source and establish baseline

**Route:** standard, explicitly selected 13 October 2026.
**Status:** Verified preservation and baseline inventory; known baseline failures recorded.
**Parent:** [Repository split plan](repository-split-plan.md).

## Goal and scope

Preserve the existing Overleaf report source and reference PDF, identify its
figure inputs/provenance, preserve current working-tree changes separately,
and establish current executable baselines before extraction. No scientific
model changes, production training, pruning, branch cleanup, or squash merge.

## Dependencies

Read-only access to Overleaf project `6a58fe9fc046d5559b8a6537`, existing
report artifacts and study output directories, current Python environment,
and Git source history. The report's source is not assumed to match HEAD.

## Tasks

- [v] Export complete Overleaf source and current PDF into an external
  preservation directory; record checksums, project identity, and export time.
- [v] Inspect manuscript figure references and match them to existing scripts,
  manifests, result records, policy artifacts, and recorded revisions/seeds.
- [v] Capture current Git/source state and preserve existing uncommitted
  changes separately without committing or stashing them indiscriminately.
- [v] Identify active processes and existing data roots; leave them intact.
- [v] Run the relevant current tests and representative artifact checks;
  record baseline failures without confusing them with migration regressions.
- [v] Preserve known original source revisions with accessible refs after
  confirming report/run attribution; explicitly record unresolved provenance.
- [v] Complete a second verification pass and review the preservation set,
  checksums, source mapping, and missing evidence before declaring P0 complete.

## Verification and exit criteria

Verify source archive integrity and required manuscript files; confirm the PDF
is readable and its source/figure references match the captured project;
verify preservation hashes and compare protected tracked-file hashes; inspect
recorded report-to-run mappings and identify any essential missing input.
Use current tests and existing numerical artifacts rather than retraining.
P0 remains incomplete if report source or essential evidence cannot be
preserved. Original-source attribution gaps must be documented, not invented.

## Evidence

Preservation root:
`/Users/campbelli/Documents/MARL-community-assembly/code/mycormarl-data/preservation/report-2026-10-13`.
Contains the complete Overleaf source ZIP and its 39 extracted files, reference
PDF, complete Git history bundle, tracked working-tree patch, untracked source
documents, selected compact numerical inputs, and SHA256 inventory.

[Report asset/provenance map](repository-split-report-map.json) records all
24 image assets: 20 scientific images exactly match local output bytes; four
static assets are preserved directly from Overleaf. Thirteen compact input
files are additionally copied. All detailed outputs and policy/checkpoint
records remain in their original locations.

`git bundle verify` confirms a complete, valid history bundle, including the
original learned-policy run commit `95d36e1c25fd61f413ccf1ca5f17595595c35a3f`
and report static-control commit `35b8a4520718638e3b9742c80985df6a8a36295b`.
No tag falsely attributing the report to current HEAD was created.

Second verification checked all 62 preservation-file hashes, the ZIP CRC
integrity, PDF signature, and hashes of the three protected pre-existing
modified files (`study.py`, `test_study_runner.py`, runner guide): all passed.
The phase review identified and explicitly retained the gaps below.

## Baseline tests and known failures

`uv run pytest -q` failed collection because ignored generated file
`media/phosphate-uptake/test_continuous_grid_historical.py` references a missing
`media/scripts/phosphate_uptake_video_data.py`. No historical file was deleted.

`uv run pytest -q tests` completed in 226.64 seconds: **500 passed, 9 failed,
1 skipped**. Failures present before extraction:

- Five `test_checkpoint_evaluation.py` cases and the trade-only evaluator case
  encounter missing `reproductive_fitness_from_components` on direct base-env
  evaluation. The field is currently added by the policy-interval wrapper.
- `test_policy_artifacts.py::test_training_cli_selects_consumer_mode_and_writes_versioned_policy`.
- `test_review_repairs.py::test_ppo_stack_accepts_current_agent_identifiers`.
- `test_study_runner.py::test_phase_1_fixture_resumes_only_the_missing_condition`;
  this lies in an already modified module and must be investigated without
  overwriting the user's pending changes.

P0 verified baseline capture, not a passing product suite. P1 must diagnose
the evaluator/CLI/training prerequisites and distinguish committed failures
from worktree-specific failures before claiming its vertical slice works.

## Explicit provenance and reproduction gaps

- Overleaf's original Git revision is unavailable from the export; source and
  PDF are identified exactly by their checksums. The live project is untouched.
- Static-screen records contain scientific configuration and seeds but not
  original Git commit identity. Do not invent missing provenance.
- The related depletion comparison script generates a different figure name;
  the exact panels-composition script for the preserved supplementary PNG is
  not identified. Matching CSV evidence and reference pixels are retained.
- A candidate plotting script is not proof of the original rendering revision.
  Some source assets/sections are unused by the main report; retain them too.
- Preserve raw evidence and historical figures while resolving these gaps
  during extraction; do not claim complete historical regeneration yet.

These gaps do not prevent additive preparation of the installed-core slice.
They remain gates for faithful report reproduction and final cutover.
