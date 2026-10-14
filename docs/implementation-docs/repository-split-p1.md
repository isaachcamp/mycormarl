# P1 — Installed-core-to-publication-figure walking skeleton

**Route:** standard. **Status:** Verified; paused before P2.
**Parent:** [Migration plan](repository-split-plan.md).
**Dependency:** P0 preservation verified; baseline failures recorded.

## Scope

Prove one independent research figure workflow using an immutable installed
core revision. Prepare code in the managed `repository-split` worktree based
on preserved HEAD, leaving existing original worktree edits untouched.
Repair only demonstrated diagnostic/runtime prerequisites, without altering
model dynamics or named scientific protocols. Retain and label baseline
failures outside the selected slice until their owning phase addresses them.

## Task checklist

- [v] Diagnose base-environment evaluator diagnostic and CLI/fixture failures;
  distinguish missing diagnostics from intentionally rejected PPO protocols.
- [v] Verify the reward-decomposition diagnostic for base and held-interval
  evaluation; preserve physical dynamics/reward numerical values.
- [v] Declare core's actual runtime/build requirements and install noneditable
  core into an isolated environment from an immutable staging commit.
- [v] Create REDbyRBD scaffold and exact uv core pin with extraction provenance.
- [v] Evaluate a tiny explicitly configured policy and regenerate a figure.
- [v] Build a minimal manuscript page through a tested, pinned showyourwork
  tool environment that invokes uv-locked research Python.
- [v] Verify imported core path/revision, output provenance, baseline numeric
  agreement, repeatability, and absence of sibling-source import leakage.
- [v] Review/fix the slice and repeat verification before marking P1 complete.

## Checks and exit

Focused diagnostic, artifact, PPO, and policy-interval tests; package install
and import checks; repeated tiny deterministic evaluation; headless figure
generation; showyourwork manuscript build with live Overleaf sync disabled.
P1 exit checks passed on 13 October 2026. Pause before starting P2.

## Completion evidence — 2026-10-13

- Core staging commit: `72e5336c6f3f8e1121554b6a1c816281750f8103`;
  REDbyRBD source commit: `4558c808afd6966691913e4fdb0cba27a90e79dc`.
- Clean noneditable core installation is imported from REDbyRBD's
  `.venv/lib/python3.12/site-packages/mycormarl`, with Git revision read from
  distribution `direct_url.json`. JAX/JAXlib 0.9.0 and JaxMARL 0.0.2 are
  independently locked in the research environment. No PYTHONPATH is needed.
- Full staged core suite: **513 passed, 1 skipped**. Second-pass focused
  reward/autoreset checks: **5 passed**. Original baseline failures and the
  untouched study-runner WIP remain recorded in P0.
- Independently initialized policy weights are byte-identical. Complete
  trajectories and summaries agree exactly across repeat evaluations and
  against the staged source-worktree evaluator. Equality is established on
  this runtime; no cross-platform bitwise guarantee is implied.
- The initial and forced showyourwork builds both completed successfully.
  Final command: `showyourwork build --no-conda --cores 1 --forceall`, with
  both showyourwork and Tectonic tool `bin` directories on PATH. Final build
  regenerates weights, evaluation JSON, the figure, and one-page manuscript.
- `tools/verify_walking_skeleton.py --reference <independent-evaluation.json>`
  passed: exact pin, installed path, clean research source revision, hashes
  of script/manifest/lock/policy, fixture protocol/length, PDF signatures,
  and exact repeated trajectories/summaries.
- One-page PDF rendered and visually inspected: legible two-panel figure,
  caption, and explicit untrained-fixture label. FontAwesome ToUnicode
  warnings did not affect the visual check. Optional JaxMARL warp imports
  emit warnings; this workflow does not use those optional environments.
- Review fixes: documented Snakemake's PATH requirement; recorded tracked
  source dirtiness and script/policy hashes alongside existing provenance;
  committed a reusable verification command. Forced rebuild and verification
  passed after those fixes. Both staging checkouts are clean.
- SHA256 checks reconfirm the original checkout's three protected WIP files
  are unchanged. No merge, squash, deletion, production training, live
  Overleaf sync, or new GitHub repository creation occurred in P1.

### Retained limitations and next phase

This is a deliberately untrained four-step build fixture, not report evidence
or a production biological protocol. The source-worktree comparison verifies
packaging equivalence to the prepared core; it does not retroactively assign
that core revision to historical report results. Their provenance remains P0's
preserved record. Production protocols, pending original study changes, and
shared-contract extraction belong to P2; manuscript import belongs to P3.
The complete publication-tool lock and CI bootstrap remain later work.
New private remote creation remains authorized but unexecuted.

Next phase is P2, followed by P3, P4, P5, P6. **Await explicit user resumption**;
pause again at every phase boundary.

### Post-P1 decision — 2026-10-14

Before P2, the user requested removal of the original study runner's pending
`allow_dirty_worktree` override. The public argument, private provenance
argument, and bypass were removed, restoring mandatory clean-checkout
execution. The pending learning-curve and checkpoint-recovery changes remain
for separate discussion. This is an authorized change to the original WIP
after P1 preservation; the P0 backup still contains the previous snapshot.
P2 remains paused.

The recovery work is deferred to the published parent spec
[mycormarl #65](https://github.com/isaachcamp/mycormarl/issues/65), with native
sub-issues #66 (durable identity), #67 (validated continuation), #68 (missing
bundle/evaluation reconstruction), and #69 (explicit earlier recovery).
Dependencies: #67 blocked by #66; #68 and #69 blocked by #67. All issues are
labelled `ready-for-agent`. Publishing these issues does not start recovery
implementation or migration P2.

## Status at user-requested pause — 2026-10-13

Pause between all implementation phases. P2 must not begin automatically
after P1 verification. The following records the earlier intermediate pause;
the completion evidence above supersedes its build status.

Core staging branch `codex/repository-split-core` starts at feature-branch
HEAD `22bb55a98bd38ba7a17dc3e5c6aa8805da62b21d`, not at main. Its single
preparation commit is `72e5336c6f3f8e1121554b6a1c816281750f8103` and the
merge-base with `feat/31-phase-1-association-study` is the preserved feature
HEAD. Core staging tests passed: **513 passed, 1 skipped**. The original
checkout's three pre-existing modified files remain unchanged and preserved
in the P0 patch; those uncommitted edits are not in the staging commit.

The local REDbyRBD scaffold has a fresh independent `main` history and an
exact uv core pin. Its GitHub repository has not yet been created; private
creation is authorized. No animation repository, merge, or squash yet.

First showyourwork build failed before figure generation because its
subprocess could not find `snakemake` on PATH. Next: add the showyourwork
tool environment's `bin` directory alongside Tectonic's `bin` on PATH,
rerun the manuscript build, then verify provenance and repeatability. P1
remains incomplete until those checks pass; pause again before P2.

## Initial diagnosis

Six evaluator failures share a KeyError: base environment info lacks
`reproductive_fitness_from_components`, while the policy-interval wrapper
adds it. Emit the existing per-step decomposition in the base diagnostic
contract; retain summation over numerical substeps in the wrapper.

The training CLI failure is an intentional undiscounted-lifetime guard:
defaults do not guarantee lifetime, and the CLI does not declare finite-horizon
returns. Do not weaken the guard or silently change scientific training
protocols. The walking skeleton must explicitly declare its finite horizon.
The legacy SimpleNamespace PPO test lacks newly required config fields;
modernize that fixture to the public typed config rather than adding hidden
runtime defaults solely to satisfy an outdated fixture.

The study-runner resume failure concerns pending original-worktree changes;
retain its baseline record and address it when porting research orchestration.
