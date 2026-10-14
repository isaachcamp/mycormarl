# Repository split implementation plan

## Summary

Split the current project into core `mycormarl`, combined research repository
`REDbyRBD`, and `mycormarl-animations`. Preserve the recent Overleaf report and
its scientific provenance before the intended squash merge. Establish one
installed-core-to-figure workflow first, then extract the remaining studies
and animations. Keep detailed exploration records available outside Git and
publish selected evidence sufficient for reproducible reporting.

**Status:** In progress; P0, P1 and P2 verified. Paused before P3.
**Created:** 13 October 2026.
**Authoritative decisions:** [ADR-0014](../adr/0014-organize-repositories-around-core-and-scientific-studies.md)
and [ADR-0015](../adr/0015-pin-core-dependencies-and-retain-executable-policy-evidence.md).
**File ownership:** [Extraction ownership map](repository-split-ownership.md).

## Execution Route

`standard` — explicitly selected by the user on 13 October 2026 after the
constrained runtime bootstrap failed and its supported operations proved
insufficient for downloads, tests, and Git preservation.

Use `rb-execute-plan` as phase owner with granular phase checklists,
implementation checks, a second verification pass, and phase review/fix gates.
The explicit `leave_constrained_pipeline: true` choice is recorded in the
canonical external diary. No constrained plan or assessment was persisted,
and no rejected artifact is being reused. Record progress both here and in
the diary. All original scientific preservation constraints still apply.

**Current phase:** P2 verified; await explicit resumption before P3.
**P0 evidence:** [Preservation phase record](repository-split-p0.md).
**P1 evidence:** [Installed-core workflow verification](repository-split-p1.md).
**P2 evidence:** [Research orchestration extraction](repository-split-p2.md).
**Execution cadence:** Pause after each implemented and verified phase.
**Ordered phase IDs:** P0, P1, P2, P3, P4, P5, P6.

## Goals and users

- Make model/algorithm development independent of named research protocols.
- Give the researcher one combined home for the current scientific story:
  fungal foraging, resource-pressure parameter/uncertainty exploration, and
  simple MARL/association studies.
- Let readers regenerate reported figures from selected archived evidence and
  inexpensive policy evaluations, without retraining every policy.
- Let the researcher explore detailed run and per-checkpoint records without
  repeating simulations.
- Keep animations independently executable against core biological code.
- Preserve the existing report and the source/results that produced it.

## Non-goals

Changing biological equations, Rate action semantics, reward definitions,
training protocols, or scientific conclusions; splitting the research story
into additional repositories; rewriting existing main history; pruning run
data; running full production studies; requiring resumable training state in
every publication bundle; replacing Overleaf's existing project contents.

## Requirements and constraints

1. Core owns the environment, algorithms, reusable action constraints,
   persistence, random-stream contracts, and generic rollout/evaluation APIs.
2. Research owns treatment matrices, named study stages, study stopping and
   qualification gates, exploratory diagnostics, interpretation, figures,
   manuscript, and Slurm launchers. No core-to-research import is permitted.
3. Animations own scenes, visual-specific preparation, render settings, and
   validators. Research cannot require an animation checkout for numerical
   evidence generation.
4. `REDbyRBD` pins an immutable core Git commit through uv, with committed
   `pyproject.toml` and `uv.lock`. Record study source, core source, manifest,
   artifact schemas, seeds, and runtime separately. Package version `0.1.0`
   alone is not sufficient provenance.
5. Preserve legacy records with their original identities. New provenance
   schemas must not make old results appear to have used the extracted core.
6. Report/submission milestones use fixed tags. Branches support active work,
   upgrade trials, and independently maintained versions. Future working
   branches use `codex/` unless the user specifies otherwise.
7. Retain exploration records outside checkouts, identified by run ID and a
   configurable data root. Existing ignored outputs may remain at their
   present location during migration. No move, deletion, or expensive copy of
   all run data is part of extraction.
8. Publication evidence includes summary results, selected executable policy
   weights and their configuration, all reported learning-curve inputs,
   non-regenerable/expensive figure inputs, environment requirements, and
   training/evaluation/design seeds. It is an export, not a replacement for
   the exploration record.
9. Slurm scripts declare and request resources for expensive computation;
   execution on a cluster is not required to validate extraction. Record
   measured resource evidence or clearly label initial estimates.
10. Document version-control conventions in core and research, as requested.
    Maintain `CONTEXT.md`, `docs/adr/`, and `docs/agents/` conventions in each
    repo, with relevant ADR ownership and original-ID traceability. Preserve
    supersession status, including ADR-0005's replacement by ADR-0007.
11. Integrate the Overleaf manuscript into research. Treat live synchronization
    as a separate configured capability from importing a fixed source snapshot.
    Historical builds must use the committed manuscript, not pull current text.
12. Preserve unrelated existing edits. The planning-time worktree contains
    changes to `study.py`, `test_study_runner.py`, and the runner guide in
    addition to this session's domain docs; do not stash, squash, or commit
    those indiscriminately.

## Evidence and unresolved prerequisites

- Planning-time branch is `feat/31-phase-1-association-study`; HEAD was
  `22bb55a`. These are observations, not a claim that this commit produced the
  report. No local tags were listed.
- The report source is in an Overleaf project. Its URL/access, exact report
  revision, reference PDF, and figure-to-run mapping are not yet identified.
- `study._repository_state()` starts from the package's `__file__`, and
  `_provenance()` reads that Git root's lockfile. This fails as a research/core
  distinction once core is installed as a dependency.
- PPO and checkpoint evaluation import a fixed-allocation action helper from
  `trade_only.py`. Its current 90/10 allocation defaults are scientific policy
  context, not permission to alter existing policy behaviour while extracting.
- Plotting uses private helpers in `study.py`, `static_controls.py`, and
  `checkpoint_evaluation.py`. Give the consuming workflow a supported seam
  before removing those imports.
- `plot_historical_grid_checkpoint_resource_trajectories.py` loads checkpoints
  and constructs an environment even with a complete numerical cache. Separate
  evaluate/cache from plot/cache instead of treating the whole CLI as plotting.
- A scientific decay plot imports a video-data helper. Resolve its shared
  calculation before making research independent of animation code.
- Prior inspection found approximately 91 GB of ignored `outputs/`, 178 MB of
  ignored `media/`, and 101 MB of `.git`. This split solves ownership, not
  existing disk usage. These measurements are approximate and time-dependent.

## Proposed layouts and dependency flow

```text
mycormarl                         REDbyRBD
  mycormarl/mycormarl/               pyproject.toml, uv.lock
  tests/                            src/scripts/, src/data/ (generated)
  scripts/ (examples/qualification)  src/tex/, src/tex/figures/ (generated)
  docs/, CONTEXT.md                  Snakefile, showyourwork.yml
  pyproject.toml, uv.lock            environment.yml, slurm/
                                    exploration/, manifests/, tests/, docs/

mycormarl-animations
  scenes/, data-preparation/, tests/, docs/
  rendering dependencies and reproducible workflow
```

These are proposed destinations, not a requirement to refactor the existing
core package layout during extraction. REDbyRBD may use a small local Python
package for reusable study orchestration; it must not shadow `mycormarl`.
Exploration scripts can share research helpers with figure scripts, while the
publication DAG explicitly enumerates only reported outputs.

Core imports neither downstream repo. Both downstream repos install core.
Research may consume versioned animation assets; no shared biological source
is copied into downstream repos.

### uv and showyourwork environment seam

The showyourwork template uses Conda via `environment.yml`; uv source overrides
are not automatically honored by other installers. In P1, use a pinned
showyourwork/tool environment to invoke research scripts through an explicitly
uv-locked runtime (e.g. `uv run --locked` under a controlled working directory).
Include dependency/lock inputs in affected Snakemake rules. Confirm which
Python executable and core revision every representative rule actually uses.
Do not maintain independently hand-edited pip/Conda and uv copies of the same
scientific dependency specification. If the selected showyourwork revision
requires a different bridge, resolve it visibly in P1 before expansion.

The current documentation recommends the development version of showyourwork;
pin a tested commit rather than tracking its moving branch. Build reference
figures headlessly and measure the inexpensive evaluation budget.

### Overleaf source handling

Import/export the existing report's complete source, bibliography, custom
styles, and necessary assets before initialization that could overwrite files.
First prove a local build from that fixed source. Record manuscript revision
or export checksum, original core/report source, run identities, and reference
outputs together. Do not label a new extraction commit as the original run.

For ongoing editing, the optional showyourwork bridge normally pulls manuscript
files from Overleaf and pushes generated figures; this needs an explicit file
ownership list and configured credentials. Existing-project integration must
be tested on a copy or isolated setup before affecting the original. Tagged
reproduction and PR validation disable live synchronization. No credentials
belong in Git. The user requested manuscript integration, not an automatic
choice of live bidirectional editing.

## Implementation phases

Phase task checklists are compiled only when their phase becomes current.
Every phase has a review/fix pass and terminal verification before advancing.
The overview below remains unchanged under the constrained status overlay.

### P0 — Preserve source and establish a baseline

Inventory tracked/untracked changes, active runs, report source, data roots,
and available policy/result artifacts. Obtain the Overleaf source snapshot and
map each report figure to its script, manifest, core revision, run, and seeds.
Capture reference numerical outputs and representative policy executions
using existing artifacts; run appropriate current tests once. Identify any
report input that cannot be regenerated before code moves. Preserve original
source revisions with accessible tags/backup refs and record dirty-source
patches separately where relevant. Do not tag current HEAD as report source
without evidence. Record missing provenance explicitly.

**Gate:** Reference report source and evidence are preserved; unrelated edits
and active outputs are accounted for; baselines are recorded. Missing report
access or unknown essential inputs blocks destructive cutover, not planning.

### P1 — Prove one installed-core-to-publication-figure workflow

Prepare an installable core commit with only the seams required for the first
slice, including explicit package dependencies/build metadata verified from
actual imports. Create a local REDbyRBD scaffold with provenance identifying
its original source and the accepted README title/subtitle. Pin core by commit.
Run one small existing policy through the public evaluator, save its summary
and selected numerical inputs, regenerate a representative figure, and build
a minimal manuscript page using showyourwork and the uv environment bridge.
Record both Git revisions; execute from outside the core checkout without
`PYTHONPATH` or editable sibling-source leakage. Pin the tested toolchain.

**Gate:** The thin vertical slice works in a fresh environment against
installed core; metadata identifies research and core correctly; the figure
matches its baseline within declared tolerances. No production retraining.

### P2 — Extract shared contracts and research orchestration

Separate generic action constraints and static/policy rollout facilities from
named protocols. Keep current fixed-allocation behaviour and policy artifact
compatibility intact. Add narrowly scoped public entry points for downstream
evaluation and actor construction where needed. Move named stages, matrix
orchestration, stopping logic, progress/resume ownership, and study result
contracts into research. Migrate study tests with their implementation.
Record explicitly provided research/core identities rather than inferring a
study repository from installed core. Version any changed provenance contract;
retain read-only interpretation of old records and strict incompatible-resume
rejection. Preserve checkpoint diagnostics, scientific manifest identity, and
RNG stream derivation. Keep legacy entry points only as documented temporary
forwarders if necessary during the staged migration.

**Gate:** Tiny interrupted/resumed research runs and static controls match
baseline semantics; changed pins/configurations cannot resume silently; core
tests pass without a research checkout. Test both deterministic evaluation and
seeded sampled evaluation, plus Rate action/lifecycle/PPO boundary contracts.

### P3 — Extract the combined study and Overleaf manuscript

Move foraging diagnostics, resource-pressure runners/analyses, MARL studies,
manifests, research plots, exploration helpers, and study-specific docs into
REDbyRBD according to the ownership map. Separate policy evaluation/data
generation from plotting, including the cached checkpoint trajectory script.
Import the preserved Overleaf source and make reported figure dependencies
explicit. Keep nonreported exploratory analyses executable outside the article
DAG. Configure a data-root/run catalog referencing existing records without
moving all outputs. Distribute ADRs and glossary terms by decision ownership,
with links back to originating records; retain default-parameter references
and numerical qualification evidence in core.

**Gate:** Representative foraging, fixed-policy, and learned-policy figures
rebuild with no animation imports; the report compiles from committed source
with live Overleaf sync disabled. Every current script/test/doc has an owner;
historical protocols are labeled and not promoted to current defaults.

### P4 — Extract the complete animation workflow

Create the animation repo with source provenance, move scenes and their data
preparation/validators/tests, and record the rendering environment. Resolve
the scientific/video helper dependency using core kernels and separately owned
preparation. Use core as an installed dependency. Since the user requested
showyourwork in downstream repos, provide a reproducible workflow for the
animation outputs, with a small figure/contact-sheet document if needed; do
not invent a scientific manuscript or require a long render for every check.
Keep generated video outputs outside Git and define short preview validation.

**Gate:** One short representative render/data-validation workflow passes
against installed core without REDbyRBD; research still needs no animation
checkout. No biological kernel is duplicated.

### P5 — Complete reproducibility and execution documentation

Export publication evidence bundles from preserved exploration records with
checksums, summary results, selected weights, actor/model config, original
provenance, seeds, and reported learning-curve data. Declare archive retrieval,
inexpensive evaluation, analysis, figures, and manuscript dependencies in
showyourwork. Cache/archive expensive results rather than launch training in
an ordinary build. Select archive hosting and access at this phase; Zenodo is
a supported candidate, not an already authorized publication.

Supply Slurm scripts for study simulation/training and expensive evaluations,
with named CPU/GPU requirements, memory, time, concurrency, seed/run mapping,
environment setup, logs, output location, and resume semantics. Distinguish
measured resource budgets from estimates, and include small local smoke paths.
Document pin upgrades, report tags, data retention, and relevant ADR conventions
in core/research. Configure CI with an explicit numerical-cost budget. Set up
ongoing Overleaf sync only if its ownership/access choices have been resolved.

**Gate:** A fresh checkout rebuilds reported evidence from the selected bundle
without exploration storage, undocumented paths, Overleaf login, or training.
An ordinary build does not mutate historical source or archived artifacts.
Slurm scripts pass syntax/configuration checks; production cluster execution
and actual public archive upload are separately recorded operations.

### P6 — Final cutover, squash integration, and dependency pins

Review the full extraction diff and confirm runnable downstream replacements
before removing study/animation implementations from the active core tree.
Verify that all files, tests, relevant docs, and tracked small evidence have
destinations. Perform the intended squash merge through the applicable Git/PR
workflow, preserving original report and staging source references. Record
the pre-squash-to-squash-to-extraction mapping. Update downstream pins to the
final accessible core commit and rerun installed-package integration checks.
Never silently replace old result provenance with the squash commit. Decide
visibility, license inheritance, remote ownership, and locations before
creating/pushing the new repositories. Reconfirm names when creating them.
If creating PRs, attach each created PR to this chat.

**Gate:** All three repos run independently; core tests and downstream clean
builds pass with final pins; preserved report source/results are reachable;
core has no named-study orchestration or downstream imports; detailed data is
intact. No force rewrite of main and no deletion of retained branch evidence.

## Validation plan

- Protect scientific invariants with existing core tests: units, finite-P
  conservation, geometry, Rate actions, transitions, lifecycle, policy
  intervals, observations, and PPO/bootstrap semantics. Do not require full
  study retraining to validate file extraction.
- Compare migrated numerical inputs to baseline before comparing figures.
  Use recorded tolerances and identical seeds; distinguish numeric agreement
  from image/TeX differences caused by fonts or toolchain changes.
- Verify policy export/import, small evaluator runs, checkpoint summaries,
  learning-curve evidence, RNG streams, interruption/resume, incompatible
  provenance rejection, and explicit core-commit attribution.
- Install noneditable core in fresh environments; run from directories with
  no access to original source on import paths. Check actual imported module
  locations and resolved package dependencies.
- Test the showyourwork dependency graph: changing a core pin, manifest,
  selected weights, seed/protocol, or plotting script invalidates the affected
  outputs; unchanged inputs do not trigger expensive computation. A dirty
  manuscript or live Overleaf state cannot silently alter a tagged rebuild.
- Exercise an ordinary build with only the publication evidence bundle,
  and exploration separately against the existing external run catalog.
- Keep tests with owners. Split mixed core/scientific tests at their true
  contracts rather than leaving duplicate production implementations.

## Rollout and rollback

Keep extraction additive until each downstream slice verifies. Original
report refs and existing external run data remain available throughout.
Rollback a code move through Git, or return to the last tagged/pinned verified
source; never repair a failed extraction by rewriting result identities or
deleting data. Do not squash/delete the only reachable source referenced by
old artifacts. Public archive deposits and Overleaf writes need their own
reviewable operation scope before execution.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Report state differs from current code | Match Overleaf snapshot and run provenance before tagging. |
| Squash makes old commits inaccessible | Preserve original refs and record commit mapping before branch cleanup. |
| Core installs with hidden transitive dependencies | Inspect imports and test wheel/noneditable installs in a clean environment. |
| uv pin is ignored by the article build | Prove interpreter/core revision in P1 and declare lockfile DAG inputs. |
| Private helpers become accidental public API | Extract minimal supported functions and verify their contracts. |
| Mixed fixed-allocation helpers alter policies | Preserve defaults and test latent-to-Rate-action and saved-policy behaviour. |
| Compact publication bundle cannot redraw learning curves | Retain reported per-checkpoint metrics and selected additional inputs. |
| Reduced publication retention deletes exploratory evidence | Export bundles; never prune exploration records as a migration operation. |
| Tagged build pulls newer Overleaf text | Separate editorial sync from fixed-source reproduction. |
| Study-specific constraints are left in core | Ownership audit covers modules, tests, docs, manifests, and ADRs. |

## Success criteria

The reader can clone REDbyRBD, install its pinned requirements, retrieve its
publication evidence, and rebuild its reported figures/manuscript without
the original worktree. The researcher can still explore detailed checkpoints
and run data without repeating simulations. Core and animation workflows run
independently, with ownership and provenance documented. Report snapshots
remain reproducible across the intended squash merge and future core updates.

## Open questions, resolved at their phase gates

- P0: Overleaf project URL/access and report revision; exact figure/run
  inventory; disposition of existing unrelated worktree changes.
- P1: Tested showyourwork commit, uv/runtime bridge, and tiny reference policy.
- P3/P5: Editorial source-of-truth and whether ongoing Overleaf sync is wanted
  beyond importing the report source; historical-build sync configuration.
- P4: Final animation repo name and rendering/tool versions.
- P5: Storage/archive service, artifact size, CI evaluation budget, and Slurm
  hardware/partition/account settings. Do not invent cluster credentials.
- P6: GitHub ownership/visibility, local checkout paths, license choices if no
  license exists, and exact report/submission tag names.

## Sources checked

- [showyourwork repository layout](https://show-your.work/en/latest/layout/):
  manuscript/scripts/data conventions and Conda environment specification.
- [showyourwork installation](https://show-your.work/en/latest/install/):
  current toolchain/development-version guidance.
- [showyourwork custom Snakemake rules](https://show-your.work/en/latest/snakefile/).
- [showyourwork Overleaf integration](https://show-your.work/en/latest/overleaf/):
  file-direction ownership, current-source sync, and existing-project caveats.
- [showyourwork Zenodo integration](https://show-your.work/en/latest/zenodo/).
- [uv Git sources](https://docs.astral.sh/uv/concepts/projects/dependencies/#git).
