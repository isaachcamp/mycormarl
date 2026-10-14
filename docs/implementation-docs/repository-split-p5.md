# P5 — Publication evidence and execution requirements

**Route:** standard. **Status:** Verified; paused before P6.
**Starting source:** original `c390d8c`, core staging `85bd2eb`, research
`1cf0c74`, animations `1e14219`.

## Scope and gate

Build a checksum-indexed publication export containing reported summaries,
selected executable policy weights/configuration, original source/runtime
requirements, training/evaluation/design seeds and learning-curve inputs.
Keep rich exploration records intact. Host initially as private GitHub release
assets, then deposit on Zenodo for publication (user-selected); no upload or
live Overleaf synchronization is part of this phase.

- [v] Inventory selected report evidence, checkpoint representations, seeds and
  original runtime/source requirements; measure bounded execution costs.
- [v] Export and validate a versioned archive with selected policy weights,
  reported metrics, source identities/configuration and checksums; implement
  explicit local/private-release retrieval without mutating exploration data.
- [v] Integrate archive retrieval, cheap evaluation, analysis, figures and
  manuscript dependencies; prove a fresh checkout rebuild from the bundle.
- [v] Provide Slurm resource requests, job/seed/output/log mapping and strict
  resume documentation, with initial estimates clearly separated from measurements.
- [v] Complete tool/runtime bootstrap, CI cost budgets, version-control and
  retention/ADR documentation in the relevant repositories.
- [v] Run second-pass verification and phase review/fix; commit source/evidence,
  update durable records and pause before P6.

Fresh builds must need no original exploration directory, undocumented paths,
Overleaf credentials or training. Test archive integrity, path containment and
seed/source contracts; check Slurm syntax and local smoke paths. Cluster jobs,
public archive upload, live editorial sync and final squash/cutover are deferred.

## Second-pass evidence and phase review

Verified 14 October 2026. This phase prepares reproducibility locally; publishing
and final repository cutover remain P6. Recovery spec isaachcamp/mycormarl#65
and its child tickets remain incomplete.

The selected `report-2026-10-13-v1.tar.gz` is 12,029,727 bytes, SHA256
`e8440c20ef47d0da483cf0026a35d4e9420b6ca9786336446c31ec36b8b8b661`.
It has 95 indexed entries and all 30 final learned policy conditions (six P
settings, seeds 0–4), original actor/source/runtime configuration and stream
keys, retained checkpoint/learning-curve summaries, selected numerical inputs
and the preserved manuscript assets. Both actor parameter trees and saved
inference observations match every original checkpoint exactly; all original
checkpoint hashes remain unchanged. No exploration files were pruned. These
weights omit optimizer/environment state and cannot resume training.

The clean fresh checkout at `85ecf0177356f942f547f82d53710f3bb6b585f6`
installed three independent locked Python environments and forced the complete
showyourwork workflow. It rebuilt the 34-page report in 318.68 seconds against
a 900-second budget, with zero unresolved reference tokens. All four generated
scientific figures matched the preserved reference pixels exactly on this
macOS runtime. Historical actor inference passed all 30 conditions using the
original core `95d36e1`; it is not a full-episode return evaluation. The fixture
uses current installed core `7022a1c`, archived untrained weights, initialization
seed 41 and evaluation seed 17, and only four environment steps. No training
or full historical episode evaluation occurs in the ordinary build.

The fresh checkout had no exploration output tree or local catalog paths.
A Python audit hook in all three environments rejected access to original and
sibling source directories; no forbidden access occurred. The archive and
verified native Tectonic/Biber prefixes were explicit external inputs. This
is Python file-access evidence, not an OS process sandbox. A subsequent fresh
native compiler installation from the explicit package lock also passed.

Research tests: 163 passed. Animation tests: 46 passed; its separately locked
publication tools forced a complete preview/PDF build, yielding 40 frames at
854×480 and 6 fps. P5 changes in core are documentation only; the package,
metadata and numerical lock are unchanged from the already verified core.

Slurm scripts passed shell syntax checks. The committed compute wrapper
executed the two-transition fixture locally and repeated it without changing
its content-addressed execution plan. The final compute fix is `c3e198a`;
report workflow dependencies are unchanged from the fresh-build commit. Job
resource requests are initial estimates, not measured production requirements.
No cluster job was submitted.

Review fixes retain execution declarations across resubmissions, reject
non-positive worker counts, clear only empty generated extraction directories
after forced rebuilds, and distinguish checked numerical evidence from portable
image encoding. The archive stays local until an explicitly configured private
GitHub release; Zenodo is the planned publication deposit. Hosted Linux CI,
full historical re-evaluation, production Slurm runs and archive uploads have
not run. CI jobs are bounded and the report job waits for its archive release.
Original renderer attribution and historical installation timing of the observed
JaxMARL compatibility patch remain documented gaps. Existing TeX layout and
algorithm.sty encoding warnings persist; no manuscript content was rewritten.

Durable evidence is in the sibling data root at
`verification/repository-split-p5/verification.json`, together with the verified
report, preview, actor/fixture outputs, source checks and complete logs.

P5 is verified. Await explicit user resumption before P6.
