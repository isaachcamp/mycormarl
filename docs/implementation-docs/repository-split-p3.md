# P3 — Combined research and preserved Overleaf manuscript

**Route:** standard. **Status:** Verified; paused before P4.
**Starting source:** original `aa58b7c`, core staging `9d586fa`, REDbyRBD `9f3a198`.

## Scope

Extract foraging diagnostics, resource-pressure exploration, research launchers,
figures, manifests, and their tests/docs into REDbyRBD. Core remains the
authority for biological kernels, default parameters, numerical invariants and
algorithm contracts. Preserve historical source/data attribution and all
exploration records. Import the fixed Overleaf export, not live editing state.
Recovery issue #65 and animation extraction remain deferred.

## Checklist

- [v] Capture representative diagnostic/rendering baselines and inventory all
  tracked source/tests/docs by final owner.
- [v] Extract remaining research modules, scripts and tests against installed
  core; expose only needed supported execution seams and remove animation imports.
- [v] Separate cached checkpoint rendering from evaluation/preparation and
  configure a data-root/run catalog without moving existing simulation outputs.
- [v] Distribute glossary terms, ADRs and scientific docs with original IDs,
  supersession state, and links to authoritative core defaults/contracts.
- [v] Import all preserved manuscript source/assets; explicitly declare figure
  dependencies, distinguish regenerated evidence from frozen reference assets,
  and compile the report with bibliography and live sync disabled.
- [v] Verify representative foraging, static-control and learned-policy figures,
  research/core tests and a clean committed-source report build; review/fix,
  record evidence and pause before P4.

## Verification boundaries

Use existing scientific tests and small numerical fixtures. Preserve units,
seeds, actor semantics and recorded output identities. No production training
or historical policy re-evaluation with a different core is implied. Render
reported historical data where attribution is established; retain checksum
identified reference figures for unresolved original renderers and explain
those boundaries. Manuscript differences caused by compiler/fonts/date must
be recorded rather than confused with numerical changes. Every original
tracked file receives an explicit final owner or historical-retention status.

## Implementation and second-pass evidence

- Research now owns six diagnostic/screen modules, 26 figure/exploration
  launchers, five migrated research test modules, study manifests, scientific
  reading and qualification evidence. Supported core actor/action seams remain
  generic; the installed immutable pin is `7022a1c`.
- All 237 files tracked at original `aa58b7c` have a final owner/status in
  `catalog/file-ownership.json`. Research destinations exist; animations await
  P4 and final core implementation removal awaits P6. The original feature
  branch and detailed exploration records are retained.
- Core: 378 passed, one skipped independently of research. Research: 154
  passed against the noneditable installed core. Six focused boundary tests
  include path containment, original far-field numerical baseline (rtol 2e-6),
  cache-only plotting and incompatible historical preparation rejection.
- All 39 exported manuscript files pass their original SHA256 checks. ADR IDs,
  supersession states and core authority links are preserved. Research has no
  animation imports; all figure modules and checkpoint preparation import
  successfully against the installed core.
- Foraging uptake, static controls, learned-policy performance and checkpoint
  phosphorus intake regenerate pixel-identically to the original report PNGs
  on this runtime. These four now feed the manuscript. The other reference
  images remain checksum-identified assets; full original renderer attribution
  remains incomplete. Historical weights were not re-evaluated.
- Clean committed-source forced showyourwork build passed. It generates both
  supplementary label sets and the complete 34-page A4 report, with 67 citekeys
  resolved, the frozen 13 October 2026 date and zero unresolved references.
  All pages were inspected as a contact sheet, with detailed checks of a
  representative scientific figure and bibliography. Live Overleaf sync is off.
- Evidence: `mycormarl-data/verification/repository-split-p3/verification.json`,
  report PDF, contact sheet, auxiliary label files, build/test logs and audit
  script. Research implementation commit: `d5602fa3d9122675e8fd9d27f2481a38dff901ff`.

## Review and fixes

The phase review corrected hyperref package load order, removed an obsolete
demo-only manuscript dependency, placed generated reference labels under
`src/tex/output`, and supplied standalone subfile asset paths only inside a
temporary compilation directory. It also separated evaluation/preparation from
cached rendering and added strict incompatible-provenance rejection before
historical checkpoint evaluation. Missing caches fail without evaluation.

The pinned showyourwork DAG traversal repeatedly queries Git for the same
repository root. A documented process-local root cache in the project Snakefile
removed more than seven minutes of pre-execution overhead; commit/provenance
queries and scientific computations are unchanged. The final complete forced
build passed with that shim.

## Accepted boundaries and next phase

The regenerated PDF is not claimed byte-identical to the pdfTeX Overleaf PDF;
the tested Tectonic/Biber toolchain has font/layout differences and retains a
nonblocking algorithm-package encoding warning, one underfull paragraph and
a bibliography overfull-line warning. The original source and reference PDF
remain preserved. Biber 2.17 is required by the tested biblatex 3.17 bundle;
2.20 rejects its BCF format. Cross-platform rendering has not been verified.

Detailed exploration storage is not moved/pruned. Existing source identities
remain attached to historical records. Recovery hardening is deferred to
parent spec isaachcamp/mycormarl#65 and its subissues. Archive hosting, selected
policy export, complete CI/tool lock and Slurm declarations remain P5. The new
private GitHub repository is authorized but remains uncreated until cutover.

P3 is verified. Pause before P4, the animation extraction; do not proceed
without the user's resumption.
