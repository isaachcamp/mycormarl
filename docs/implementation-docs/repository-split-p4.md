# P4 — Independent animation workflow

**Route:** standard. **Status:** Verified; paused before P5.
**Starting source:** original `2d22463`, core `7022a1c`, research `683609b`.

## Scope and exit criteria

Extract all nine animation/preparation/validation scripts and four animation
test modules into local `mycormarl-animations`. Use installed core kernels,
preserve numerical assumptions and visual seeds, and provide a showyourwork
workflow that prepares data, validates it, renders a short preview and builds
a small visual verification document. Keep full renders explicit and generated
media outside Git. Original implementations remain until P6; no GitHub creation,
production study, source squash or research dependency is part of this phase.

- [v] Capture numerical preparation baselines and inspect scene/tool dependencies.
- [v] Extract scripts/tests/docs into an installed animation package with an
  independent locked runtime and noneditable core dependency.
- [v] Build explicit preparation/validation/short-preview/contact-sheet rules
  and document full-render entry points, assumptions and visual seed identities.
- [v] Preserve single-context agent conventions and ADR authority links, with
  animation vocabulary distinct from biological/scientific model parameters.
- [v] Verify original numerical agreement, animation tests and a short render
  without REDbyRBD; inspect the preview and build the showyourwork document.
- [v] Run a second verification and phase review/fix pass; commit source and
  evidence, update ownership/phase/diary records, and pause before P5.

Use existing conservation, radial-geometry, contact-time and render-object
tests. Compare preparation arrays exactly on the same runtime where possible;
retain existing scientific tolerances. Full-length renders are optional manual
commands, not required for the inexpensive default workflow.

## Implementation and second-pass verification

- All nine source modules, four animation test modules and the animation source
  document are extracted into `mycormarl-animations`. Installed package imports
  replace sibling-script path injection; scene wrappers preserve the original
  implementations and seeds. The nine module bodies are AST-equivalent after
  normalizing imports, removed path injection/unused imports and CLI docstrings.
  All 13 original code/test SHA256 values remain unchanged in the original tree.
- The independent uv runtime pins installed noneditable core `7022a1c`, with
  JAX/JAXlib0.9.0, NumPy2.3.5, Manim0.22.0, PyAV19.0.1 and ManimPango0.7.0.
  Its lock is separate from research. Both downstream repos retain the same
  tested core implementation pin. Core staging `85bd2eb` only relocates tests;
  its package source, project metadata and scientific lock are unchanged.
- Original baseline: 44 animation tests passed. Extracted first/second pass:
  46 passed, including exact original isolated/two-day-colony preparation arrays
  and actual installed-core/no-research checks. Core: 348 passed with neither
  downstream package nor Manim installed. Research boundary checks: six passed.
- Both full numerical validators passed. Closed-colony 24-day phosphorus
  balance error was 1.39453e-6 relative, below the unchanged 1e-5 bound;
  timestep/grid refinements, shell-integrated flux and moving profile peaks
  passed. The isolated diagnostic retains prescribed decay, local-age contact
  shifts, supply-demand checks and its explicit nonconservative interpretation.
- Clean committed-source forced showyourwork build passed twice after review:
  preparation/validation, a 40-frame 854x480 preview at6fps (6.67 seconds),
  three-frame contact sheet and one-page A4 verification document. Preview and
  document were visually inspected. A four-frame geometry GIF also rendered.
  Full videos remain explicit commands; no policy training/re-evaluation ran.
- The source manifest preserves visual seeds19/8,73 and20261008; generated
  runtime records actual source/core commits, lock hash and package versions.
  All generated media is ignored. Original exploration/media records stay intact.
- Core test relocation committed/pushed to the existing staging branch. New
  animation repo remains local, with its intended remote configured but not
  created/pushed. Source remains on the original feature branch until P6.

## Review fixes and accepted boundaries

Review corrected one remaining legacy test path, stale CLI examples, duplicate
literal pin checks (runtime now reads uv.lock), and a paragraph split around the
document figure. Scientific kernels, visual topology and seeds are unchanged.
The numerical tests, explicit validators, source-body audit and forced workflow
checks cover the extraction. There are no unresolved blocking P4 findings.

Historical rendering dependencies were not recorded; this phase establishes a
tested rendering runtime, rather than assigning it to old videos. Cross-platform
font/raster/video identity and full-length video rendering were not tested.
Existing render-object tests cover the complete scene's topology, timing,
geometry and frame bounds. The smoke preview covers only the opening excerpt.
Complete portable CI/tool bootstrap remains P5; final core implementation
removal, squash and private downstream GitHub publication remain P6. Recovery
parent spec isaachcamp/mycormarl#65 remains deferred.

Evidence: `mycormarl-data/verification/repository-split-p4/verification.json`,
source/runtime identities, preview/document/GIF, full validator reports/plots,
test/workflow logs, original numerical baselines and source-audit script.
Verified animation execution commit: `07884db6c3fb18800f5197939f6d50374b7c398b`.

P4 is verified. Pause before P5 and await the user's resumption.
