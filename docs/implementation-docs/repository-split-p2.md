# P2 — Extract study orchestration and shared execution contracts

**Route:** standard. **Status:** Verified, 14 October 2026; paused before P3.
**Source:** original feature commit `e858202`; prepared core `72e5336`.

## Scope and ownership

REDbyRBD owns the public study runner, named trade-only protocols, static
treatment matrices, result identities, stopping interpretation, and their
tests. Core owns environment/PPO/action/policy/RNG facilities and reusable
static rollout execution. Biological numerical kernels and fixed-allocation
actor semantics are unchanged. Existing scripts/manifests and manuscript
import remain P3; final removal of legacy entry points remains P6.

The committed recovery prototype is preserved for #65, but automatic recovery
without validated execution identity must not be enabled by this extraction.
Existing validated single-condition resume stays supported. Strict committed
source execution remains mandatory. Legacy study results remain readable;
they cannot be silently resumed as the new cross-repository result contract.

## Tasks

- [v] Capture a tiny static numerical baseline and define installed-package
  study/provenance acceptance checks before extraction.
- [v] Expose reusable static execution in core without study manifests or
  treatment-matrix orchestration; retain fixed-allocation action semantics.
- [v] Extract study/static/trade-only orchestration and tests into a distinct
  installed `redbyrbd` helper package, carrying original committed changes.
- [v] Version research/core provenance separately and reject incompatible
  resume; preserve legacy read-only analysis and fail-closed orphan behavior.
- [v] Validate tiny resumed training, static controls, deterministic and
  sampled evaluation, and original core numerical/PPO contracts.
- [v] Commit/pin verified core and research revisions, run a second installed
  workflow check, review/fix integration, and pause before P3.

## Verification

Public runner tests and static controls use the uv-locked installed core.
Compare original versus extracted tiny static results; compare interrupted
versus uninterrupted training endpoint and checkpoint state on one runtime.
Run core tests without a research checkout on its import path. Confirm source,
core pin, lock, RNG, and interface drift rejection and unchanged legacy saved
policy semantics. Rebuild P1 figure/article when its pin changes. Document
deferred #65 work, retained compatibility adapters, and all skipped checks.

## Verified implementation

- Core staging: `9d586fa5b12004a87b87cd95d588d28db39e616d`, on
  `codex/repository-split-core`, pushed to the existing core remote. Research:
  `9f3a198db076aaf039fb4ad5ae7400d26c9911d1` in local REDbyRBD, with that exact
  core pin in pyproject and uv.lock. Both checkouts are clean.
- The installable research package `redbyrbd` owns study/static/trade-only
  orchestration, stopping interpretation and result contracts. Associated
  tests and two protocol declarations moved with it. The remaining scripts,
  manifests, diagnostic/qualification inventory and manuscript remain P3.
- `mycormarl.static_rollout` accepts supplied environments and physical Rate
  actions. Core retains PPO, generic checkpoint evaluation, policy artifacts,
  fixed-allocation actor semantics, RNG and biological kernels. Static
  treatment adapters invoke those numerical facilities without duplicating
  environment dynamics.
- New research results use version 3, recording research/core revisions
  separately. The imported core must match its installed distribution and
  the immutable revision in the research lock. Tracked or untracked dirty
  source is rejected. Research/core/lock/interface drift cannot silently
  resume a bundle. Historical v2 bundles remain read-only through a reader
  and pilot analysis; new continuation requires the current contract.
- The historical trade launcher also gates reuse by source/core execution
  provenance. It does not automatically claim old directories without that
  record. Its checkpoint representation remains historical rather than
  being relabelled as a new v3 result.
- Original realized-return diagnostics are preserved. The original orphan
  recovery helper is retained as deferred prototype code, but the public
  comparison runner does not call it. Missing identity or a pending condition
  with checkpoints fails clearly; robust recovery remains #65/#66–#69.
  Existing validated single-condition continuation remains operational.

## Verification evidence

- Final core suite: **423 passed, 1 skipped**, without research on its import
  path. Final research suite: **102 passed** against the installed final core.
  Fourteen focused action/PPO/reward/autoreset/static checks passed after the
  final diagnostic repair. The sole core skip is inherited from the baseline.
- Captured serial and batched static results match exactly before/after
  extraction. The final installed research run reconfirms that equality.
- Tiny interrupted and uninterrupted training matches complete runner state
  exactly: policy/optimizer/critic-normalizer, environment, RNG and counters,
  plus the final evaluation. This is verified both in tests and through an
  unmocked public runner from clean committed research source.
- Deterministic latent-location and sampled-policy evaluation repeat exactly
  with fixed evaluation seed 17 on this runtime. Training uses master seed 7
  and existing stream derivation. No cross-hardware bitwise claim is made.
- A forced showyourwork build passed against the final pin. The walking-skeleton
  provenance verifier passed against P1 numerical evidence; the final rendered
  manuscript PNG has the same SHA256 as the visually reviewed P1 page.
- Evidence and tiny integration outputs are preserved under the external
  `mycormarl-data/verification/repository-split-p2` directory. Real study
  exploration records were not moved, modified or pruned.
- Original feature history remains intact at `e858202` before this phase's
  documentation checkpoint. No core merge/squash or new private remote
  creation occurred; REDbyRBD remains local. Creation is still authorized
  for the later cutover phase.

## Review fixes and residual scope

Review repaired a missing batched compatibility adapter and confirmed exact
baseline agreement. Two declaration fixtures were copied with their owning
tests. A stale comparison test was corrected to represent a genuinely unrun
condition rather than asking the deferred recovery path to reclaim an orphan
checkpoint. Source-import consistency and dirty/core-drift rejection gained
explicit public-runner coverage.

The generic static interface exposed an inherited empty-inventory diagnostic
indexing error. A zero-P regression fails on the previous installed core and
passes on the final core. The fix recognizes a uniform zero field and changes
no biological dynamics or positive-inventory numerical outputs.

Temporary lazy study/named-baseline forwarders and the old static manifest
adapter remain in staging core for P3 callers and final P6 removal. Broader
qualification distribution and file ownership audit remain later work.
Optional JaxMARL warp-import/JAXopt maintenance warnings and the existing
FontAwesome PDF mapping warnings are nonblocking for the verified workflows.

**Next:** P3, then P4, P5, P6. Await explicit user resumption. Pause again at
each phase boundary. This verification does not mark recovery issue #65 done.
