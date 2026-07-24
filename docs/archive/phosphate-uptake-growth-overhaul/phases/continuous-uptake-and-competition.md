# Conservative continuous uptake and root–fungus competition

## Phase goal

Replace the fail-closed soil boundary with a runnable, amount-conservative
continuous-uptake step. Root and fungal requests are calculated simultaneously
from the same derived bulk solution concentration, capped once against each
cell's labile inventory, and credited to the correct organism pools in mg P.

## Walking skeleton

The soil step derives solution concentration from canonical labile amount,
turns post-growth root and hyphal length densities into absorbing surface,
calculates continuous Michaelis–Menten requests for the biological timestep,
applies one shared cellwise competition cap, subtracts accepted µmol P, and
credits the two scalar organism pools after converting to mg P.

## Scope

- Provisional shared root/fungal kinetics from Tinker & Nye (2000), following
  Schnepf & Roose (2006): `J_max = 3.26e-6 µmol cm^-2 s^-1` and
  `K_m = 5.8e-3 µmol cm^-3`.
- Pure continuous request, competition, and accounting kernels.
- Plant-only, fungus-only, and mixed uptake on the P1/P2 axisymmetric fields.
- Growth-before-uptake environment integration and mg-P pool credit.
- Accumulation of reproduction P export and structural-P mortality loss.

## Non-scope

- Diffusion, CFL calculation, or subcycling (P4).
- Sparse cylindrical surface concentration or smooth regime blending (P5).
- Empirical calibration, root/hyphal activity fractions, or individual
  absorber ownership maps.
- Redesign of the pre-existing maintenance-resource model. Phase-wide balance
  fixtures isolate the new soil transaction and explicitly represented
  reproduction/mortality terms.

## Dependencies and existing stack

- Verified P0 unit conversions and Michaelis–Menten helper.
- Verified P1 canonical `µmol P cell^-1` state and concentration transform.
- Verified P2 post-growth `(n_r, n_z)` root and hyphal length densities.
- Existing JAX/Flax immutable state and JaxMARL environment sequencing.
- No new dependency or framework.

## Scientific contracts

- For consumer `a` in cell `i`, represented length is
  `L_a,i = lambda_a,i V_i` and lateral area is
  `A_a,i = 2 pi r_a L_a,i`; end caps are excluded.
- The continuous request is
  `Q_a,i = J_max,a C_i / (K_m,a + C_i) A_a,i dt_seconds`, in `µmol P`.
- Root and fungal requests use the same pre-uptake `C_i` and are combined
  before any amount is removed.
- Per cell, `Q_accept = min(M_available, Q_root + Q_fungus)`. Each consumer
  receives its proportional request share, with an exact zero-demand guard.
- `M_after = M_available - Q_root,accepted - Q_fungus,accepted >= 0` within
  float32 arithmetic; accepted uptake cannot exceed either request.
- Accepted spatial amounts are summed per consumer, converted exactly once
  with `1 µmol P = 0.0309738 mg P`, and added after growth/allocation.
- Reproduction allocation is accumulated as P export in mg. Structural P
  removed with realised maintenance-deficit biomass loss is accumulated as
  mortality loss using the relevant `gamma_p`.

## Granular task checklist

- [v] **P3.1 — Parameterise and calculate continuous requests.**
  - Replace placeholder trait kinetics with the agreed units and defaults.
  - Validate finite non-negative `J_max` and finite positive `K_m`.
  - Implement the pure length-density-to-cell-request calculation.
  - Verify units, reference arithmetic, zeros, monotonicity, and JIT use.
- [v] **P3.2 — Implement shared cellwise competition.**
  - Calculate both requests from one concentration field.
  - Apply one proportional inventory cap with a zero-demand guard.
  - Verify uncapped, oversubscribed, symmetry, zero-inventory, non-negative,
    and exact cellwise accounting cases.
- [v] **P3.3 — Integrate soil uptake and organism-pool credit.**
  - Derive concentration from canonical amount at the soil boundary.
  - Subtract accepted µmol and credit summed mg to the matching pools.
  - Preserve post-growth geometry exposure and next-step resource credit.
  - Verify plant-only, fungus-only, mixed, JIT, and transaction mass balance.
- [v] **P3.4 — Activate P export/loss diagnostics and document P3.**
  - Accumulate reproduction P export for each consumer.
  - Accumulate structural-P loss from realised biomass mortality.
  - Update the function/test reference and remove the obsolete P1–P3 failure
    expectation and documentation.

## Tests to add or run

- Add focused pure-kernel tests in `tests/test_continuous_phosphate_uptake.py`.
- Add small-grid environment tests for sequencing, credit, accounting, and
  diagnostic accumulation.
- Retain P0–P2 tests, especially amount/concentration and geometry invariants.
- Run focused tests after each red/green increment and the complete suite
  before and after the phase review.

## Verification checklist

- [v] Kinetic defaults and validation match the agreed physical units.
- [v] Request reference arithmetic independently matches flux × area × time.
- [v] Plant-only, fungus-only, and mixed requests have correct limiting cases.
- [v] Shared competition never overdraws a cell and preserves request shares.
- [v] Soil loss in µmol equals accepted root plus fungal uptake per cell.
- [v] Organism pool gain in mg equals accepted soil uptake times `0.0309738`.
- [v] Uptake observes post-growth geometry but cannot fund same-step growth.
- [v] Reproduction and mortality diagnostics accumulate their represented P.
- [v] Focused and complete suites pass; phase review has no blocking finding.

## Phase exit criteria

- A full environment step runs through continuous uptake without the P1
  `NotImplementedError`.
- Soil amount stays non-negative and the uptake transaction closes mass
  balance in a common unit within relative tolerance `1e-5`.
- Root-only, fungus-only, and mixed simulations are covered by tests.
- All tasks and checks are `[v]` after the phase-level review/fix pass.

## Phase notes

- Phase status: **verified** on 2026-07-21.
- This is deliberately the continuous walking skeleton. P5 will replace each
  raw continuous request with the agreed sparse/continuous blended request
  without changing the shared competition or pool-credit boundary.
- P3.1/P3.2 red evidence: the focused test module failed during collection
  because `phosphate_uptake` did not exist. After implementation, request,
  competition, retained unit, and geometry tests passed `53 passed, 1
  warning`.
- P3.3 red evidence: five integration tests reached the retained P1
  `NotImplementedError` or the old three-argument soil signature. The
  continuous soil transaction and environment integration then passed with
  plant-only, fungus-only, mixed, JIT, sequencing, and balance coverage.
- P3.4 red evidence: reproduction and mortality tests showed all four
  cumulative diagnostics remained unchanged. The environment now accumulates
  allocated reproductive P and the structural P associated with realised
  biomass loss.
- Second verification pass: the focused P0–P3/environment set passed `88
  passed, 1 warning`; the complete suite passed `120 passed, 1 warning`.
  Function/class documentation coverage and `git diff --check` also passed.
- Phase review found one numerical hygiene issue: the zero-demand competition
  branch selected a finite result but still formed `0/0` in the unused branch.
  A safe denominator now prevents the invalid intermediate; all focused and
  complete checks passed after the fix. No blocking findings remain.
- Accepted limitation: the pre-existing biological maintenance calculation
  consumes allocated P without a separate fate diagnostic. P3 transaction
  balance and end-to-end fixtures therefore isolate steps with zero
  maintenance P use. The human explicitly rejected silently changing this to
  carbon-only maintenance; retain C-and-P maintenance until the P destination
  is deliberately specified and tested.
