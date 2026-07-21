# P4 — Conservative axisymmetric diffusion and subcycling

## Phase goal

Transport canonical labile phosphate conservatively between neighbouring
axisymmetric soil cells, enforce the exact explicit finite-volume stability
ceiling, and subcycle the complete fixed-geometry diffusion-plus-uptake soil
transaction when the biological timestep is longer than that ceiling.

## Walking skeleton

Environment construction converts the static grid and physical diffusion
parameters into radial/vertical face conductances, the exact minimum cellwise
CFL time, and a fixed substep count. Each soil substep derives concentration
from canonical amount, applies pairwise conservative diffusion across internal
faces only, derives the updated concentration, and then runs the existing P3
uptake/competition transaction for the same substep duration.

## Scope

- Molecular phosphate diffusion coefficient
  `D_l = 1e-5 cm^2 s^-1` and impedance factor `f_l = 0.308`.
- Amount-flux coefficient `D_flux = D_l theta f_l`.
- Exact axisymmetric radial and vertical centre distances, face
  conductances, and closed external boundaries.
- Exact cellwise explicit CFL ceiling with safety factor `0.8`.
- Static substep-count calculation and complete diffusion-plus-uptake
  subcycling with fixed post-growth geometry.
- Diffusion diagnostics exposed on the environment for later convergence and
  performance work.

## Non-scope

- Sparse/continuous uptake blending or colony-age effects (P5).
- Advection, hydrodynamic dispersion, changing water content, kinetic
  sorption, or open soil boundaries.
- Production-grid/timestep selection and annual performance benchmarking
  (P6).
- Resolution of the separately recorded maintenance-P accounting issue.

## Dependencies and existing stack

- Verified P1 axisymmetric cell volumes and face areas, including shortened
  final cells.
- Verified P1 linear-buffer amount/concentration transform.
- Verified P3 continuous uptake, shared competition, and pool-credit boundary.
- Existing JAX/Flax immutable state and JaxMARL environment interface.
- No new dependency or framework.

## Scientific contracts

- Use `D_flux = D_l theta f_l` in the amount flux. Do not use buffered
  `D_app` there, because concentration already derives from the full buffered
  inventory.
- For an internal face between cells `i` and `j`, define conductance
  `G_ij = D_flux A_ij / d_ij` and signed transfer rate
  `T_ij = G_ij (C_i - C_j)`. Subtract `T_ij dt` from `i` and add the same
  amount to `j`.
- Omit surface, bottom, outer-radius, and central-axis transfers. The central
  radial face also has zero physical area.
- With `S = theta + B`, the exact positivity ceiling for cell `i` is
  `dt_CFL,i = V_i S / sum_j G_ij`, equivalent to
  `V_i / sum_j(D_app A_ij/d_ij)` for constant properties.
- The permitted explicit substep is at most
  `safety * min_i(dt_CFL,i)`, with provisional `safety = 0.8`.
- Set `N_sub = max(1, ceil(dt_bio_seconds / dt_allowed))` and use equal
  substeps whose durations sum exactly to the biological timestep.
- Each substep performs diffusion first and P3 uptake second. Root and hyphal
  geometry is fixed at its already computed post-growth value throughout the
  biological step; accepted uptake accumulates in organism pools.

## Granular task checklist

- [v] **P4.1 — Parameterise diffusion and precompute stability geometry.**
  - Replace ambiguous placeholder parameter names with unit-labelled
    `D_l`, `f_l`, and CFL safety controls.
  - Validate finite physical ranges.
  - Compute radial/vertical conductances, per-cell outgoing conductance,
    exact minimum CFL time, and required substeps.
  - Verify the provisional `D_app`, shortened-cell distances, zero-diffusion
    limit, and independently calculated small-grid CFL.
- [v] **P4.2 — Implement conservative explicit axisymmetric diffusion.**
  - Derive concentration from canonical amount at each diffusion call.
  - Apply pairwise internal-face transfers with equal and opposite updates.
  - Verify uniform-field invariance, radial and vertical references, closed
    boundaries, conservation, radial-axis behaviour, JIT, and non-negativity
    at the safety limit.
- [v] **P4.3 — Subcycle the complete soil update.**
  - Cache static conductances, CFL time, and substep count on the environment.
  - Run diffusion then uptake/competition in every equal substep.
  - Verify one-substep equivalence, multi-substep agreement with explicit
    repetition, accumulated consumer uptake, and concentration recovery after
    each diffusion update.
- [v] **P4.4 — Document and review the integrated P4 pipeline.**
  - Update public exports, function/test reference, plan links, and source
    docstrings.
  - Remove obsolete claims that diffusion is absent.
  - Run the second verification and phase review/fix cycle.

## Tests to add or run

- Add pure numerical tests in `tests/test_phosphate_diffusion.py`.
- Extend P3 environment tests for one- and multi-substep integrated updates.
- Retain P0–P3 unit, amount, geometry, uptake, and sequencing tests.
- Run the full suite before and after phase review.

## Verification checklist

- [v] Defaults and validation match `D_l`, `f_l`, and safety units/ranges.
- [v] `D_app` reproduces approximately `3.86e-9 cm^2 s^-1`.
- [v] Conductances use actual centre distances on shortened boundary cells.
- [v] Pairwise radial and vertical transfers are antisymmetric.
- [v] Uniform concentration is invariant and closed-domain amount is conserved.
- [v] No external or central-axis flux is introduced.
- [v] Stable diffusion remains non-negative at `0.8 dt_CFL,min`.
- [v] Required substeps are the exact ceiling and sum to the biological step.
- [v] Subcycled diffusion-plus-uptake accumulates correct consumer pool gains.
- [v] Focused and complete suites pass; review has no blocking finding.

## Phase exit criteria

- Integrated environment steps run stable, conservative diffusion followed by
  uptake on every required soil substep.
- Diffusion-only total labile P changes by no more than relative `1e-5` and
  stable test states remain non-negative.
- Static environment diagnostics expose the CFL ceiling and substep count.
- All tasks and checks are `[v]` after the phase-level review/fix pass.

## Phase notes

- Phase status: **verified** on 2026-07-21.
- The provisional default `0.1 cm` grid with strong buffering is expected to
  need one soil substep for `dt = 0.05 day`; unbuffered and finer-grid tests
  deliberately exercise multi-substep behaviour.
- P4.1 red evidence: the diffusion test module failed during collection
  because `phosphate_diffusion` did not exist. Parameter defaults, apparent
  diffusivity, nonuniform centre distances, exact CFL, substep ceiling,
  zero-diffusion limit, and validation then passed with retained state tests:
  `28 passed, 1 warning`.
- P4.2 red evidence: focused tests failed during collection because
  `diffuse_labile_amount` did not exist. The pairwise amount update then passed
  radial/vertical hand calculations, uniform concentration, closed single-cell
  boundaries, conservation, positivity at the safety limit, and JIT checks;
  the broader P1/P4 set passed `47 passed, 1 warning`.
- P4.3 red evidence: integrated tests failed during collection because the
  complete diffusion–uptake substep did not exist. After implementation, the
  environment caches conductances, exact CFL, substep count, and duration;
  one-step equivalence, three-step explicit repetition, diffusion-before-
  uptake concentration refresh, JIT, and retained P1–P3 behaviour passed `86
  passed, 1 warning`.
- Second verification pass: the focused P0–P4/environment set passed `121
  passed, 1 warning`; the complete suite passed `148 passed, 1 warning`.
  Function/class documentation coverage and `git diff --check` passed.
- Default-grid diagnostic: for `(500, 1000)` cells at `0.1 cm`,
  `D_app = 3.861262e-9 cm^2 s^-1`, `dt_CFL,min = 647,432 s`, and the `0.8`
  permitted duration is approximately `5.995 days`; `dt = 0.05 day` therefore
  uses one soil substep.
- Phase review found and fixed a latent zero-conductance division in the CFL
  diagnostic by using an explicit safe denominator. It also updated the
  former cellwise-decrease assertion—invalid once diffusion may increase a
  receiving cell—to check domain-total soil loss, and removed stale source
  documentation that described diffusion as future work. No blocking finding
  remains.
- Residual risks: P6 must quantify grid/timestep convergence and target-grid
  runtime. The substep trigger is diffusion-stability based only; whether
  uptake stiffness needs an additional accuracy trigger remains an open P6
  question. The maintenance-P accounting issue remains unchanged and explicit.
