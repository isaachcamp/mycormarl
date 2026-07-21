# P5 — Analytical sparse closure and smooth regime transition

## Phase goal

Add an inexpensive analytical uptake closure for isolated roots and hyphae,
then blend it smoothly with the existing continuous closure according to the
static diffusion-overlap timescale implied by local hyphal density.

## Walking skeleton

At the start of each biological soil step, derive geometry-only sparse
resistances for roots and fungi and one shared fungal-overlap weight. Keep
those coefficients fixed while diffusion and uptake are subcycled. After each
diffusion update, recover the current bulk solution concentration, calculate
both sparse and continuous requests, blend each consumer's requests with the
same weight, and apply the existing shared-inventory cap once.

## Scope

- Cylindrical territory radii for root and fungal length densities.
- Diffusion-limited effective radii and logarithmic transport resistance.
- Numerically stable analytical surface concentration and sparse uptake.
- Static `T_ref / t_diff` overlap transition driven only by hyphal density.
- Shared blending for roots and fungi before conservative competition.
- Geometry-coefficient calculation once per biological soil step.

## Non-scope

- Colony age or local time-since-colonisation.
- Root density as a driver of the shared continuous-regime weight.
- Per-cell radial PDEs or iterative nonlinear surface-concentration solves.
- Timestep/grid convergence and performance qualification (P6).
- Resolution of the separately recorded maintenance-P accounting issue.

## Scientific contracts

- `R_soil = 1 / sqrt(pi lambda)` for positive consumer length density.
- `D_app = D_l theta f_l / (theta + B)` sets propagation distance and overlap
  time; `D_flux = D_l theta f_l` sets steady amount-flux resistance.
- `R_eff = r_a + min(sqrt(D_app T_ref), max(R_soil-r_a, 0))`.
- `k = r_a J_max ln(R_eff/r_a) / D_flux`.
- Use the stable positive solution of the uptake/supply quadratic and enforce
  `0 <= C_s <= C_b` without iterative solving.
- `t_diff = max(R_soil-r_h, 0)^2 / D_app`; zero hyphal density gives infinite
  overlap time and exactly zero continuous weight.
- `w_cont = 1 / (1 + (t_diff/T_ref)^p)`, equivalent to the agreed Omega form
  but numerically well behaved at zero and infinite overlap time.
- Use the fungal-derived `w_cont` for both consumers. Root-only cells are
  sparse. Blend requests, then cap their sum once against cell inventory.

## Granular task checklist

- [v] **P5.1 — Parameterise and test geometry/transition primitives.**
  - Add configurable `T_ref = 1 day` and transition exponent `p = 2`.
  - Implement territory/effective radii, overlap time, and stable blend weight.
  - Verify the nominal saturated-hypha result and all zero/limit cases.
- [v] **P5.2 — Implement and test the analytical sparse closure.**
  - Implement geometry-dependent resistance, stable `C_s`, and sparse request.
  - Verify the quadratic residual, bounds, limiting behaviour, and JIT.
- [v] **P5.3 — Integrate blended requests into the soil pipeline.**
  - Calculate geometry-only coefficients once outside the substep loop.
  - Recalculate concentration, `C_s`, and requests after every diffusion step.
  - Preserve one shared post-blend cap and exact organism-pool crediting.
  - Verify root-only, fungus-only, mixed, sparse, and continuous-limit modes.
- [v] **P5.4 — Document, review, and verify the phase.**
  - Update exports, source/test documentation, and planning records.
  - Run focused and complete tests plus a phase review/fix cycle.

## Tests to add or run

- Add pure numerical contracts in `tests/test_sparse_phosphate_uptake.py`.
- Update integrated uptake/subcycling tests for the P5 call boundary.
- Retain P0–P4 unit, state, geometry, uptake, accounting, and sequencing tests.
- Run the complete suite before and after the phase review.

## Verification checklist

- [v] Defaults and validation match the agreed units and ranges.
- [v] Saturated hyphal density gives `t_diff` approximately 5.5 days.
- [v] Zero hyphal density gives infinite `t_diff` and `w_cont = 0`.
- [v] Dense/vanishing-gap hyphae give `w_cont = 1` without NaNs.
- [v] Stable `C_s` satisfies its analytical equation and physical bounds.
- [v] Sparse requests recover zero, no-resistance, and high-resistance limits.
- [v] Root-only uptake uses the sparse closure.
- [v] Both consumers use the same fungal-derived weight in mixed cells.
- [v] Competition is applied exactly once after blending and remains conservative.
- [v] Geometry-only coefficients are outside the soil substep loop.
- [v] Focused and complete suites pass; review has no blocking finding.

## Phase exit criteria

- Root-only, fungus-only, and mixed simulations all run through the blended
  closure without non-finite values.
- Sparse and continuous limiting cases match analytical expectations.
- Soil loss equals accepted plant-plus-fungal uptake at the existing unit
  boundary, with the shared inventory cap still exact.
- All tasks and checks are `[v]` after the phase-level review/fix pass.

## Phase notes

- Phase status: **verified** on 2026-07-21.
- The static reference time is a provisional exposure/lifetime proxy while
  absorbers remain indefinitely active; age dependence is explicitly deferred.
- The maintenance-P accounting issue remains unchanged and explicit.
- P5.1/P5.2 red evidence: the focused test module failed during collection
  because the sparse/blending functions did not yet exist. The implemented
  analytical primitives then passed `28 passed, 1 warning`, including the
  nominal `t_diff`, exact overlap limits, quadratic residuals, and JIT.
- P5.3 focused integration passed `84 passed, 1 warning`. The mixed-cell test
  independently reconstructs root and fungal sparse/continuous requests with
  one shared weight and applies the cap once; root-only, fungus-only, diffusion
  subcycling, validation, conservation, and pool credit tests are retained.
- The first and second complete verification passes each passed `182 passed,
  1 warning`; the warning is the unchanged upstream JAXopt deprecation.
  `git diff --check`, bytecode compilation, and function/test documentation
  coverage also passed.
- Phase review found and fixed a potential cancellation problem at very large
  positive `C_b - K_m - k`: `C_s` now selects the direct or rationalised
  quadratic expression by sign, with a dedicated large-concentration test.
  No blocking finding remains.
- Residual risks move to P6: the provisional `T_ref` and exponent require
  sensitivity analysis, and uptake/grid/timestep convergence and runtime have
  not yet been qualified. Local colony age remains a deferred extension.
