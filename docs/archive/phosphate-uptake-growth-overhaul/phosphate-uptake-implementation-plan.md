# Buffered Phosphate Transport and Root–Fungus Uptake Implementation Plan

**Status:** Implementation, numerical qualification, documentation, and the
post-implementation audit are complete. This plan is retained as a historical
record; current behaviour is described in
[`../../phosphate-model.md`](../../phosphate-model.md).

## Summary

Replace the current dimensionally inconsistent phosphate placeholder with a
conservative, axisymmetric soil-phosphate model coupled to plant-root and
fungal-hyphal growth. The model will store total labile phosphate amount in
each soil cell, derive soil-solution concentration through linear buffering,
transport phosphate by finite-volume diffusion, blend analytical sparse and
continuous uptake closures according to local hyphal density, and allocate the
accepted uptake conservatively between plant and fungus.

The implementation will retain the current JAX/JaxMARL environment interface
and the agreed growth-before-uptake credit-assignment order. It will support
plant-only, fungus-only, and mixed simulations.

Current unresolved decisions are maintained in
[`../../open-questions.md`](../../open-questions.md), and implemented module
boundaries are summarised in [`../../module-map.md`](../../module-map.md).

## Execution Route

`standard`

Execute the plan with `$rb-execute-plan` using its ordinary verified,
phase-by-phase workflow.

## Goals

- Make phosphate state, transport, uptake, competition, and organism-pool
  transfers dimensionally consistent.
- Guarantee non-negative cell inventories and conservative allocation of
  phosphate accepted from each cell.
- Represent soil on an axisymmetric `r-z` cylinder configured directly by
  maximum radius, maximum depth, and radial/depth intervals.
- Support root-only uptake, fungus-only uptake, and root–fungus competition.
- Couple realised, stoichiometrically limited biomass growth immediately to
  root and hyphal absorbing geometry.
- Use a computationally inexpensive analytical sparse-uptake closure and a
  smooth, density-dependent transition to continuous uptake.
- Establish unit, invariant, limiting-case, convergence, and performance tests
  suitable for scientific model development.

## Non-goals

- Kinetic sorption/desorption, nonlinear buffering, irreversible fixation, or
  mineral precipitation/dissolution.
- Water flow, advection, hydrodynamic dispersion, or changing water content.
- Explicit three-dimensional angular structure; the first implementation is
  axisymmetric.
- Explicit individual roots or hyphae, local depletion-profile state, or a
  per-cell radial PDE.
- Local colony-age tracking in the sparse-to-continuous transition.
- Root or hyphal senescence schedules, litter decomposition, P mineralisation,
  or recycling of organism P to soil.
- Multiple plants or fungi with separate spatial ownership maps.
- Final empirical calibration of the provisional soil, plant, fungal, or
  kinetic parameters.

## Users

- The model developer validating phosphate transport and uptake mechanisms.
- Researchers running plant-only, fungus-only, or plant–fungus MARL
  experiments.
- Future calibration work comparing simulation outputs with measured
  soil-solution P, tissue P, uptake, biomass, and spatial extent.

## Requirements

### State and units

- Use centimetres for spatial calculations, seconds for physical rates,
  days for the user-facing biological timestep, grams of dry biomass,
  grams of carbon, and milligrams of organism P.
- Store soil `M_labile` in `µmol P` per axisymmetric cell.
- Derive solution concentration as
  `C = M_labile / (V_cell * (theta + B))` in `µmol cm^-3`.
- Convert accepted uptake at the soil–organism boundary with
  `1 µmol P = 0.0309738 mg P`.
- Keep plant and fungal `gamma_C` and `gamma_P` parameters distinct.
- Track P lost through mortality and other explicit biological exports so
  extended whole-system mass balance remains diagnosable.

### Domain and initial state

- Use an axisymmetric `r-z` grid with provisional maximum radius `50 cm`,
  maximum depth `100 cm`, and radial/depth intervals of `0.1 cm`.
- Generate edges at the requested intervals and end exactly at each maximum;
  if an extent is not divisible by its interval, shorten only the final cell.
- Initialise solution concentration to `1 µM` in the upper `25 cm` and zero
  below, volume-averaging a cell crossed by the boundary.
- Map concentration to labile amount using `theta = 0.3` and `B = 239`.
- Reproduce the analytical initial inventory implied by the configured
  cylindrical extents; with the defaults this is approximately
  `46,986.45 µmol = 1,455.35 mg P`.
- Apply no-flux conditions at the surface, bottom, and outer radius, and
  cylindrical symmetry at `r = 0`.

### Growth and absorbing geometry

- Treat realised growth from the essential-resource growth function as the
  sole source of new structural geometry.
- Use plant
  `gamma_C = 0.402 g C g^-1 dry biomass`,
  `gamma_P = 1.92 mg P g^-1 dry biomass`,
  `k_root = 0.62`, and
  `SRL = 25,434.3 cm g^-1 root dry mass` provisionally.
- Increase root length by
  `delta_L_root = SRL * k_root * delta_G_plant`.
- Use fungal
  `gamma_C = 0.5 g C g^-1 dry biomass`,
  `gamma_P = 40 mg P g^-1 dry biomass`, and
  `M_C = 0.1155 g C cm^-3 fungal tissue` provisionally.
- Increase hyphal length by
  `delta_L_hypha = gamma_C * delta_G_fungus /
  (M_C * pi * r_h^2)`.
- Use `r_root = 0.01 cm` and `r_h = 5e-4 cm` provisionally.
- Conserve total represented length when distributing root and fungal length
  onto the grid.
- Retain stacked-disc root distribution and saturated hemispherical fungal
  expansion, with `lambda_sat = 168.75 cm^-2`.

### Transport and uptake

- Transfer labile P conservatively between radial and vertical neighbours
  using finite-volume flux
  `q = -D_l * theta * f_l * grad(C)`.
- Use `D_l = 1e-5 cm^2 s^-1`, `f_l = 0.308`, and the implied
  `D_app approximately 3.86e-9 cm^2 s^-1`.
- Use separate root and fungal interfaces with provisional shared kinetics:
  `J_max = 3.26e-6 µmol cm^-2 s^-1` and
  `K_m = 5.8e-3 µmol cm^-3`.
- Convert length density to lateral absorbing surface using
  `A = 2 * pi * radius * length`; exclude end caps.
- Implement the agreed analytical cylindrical sparse closure, including a
  numerically stable physical root for `C_s`.
- Use local hyphal density to calculate `t_diff`, `Omega`, and the shared
  continuous weight
  `w_cont = Omega^p / (1 + Omega^p)`, with provisional
  `T_ref = 1 day` and `p = 2`.
- Force `w_cont = 0` in cells without hyphae so plant-only uptake remains
  sparse.
- Blend sparse and continuous requests before applying competition.
- Calculate plant and fungal requests from the same post-diffusion bulk
  concentration and proportionally scale their sum to available cell
  inventory.

### Timestep order

- Use only pools present at the beginning of the biological step for
  allocation and growth.
- Apply realised growth and update absorbing geometry before soil uptake.
- Run diffusion, uptake blending, and competition with the post-growth
  geometry.
- Credit accepted P after growth so it becomes available at the next step.
- If the biological timestep exceeds the finite-volume diffusion stability
  ceiling, subcycle the complete fixed-geometry soil update and accumulate
  accepted uptake.

## Assumptions

- Linear buffering has zero intercept and equilibrates instantaneously within
  each cell.
- All represented root and hyphal length is initially uptake-active.
- Root mass fraction is an acceptable provisional proxy for marginal root
  allocation, and GRooT fine-root SRL can be combined with it.
- The converted two-dimensional fungal saturation density is an acceptable
  provisional three-dimensional length density.
- `T_ref = 1 day` is a static exposure-time proxy while roots and hyphae have
  indefinite active lifetimes.
- The 4% fungal P mass fraction is an upper-bound provisional growth cost, not
  a validated representative AM-fungal concentration.
- Existing initial organism-pool values will be interpreted in the newly
  documented units until separately calibrated.

## Constraints

- Keep state and calculations compatible with JAX transformation and JIT
  compilation; avoid Python-side data-dependent branches in model kernels.
- Preserve the JaxMARL-facing reset, step, observation, reward, and action
  interfaces unless a tested migration is necessary.
- Use an `r-z` state shape for active soil and density fields; remove the
  inactive angular dimension from the physical calculation rather than paying
  its memory and compute cost.
- Target approximately `500 x 1000 = 500,000` cells at `0.1 cm` resolution,
  while keeping small grids available for unit and integration tests.
- Avoid iterative nonlinear solves and per-cell radial submodels.
- Preserve unrelated working-tree changes and introduce the model in small,
  reviewable phases.
- The current shell does not expose `pytest`; environment setup and a clean
  baseline test run are required before implementation begins.

## Proposed Approach

### Model boundaries

- `params.py` and organism trait modules own explicit, documented parameters
  and units.
- `phosphate_grid.py` owns axisymmetric edges, cell volumes, face geometry,
  topsoil fractions, concentration/amount conversions, and stability limits.
- A focused phosphate transport module owns conservative diffusion.
- A focused phosphate uptake module owns kinetics, surface concentration,
  sparse/continuous requests, blending, and competition.
- Root and mycelium modules own biomass-to-length conversion and conservative
  spatial density fields.
- `BaseMycorMarl` owns biological sequencing and transfers scalar accepted
  uptake into organism pools.
- `State` stores canonical cell amounts, geometry fields, organism pools, and
  cumulative P-loss diagnostics; temporary concentration and `C_s` fields are
  derived rather than stored.

### Migration strategy

- Establish unit-labelled parameter and geometry helpers before changing the
  environment state.
- Introduce the new soil amount state in one end-to-end walking skeleton:
  initialise, derive concentration, request uptake, cap uptake, subtract
  amount, convert accepted uptake, and credit scalar organism pools.
- Add diffusion and sparse/continuous physics behind the same stable soil-step
  interface.
- Remove the legacy concentration-as-inventory path once the walking skeleton
  passes conservation and integration tests; do not maintain two long-lived
  phosphate models.
- Keep every phase independently testable and preserve the previous passing
  phase as the rollback point.

## Implementation Phases

### Baseline, terminology, and unit contracts

Detailed phase checklist:
[`phases/baseline-and-unit-contracts.md`](phases/baseline-and-unit-contracts.md).

- Restore or document the project test environment and run the existing suite.
- Record baseline failures separately from new work.
- Define canonical symbols, state shapes, units, conversion constants, and
  parameter names.
- Add focused tests for unit conversions, reference kinetics, buffer
  identities, and absorbing surface area.
- Decide the compatibility migration from current
  `carbon_per_growth`/`phosphorus_per_growth` names to organism-specific
  `gamma_C`/`gamma_P`.

**Gate:** existing tests are runnable; unit-contract tests pass; no physics
state has yet been migrated.

### Axisymmetric domain and canonical labile-phosphate state

Detailed phase checklist:
[`phases/axisymmetric-labile-phosphate-state.md`](phases/axisymmetric-labile-phosphate-state.md).

- Generate exact radial and depth edges, annular cell volumes, radial and
  vertical face geometry, and partial topsoil occupancy.
- Migrate active soil and length-density fields to `(n_r, n_z)`.
- Initialise `M_labile` from solution concentration and linear buffering.
- Expose pure helpers to recover solution concentration.
- Add state diagnostics for cumulative P losses and exports.
- Test exact configured boundaries, interval behaviour, analytical inventory,
  partial layers, zero-P subsoil, shape invariants, and concentration–amount
  round trips.

**Gate:** reset produces the correct two-dimensional canonical amount state
and the inventory implied by its configured extents without changing
biological pools.

### Realised growth to conservative root and fungal geometry

Detailed phase checklist:
[`phases/biomass-to-absorbing-geometry.md`](phases/biomass-to-absorbing-geometry.md).

- Replace placeholder density updates with total-length calculations driven
  by realised growth.
- Correct root allocation so `k_root` is applied exactly once.
- Implement stacked-disc root length density with uniform provisional
  `lambda_root = 1.0 cm cm^-3` inside occupied discs and depth-dependent radii
  derived from beta-weighted layer lengths.
- Implement fungal carbon-to-volume-to-length conversion and saturated
  hemispherical occupancy.
- Handle zero biomass, mortality-related biomass decline, domain clipping,
  and colonies reaching domain boundaries explicitly.
- Test total represented length, partial-cell occupancy, monotonic extent,
  and the provisional per-gram reference calculations.

**Gate:** integrated growth updates produce post-growth density fields whose
volume integrals equal analytically expected root and in-domain hyphal length;
fungal structure outside the hemisphere–soil intersection is explicitly
clipped, with saturation density never exceeded.

### Conservative continuous uptake and root–fungus competition

Detailed phase checklist:
[`phases/continuous-uptake-and-competition.md`](phases/continuous-uptake-and-competition.md).

- Implement Michaelis–Menten surface flux using derived solution
  concentration and absorbing surface area.
- Calculate simultaneous continuous root and fungal requests.
- Apply the shared per-cell inventory cap once and aggregate accepted amounts
  into scalar consumer totals.
- Convert accepted `µmol P` to `mg P` only when crediting organism pools.
- Integrate the walking skeleton into the growth-before-uptake environment
  sequence.
- Track reproduction export and mortality loss consistently with the mass
  balance.
- Test plant-only, fungus-only, mixed, zero-demand, zero-inventory,
  oversubscribed, and consumer-symmetry cases.

**Gate:** end-to-end steps are non-negative and close soil-plus-organism-plus-
loss P balance in a common unit.

### Conservative axisymmetric diffusion and subcycling

Detailed phase checklist:
[`phases/conservative-diffusion-and-subcycling.md`](phases/conservative-diffusion-and-subcycling.md).

- Implement radial and vertical finite-volume fluxes with closed boundaries.
- Calculate the exact cellwise explicit stability ceiling and enforce the
  `0.8` safety factor.
- Add fixed-geometry soil subcycling when required by the biological
  timestep.
- Recover concentration after every diffusion and uptake substep.
- Test uniform-field invariance, pairwise flux antisymmetry, no-flux
  boundaries, diffusion-only conservation, radial-axis behaviour,
  non-negativity within the stability limit, and a manufactured or
  independently calculated small-grid case.

**Gate:** diffusion conserves total P and converges against a refined
timestep on representative small grids.

### Analytical sparse closure and smooth uptake-regime transition

Detailed phase record:
[`phases/sparse-continuous-uptake-transition.md`](phases/sparse-continuous-uptake-transition.md).

- Implement root and fungal territory radii, `R_eff`, logarithmic resistance,
  stable `C_s`, sparse requests, and diagnostics.
- Implement static fungal-overlap `t_diff`, shared `w_cont`, and request
  blending.
- Cache geometry-only coefficients and recompute concentration-dependent
  quantities each soil step.
- Handle no absorber, no hyphae, zero concentration, vanishing gap, and very
  large or small resistance without NaNs.
- Test analytical residuals, `0 <= C_s <= C_b`, sparse and continuous limits,
  `w_cont` limits, the nominal `t_diff approximately 5.5 days` result, and
  root-only sparse behaviour.

**Gate:** all three simulation modes run through the blended closure, and
limiting cases reproduce their analytical expectations.

### Numerical convergence and performance qualification

Detailed phase record:
[`phases/numerical-qualification-and-performance.md`](phases/numerical-qualification-and-performance.md).

- Run concentration scenarios `0.1, 0.3, 1, 3, 10 µM`.
- Run root-only, fungus-only, and mixed timestep studies over candidate
  biological timesteps, subject to the stability ceiling.
- Compare `0.1`, `0.05`, and `0.025 cm` grids on uptake, consumer shares,
  biomass, colony extent, and P balance.
- Record `C_s/C_b`, overlap weights, capped-demand frequency, and cumulative
  loss diagnostics.
- Benchmark JIT compilation, memory use, and per-step runtime on reduced and
  target grids; project the cost of a one-year target-domain simulation.
- Select the coarsest grid and largest timestep meeting the agreed accuracy
  criteria; do not silently change provisional scientific parameters to make
  the model faster.

**Gate:** the selected production configuration has documented accuracy,
stability, conservation, memory, and runtime evidence.

### Documentation, provenance, and handoff

Detailed phase record:
[`phases/documentation-provenance-and-handoff.md`](phases/documentation-provenance-and-handoff.md).

- Document equations, units, step order, state meanings, configuration, and
  expected diagnostics.
- Preserve source and publisher provenance for every provisional parameter.
- Record unvalidated assumptions, especially fungal P mass fraction,
  root-radius sourcing, root allocation/SRL compatibility, and MDPI-derived
  plant P.
- Provide small reproducible plant-only, fungus-only, and mixed examples.
- Remove obsolete placeholder code and update the implementation status.

**Gate:** another developer can configure, run, validate, and interpret the
new model without relying on conversation history.

## Validation Plan

### Unit and invariant tests

- Concentration, amount, mass, length, area, flux, and timestep conversions.
- Buffer-factor, dissolved-fraction, `D_app`, kinetic, surface-area, initial
  inventory, root-length, and hyphal-length reference values.
- Non-negative amounts and pools.
- Exact request accounting:
  `M_before - M_after = U_root + U_fungus`.
- Soil-to-organism conversion and extended whole-system P balance.
- Length-density integrals equal total root and hyphal length.

### Numerical tests

- Closed-domain diffusion conservation and uniform-field invariance.
- Axisymmetric radial geometry, including the zero-area central face.
- Stable explicit steps at or below the safety limit and detected rejection or
  subcycling above it.
- Analytical sparse-closure residual and stable quadratic evaluation.
- Timestep and grid convergence with quantitative error reports.

### Integration tests

- Reset and one-step transitions on small grids.
- Plant-only survival and uptake with no fungus.
- Fungus-only uptake with no plant roots.
- Mixed uptake below and above the inventory cap.
- Correct growth-before-uptake credit assignment.
- Death and reproduction exports reflected in diagnostics.
- JIT compilation and deterministic repeated runs.

### Acceptance tolerances

- Use relative tolerance `1e-6` for small deterministic float32 unit fixtures
  where conditioning permits, relaxing only with a documented numerical
  reason.
- Require closed-domain mass-balance relative error no greater than `1e-5`
  in integration tests.
- Require represented-length relative error no greater than `1e-5`.
- Provisionally accept no more than `5%` change in selected scientific outputs
  between the retained grid/timestep and the next finer reference.

## Risks

- **Fungal P cost:** `40 mg P g^-1` is a reported maximum and may severely
  suppress fungal growth. Mitigation: keep configurable, expose limitation
  diagnostics, and validate the underlying literature before calibration.
- **Root trait mismatch:** GRooT root mass fraction and fine-root SRL are
  separate species aggregates and may overstate absorbing length when
  combined. Mitigation: document, sensitivity-test, and later replace with
  matched cultivar/ontogenetic data.
- **Strong buffering:** `B = 239` creates a very small dissolved fraction and
  large inventory. Mitigation: explicit unit tests and no double use of
  `D_app` in the amount flux.
- **Sparse-closure conditioning:** logarithms and the quadratic root can lose
  precision near degenerate limits. Mitigation: stable algebra, explicit
  branches expressed with JAX-safe operations, and limiting-case tests.
- **Growth-front approximation:** the static transition treats newly occupied
  cells as density-equilibrated. Mitigation: report as accepted first-model
  error and retain colony-age modelling as an extension.
- **Coarse-step bias:** new surface is exposed for the whole discrete step.
  Mitigation: convergence testing and an optional midpoint-geometry fallback.
- **Axisymmetric migration (resolved in P1):** active soil and density state is
  now `(n_r, n_z)` with no inactive angular dimension; reset, observations,
  and step integration are covered by tests.
- **Target-grid cost:** several temporary arrays over approximately 500,000
  cells can raise compile time and memory pressure. Mitigation: fuse kernels
  where clear, cache geometry, avoid stored derived fields, and benchmark
  before annual production runs.
- **Dirty working tree:** existing uncommitted and untracked work overlaps the
  target files. Mitigation: inspect ownership before every phase, avoid
  destructive Git operations, and keep changes narrowly scoped.

## Success Criteria

- The canonical soil state is labile P amount per axisymmetric cell, with
  solution concentration derived consistently everywhere.
- Reset reproduces the specified `1 µM`, upper-`25 cm` field and the analytical
  inventory implied by the configured cylindrical domain.
- Diffusion conserves P under closed boundaries and uptake is the only soil
  sink.
- Accepted uptake never exceeds cell inventory and is credited to the correct
  scalar organism pool in milligrams.
- Plant-only, fungus-only, and mixed simulations run without special-case
  failures or NaNs.
- Growth produces analytically consistent root and hyphal length and surface
  area.
- Sparse/continuous blending reaches correct limiting behaviour and matches
  the nominal overlap-timescale checks.
- Extended P balance includes living pools, soil inventory, mortality losses,
  and reproduction exports.
- The selected grid and timestep meet the provisional `5%` convergence
  criterion and have a documented annual-run compute estimate.
- Tests, equations, parameter provenance, limitations, and run examples are
  durable in the repository.

## Open Questions

- Does Bisot et al.'s underlying literature support using `4%` as a default
  fungal P mass fraction rather than only an extreme upper bound?
- Can the provisional carrot tissue-P value from the MDPI journal
  *Agronomy* be corroborated in an independent source?
- What measured root radius, ontogenetic root allocation, and matched
  fine-root SRL should replace the provisional carrot traits?
- Should initial plant and fungal resource-pool values be calibrated before
  scientific production runs?
- Does convergence require midpoint rather than full-step exposure of new
  absorbing geometry?
- Should an uptake-based substep trigger supplement the diffusion stability
  ceiling when demand is strongly inventory-limited?
- What physical fate should be assigned to `maint_p_used`? It is currently
  removed from the free organism pool without entering biomass, soil, or a
  loss/export diagnostic. Preserve C-and-P maintenance until this is resolved;
  do not silently reinterpret maintenance as carbon-only.
- When turnover is introduced, should local colonisation age or active-length
  fraction control the sparse-to-continuous transition first?
