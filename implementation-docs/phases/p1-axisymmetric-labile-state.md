# P1 — Axisymmetric domain and canonical labile-P state

## Phase goal

Make environment reset produce a two-dimensional axisymmetric soil domain
whose sole phosphate state is total buffered labile P amount per cell.

## Walking skeleton

Configuration in centimetres and micromolar solution concentration flows
through exact grid construction, partial-topsoil occupancy, linear buffering,
and state initialisation. A caller can reset the environment, inspect
`soil_labile_p`, and recover the configured solution concentration. Uptake and
diffusion are intentionally unavailable until their later phases; the legacy
dimensionally inconsistent uptake path must fail clearly rather than mutate
the new amount state.

## Scope

- Radial and depth edges generated from configured intervals and maxima.
- Annular cell volumes and radial/vertical face areas.
- Fractional occupancy for a topsoil boundary crossing a cell.
- Reversible concentration/amount conversion under linear buffering.
- Axisymmetric `(n_r, n_z)` phosphate and length-density state fields.
- Zero-initialised cumulative P mortality-loss and reproduction-export
  diagnostics in organism-pool units (`mg P`).
- Reset integration and configuration validation.

## Non-scope

- Uptake requests, competition, or organism-pool credits.
- Diffusive transport, stability limits, or subcycling.
- Growth-driven changes to root or fungal geometry.
- Sparse/continuous uptake blending.
- Mortality/export accounting updates beyond zero-initialised diagnostics.

## Dependencies and existing stack

- Verified P0 dimensional helpers and Python/JAX/pytest stack.
- Existing JAX/Flax immutable state conventions.
- Existing uncommitted refactor work is the starting state and must be
  preserved.
- No new dependency or framework is proposed.

## Configuration migration

P1 uses explicit centimetre- and unit-bearing names:

- `soil_radius_cm`, `soil_depth_cm`
- `radial_interval_cm`, `depth_interval_cm`
- `topsoil_depth_cm`
- `initial_solution_p_um`
- `theta_water`, `buffer_power`

The pre-release legacy names (`r_max`, `r_size`, `theta_size`,
`max_soil_depth`, `n_layers`, `topsoil_depth`, `initial_soil_p`,
`volum_water_content`, and `soil_buffer_power`) are removed atomically from
code and tests rather than retained as ambiguous dual configuration fields.

## Granular task checklist

- [v] **P1.1 — Complete and validate axisymmetric grid geometry.**
  - Generate interval-spaced edges that end exactly at physical maxima,
    shortening only the final cell when necessary.
  - Compute annular volumes plus radial and vertical face areas.
  - Compute volume-equivalent topsoil fractions.
  - Reject invalid scalar geometry before JAX kernels are constructed.
- [v] **P1.2 — Add buffered concentration/amount transformations.**
  - Convert configured `µM` to kernel concentration `µmol cm^-3`.
  - Initialise `M_labile = V_cell * (theta + B) * C`.
  - Recover solution concentration without storing it.
  - Cover zero concentration and round-trip invariants.
- [v] **P1.3 — Migrate configuration, state, and reset to the P1 slice.**
  - Use provisional defaults: maximum radius `50 cm`, maximum depth `100 cm`,
    and radial/depth intervals `0.1 cm`.
  - Use `1 µM` in the upper `25 cm`, `theta = 0.3`, and `B = 239`.
  - Store only `soil_labile_p` with `(n_r, n_z)` root/hyphal fields.
  - Store grid geometry on the environment and initialise P-loss/export
    diagnostics to zero without changing biological pools.
  - Make the legacy soil evolution path fail with a clear P3 handoff message.
- [v] **P1.4 — Add reset-level integration and invariant tests.**
  - Test exact configured boundaries, interval behaviour, and the analytical
    inventory implied by the configured cylinder.
  - Test partial layers, zero-P subsoil, shape invariants, round trips, and
    unchanged initial biological pools.
  - Test reset under JIT and explicit rejection of invalid geometry/buffering.
  - Test that unsupported legacy uptake cannot silently consume amount state.

## Tests to add or run

- Extend `tests/test_axisymmetric_geometry.py` for grid/face invariants.
- Add `tests/test_labile_phosphate_state.py` for conversion and reset.
- Run focused P1 tests after each red/green increment.
- Run the complete suite before edits, after implementation, during the
  second verification pass, and after review fixes.

## Verification checklist

- [v] Exact maxima, interval behaviour, cell volumes, and face shapes are
  verified.
- [v] Partial topsoil occupancy and zero subsoil are verified.
- [v] Concentration/amount round trips are verified.
- [v] Analytical configured-domain inventory is within documented float32
  tolerance.
- [v] Reset produces only `(n_r, n_z)` active soil/density fields.
- [v] Initial biological pools are unchanged.
- [v] Unsupported old uptake fails clearly.
- [v] Focused and complete suites pass.
- [v] Diff review finds no P2–P5 physics implemented early.

## Phase exit criteria

- Reset produces the correct two-dimensional canonical amount state and
  analytical configured-domain total.
- Solution concentration is always derived from amount, volume, and buffering.
- Biological pool initial values are unchanged.
- Invalid configuration and the obsolete uptake path fail explicitly.
- All tasks and verification checks are `[v]`, followed by phase review/fix.

## Phase notes

Function-level and test-level documentation:
`implementation-docs/phosphate-foundations-reference.md`.

- Phase status: **verified** on 2026-07-21.
- Pre-P1 baseline on 2026-07-21: `47 passed, 1 warning` in 33.43 seconds.
- The warning is the unchanged upstream JAXopt deprecation notice.
- P1.1/P1.2 red evidence: focused collection failed because the new grid and
  buffered-state functions did not exist. The first implementation run also
  caught and fixed a malformed soil-package export list before product tests
  could execute.
- P1.1/P1.2 green evidence: `17 passed, 1 warning` in 29.97 seconds.
- P1.3/P1.4 red evidence: collection failed because `EnvConfig` did not yet
  accept the then-planned grid configuration fields.
- P1.3/P1.4 green evidence: geometry, labile-state, and environment integration
  tests passed `34 passed, 1 warning` in 32.22 seconds.
- Post-implementation regression evidence: `67 passed, 1 warning` in 30.79
  seconds.
- Second verification pass: focused P1/integration tests passed `34 passed, 1
  warning`; the independently rerun complete suite passed `67 passed, 1
  warning` in 31.24 seconds.
- The grid configuration was revised on 2026-07-21 at the user's request:
  exact-volume preservation and explicit cell counts were removed in favour
  of maximum radius/depth plus interval sizes. The revised defaults produce a
  `500 x 1000` grid and an analytical initial inventory of approximately
  `46,986.45 µmol P`.
- Revision verification: the interval/maxima red tests initially failed at
  collection because the new API did not exist; after implementation the
  focused geometry/reset/environment suite passed `34 passed, 1 warning` and
  the complete suite passed `67 passed, 1 warning`.
- Static migration audit found no remaining `state.soil_p`, angular-grid, or
  legacy soil-configuration references. `git diff --check` is clean after
  removing trailing whitespace in the touched environment module.
- Phase completion review found one actionable boundary defect: obsolete 3D
  concentration construction and dimensionally incomplete uptake/competition
  helpers remained publicly accessible after the canonical state migration.
  Those dead legacy entry points were removed; no P3 uptake physics was added.
- Post-review verification: focused P1/integration tests passed `34 passed, 1
  warning`; the complete suite passed `67 passed, 1 warning` in 36.64 seconds;
  `git diff --check` remained clean. No blocking review findings remain.
- Accepted residual risk: reset is the only complete P1 workflow. A biological
  step deliberately stops at soil uptake until P3; P2 must first replace the
  remaining placeholder density-update functions. Loss/export diagnostics are
  zero-initialised but are not accumulated until the biological accounting
  phase. Target-grid runtime and memory remain for P6 qualification.
