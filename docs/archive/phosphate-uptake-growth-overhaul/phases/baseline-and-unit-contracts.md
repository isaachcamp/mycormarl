# Baseline, terminology, and unit contracts

## Phase goal

Establish a runnable test baseline and a single, explicit dimensional contract
for phosphate calculations before migrating any physical state.

## Walking-skeleton contribution

P0 supplies the pure numerical contract used by the later end-to-end soil
phosphate slice: configured solution concentration, buffered labile amount,
surface uptake, and organism-pool mass must be connected by named conversions
with reference tests. It deliberately does not migrate `soil_p` or environment
state.

## Scope

- Confirm and document the existing Python and pytest environment.
- Define canonical phosphate symbols, units, state-shape conventions,
  conversion constants, and parameter names.
- Add pure helpers for unit conversion, linear-buffer identities,
  Michaelis–Menten surface flux, and cylindrical lateral area.
- Add focused scientific reference and invariant tests.
- Decide how legacy growth-cost names migrate to `gamma_c` and `gamma_p`.

## Non-scope

- Changing `soil_p` from concentration to amount.
- Changing environment reset or step sequencing.
- Adding axisymmetric diffusion, uptake competition, or organism-pool credit.
- Changing root or fungal growth geometry.
- Calibrating provisional parameters.

## Dependencies and existing stack

- Python `>=3.12`, currently `.venv/bin/python` 3.12.9.
- JAX and JAX-compatible pure numerical helpers.
- pytest, currently 9.0.2.
- Existing package layout under `mycormarl/mycormarl`.
- No new dependency or framework is proposed in P0.

## Canonical contract

| Symbol / name | Meaning | Unit / shape |
|---|---|---|
| `C_solution` | Soil-solution inorganic phosphate concentration | `µmol cm^-3`; configuration may use `µM` |
| `M_labile` | Total reversibly labile phosphate in one soil cell | `µmol P cell^-1`, future `(n_r, n_z)` state |
| `V_cell` | Axisymmetric bulk-soil cell volume | `cm^3` |
| `theta_water` | Volumetric water content | `cm^3 water cm^-3 bulk soil` |
| `buffer_power` | `dC_sorbed,bulk / dC_solution` | effective `cm^3 water cm^-3 bulk soil` |
| `J_max` | Maximum surface uptake flux | `µmol P cm^-2 s^-1` |
| `K_m` | Michaelis–Menten half-saturation concentration | `µmol P cm^-3` |
| `length_density` | Uptake-active root or hyphal length per bulk-soil volume | `cm cm^-3`, future `(n_r, n_z)` field |
| `absorbing_area` | Lateral cylindrical absorbing area; end caps excluded | `cm^2` |
| `dt_physical` | Physical transport/uptake interval | `s` |
| organism P pools | Plant or fungal phosphorus mass | `mg P` per organism |
| `gamma_c_plant`, `gamma_c_fungus` | Carbon required per realised dry-biomass growth | `g C g^-1 dry biomass` |
| `gamma_p_plant`, `gamma_p_fungus` | Phosphorus required per realised dry-biomass growth | `mg P g^-1 dry biomass` |

Configuration and code use lowercase snake-case names. Mathematical
documentation may retain `C`, `M_labile`, `J_max`, `K_m`, `theta`, `B`, and
`gamma` notation where the mapping is stated.

## Growth-parameter compatibility decision

`gamma_c` and `gamma_p` are the canonical trait names. The current
`carbon_per_growth` and `phosphorus_per_growth` fields remain unchanged during
P0 so this phase cannot alter biological behaviour. In P2 they will be renamed
atomically across trait constructors, environment call sites, fixtures, and
configuration ingestion. The codebase is pre-release and has no documented
external configuration compatibility contract, so two writable aliases will
not be retained: that would create two sources of truth. If a legacy
configuration loader is introduced before P2, it may translate legacy keys at
the input boundary while rejecting files that provide both old and new names.

## Granular task checklist

- [v] **P0.1 — Establish the baseline test environment.**
  - Confirm the local Python and pytest versions.
  - Run the complete existing suite before P0 product edits.
  - Record failures separately from new work.
  - Evidence: Python 3.12.9; pytest 9.0.2; `29 passed, 1 warning` on
    2026-07-20. The warning is the upstream JAXopt deprecation notice.
- [v] **P0.2 — Add the pure phosphate unit-contract helpers and constants.**
  - Keep helpers JAX-compatible and independent of environment state.
  - Include `µM -> µmol cm^-3`, `µmol P -> mg P`, days-to-seconds,
    linear-buffer capacity/retardation/dissolved fraction,
    Michaelis–Menten surface flux, and cylindrical lateral area.
  - Fail clearly for invalid scalar parameters where checks can occur outside
    transformed model kernels; do not add silent numerical fallbacks.
- [v] **P0.3 — Add focused scientific contract tests.**
  - Write the reference tests before implementing P0.2.
  - Cover exact conversions, provisional buffer identities, the `1 µM`
    kinetic reference, root and hyphal one-centimetre surface areas,
    zero-concentration/zero-length limits, and JIT compatibility.
  - Use relative tolerance `1e-6` where float32 conditioning permits.
- [v] **P0.4 — Verify and document the canonical naming/state contract.**
  - Confirm the table above agrees with the top-level plan and modelling TODO.
  - Confirm the staged `carbon_per_growth`/`phosphorus_per_growth` migration
    has one canonical destination and no dual writable fields.
  - Confirm no physical soil or organism state was migrated in P0.

## Tests to add or run

- `tests/test_phosphate_unit_contracts.py`
- Focused P0 contract tests.
- Complete pytest suite before edits, after implementation, during the second
  verification pass, and after any review fix.

## Verification checklist

- [v] Focused contract tests pass.
- [v] Complete repository suite passes.
- [v] Reference values agree with independently evaluated dimensional
  calculations.
- [v] Helpers are JIT-compatible where intended.
- [v] Diff review confirms P0 did not migrate physics state or change
  environment sequencing.
- [v] Every granular task has automated or explicit verification evidence.

## Phase exit criteria

- The existing suite remains runnable with baseline failures distinguished
  from P0 regressions.
- Unit-contract tests pass at the documented tolerances.
- Canonical symbols, shapes, units, constants, and parameter names have one
  durable definition.
- The legacy growth-cost migration strategy is explicit.
- No physics state has been migrated.

## Phase notes

Function-level and test-level documentation:
the current [`../../../module-map.md`](../../../module-map.md).

- Phase status: **verified** on 2026-07-20.
- P0 baseline: `29 passed, 1 warning` in 33.32 seconds.
- Accepted baseline warning: JAXopt announces that it is no longer maintained;
  P0 neither adds nor changes this dependency.
- P0.2 implementation: added JAX-compatible pure helpers in
  `mycormarl/mycormarl/soil/phosphate_units.py` and exposed them through the
  soil package. Scalar parameter validation is deliberately separate from
  transformed numerical kernels.
- P0.3 red evidence: the focused test initially stopped at collection with
  `ModuleNotFoundError: mycormarl.soil.phosphate_units`.
- P0.3 green evidence: `14 passed, 1 warning` in 26.04 seconds.
- Post-implementation regression evidence: `43 passed, 1 warning` in 28.01
  seconds. The warning remains the unchanged JAXopt deprecation notice.
- P0.4 review: the contract agrees with the top-level plan and modelling TODO;
  P0 introduces no `MycorMarlState`, reset, step, diffusion, uptake,
  competition, or growth-geometry migration.
- Second verification pass: focused `14 passed, 1 warning`; complete
  `43 passed, 1 warning`. Independent scalar calculations reproduced capacity
  `239.3`, retardation `797.6666667`, dissolved fraction `0.0012536565`,
  flux `4.794117647e-7 µmol cm^-2 s^-1`, hyphal area
  `0.00314159265 cm^2`, and root area `0.0628318531 cm^2`.
- Phase completion review found one actionable gap: non-finite scalar
  configuration values were rejected by the validators but not covered by
  tests. NaN and infinity cases were added for both buffer and kinetic
  validation.
- Post-review verification: focused `18 passed, 1 warning`; complete
  `47 passed, 1 warning`. No blocking findings remain.
- Accepted residual risk: P0 documents but does not yet wire configuration
  validation into environment construction, and it intentionally leaves the
  legacy uptake implementation and growth-cost field names in place for later
  phases. The physical kernel contract assumes non-negative concentration.
