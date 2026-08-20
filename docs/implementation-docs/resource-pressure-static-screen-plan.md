# Resource-pressure static-screen implementation and execution plan

## Summary

This plan defines the maintenance-only reset-pool contract and the subsequent
static-policy resource-pressure screens. The first 48-condition screen was a
discovery sample at uniform `1 µM` P. The follow-up expands the design to trade
fractions and a range of initial P concentrations while retaining deterministic,
auditable sampling rather than enumerating every condition in one run.

## Goals

- Derive unspecified reset C and P pools from exactly one next-step maintenance
  requirement, while preserving explicit numeric overrides.
- Screen plant/fungus carbon maintenance, plant/fungus P-loss maintenance,
  initial AMF biomass, bilateral trade fractions, and initial solution P.
- Record biomass, growth, uptake, fixation, geometry, and timestep limitation
  diagnostics under static full-growth policies.
- Use the screen to identify robust candidate regions before any learned-policy
  or reproductive-fitness experiment.

## Non-goals

- No PPO training or reproductive-fitness endpoint.
- No change to the current within-step reserve/maintenance ordering in this
  screen; its timing limitation remains separately documented.
- No claim that a static-policy biomass optimum is an ecological optimum.

## Fixed protocol

- Mode: `mixed`; horizon `120 d`; `dt = 0.025 d`.
- Grid: radius `20 cm`, depth `60 cm`, resolution `0.1 cm`.
- Policies: plant `[trade, 1, 0, 0]`, fungus `[trade, 1, 0, 0]`.
- Initial P concentrations: `0.1, 0.3, 0.5, 0.7, 0.9 µM` (0.1 to 1.0 in
  increments of 0.2; the arithmetic sequence stops at 0.9).
- Primary endpoint: final living plant biomass.
- Secondary endpoints: fungal biomass, gross growth, direct P uptake, C
  fixation, final geometry, and C/P limitation traces.

## Factor levels

- Plant and fungus `kappa_c`: five logarithmic multipliers, `0.01` through the
  former `0.693145` level; the upper `2x` level is excluded as requested.
- Plant and fungus `kappa_p`: zero plus six logarithmic multipliers from
  `0.01` to `2.0`.
- Initial AMF biomass: six logarithmic multipliers from `1x` to `100x`.
- Plant-to-fungus trade: `0.05, 0.10, 0.15, 0.20`.
- Fungus-to-plant trade: `0.5, 0.6, 0.7, 0.8`.
- Initial P: the five levels listed above.

## Condition count

The full Cartesian design contains:

```text
5 plant-kappa_c × 5 fungus-kappa_c
× 7 plant-kappa_p × 7 fungus-kappa_p
× 6 AMF-biomass × 4 plant-trade × 4 fungus-trade
× 5 initial-P levels = 588,000 conditions
```

This is a design-space size, not a recommendation to execute all conditions.
The practical run should use independent Latin-hypercube/permuted balanced
columns and a declared sample budget. If the focused trade arm fixes both
`kappa_p` values to zero, its corresponding design space is
`5 × 5 × 6 × 4 × 4 × 5 = 12,000` conditions.

## Sampling and selection

1. Preserve the checked-in 48-condition discovery bundle as provenance.
2. Generate a reproducible discrete Latin-hypercube sample for the expanded
   design, recording seed, levels, and every sampled row.
3. Execute each row through the public static-control seam.
4. Retain completed conditions with non-negative fungal net biomass and rank by
   final living plant biomass.
5. Re-run a small retained set at neighboring P levels and inspect limitation
   traces before selecting any learned-policy study.

### Agreed first comprehensive sample

The next executable screen fixes both `kappa_p` values to zero and draws **120
unique Latin-hypercube conditions** from the 12,000-condition reduced design
(`5 × 5 × 6 × 4 × 4 × 5`). The sample is balanced as evenly as possible over
each discrete factor, uses a recorded seed, and retains every sampled row in
the result bundle.

## Reset-pool contract

For an active organism, an unspecified pool is resolved at reset as:

```text
initial C pool = kappa_c × initial_biomass × dt
initial P pool = kappa_p × initial_biomass × dt
```

Explicit numeric pools remain overrides. Inactive organisms receive zero pools.

## Limitation diagnostics

Each traced timestep records gamma-normalized C/P growth allocations and use,
the limiting resource, C/P acquired, C/P maintenance use, and maintenance
use divided by the corresponding resource acquired in the preceding timestep.
The preceding-step alignment reflects that uptake, fixation, and transfers are
credited after maintenance in the current transition.

For visualization, define `C_eq = allocated_C / gamma_C` and
`P_eq = allocated_P / gamma_P`. The signed pressure is `P_eq - C_eq`:
positive values indicate C limitation (P surplus), negative values indicate P
limitation (C surplus), and zero is balanced. Heatmaps use a divergent scale
centered at zero, with balanced white and no-realized-growth timesteps masked
grey. Trade panels retain raw transfers and also show recipient-equivalent
amounts: plant-to-fungus C divided by fungal `gamma_C`, and fungus-to-plant P
divided by plant `gamma_P`.

## Implementation record

### Reset-pool contract

- Traits accept unspecified (`None`) pools and validate explicit overrides.
- Reset resolves defaults using the configured timestep and initial biomass.
- Model overview, parameter register, provenance note, and open questions were
  updated.
- Verification: `uv run pytest tests/test_reset_pools.py
  tests/test_growth_geometry.py -q` — 40 passed.

### Screen infrastructure

- Deterministic discrete sampling, static-control execution, viability ranking,
  immutable JSON bundles, focused trade/P manifests, and retained diagnostics
  were added.
- Verification: `uv run pytest tests/test_resource_pressure_screen.py -q` and
  static-control tests pass.

### Executed screens

- Uniform-1-µM discovery screen: 48/48 complete.
- Focused trade screen with both `kappa_p=0`, five κC levels, six AMF levels,
  and requested trade levels: 48/48 complete.
- P-limitation calibration: 12/12 complete across `1.0, 0.1, 0.01 µM` and
  `0x, 1x, 10x, 20x` κP multipliers at `0.01x` κC.

## Artifacts

- Discovery manifest: `docs/qualification/resource-pressure-static-screen-manifest.json`
- Focused trade manifest: `docs/qualification/resource-pressure-focused-trade-screen-manifest.json`
- P-limitation manifest: `docs/qualification/p-limitation-experiment-manifest.json`
- Analysis reports: `docs/analysis/resource-pressure-static-screen-report.md`,
  `docs/analysis/p-limitation-experiment-report.md`
- Result bundles: `outputs/resource-pressure-static-screen/`,
  `outputs/resource-pressure-focused-trade-screen/`,
  `outputs/p-limitation-experiment/`

## Risks and open questions

- The reserve component does not currently guarantee next-step maintenance
  because acquisition is credited after allocation; interpret reserve results
  cautiously.
- The full Cartesian design is too large for routine execution; every sampled
  run must retain its design provenance and sample budget.
- Very low P can produce near-zero growth, making allocated-to-used ratios
  numerically ill-conditioned; limitation labels and absolute normalized
  allocations should be inspected together.
