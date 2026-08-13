## Problem Statement

The model needs a reproducible diagnostic relating phosphate uptake to the
root-tissue carbon required to construct absorbing cylinders across absorber
radius and length density. The first implementation held plant construction
cost per unit length fixed by SRL while sweeping uptake radius, producing an
inconsistent equal-geometry comparison with native fungal economics. It also
reported touching/overlapping cylinder geometries and searched for depletion
only within the one-day uptake horizon.

## Solution

Run a standalone closure-level, single-cell diagnostic over logarithmic grids
of absorber radius (`1e-4` to `3e-2 cm`) and length density (`1e-1` to
`1e4 cm cm^-3`). Do not construct or step `BaseMycorMarl`, organism growth,
policy code, or the multi-agent model.

For supplied plant traits infer root-tissue structural-carbon density:

```text
rho_C,root = gamma_C,plant / (SRL pi r_root,ref^2)
C_construction = (lambda V_i) pi r_a^2 rho_C,root
```

The default is approximately `0.05031 g C cm^-3`. Exclude `kfroot`, above-ground
growth, maintenance, reproduction, reserves, and whole-organism budgets.
Custom plant traits redefine the inferred density.

Run two experiments over the common configured `T_ref` horizon:

1. fixed bulk concentration, isolating intrinsic closure performance; and
2. finite inventory, deducting accepted uptake from canonical labile P.

Produce construction-efficiency and absolute-uptake figures with integrated
one-horizon uptake and maximum instantaneous uptake rate. Define maximum rate
as the uncapped blended closure rate at the initial phosphate state and report
it in `micromol P s^-1`, or `micromol P g C^-1 s^-1` after normalization.
Retain the fixed-reservoir rate panel even though its structure is proportional
to one-day integrated uptake.

## Geometry validity

For every grid coordinate calculate

```text
R_terr = 1 / sqrt(pi lambda).
```

Only `r_a < R_terr` is valid. Retain all cells so CSV output remains
rectangular, add explicit territory-radius and validity fields, and leave
scientific result metrics blank for invalid cells. Mask invalid cells in every
figure, draw `r_a = R_terr`, and label the blocked region as absorbers touching
or overlapping. Equivalent-geometry solvers must search only the valid plotted
domain.

## Reference markers

Construction-efficiency panels show:

- plant-native geometry; and
- fungal geometry evaluated with plant uptake and root-tissue construction
  economics; and
- panel-specific fungus-equivalent plant geometry.

For the latter, compute the native fungal P-per-C target using fungal uptake
traits, fungal tissue construction, and actual fungal geometry. Hold
`lambda = lambda_fungus` and solve continuously for the valid plant radius
whose plant-economics value equals that panel's target. Solve separately for
integrated and maximum-rate metrics and for each experimental condition. Do
not select the nearest grid cell. Record an explicit unavailable marker if no
valid in-range root is bracketed.

Absolute-uptake and depletion-timescale panels instead show plant-native and
actual fungus-native geometry. Marker labels and annotations omit `r_a` and
`lambda`; machine-readable marker rows retain coordinates and metric values.
Use one shared marker legend outside the upper-right panel of each multi-panel
figure. The blocked overlap region is shown through masking and its boundary;
its explanatory text belongs in the figure caption, not the plot itself.

## Depletion timescale

For every valid finite-inventory geometry calculate

```text
t_1% = min {t : C_s(t) = 0.01 C_s(0)}.
```

Do not truncate this calculation at `T_ref` and do not run a fixed-timestep
simulation until depletion. Keep sparse resistance and the continuous-regime
weight frozen at configured `T_ref`. Reduce the surface threshold to its bulk
concentration algebraically, then evaluate

```text
t_1% = capacity * integral[C_b,* to C_b,0] dC_b / Q(C_b)
```

with deterministic numerical quadrature of the autonomous blended uptake rate
`Q`. Report event time in days and plot it with logarithmic colour scaling.
Very long values are frozen-geometry closure diagnostics, not predictions of
root or hyphal longevity. This does not introduce ADR-0006's elapsed-time
depletion-gradient model.

## User Stories

1. Compare integrated P uptake and instantaneous uptake power per root-tissue
   construction C over valid absorber geometries.
2. Inspect corresponding absolute uptake without confusing scale and
   efficiency.
3. Compare native fungus P-per-C to an exact, on-surface equivalent plant
   geometry for each efficiency metric.
4. Identify touching or overlapping absorber geometries and prevent their
   interpretation as valid closure results.
5. Obtain `t_1%` for every valid positive-density geometry without an arbitrary
   simulation horizon.
6. Preserve the distinction between amount-flux diffusivity, buffered
   propagation diffusivity, and fixed configured `T_ref`.
7. Receive deterministic, rectangular, machine-readable output plus
   publication-quality SVG and high-resolution PNG figures.

## Implementation Decisions

- Keep one public high-level sweep runner and isolated construction helpers.
- Reuse existing territory, uptake, resistance, diffusivity, concentration,
  unit, and fungal biomass-to-hypha APIs.
- Keep plotting downstream of tabular results; plotting does not recompute
  scientific values.
- Keep the common efficiency and scale horizon configurable and one day by
  default.
- Sum accepted timestep uptake for integrated finite-inventory amounts and
  preserve conservation accounting.
- Calculate maximum instantaneous rate directly from initial concentration,
  independent of integration timestep.
- Do not add a length-normalized surface. After radius-consistent pricing it
  would be scientifically distinct, but remains outside this diagnostic.
- Record inferred root carbon density, `R_terr`, geometry validity, marker
  semantics, solve status, uptake, construction C, resistance, concentrations,
  conservation, and event time with unambiguous units.

## Testing Decisions

- Recover inferred root carbon density from supplied `gamma_c`, SRL, and
  reference root radius; verify `kfroot` and unrelated organism economics do not
  enter it.
- Verify construction C scales linearly with length and quadratically with
  candidate radius.
- Test strict validity at, below, and above `r_a = R_terr`; ensure invalid rows
  remain present with blank result metrics and are masked in every artifact.
- Test continuous, valid-domain equivalent-radius solves against native fungal
  targets, panel specificity, determinism, and unavailable targets.
- Test marker selection by figure type and absence of coordinate text in
  marker labels and annotations.
- Test initial instantaneous rates in per-second units and their independence
  from integration timestep.
- Test fixed-reservoir one-day amount/rate proportionality.
- Test semi-analytical `t_1%` against high-accuracy reference integrations,
  monotonic limiting cases, and conservation; ensure it is not bounded by
  `T_ref`.
- Retain tests for fixed reservoir, finite inventory, zero density, invalid
  scientific inputs, closure diagnostics, tabular schema, plot formats, and
  visual quality.

## Out of Scope

- Production uptake or growth changes.
- Elapsed-time replacement of fixed `T_ref` in resistance or blending.
- Absorber lifespan, turnover, cohort age, or ecological persistence.
- Above-ground or whole-organism construction economics.
- A length-normalized sweep, policy training, biological calibration, or a
  general plotting framework.

Related time-dependent depletion-gradient diagnostic: #27.
