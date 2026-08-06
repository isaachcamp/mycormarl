# Open scientific and modelling questions

This page contains only unresolved decisions. Implemented behaviour is
described in the [phosphate model](phosphate-model.md); completed planning and
verification records are retained under [`archive/`](archive/).

## Phosphorus accounting

- **Mortality and recycling:** structural P lost through mortality is recorded
  as leaving the simulated system. Decide whether litter, mineralisation, and
  re-uptake pools are needed for the intended experiments.

## Soil calibration and observation

- Calibrate initial solution Pi, buffer power, water content, diffusion
  impedance, and uptake kinetics for a specific soil and experimental setup.
- Define an observation model linking simulated solution/labile P to the
  extraction method used experimentally. Extractable P must not be compared
  directly with solution concentration without this mapping.
- Determine whether instantaneous homogeneous linear buffering is adequate or
  whether nonlinear sorption, kinetic desorption, precipitation, or spatially
  varying soil properties are required.

## Organism parameters and geometry

- Obtain a direct *R. irregularis* external-mycelium dry-mass measurement to
  validate or replace the current `0.0001 g` early-established fixture. The
  prior `7.97e-7 g` cross-species spore estimate remains a sensitivity case;
  at reset, the model maps the configured fungal biomass to external
  absorptive hyphae rather than simulating spores or inoculum.
- Corroborate the provisional *Daucus carota* P concentration independently of
  the current MDPI source and obtain a whole-plant carbon measurement if
  possible.
- Replace the unsourced `0.01 cm` root absorbing radius with data for the
  relevant cultivar and absorbing root orders.
- Test whether GRooT standing root-mass fraction is an adequate proxy for
  marginal growth allocation and whether its SRL is compatible with the plant
  representation.
- Determine when the stacked-disc root and saturated-hemisphere fungal
  closures need explicit branching, directional growth, or intraradical and
  spore compartments.
- Re-run the canonical numerical qualification when the `2,000 cm cm^-3`
  fungal saturation density or deep-soil fixture changes; the current run
  includes the deep-soil confinement and extended-P balance check. Continue
  tracking follow-up work in [issue #18](https://github.com/isaachcamp/mycormarl/issues/18).

## Carbon accounting and light

- Add growth respiration or construction efficiency as a future parameter
  separate from standing-biomass maintenance `kappa_c`. Until then, growth
  consumes only structural carbon through `gamma_c` and therefore implies
  100% conversion of allocated substrate C into structural C.
- When adding a diurnal cycle, retain `amass` as an apparent-gross carbon
  budget for one stated reference day and multiply it by a dimensionless
  `f_light(t)`. Normalise the profile so its one-day integral is one day when
  preserving the calibrated daily budget; `f_light = 1` is the current
  uniform-in-time special case.

## Uptake regimes and time

- Determine a biologically motivated value for $T_{ref}$.
- Add local time since colonisation if growth-front age or structure turnover
  materially changes the sparse-to-continuous transition.
- Assess sub-cell interference between sparse root and fungal depletion zones;
  proportional inventory scaling prevents overdraw but does not resolve this
  interaction below the cell scale.
- Separate policy decision timing from numerical integration timing under
  [issue #24](https://github.com/isaachcamp/mycormarl/issues/24). Until then,
  coupled fixed-action timestep comparisons change action frequency and remain
  sensitivity diagnostics rather than soil-solver convergence evidence.
- Extend convergence tests to long-horizon trajectories and learned policies;
  the current deterministic two-day fixture is numerical evidence, not proof
  for every MARL trajectory.
