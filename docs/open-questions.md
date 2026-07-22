# Open scientific and modelling questions

This page contains only unresolved decisions. Implemented behaviour is
described in the [phosphate model](phosphate-model.md); completed planning and
verification records are retained under [`archive/`](archive/).

## Phosphorus accounting

- **Maintenance-P accounting implementation:** current maintenance removes P
  from the free organism pool without assigning it to structure, recycling,
  soil, or a loss diagnostic. The agreed next treatment is to record it as an
  explicit unmodelled maintenance/turnover loss in separate plant and fungal
  counters. Until that change is implemented and tested, whole-system
  conservation must not be claimed for maintenance-active runs.
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

- Validate the provisional fungal P mass fraction of `40 mg P g^-1 dry mass`
  against the underlying studies rather than treating the reported maximum as
  representative.
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

## Uptake regimes and time

- Add local time since colonisation if growth-front age or structure turnover
  materially changes the sparse-to-continuous transition.
- Assess sub-cell interference between sparse root and fungal depletion zones;
  proportional inventory scaling prevents overdraw but does not resolve this
  interaction below the cell scale.
- Investigate the endpoint free-P pool sensitivity that caused every tested
  timestep above `0.025 day` to fail the current 5% qualification rule. Run a
  finer timestep study or justify a better endpoint metric before claiming
  temporal convergence.
- Extend convergence tests to long-horizon trajectories and learned policies;
  the current deterministic two-day fixture is numerical evidence, not proof
  for every MARL trajectory.
