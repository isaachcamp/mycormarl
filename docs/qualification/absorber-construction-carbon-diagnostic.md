# Absorber construction-carbon diagnostic

This workflow is an isolated, fixed-geometry, one-cell absorber qualification.
It calls phosphate uptake closures directly and does not step `BaseMycorMarl`,
update biomass, run growth or maintenance, or invoke policy code.

## Command and artifacts

```bash
uv run python scripts/absorber_construction_carbon_diagnostic.py \
  --output-dir outputs/absorber-construction-cost-diagnostic
```

The canonical sweep uses 40 logarithmic absorber radii from `1e-4` to
`3e-2 cm`, 40 logarithmic length densities from `1e-1` to
`1e4 cm cm^-3`, a one-day configured `T_ref`, and a configurable integration
timestep for one-day finite-inventory uptake. It writes rectangular CSV output,
construction-efficiency, absolute-uptake, and depletion-timescale SVG and PNG
artifacts.

The canonical run contains 3,200 rectangular surface rows across both
conditions. Of these, 156 are retained but invalid touching/overlapping cells.
All valid scientific metrics are finite. The default `t_1_percent_days` range
is approximately `0.0718` to `483,045` days, with a median of approximately
`316` days; the upper values are diagnostic closure times, not organism
lifespans.

## Root-tissue construction economics

For the supplied plant traits:

```text
rho_C,root = gamma_C,plant / (SRL pi r_root,ref^2)
L = lambda V_i
C_construction = L pi r_a^2 rho_C,root
```

The defaults imply `rho_C,root ~= 0.05031 g C cm^-3`. This is absorbing root
tissue only. `kroot`, shoots, maintenance, reproduction, reserves, and
whole-organism budgets are excluded. Custom plant traits redefine the inferred
density.

Because construction cost now scales with candidate radius, uptake per
construction C is not a constant rescaling of uptake per length. A separate
length-normalized surface is not part of this diagnostic.

## Valid geometry

The assigned cylindrical territory radius is

```text
R_terr = 1 / sqrt(pi lambda).
```

Cells with `r_a >= R_terr` represent touching or overlapping absorbers and are
invalid for this diagnostic. They remain flagged in the rectangular CSV, with
scientific result fields blank, but are masked in every figure with their
boundary drawn. The figure caption explains the masked region.

## Conditions, rates, and markers

Fixed-reservoir panels hold bulk concentration constant. Finite-inventory
panels deduct accepted uptake from canonical labile P and retain conservation
diagnostics. Both freeze sparse resistance and continuous-regime blending at
configured `T_ref` and integrate uptake over the same one-day horizon.

Maximum instantaneous uptake is the uncapped initial blended closure rate,
reported in `micromol P s^-1`; normalized rate is
`micromol P g C^-1 s^-1`. In a fixed reservoir, one-day integrated uptake is
that rate multiplied by 86,400 seconds, so those two panels intentionally have
the same spatial structure.

Efficiency panels mark plant-native geometry, fungal geometry evaluated with
plant economics, and a panel-specific fungus-equivalent plant geometry. At
native fungal length density, the equivalent geometry is solved continuously
so plant-economics P/C equals native fungal P/C for the displayed metric.
Absolute-uptake and depletion panels instead mark actual fungus-native
geometry. Each figure has a single external marker legend; the CSV retains all
coordinates and metric values.

For the default traits, the fungus-equivalent plant radii are approximately
`1.13977e-3 cm` for both fixed-reservoir metrics and for the finite-inventory
instantaneous-rate metric, and `8.60425e-4 cm` for finite-inventory integrated
uptake.

## Reference depletion timescale

`t_1_percent_days` is the finite-inventory time at which surface concentration
reaches one percent of its initial value. It is not truncated at the one-day
surface horizon. The implementation obtains the corresponding bulk threshold
algebraically and evaluates the autonomous uptake-rate integral with
deterministic numerical quadrature. It does not run millions of explicit
timesteps.

The timescale plot uses logarithmic colour scaling in days. Long event times
describe a fixed-geometry, fixed-closure qualification and must not be read as
predictions that absorbers live unchanged for those durations. Elapsed time
does not replace `T_ref` in the depletion-gradient closure.
