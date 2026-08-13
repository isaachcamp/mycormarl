# Time-dependent depletion-gradient diagnostic

This fixed-reservoir diagnostic isolates the time dependence hidden by the
production sparse-uptake closure's fixed `T_ref`. It is an experiment only:
all absorbing cylinders exist at time zero, retain fixed geometry, and draw
from a fixed bulk phosphate concentration. It neither constructs nor steps
`BaseMycorMarl`, and it must not be used as a production global-time switch.

Run it with:

```bash
uv run python scripts/depletion_gradient_diagnostic.py \
  --output-dir outputs/depletion-gradient-diagnostic
```

The command writes `depletion_gradient_time_series.csv`,
`depletion_gradient_summary.csv`, and a two-panel
`depletion_gradient_cumulative_uptake` figure in SVG and PNG formats. The time
series contains the fixed reservoir, geometry, plant kinetic baseline,
territory radius, time-dependent effective radius, sparse resistance, the
`t_sim`-controlled continuous weight, component rates, blended rate, and
cumulative uptake. The summary reports the endpoint cumulative blended uptake.

Both panels use the same density fixtures (1, 100, and 2,000 cm cm^-3) and a
single plant kinetic baseline. They differ only in absorber radius: 0.01 cm
(plant scale) and 0.0005 cm (fungus scale). The latter is a geometric scale,
not a fungal trait bundle. A vertical line marks the configured one-day
`T_ref`; each curve receives a small marker at its computed overlap time
`t_diff` when it occurs in the plotted window. Construction carbon, finite
inventory, growth, and organism age are outside the diagnostic.

The native plant geometry (`r_a=0.01 cm`, `lambda=1 cm cm^-3`) and native
fungus geometry (`r_a=0.0005 cm`, `lambda=2000 cm cm^-3`) are drawn fully
opaque. The other radius-density fixtures, including their in-window overlap
markers, are drawn at 40% opacity so that the defaults remain prominent. The
colour legend itself remains fully opaque.

`native_geometry_closure_comparison` is a second two-panel SVG/PNG figure. It
compares sparse and continuous limits for the native plant and fungus
geometries in the first panel, then fixed `omega(T_ref)` and diagnostic
`omega(t_sim)` blends in the second. The second panel additionally includes a
neutral transition-scale geometry: it retains the fungal radius but derives its
length density from the configured apparent diffusivity so that `t_diff` is 10
days. This exposes the consequence of holding the blend at one-day `T_ref`
instead of advancing it with `t_sim`. All geometries use the shared plant
kinetic baseline. Colour denotes treatment and line style distinguishes plant
(solid), fungus (dashed), and transition scale (dash-dot); each panel has its
own upper-left treatment legend, with no line-style legend. Its CSV and
endpoint-summary CSV use the same `native_geometry_closure_comparison` stem.
The blend-clock panel marks every in-window `t_sim = t_diff` point on its
`t_sim` trajectories; both panels show the configured one-day `T_ref` as an
unlabeled vertical dashed line.
