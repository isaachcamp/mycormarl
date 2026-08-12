# ADR-0007: Use geometry-consistent root-tissue economics

**Status:** Accepted

## Context

ADR-0005 defined plant construction carbon from specific root length (SRL),
plant root allocation fraction, and structural carbon fraction. It then swept
absorber radius only through the uptake closure. This decoupled uptake radius
from construction geometry: a candidate with fungal-scale radius retained the
mass per unit length implied by the default plant root radius and SRL. The
result made native fungal construction appear hundreds of times cheaper even
when comparing equal absorber radius and length.

The diagnostic also evaluated cells where the absorber radius equalled or
exceeded its assigned cylindrical soil-territory radius. Such cylinders touch
or overlap and do not satisfy the geometry assumed by the sparse closure.

Finally, the first implementation searched for `t_1%` only during the common
one-day uptake horizon. Many valid geometries require much longer to reach the
threshold, while explicit fixed-timestep simulation can require millions of
steps.

## Decision

Infer root-tissue structural-carbon density from the plant traits supplied to
the diagnostic:

    rho_C,root = gamma_C,plant / (SRL * pi * r_root,ref^2)

The default traits give approximately `0.05031 g C cm^-3`. The root allocation
fraction `kroot` is excluded because this diagnostic prices absorbing root
tissue only. For represented length `L = lambda V_i`, every plant-economics
surface cell uses

    C_construction = L * pi * r_a^2 * rho_C,root.

Custom plant traits redefine the inferred density. The density remains fixed
within a sweep while candidate `r_a` varies.

Define the cylindrical territory radius as

    R_terr = 1 / sqrt(pi * lambda).

A cell is valid only when `r_a < R_terr`. Retain invalid cells in rectangular
tabular output, flag them explicitly, leave scientific result metrics blank,
mask them in every figure, and draw the touching/overlap boundary. Explain the
masked region in the figure caption rather than with in-plot text.

On each construction-efficiency panel, show fungal geometry evaluated with
plant uptake and root-tissue construction economics, plus a panel-specific
**fungus-equivalent plant geometry**. Hold length density at the native fungal
value and solve continuously for the valid plant radius whose plant-economics
P-per-C value equals the native fungus P-per-C value for that panel's metric.
Do not select the nearest sampled grid cell. If no valid in-range solution is
bracketed, record the marker as unavailable. The integrated and maximum-rate
panels may therefore place this marker at different radii.

On absolute-uptake and depletion-timescale panels, show actual fungus-native
geometry rather than the P-per-C-equivalent plant geometry. Also show the
plant-native reference where applicable. Use one external shared marker legend
per figure; labels and annotations do not include `r_a` or `lambda`, while
coordinates remain available in tabular output.

Define maximum instantaneous uptake rate as the uncapped initial blended
closure rate. Report absolute rate in `micromol P s^-1` and normalized rate in
`micromol P g C^-1 s^-1`. This removes timestep dependence. In the fixed
reservoir, one-day integrated uptake is necessarily the instantaneous rate
multiplied by 86,400 seconds; retain both panels for symmetric, explicit
reporting.

Keep the efficiency and absolute-uptake experiments on their common configured
`T_ref` horizon. Calculate `t_1%` separately as the time at which

    C_s(t) = 0.01 * C_s(0).

Freeze resistance and blending at the configured `T_ref`; do not substitute
elapsed time into the depletion-gradient closure. Obtain the threshold bulk
concentration algebraically and evaluate elapsed time from the autonomous
finite-inventory rate integral using deterministic numerical quadrature. This
is a semi-analytical event-time calculation, not a timestep simulation. Plot
valid event times with logarithmic colour scaling in days.

## Consequences

- Equal absorber radius and length are compared using tissue densities rather
  than inconsistent plant and fungal mass geometries.
- The default fungal tissue carbon density is about 2.3 times the inferred
  default root-tissue carbon density, so equal cylinders are comparable rather
  than separated by orders of magnitude.
- Construction-carbon and length normalization are no longer constant
  multiples across radius because construction cost scales with `r_a^2`. A
  separate length-normalized surface remains outside this diagnostic's scope,
  but is not described as algebraically redundant.
- Every fungus-equivalent marker on an efficiency panel has an exact metric
  interpretation, although its coordinate can move between panels.
- Invalid touching or overlapping geometries remain machine-auditable without
  being presented as scientific results.
- `t_1%` is available for slow valid geometries without an arbitrary horizon or
  timestep quantization. Very long times describe a frozen closure-level
  experiment, not absorber lifespan or ecological persistence.
- The production uptake equations and ADR-0006's separation of elapsed-time
  depletion-gradient experiments are unchanged.
