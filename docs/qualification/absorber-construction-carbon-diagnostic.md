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
tissue only. `kfroot`, shoots, maintenance, reproduction, reserves, and
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
surface horizon.

### Quantities held fixed

This is an event-time calculation for one isolated cell, not an additional run
of the full model. The absorber radius $r$, length density $\lambda$, cell
volume $V$, Michaelis--Menten traits $J_{max}$ and $K_m$, sparse resistance
$k$, and sparse-to-continuous weight $w$ are all held fixed. In particular,
$k$ and $w$ are calculated once using the configured $T_{ref}$. They are not
recalculated using the potentially much longer depletion time.

The represented absorber length and lateral absorbing area are therefore

$$
L=\lambda V,\qquad A=2\pi rL.
$$

Let $C_b$ be bulk solution-P concentration and $C_s(C_b)$ the concentration at
the absorber surface predicted by the sparse closure. Both concentrations are
in $\mathrm{\mu mol\,cm^{-3}}$. The closure satisfies

$$
C_b=C_s+k\frac{C_s}{K_m+C_s}.
$$

The implementation normally evaluates the equivalent quadratic for $C_s$ in
a numerically stable form; see
[`sparse_surface_concentration`](../../mycormarl/soil/phosphate_uptake.py#L162-L187).

### Converting the surface threshold to a bulk threshold

First, the diagnostic evaluates the initial surface concentration

$$
C_{s,0}=C_s(C_{b,0}).
$$

The event is defined by $C_{s,*}=0.01C_{s,0}$. Substituting that value into the
closure above gives the associated bulk concentration directly:

$$
C_{b,*}=C_{s,*}+k\frac{C_{s,*}}{K_m+C_{s,*}}.
$$

Thus the code does not search through simulated timesteps to discover when the
surface crosses one percent. It first converts the desired surface value into
the exact corresponding bulk value. This conversion is implemented in
[`_depletion_event_times_days`](../../mycormarl/soil/absorber_diagnostic.py#L151-L158).

The threshold always refers to the sparse-closure surface concentration, which
is also the surface concentration reported in the CSV. The rate used to reach
that threshold is nevertheless the configured blend of sparse and continuous
uptake.

### Deriving elapsed time from inventory conservation

The cell's labile-P amount is

$$
M=B C_b,\qquad B=V(\theta+b_p),
$$

where $\theta$ is volumetric water content, $b_p$ is linear buffer power, and
$B$ is the cell's concentration-to-inventory capacity. The uptake rate at a
given bulk concentration is

$$
Q(C_b)=A\left[
(1-w)J_{max}\frac{C_s(C_b)}{K_m+C_s(C_b)}
+wJ_{max}\frac{C_b}{K_m+C_b}
\right],
$$

in $\mathrm{\mu mol\,s^{-1}}$. The first term is sparse uptake evaluated at
the absorber surface; the second is the continuous approximation, which uses
bulk concentration. The two alternatives are blended rather than added.

With no replenishment, inventory conservation gives

$$
\frac{dM}{dt}=B\frac{dC_b}{dt}=-Q(C_b).
$$

All parameters in $Q$ are fixed, so the rate depends only on the current
concentration and not explicitly on time. This is what **autonomous uptake
rate** means here. Separating variables and integrating from the initial bulk
concentration to the threshold gives

$$
t_{1\%}=B\int_{C_{b,*}}^{C_{b,0}}\frac{1}{Q(C_b)}\,dC_b.
$$

In plain language, the interval is divided conceptually into small
concentration losses. Each loss takes longer when uptake is slow and less time
when uptake is fast; the integral adds those durations. The result is in
seconds and is divided by $86{,}400$ for `t_1_percent_days`.

### What deterministic numerical quadrature means

The final integral is one-dimensional but does not have a convenient closed
form because $Q$ contains both the sparse surface-concentration root and the
blended Michaelis--Menten rate. The code therefore approximates the integral
directly with a fixed 64-point Gauss--Legendre rule. If $x_i$ and $q_i$ are its
predefined nodes and weights on $[-1,1]$, and

$$
m=\frac{C_{b,0}+C_{b,*}}{2},\qquad
h=\frac{C_{b,0}-C_{b,*}}{2},
$$

then

$$
t_{1\%}\approx B h\sum_{i=1}^{64}
\frac{q_i}{Q(m+h x_i)}.
$$

"Numerical quadrature" simply means approximating a definite integral with a
weighted sum of function evaluations. "Deterministic" means every geometry
uses the same fixed rule and therefore the same concentration sampling pattern:
there is no random sampling, adaptive timestep choice, or event-search
tolerance. NumPy supplies the nodes and weights through
[`leggauss(64)`](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html),
and the mapping and weighted sum are implemented at
[`absorber_diagnostic.py:160-183`](../../mycormarl/soil/absorber_diagnostic.py#L160-L183).
The method is described as semi-analytical because the threshold conversion
and separation of the conservation equation are algebraic, while only this
single definite integral is evaluated numerically.

An absorber-free cell, a zero initial surface concentration, or a non-finite
integral has no reported event time. Regression tests verify that the event
time is independent of the one-day integration timestep and preserve reference
values for slow canonical cells; see
[`test_absorber_diagnostic.py:133-172`](../../tests/test_absorber_diagnostic.py#L133-L172).

The timescale plot uses logarithmic colour scaling in days. Long event times
describe a fixed-geometry, fixed-closure qualification and must not be read as
predictions that absorbers live unchanged for those durations. Elapsed time
does not replace `T_ref` in the depletion-gradient closure.
