# Soil phosphate transport and uptake model

> **Scope:** This document covers labile-P state, concentration, buffering,
> finite-volume diffusion, sparse/continuous uptake, root–fungus competition,
> soil-to-organism unit conversion, and numerical verification. Biomass and
> absorber geometry are inputs described in the [growth model](growth-model.md);
> shared scheduling is described in the [model overview](model-overview.md).

## Executive summary

The phosphate model stores a finite amount of labile inorganic P in every
axisymmetric soil cell. Soil-solution concentration is derived from that amount
through a linear instantaneous buffer. Diffusion conservatively transfers
amount between neighbouring cells; roots and fungi then request P according to
Michaelis–Menten surface kinetics.

Two uptake closures are blended. Sparse absorbers use an analytical depleted
surface concentration, while the continuous closure approximates surface
concentration by local bulk concentration. A smooth weight based on hyphal
depletion-zone overlap selects between them. Root and fungal requests are
finally capped together against the same cell inventory, so uptake cannot
create negative soil P or credit more P than was removed.

## State, units, and buffering

`State.soil_labile_p` is the canonical state in µmol P cell⁻¹. Configured
solution concentration enters in µM and is converted by
$1\,\mu\mathrm{M}=10^{-3}\,\mu\mathrm{mol\,cm^{-3}}$. Organism P pools use mg.
The unit boundary is explicit in
[`phosphate_units.py`](../src/mycormarl/soil/phosphate_units.py).

For cell volume $V$ (cm³ bulk soil), solution concentration $C$
(µmol cm⁻³), volumetric water content $\theta$ (cm³ water cm⁻³ bulk soil), and
linear volumetric buffer power $b_p$ (cm³ water-equivalent cm⁻³ bulk soil),

$$
M=C\,V(\theta+b_p),\qquad C=\frac{M}{V(\theta+b_p)}.
$$

$M$ is the total instantaneously labile amount: dissolved plus reversibly
sorbed. The dissolved fraction is $\theta/(\theta+b_p)$ and retardation relative
to solution-only transport is $(\theta+b_p)/\theta$. Buffering increases the
available labile inventory at a given $C$ while slowing propagation of changes
in that inventory. It does not imply that total, occluded, or extractable P is
labile on model timescales.

```python
return (
    jnp.asarray(concentration_micromol_cm3)
    * jnp.asarray(cell_volumes_cm3)
    * labile_capacity_factor(theta_water, b_p)
)
```

Implemented by
[`solution_concentration_to_labile_amount`](../src/mycormarl/soil/phosphate_grid.py)
and its inverse
[`labile_amount_to_solution_concentration`](../src/mycormarl/soil/phosphate_grid.py).
Round-trip, initial-field, and canonical-state behaviour are tested in
[`test_labile_phosphate_state.py`](../tests/test_labile_phosphate_state.py).

Initialisation immediately converts the configured solution-P field to
canonical amount. By default, `initial_solution_p_um` is uniform in every cell
of the represented domain. An explicit `initial_solution_p_depth_profile` is
the sole opt-in vertical-heterogeneity treatment: its `(depth cm, relative
factor)` knots are linearly interpolated at cell-centre depth, apply uniformly
in the radial direction, and scale `initial_solution_p_um`. The profile must
cover the configured soil depth; values above its first knot retain the first
factor. It is a relative distribution, not a pore-water calibration.

Both treatments are provisional solution-P initial conditions, not direct
equivalents of agronomic extractable-P or total soil-P measurements. Comparison
with extraction data needs an observation/calibration model. After reset the
finite-inventory model evolves normally, so diffusion and uptake may create
further spatial differences.

For a uniform initial concentration, the analytical domain inventory is
`1e-3 * initial_solution_p_um * pi * radius_cm**2 * depth_cm * (theta+b_p)`
in µmol P. At the current `50 cm × 100 cm`, `1 µM` defaults this is about
`187,945.8 µmol P` (`5,821.4 mg P`). The early upper-25-cm-only fixture held
one quarter of this amount; its archived inventory must not be reused for the
current uniform reset. All soil and density fields have shape `(n_r, n_z)`;
there is no angular state dimension.

## Conservative diffusion

Across an internal face between cells $a$ and $b$, the amount transferred in
time $\Delta t$ is

$$
\Delta M_{a\rightarrow b}
=D_l\theta f_l\frac{A_{ab}}{d_{ab}}(C_a-C_b)\Delta t,
$$

where $D_l$ is the solution diffusion coefficient (cm² s⁻¹), $f_l$ is a
dimensionless impedance/tortuosity factor, $A_{ab}$ is face area (cm²), and
$d_{ab}$ is centre distance (cm). The same transfer is subtracted from one
cell and added to the other. External faces are omitted; hence the top, bottom,
outer radius, and symmetry axis are no-flux.

Buffering appears only when $M$ is converted to $C$ (calculated as above,
with buffering). The conductance correctly
uses $D_l\theta f_l$, avoiding double retardation. The apparent diffusivity
used for depletion-zone propagation diagnostics is instead

$$
D_{app}=\frac{D_l\theta f_l}{\theta+b_p}.
$$

Implemented by
[`axisymmetric_diffusion_conductances`](../src/mycormarl/soil/phosphate_diffusion.py),
[`diffuse_labile_amount`](../src/mycormarl/soil/phosphate_diffusion.py),
and [`apparent_diffusivity_cm2_s`](../src/mycormarl/soil/phosphate_diffusion.py).
Independent two-cell transfers, closed boundaries, non-negativity, and amount
conservation are tested in
[`test_phosphate_diffusion.py`](../tests/test_phosphate_diffusion.py).

An exact cellwise explicit CFL ceiling is precomputed. If a biological step is
too large, the entire fixed-geometry diffusion-then-uptake transaction is
repeated in equal stable substeps. This is numerical subcycling, not biological
ageing. The number of substeps is cached during environment construction
([`base_mycor.py:92–118`](../src/mycormarl/environments/base_mycor.py));
the schedule is tested in
[`test_phosphate_diffusion.py`](../tests/test_phosphate_diffusion.py).

More explicitly, with face conductance
$G_{ij}=D_l\theta f_l A_{ij}/d_{ij}$, the positivity ceiling is

$$
\Delta t_{CFL}=\min_i\frac{V_i(\theta+b_p)}{\sum_jG_{ij}},\qquad
N_{soil}=\max\left(1,\left\lceil\frac{86400\,dt}
 {s\Delta t_{CFL}}\right\rceil\right),
$$

where `dt` is the numerical environment interval in days and
`s=diffusion_cfl_safety=0.8`. A cell with no conducting neighbours has an
infinite ceiling; disabled diffusion therefore needs one soil substep. Each
substep lasts `86400*dt/N_soil` seconds and performs diffusion followed by
uptake, refreshing concentration between them. Conductances use actual
neighbouring centre distances. Low-level helpers accept explicitly supplied
nonuniform edges; production grid construction requires uniform cells whose
intervals divide the domain extents exactly.

Pairwise conservation and the CFL bound are separate properties: a
conservative update can still make cells negative if stepped too far. The CFL
trigger protects diffusion positivity, while the uptake cap protects finite
inventory. Neither proves accuracy when uptake nearly exhausts a cell;
such cases need timestep refinement rather than parameter retuning.

## Surface uptake requests

This section describes the two uptake closures and how they are combined.

For root or hyphal length density $\lambda$ (cm cm⁻³), cell volume $V$, and
absorber radius $r$, represented length and lateral absorbing area are

$$
L=\lambda V,\qquad A=2\pi rL.
$$

End caps are excluded. Surface influx follows

$$
J(C_s)=\frac{J_{max}C_s}{K_m+C_s},\qquad
U=J(C_s)A\Delta t,
$$

where $J_{max}$ is µmol cm⁻² s⁻¹, $K_m$ and $C_s$ are µmol cm⁻³, and request
$U$ is µmol P cell⁻¹ per substep. These primitives are implemented in
[`phosphate_units.py`](../src/mycormarl/soil/phosphate_units.py).

### Continuous closure

The continuous closure assumes $C_s\approx C_b$, the current cellwise bulk
solution concentration. It is appropriate when depletion zones are coupled
at the represented scale. Implemented by
[`continuous_uptake_request`](../src/mycormarl/soil/phosphate_uptake.py)
and verified against an independent flux-area-time calculation in
[`test_continuous_phosphate_uptake.py`](../tests/test_continuous_phosphate_uptake.py).

### Sparse closure

For separated cylindrical absorbers, each structure receives a territory
radius

$$
R_{soil}=\frac{1}{\sqrt{\pi\lambda}}.
$$

The effective depletion radius is limited by both that territory and the
distance $\sqrt{D_{app}T_{ref}}$ that a disturbance can propagate during a
reference exposure time $T_{ref}$. The resulting logarithmic cylindrical
diffusion resistance is

$$
k=\frac{rJ_{max}}{D_l\theta f_l}\ln\left(\frac{R_{eff}}{r}\right).
$$

Precisely,
$R_{eff}=r+\min(\sqrt{D_{app}T_{ref}},\max(R_{soil}-r,0))$,
with $T_{ref}$ converted to seconds. The resistance $k$ has concentration
units. The surface concentration solves

$$
C_s^2+(K_m+k-C_b)C_s-K_mC_b=0.
$$

Equating diffusive supply to Michaelis–Menten demand produces a quadratic for
$C_s$; the code evaluates the physical root in a numerically stable form and
clips it to $0\le C_s\le C_b$. Implemented by
[`effective_uptake_radius_cm`](../src/mycormarl/soil/phosphate_uptake.py),
[`sparse_uptake_resistance`](../src/mycormarl/soil/phosphate_uptake.py),
and [`sparse_surface_concentration`](../src/mycormarl/soil/phosphate_uptake.py).
Analytical limits, bounded surface concentration, and JIT compatibility are
tested in [`test_sparse_phosphate_uptake.py`](../tests/test_sparse_phosphate_uptake.py).

The implementation switches between direct and rationalised forms of the
positive quadratic root to avoid cancellation. Zero concentration or zero
absorbing length gives zero uptake; zero resistance gives $C_s=C_b$;
infinite resistance gives $C_s=0$. With diffusion disabled, sparse supply is
zero and the overlap time is infinite. With active diffusion, a territory at
or inside the absorber surface has zero gap and zero overlap time. These are
explicit limiting cases, not additional sub-grid geometry states.

A depletion zone here means a region whose solution P is lower than the
far-field value; it need not be totally depleted. At the illustrative reference
density $\lambda=1{,}000\;\mathrm{cm\;cm^{-3}}$, the default soil parameters
give $t_{diff}\approx0.90$ day, motivating the rounded provisional choice
`T_ref=1 day`. This static exposure scale is not root/hyphal lifetime or a
numerical timestep, and it does not track time since a cell was occupied.
Agreement with an age-dependent closure is not established by this choice.

## Sparse-to-continuous transition

The hyphal territory gap defines an overlap timescale

$$
t_{diff}=\frac{\max(R_{soil,h}-r_h,0)^2}{D_{app}},\qquad
w=\frac{1}{1+(t_{diff}/T_{ref})^p}.
$$

$w\in[0,1]$ is shared by root and fungal requests, and $p=2$ controls transition
sharpness. Final demand is

$$
U_{request}=(1-w)U_{sparse}+wU_{continuous}.
$$

Only hyphal density determines $t_{diff}$. Consequently, a plant-only cell has
$w=0$ and remains sparse; roots enter the continuous regime wherever sufficiently
dense hyphae create a shared coupled depletion field. Implemented by
[`hyphal_overlap_time_seconds`](../src/mycormarl/soil/phosphate_uptake.py),
[`continuous_regime_weight`](../src/mycormarl/soil/phosphate_uptake.py),
and [`blend_uptake_requests`](../src/mycormarl/soil/phosphate_uptake.py).

This is an adaptation of the depletion-zone argument in Schnepf & Roose, not a
resolved sub-grid simulation. Colony age and front-specific exposure are
deferred. Transition limits and the nominal overlap time are tested in
[`test_sparse_phosphate_uptake.py`](../tests/test_sparse_phosphate_uptake.py).

## Root–fungus competition and pool credit

Both requests are evaluated from the same post-diffusion $C_b$, blended, and
then capped once per cell. With available amount $M$ and total request
$U_T=U_r+U_f$,

$$
U_{accepted}=\min(M,U_T),\quad
U_{r,accepted}=U_{accepted}\frac{U_r}{U_T},\quad
U_{f,accepted}=U_{accepted}-U_{r,accepted}.
$$

The guarded zero-demand case assigns zero to both consumers. This proportional
allocation prevents overdraw and preserves request shares, but it does not
resolve sub-cell competition when sparse depletion zones overlap imperfectly
and total request remains below cell inventory.

```python
accepted_total = jnp.minimum(available, total_request)
accepted_root = accepted_total * root_fraction
accepted_fungus = accepted_total - accepted_root
remaining = available - accepted_total
```

Implemented by
[`allocate_competing_uptake`](../src/mycormarl/soil/phosphate_uptake.py)
inside the shared production/qualification
[`blended_uptake_transaction`](../src/mycormarl/soil/phosphate_uptake.py).
Competition symmetry, proportionality, and conservation are tested in
[`test_continuous_phosphate_uptake.py`](../tests/test_continuous_phosphate_uptake.py)
and end-to-end in
[`test_environment_phosphate_uptake.py`](../tests/test_environment_phosphate_uptake.py).

Accepted root and fungal amounts are summed separately and converted exactly
once using $1\,\mu mol\,P=0.0309738\,mg\,P$ before being added to organism
pools. The transaction is implemented in
[`soil_diffusion_uptake_substep_with_diagnostics`](../src/mycormarl/soil/soil.py)
and its soil-to-pool balance is tested by
[`test_end_to_end_uptake_conserves_soil_plus_free_pool_p`](../tests/test_environment_phosphate_uptake.py).

Consequently, on the closed soil domain,
`sum(M_before)-sum(M_after)=sum(U_root,accepted+U_fungus,accepted)`
apart from floating-point error. This is a domain-total identity: diffusion
can increase an individual cell's inventory. The corresponding organism-pool
gain is that accepted amount times `0.0309738`; growth, internal trade, and
explicit biological exports require the wider ledger in the
[model overview](model-overview.md#phosphorus-accounting).

## Parameters and empirical interpretation

| Parameter | Default | Provenance/status |
|---|---:|---|
| Initial solution Pi | `1 µM` | Order-of-magnitude condition associated with [Vance et al. (2003)](https://doi.org/10.1046/j.1469-8137.2003.00695.x); not a universal mean or extractable-P value. |
| $\theta$, $b_p$ | `0.3`, `239` | Provisional values reported by [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x), attributed there to Barber. Soil-specific calibration is required. |
| $D_l$, $f_l$ | `1e-5 cm² s⁻¹`, `0.308` | Provisional Schnepf–Roose parameterisation; with the defaults, $D_{app}\approx3.86\times10^{-9}$ cm² s⁻¹. |
| Root/fungal $J_{max}$ | `3.26e-6 µmol cm⁻² s⁻¹` | Tinker & Nye values as reported by Schnepf & Roose; separate trait fields are retained. |
| Root/fungal $K_m$ | `5.8e-3 µmol cm⁻³` | Same provisional source and status. |
| Hyphal radius | `5e-4 cm` | Schnepf & Roose. |
| Root radius | `0.01 cm` | Derived effective fine-root radius from GRooT diameter data; rounded and combined with unmatched fine-root traits. See the [growth-model provenance](growth-model.md#parameterisation-and-literature-relationship). |
| $T_{ref}$, $p$ | `1 day`, `2` | Explicit modelling choices, not empirically inferred lifetimes. |

Configuration defaults are declared in
[`EnvConfig`](../src/mycormarl/params.py); organism kinetics and
radii are in [`PlantTraits`](../src/mycormarl/plant/traits.py)
and [`FungusTraits`](../src/mycormarl/fungus/traits.py).

## Numerical qualification and limitations

The production and offline qualification paths call the same transaction.
Focused tests cover unit conversions, buffer identities, grid geometry,
diffusion, sparse/continuous limits, competition, JIT compilation, independent
consumers, and end-to-end balance. The canonical convergence and performance
results are in the [numerical qualification report](qualification/phosphate-numerical-qualification.md).

Those retained coupled trajectories predate the three-rate allocation engine
and keep their original provenance. Soil transaction choices remain in force;
the new action, reproduction, grouping and workflow evidence is in the
[three-rate qualification](qualification/three-rate-allocation.md). It does not
relabel the older spatial/calibration study as a rerun of the new model.

- **Code-backed:** Diffusion is conservative on a closed domain and uptake is
  capped by cell inventory.
- **Code-backed:** Linear buffering is instantaneous, homogeneous, and
  zero-intercept.
- **Code-backed:** Geometry-dependent resistances and transition weights are
  computed once per biological step; concentration-dependent requests are
  refreshed after every diffusion substep.
- **Limitation:** Proportional cellwise competition is an inventory rule, not a
  spatial solution for root–fungus interference below the grid scale.
- **Limitation:** No advective transport, kinetic/nonlinear sorption,
  mineralisation, precipitation, or external boundary supply is represented.
- **Limitation:** Extractable P cannot be compared directly with model $C$ or
  $M$ without an assay-specific observation model.

Qualification separates fixed-geometry soil convergence from coupled
fixed-action convergence. The latter starts free C/P at one structural-biomass
equivalent, disables automatic maintenance, and holds rate commands across a
fixed one-day policy interval. It therefore tests numerical resolution without
changing decision frequency. Selection requires both the next-finer and the
finest tested reference to agree within 5%, with absolute scale guards near
zero. Uptake by each consumer, shares, final inventory, biomass, spatial extent,
and endpoint free pools must be considered; soil stability alone is insufficient.

The retained [JSON evidence](qualification/phosphate-numerical-qualification.json)
and comparison tables select `0.05 cm` and `0.025 day` for their tested fixture;
the runtime defaults now use that same spatial interval and timestep. The early
archive's `0.1 cm` pass and finest-tested timestep fallback are historical
decisions, not the latest qualification result. Use the explicit comparison
rows and declared fixture when choosing a study grid.

The qualification matrix covers `0.1, 0.3, 1, 3, 10 µM` and root-only,
fungus-only, and mixed fields on a reduced domain over two days, plus a
deep-soil confinement/extended-P check. No scientific matrix row was
inventory-capped, so those convergence results do not qualify severe depletion.
Balance tolerance is relative `1e-5`; small well-conditioned float32 reference
tests typically use `1e-6`. Represented-length tests use `1e-5` where applicable.
These tolerances are numerical acceptance criteria, not empirical uncertainty.

Changing `T_ref` or `p` is scientific parameter sensitivity, not numerical
refinement: `T_ref` affects both overlap weight and sparse propagation distance,
so total uptake need not be monotonic. Local absorber age, midpoint exposure of
new geometry, and a separate uptake-accuracy substep trigger remain potential
extensions. Each long-horizon or strongly depleted study needs its own
convergence evidence. Historical full-grid CPU timings are device-specific;
annual costs are projections of warmed soil/full-environment steps, excluding
training, learned inference, output and transfers. Array-memory estimates are
not peak process RSS; the latest retained run skipped the target benchmark.

For reproducible small examples and qualification commands, see the
[model overview](model-overview.md#running-and-checking-the-model).

## References

1. Schnepf, A., & Roose, T. “Modelling the contribution of arbuscular mycorrhizal fungi to plant phosphate uptake.” *New Phytologist* 171 (2006). [DOI](https://doi.org/10.1111/j.1469-8137.2006.01771.x).
2. Jakobsen, I., & Abbott, L. K. “External hyphae of vesicular-arbuscular mycorrhizal fungi associated with *Trifolium subterraneum* L. 1. Spread of hyphae and phosphorus inflow into roots.” *New Phytologist* 120 (1992). [DOI](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x).
3. Vance, C. P., Uhde-Stone, C., & Allan, D. L. “Phosphorus acquisition and use: critical adaptations by plants for securing a nonrenewable resource.” *New Phytologist* 157 (2003). [DOI](https://doi.org/10.1046/j.1469-8137.2003.00695.x).
