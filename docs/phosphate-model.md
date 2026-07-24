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
[`phosphate_units.py`](../mycormarl/mycormarl/soil/phosphate_units.py#L16-L50).

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
[`solution_concentration_to_labile_amount`](../mycormarl/mycormarl/soil/phosphate_grid.py#L195-L210)
and its inverse
[`labile_amount_to_solution_concentration`](../mycormarl/mycormarl/soil/phosphate_grid.py#L213-L227).
Round-trip, partial-topsoil, and canonical-state behaviour are tested in
[`test_labile_phosphate_state.py`](../tests/test_labile_phosphate_state.py#L49-L180).

Initialisation creates a uniform solution concentration down to the configured
topsoil depth, volume-averages a partially crossed layer, and immediately
converts it to canonical amount. The default is `1 µM` in the upper `25 cm`.
This is a provisional solution-P initial condition, not a direct equivalent of
an agronomic extractable-P measurement; comparison with extraction data needs
an observation/calibration model.

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
[`axisymmetric_diffusion_conductances`](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L41-L78),
[`diffuse_labile_amount`](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L95-L135),
and [`apparent_diffusivity_cm2_s`](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L20-L38).
Independent two-cell transfers, closed boundaries, non-negativity, and amount
conservation are tested in
[`test_phosphate_diffusion.py`](../tests/test_phosphate_diffusion.py#L193-L307).

An exact cellwise explicit CFL ceiling is precomputed. If a biological step is
too large, the entire fixed-geometry diffusion-then-uptake transaction is
repeated in equal stable substeps. This is numerical subcycling, not biological
ageing. The number of substeps is cached during environment construction
([`base_mycor.py:92–118`](../mycormarl/mycormarl/environments/base_mycor.py#L92-L118));
the schedule is tested in
[`test_phosphate_diffusion.py`](../tests/test_phosphate_diffusion.py#L100-L168).

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
[`phosphate_units.py`](../mycormarl/mycormarl/soil/phosphate_units.py#L80-L102).

### Continuous closure

The continuous closure assumes $C_s\approx C_b$, the current cellwise bulk
solution concentration. It is appropriate when depletion zones are coupled
at the represented scale. Implemented by
[`continuous_uptake_request`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L25-L54)
and verified against an independent flux-area-time calculation in
[`test_continuous_phosphate_uptake.py`](../tests/test_continuous_phosphate_uptake.py#L29-L80).

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

Equating diffusive supply to Michaelis–Menten demand produces a quadratic for
$C_s$; the code evaluates the physical root in a numerically stable form and
clips it to $0\le C_s\le C_b$. Implemented by
[`effective_uptake_radius_cm`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L70-L90),
[`sparse_uptake_resistance`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L93-L121),
and [`sparse_surface_concentration`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L162-L187).
Analytical limits, bounded surface concentration, and JIT compatibility are
tested in [`test_sparse_phosphate_uptake.py`](../tests/test_sparse_phosphate_uptake.py#L28-L206).

A depletion zone here means a region whose solution P is lower than the
far-field value; it need not be totally depleted. `T_ref=1 day` is a provisional
static exposure scale for this first model, not root/hyphal lifetime and not a
numerical timestep.

## Sparse-to-continuous transition

The hyphal territory gap defines an overlap timescale

$$
t_{diff}=\frac{(R_{soil,h}-r_h)^2}{D_{app}},\qquad
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
[`hyphal_overlap_time_seconds`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L124-L142),
[`continuous_regime_weight`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L145-L159),
and [`blend_uptake_requests`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L222-L235).

This is an adaptation of the depletion-zone argument in Schnepf & Roose, not a
resolved sub-grid simulation. Colony age and front-specific exposure are
deferred. Transition limits and the nominal overlap time are tested in
[`test_sparse_phosphate_uptake.py`](../tests/test_sparse_phosphate_uptake.py#L39-L85).

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
[`allocate_competing_uptake`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L357-L381)
inside the shared production/qualification
[`blended_uptake_transaction`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L258-L354).
Competition symmetry, proportionality, and conservation are tested in
[`test_continuous_phosphate_uptake.py`](../tests/test_continuous_phosphate_uptake.py#L99-L153)
and end-to-end in
[`test_environment_phosphate_uptake.py`](../tests/test_environment_phosphate_uptake.py#L63-L238).

Accepted root and fungal amounts are summed separately and converted exactly
once using $1\,\mu mol\,P=0.0309738\,mg\,P$ before being added to organism
pools. The transaction is implemented in
[`_apply_blended_uptake_with_diagnostics`](../mycormarl/mycormarl/soil/soil.py#L115-L151)
and its soil-to-pool balance is tested by
[`test_end_to_end_uptake_conserves_soil_plus_free_pool_p`](../tests/test_environment_phosphate_uptake.py#L260-L282).

## Parameters and empirical interpretation

| Parameter | Default | Provenance/status |
|---|---:|---|
| Initial solution Pi | `1 µM` | Order-of-magnitude condition associated with [Vance et al. (2003)](https://doi.org/10.1046/j.1469-8137.2003.00695.x); not a universal mean or extractable-P value. |
| $\theta$, $b_p$ | `0.3`, `239` | Provisional values reported by [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x), attributed there to Barber. Soil-specific calibration is required. |
| $D_l$, $f_l$ | `1e-5 cm² s⁻¹`, `0.308` | Provisional Schnepf–Roose parameterisation; with the defaults, $D_{app}\approx3.86\times10^{-9}$ cm² s⁻¹. |
| Root/fungal $J_{max}$ | `3.26e-6 µmol cm⁻² s⁻¹` | Tinker & Nye values as reported by Schnepf & Roose; separate trait fields are retained. |
| Root/fungal $K_m$ | `5.8e-3 µmol cm⁻³` | Same provisional source and status. |
| Hyphal radius | `5e-4 cm` | Schnepf & Roose. |
| Root radius | `0.01 cm` | Unsourced provisional value requiring species/root-order calibration. |
| $T_{ref}$, $p$ | `1 day`, `2` | Explicit modelling choices, not empirically inferred lifetimes. |

Configuration defaults are declared in
[`EnvConfig`](../mycormarl/mycormarl/params.py#L15-L46); organism kinetics and
radii are in [`PlantTraits`](../mycormarl/mycormarl/plant/traits.py#L17-L34)
and [`FungusTraits`](../mycormarl/mycormarl/fungus/traits.py#L17-L29).

## Numerical qualification and limitations

The production and offline qualification paths call the same transaction.
Focused tests cover unit conversions, buffer identities, grid geometry,
diffusion, sparse/continuous limits, competition, JIT compilation, independent
consumers, and end-to-end balance. The canonical convergence and performance
results are in the [numerical qualification report](qualification/phosphate-numerical-qualification.md).

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
- **Qualification caveat:** The chosen `dt=0.025 day` is the finest tested
  fallback after stricter coupled-output checks; it is not demonstrated temporal
  convergence. Run a timestep sensitivity analysis for each scientific setup.

## References

1. Schnepf, A., & Roose, T. “Modelling the contribution of arbuscular mycorrhizal fungi to plant phosphate uptake.” *New Phytologist* 171 (2006). [DOI](https://doi.org/10.1111/j.1469-8137.2006.01771.x).
2. Jakobsen, I., & Abbott, L. K. “External hyphae of vesicular-arbuscular mycorrhizal fungi associated with *Trifolium subterraneum* L. 1. Spread of hyphae and phosphorus inflow into roots.” *New Phytologist* 120 (1992). [DOI](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x).
3. Vance, C. P., Uhde-Stone, C., & Allan, D. L. “Phosphorus acquisition and use: critical adaptations by plants for securing a nonrenewable resource.” *New Phytologist* 157 (2003). [DOI](https://doi.org/10.1046/j.1469-8137.2003.00695.x).
