# Why the phosphate model needs numerical qualification

> **Scope:** This summary explains the purpose, design, mathematics, and current
> results of the phosphate qualification study. It covers spatial and temporal
> refinement, the diffusion CFL limit, conservation checks, sensitivity studies,
> and performance measurements. It does not establish empirical agreement with
> measured plant or fungal uptake.
>
> **Evidence note:** Statements about what the study executes are code-backed.
> Interpretations of why particular outputs are sensitive are identified as
> interpretations rather than demonstrated mechanisms.

## Executive summary

The model solves continuous diffusion and uptake processes using a finite number
of spatial cells and timesteps. Those numerical choices can change predicted
uptake, competition, biomass, and colony extent even when every biological
parameter is unchanged. Qualification asks whether the chosen discretisation is
fine enough that the reported scientific outputs are acceptably insensitive to
further refinement, while remaining affordable to run.

This is why the study runs several spatial intervals and several timesteps. A
coarse result is compared with the next finer or smaller result. If important
outputs agree within the declared 5% tolerance, the coarser choice is treated as
adequate for this qualification fixture. This is a practical convergence test,
not proof of an exact solution.

The CFL limit answers a narrower question: **how long may one explicit diffusion
update be before it can become unstable or create negative cell inventories?**
The code automatically divides a biological timestep into stable soil substeps.
CFL compliance is necessary for this explicit solver, but it does not show that
the timestep is accurate. The current study supports the selected spatial
interval of 0.1 cm, but the selected 0.025-day biological timestep is only the
finest tested fallback; it has not yet demonstrated temporal convergence.

## Verification, qualification, and validation

These terms answer different questions:

| Activity | Question | Evidence in this repository |
|---|---|---|
| Unit/integration verification | Does the code implement its stated numerical contracts? | Tests for conservation, positivity, subcycling, diagnostic equivalence, and comparison arithmetic |
| Numerical qualification | Are chosen \(\Delta r,\Delta z,\Delta t\) adequate for the defined scenarios and outputs? | Refinement matrices, conservation metrics, sensitivity tables, and benchmarks |
| Empirical validation | Does the model reproduce observations with credible parameters? | Not performed by this qualification study |

The distinction matters. A perfectly conservative, stable program can still
give a resolution-dependent answer; a grid-converged model can still represent
the biology poorly. Verification and solution qualification address numerical
error, whereas validation requires comparisons with independent observations.
This separation follows the standard verification-and-validation distinction
described by Oberkampf and Trucano
([DOI 10.1016/S0376-0421(02)00005-2](https://doi.org/10.1016/S0376-0421(02)00005-2)).

## Where qualification sits in the codebase

| Responsibility | Symbol | Location |
|---|---|---|
| Define scenario matrices and generate results | `run_studies` | [`scripts/phosphate_qualification.py`, lines 441–575](../scripts/phosphate_qualification.py#L441-L575) |
| Run isolated, fixed-density soil scenarios | `run_fixed_soil_scenario` | [`scripts/phosphate_qualification.py`, lines 123–233](../scripts/phosphate_qualification.py#L123-L233) |
| Run coupled growth–uptake scenarios | `run_coupled_scenario` | [`scripts/phosphate_qualification.py`, lines 249–351](../scripts/phosphate_qualification.py#L249-L351) |
| Compare a candidate against a finer reference | `reference_relative_change` | [`phosphate_qualification.py`, lines 8–23](../mycormarl/mycormarl/soil/phosphate_qualification.py#L8-L23) |
| Calculate the explicit diffusion limit | `explicit_diffusion_cfl_seconds` | [`phosphate_diffusion.py`, lines 135–155](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L135-L155) |
| Convert the biological step into stable substeps | `required_diffusion_substeps` | [`phosphate_diffusion.py`, lines 158–172](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L158-L172) |
| Execute the production subcycled solver | `evolve_soil_p` | [`soil.py`, lines 233–295](../mycormarl/mycormarl/soil/soil.py#L233-L295) |
| Observe the same solver during qualification | `evolve_soil_p_with_diagnostics` | [`soil.py`, lines 298–348](../mycormarl/mycormarl/soil/soil.py#L298-L348) |
| Project annual runtime | `annual_runtime_projection` | [`phosphate_qualification.py`, lines 26–46](../mycormarl/mycormarl/soil/phosphate_qualification.py#L26-L46) |

The diagnostic path deliberately calls the production diffusion–uptake
transaction rather than duplicating the scientific calculation. A test verifies
that diagnosed and production evolution produce the same soil and organism
pools
([`test_phosphate_qualification.py`, lines 166–195](../tests/test_phosphate_qualification.py#L166-L195)).

## Why test several spatial intervals?

The continuous soil domain is represented by annular finite volumes. A spatial
interval determines:

- how accurately concentration gradients and depletion fronts are represented;
- the distance over which neighbouring concentrations are differenced;
- cell volumes, face areas, and diffusion conductances;
- how sharply root and fungal geometry can occupy the soil; and
- the number of cells and therefore memory and compute cost.

For a characteristic uniform interval \(h\), reducing \(h\) provides more
samples of the spatial solution. If a scientific output \(Q\) changes
substantially from \(h\) to a finer \(h_{\mathrm{ref}}\), the coarse grid is
influencing that output:

\[
\epsilon_h(Q)
=
\frac{\left|Q_h-Q_{h_{\mathrm{ref}}}\right|}
{\max\left(\left|Q_{h_{\mathrm{ref}}}\right|,Q_{\mathrm{floor}}\right)}.
\]

Here \(Q_{\mathrm{floor}}\) prevents meaningless relative errors when both
values are effectively zero. The implementation is
[`reference_relative_change`](../mycormarl/mycormarl/soil/phosphate_qualification.py#L8-L23).

The study tests intervals of 0.1, 0.05, and 0.025 cm. Each candidate is compared
with the next finer grid, rather than assuming the finest tested answer is exact
([`scripts/phosphate_qualification.py`, lines 453–457 and 475–479](../scripts/phosphate_qualification.py#L453-L479)).
It examines fixed-geometry soil outputs and coupled outputs including uptake,
consumer shares, free pools, biomass, and spatial extents
([`scripts/phosphate_qualification.py`, lines 469–500](../scripts/phosphate_qualification.py#L469-L500)).

The current 0.1 cm candidate differs from 0.05 cm by at most 0.254% for the
fixed-soil metrics and 3.030% for the coupled metrics, passing the declared 5%
gate
([qualification results, lines 38–43](../docs/qualification/phosphate-numerical-qualification.md#L38-L43)).
This supports 0.1 cm **for the tested reduced-domain scenarios and selected
metrics**. It is not a universal guarantee for every parameterisation or a
sharper spatial phenomenon.

## Why test several temporal intervals?

There are two distinct time intervals:

1. the **biological timestep** \(\Delta t_{\mathrm{bio}}\), at which actions,
   trade, growth, mortality, geometry, and soil evolution are coupled; and
2. the **soil substep** \(\delta t\), used repeatedly inside a biological step
   so explicit diffusion remains stable.

Changing \(\Delta t_{\mathrm{bio}}\) can change results even if every diffusion
substep satisfies CFL. Uptake is nonlinear in concentration, finite inventory
can cap competing requests, geometry and resource pools are updated at discrete
times, and the pipeline applies processes in a defined order. A larger interval
allows more change to accumulate before those quantities and interactions are
re-evaluated.

The study tests biological timesteps of 0.025, 0.05, 0.1, 0.2, and 0.4 days
([`scripts/phosphate_qualification.py`, lines 34–39](../scripts/phosphate_qualification.py#L34-L39)).
As with the grid study, each candidate is compared with the next smaller
timestep
([`scripts/phosphate_qualification.py`, lines 448–452 and 470–474](../scripts/phosphate_qualification.py#L448-L474)).

Fixed-geometry soil outputs changed by less than 0.5% across these candidates,
but the coupled endpoint metrics changed by roughly 98–102%. Consequently, none
of the larger timestep candidates passed
([qualification results, lines 29–36 and 65–70](../docs/qualification/phosphate-numerical-qualification.md#L29-L36)).
The configuration therefore uses 0.025 day, the finest value tested
([`params.py`, lines 29–35](../mycormarl/mycormarl/params.py#L29-L35)).

This result needs careful interpretation: there is no smaller reference for
0.025 day, so its selection is a fallback, **not evidence of temporal
convergence**. A finer follow-up study, longer-horizon comparison, or revision
of overly sensitive endpoint metrics is still required. The very large coupled
changes alongside small fixed-soil changes suggest that the sensitivity arises
in the coupled biological trajectory or its endpoint metric, but the existing
study does not by itself identify the mechanism.

## What is CFL?

CFL stands for Courant–Friedrichs–Lewy, after the foundational analysis by
Courant, Friedrichs and Lewy. In general, a CFL condition relates timestep size
to spatial resolution and propagation speed so a discrete explicit update does
not advance information farther than its numerical representation can support
([DOI 10.1007/BF01448839](https://doi.org/10.1007/BF01448839)).

For this diffusion implementation, the code calculates a more specific
cellwise positivity ceiling. Let:

- \(M_i\) be labile P amount in cell \(i\), in µmol;
- \(V_i\) be the cell volume, in cm\(^3\);
- \(C_i\) be solution concentration, in µmol cm\(^{-3}\);
- \(\theta\) be volumetric water content;
- \(b_p\) be linear volumetric buffer power;
- \(G_{ij}\) be the conductance of the face between cells \(i\) and \(j\), in
  cm\(^3\) s\(^{-1}\); and
- \(\delta t\) be the explicit soil timestep, in seconds.

The stored amount and driving concentration are related by

\[
M_i=V_i(\theta+b_p)C_i.
\]

For a face of area \(A_{ij}\) and centre-to-centre distance \(d_{ij}\),

\[
G_{ij}
=D_l\theta f_l\frac{A_{ij}}{d_{ij}},
\]

where \(D_l\) is the solution diffusion coefficient in cm\(^2\) s\(^{-1}\) and
\(f_l\) is the dimensionless impedance factor. The code constructs these
conductances in
[`axisymmetric_diffusion_conductances`](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L41-L79).

The explicit amount update is

\[
M_i^{n+1}
=M_i^n+\delta t\sum_jG_{ij}(C_j^n-C_i^n).
\]

To keep the coefficient multiplying the old amount non-negative, each cell must
satisfy

\[
\delta t
\leq
\frac{V_i(\theta+b_p)}
{\sum_jG_{ij}}.
\]

The global ceiling is the most restrictive cell:

\[
\Delta t_{\mathrm{CFL}}
=
\min_i
\frac{V_i(\theta+b_p)}
{\sum_jG_{ij}}.
\]

This exact cellwise calculation is implemented at
[`phosphate_diffusion.py`, lines 135–155](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L135-L155).
The environment uses only a safety fraction \(s=0.8\) of the ceiling and chooses

\[
N_{\mathrm{sub}}
=
\max\left(
1,
\left\lceil
\frac{\Delta t_{\mathrm{bio}}}
{s\Delta t_{\mathrm{CFL}}}
\right\rceil
\right),
\qquad
\delta t=\frac{\Delta t_{\mathrm{bio}}}{N_{\mathrm{sub}}}.
\]

The CFL value and substep count are computed once because grid geometry,
buffering, and diffusion parameters are static
([`base_mycor.py`, lines 76–117](../mycormarl/mycormarl/environments/base_mycor.py#L76-L117)).
Tests verify the calculated ceiling, the substep schedule, non-negativity, and
conservation through repeated substeps
([`test_phosphate_diffusion.py`, lines 337–347](../tests/test_phosphate_diffusion.py#L337-L347);
[`test_phosphate_diffusion.py`, lines 410–424](../tests/test_phosphate_diffusion.py#L410-L424)).

### What CFL does—and does not—tell us

CFL/subcycling tells us:

- whether the explicit diffusion update is within its derived positivity limit;
- how many diffusion–uptake substeps a biological step requires; and
- part of the computational cost associated with a grid and timestep.

It does **not** tell us:

- that concentration gradients are spatially resolved;
- that uptake and coupled growth are temporally converged;
- that 0.8 of the limit is sufficiently accurate;
- that biological parameters are empirically correct; or
- that the scientific model is valid.

On a roughly uniform diffusion grid, the stable explicit timestep typically
scales like \(h^2/D_{\mathrm{app}}\). Halving the interval can therefore require
approximately four times as many diffusion substeps, as well as roughly four
times as many cells in this two-dimensional axisymmetric grid. This is why
qualification must balance refinement against cost rather than selecting the
smallest possible interval automatically.

## What the full study is doing

The deterministic qualification matrix contains several complementary checks:

| Study | Values or modes | Purpose |
|---|---|---|
| Concentration response | 0.1–10 µM; plant-only, fungus-only, mixed | Exercise nonlinear uptake across the intended concentration range |
| Timestep refinement | 0.025–0.4 day | Measure temporal discretisation sensitivity |
| Grid refinement | 0.025–0.1 cm | Measure spatial discretisation sensitivity |
| Consumer modes | plant-only, fungus-only, mixed | Separate each sink and their competition |
| Sparse/continuous transition | several \(T_{\mathrm{ref}}\) and \(p\) values | Expose sensitivity to provisional transition parameters |
| Coupled trajectories | growth, trade, geometry, uptake | Check effects that fixed-density soil runs deliberately isolate |
| P balances | soil-to-pool and extended ledger | Detect lost or created phosphorus |
| Performance | compilation, warmed runtime, array memory, annual projection | Test whether the selected resolution is usable |

The scenario construction is visible in
[`run_studies`](../scripts/phosphate_qualification.py#L441-L575). The study uses a
reduced 2 cm by 2 cm domain with an internal topsoil front for the convergence
matrix, then benchmarks the full 500 by 1000-cell target grid
([qualification results, lines 55–63](../docs/qualification/phosphate-numerical-qualification.md#L55-L63)).

The fixed-density scenarios isolate diffusion and uptake from changing organism
geometry. The coupled scenarios then test whether a numerically modest local
difference is amplified by growth, resource allocation, or spatial expansion.
This two-stage structure is useful diagnostically: it shows whether sensitivity
first appears in the soil calculation or only after feedback through the coupled
model.

## Conservation and qualitative checks

Refinement is not the only requirement. Every scenario also reports or tests:

- P balance between soil removal and organism pool credit;
- extended P balance including structural biomass and explicit exports/losses;
- non-negative final soil inventory;
- surface-to-bulk concentration ratios within \([0,1]\);
- the fraction of demand cells capped by finite inventory;
- uptake response to concentration; and
- deterministic repeatability.

The executable contracts are in
[`tests/test_phosphate_qualification.py`, lines 61–280](../tests/test_phosphate_qualification.py#L61-L280).
The present coupled balance fixture sets maintenance demand and allocation to
zero because maintenance P still lacks an explicit destination. Its successful
balance must not be generalized to maintenance-active trajectories.

## Performance qualification

A converged choice is not useful if it makes the intended annual simulation or
MARL training infeasible. The benchmark distinguishes:

- compilation plus first execution, a one-off JAX cost;
- warmed soil-only time per step;
- warmed complete deterministic environment time per step;
- concrete state and cached-geometry array bytes;
- a formula-based temporary-array estimate; and
- an annual runtime projection.

The projection is

\[
N_{\mathrm{year}}
=\left\lceil\frac{365}{\Delta t_{\mathrm{bio}}}\right\rceil,
\qquad
T_{\mathrm{year}}=N_{\mathrm{year}}T_{\mathrm{step}}.
\]

It is implemented by
[`annual_runtime_projection`](../mycormarl/mycormarl/soil/phosphate_qualification.py#L26-L46),
while measurement and memory accounting are performed by
[`benchmark_environment`](../scripts/phosphate_qualification.py#L370-L438).
The projection excludes policy inference, learning, logging, data transfer, and
other training overhead, so it is a core-model estimate rather than a complete
experiment budget.

## Relationship to the literature

| Source | Relevant contribution | Relationship to this qualification |
|---|---|---|
| [Courant, Friedrichs & Lewy (1928)](https://doi.org/10.1007/BF01448839) | Foundational condition connecting discrete time and space steps for convergence/stability | The code uses the CFL name for its derived explicit diffusion positivity ceiling and substep scheduler |
| [Eymard, Gallouët & Herbin (2000)](https://doi.org/10.1016/S1570-8659(00)07005-8) | Finite-volume discretisation of conservation laws | The soil solver adopts cell inventories, face conductances, and conservative neighbour transfers; the qualification tests the resulting discrete solution |
| [Oberkampf & Trucano (2002)](https://doi.org/10.1016/S0376-0421(02)00005-2) | Distinguishes code/solution verification from empirical model validation and emphasizes quantified numerical error | The repository’s refinement and conservation studies are numerical verification/qualification, not experimental validation |
| [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x) | Mechanistic model of phosphate uptake by roots and external mycorrhizal hyphae | Provides scientific precedent for the uptake problem; it does not itself validate this repository’s axisymmetric discretisation, coupling, or chosen intervals |

## Assumptions, limitations, and next evidence needed

- **Code-backed:** Qualification diagnostics use the production scientific
  transaction, and tests check equivalence.
- **Code-backed:** The 5% rule is applied to selected outputs, not the entire
  spatial concentration field.
- **Code-backed:** Spatial candidates passed the current reduced-domain gate.
- **Limitation:** The selected 0.025-day timestep has no finer reference and is
  not demonstrated converged.
- **Limitation:** Successive-pair agreement is practical evidence, not an
  estimated formal order of convergence or comparison with an analytical
  solution.
- **Limitation:** The two-day reduced-domain fixture cannot exercise every
  spatial and temporal scale in a year-long full-domain run.
- **Limitation:** Fixed actions and disabled maintenance simplify the coupled
  qualification trajectory.
- **Limitation:** The 5% tolerance is a declared modelling choice; it is not
  derived from experimental uncertainty or a downstream decision threshold.
- **Interpretation:** The coupled temporal failure may reflect sensitive
  endpoint pools or discrete biological feedback rather than the soil solver,
  but targeted diagnostics are needed to establish the cause.
- **Next evidence:** Add at least one timestep below 0.025 day, inspect
  time-series or integrated coupled metrics rather than endpoints alone, and
  repeat selected cases over a longer horizon before making a strong temporal
  adequacy claim.

## Conclusion

Qualification exists because stability, conservation, and scientific accuracy
are different properties. CFL-based subcycling prevents the explicit diffusion
step from exceeding its derived positivity limit. Conservation tests ensure P
is not numerically created or lost along the covered pathways. Spatial and
temporal refinement tests ask whether predictions depend materially on
discretisation. Sensitivity studies expose provisional model choices, and
benchmarks test affordability.

The current evidence supports the 0.1 cm grid for the defined qualification
scenarios. It does not yet establish temporal convergence at 0.025 day, nor does
it validate the model against experiments.

## References

1. Courant, R., Friedrichs, K., and Lewy, H. “Über die partiellen
   Differenzengleichungen der mathematischen Physik.” *Mathematische Annalen*
   100 (1928): 32–74.
   [DOI 10.1007/BF01448839](https://doi.org/10.1007/BF01448839).
2. Eymard, R., Gallouët, T., and Herbin, R. “Finite Volume Methods.”
   *Handbook of Numerical Analysis* 7 (2000): 713–1020.
   [DOI 10.1016/S1570-8659(00)07005-8](https://doi.org/10.1016/S1570-8659(00)07005-8).
3. Oberkampf, W. L., and Trucano, T. G. “Verification and Validation in
   Computational Fluid Dynamics.” *Progress in Aerospace Sciences* 38(3)
   (2002): 209–272.
   [DOI 10.1016/S0376-0421(02)00005-2](https://doi.org/10.1016/S0376-0421(02)00005-2).
4. Schnepf, A., and Roose, T. “Modelling the Contribution of Arbuscular
   Mycorrhizal Fungi to Plant Phosphate Uptake.” *New Phytologist* 171(3)
   (2006): 669–682.
   [DOI 10.1111/j.1469-8137.2006.01771.x](https://doi.org/10.1111/j.1469-8137.2006.01771.x).
