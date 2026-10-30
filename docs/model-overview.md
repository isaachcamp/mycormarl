# Coupled growth and phosphate model

> **Scope:** This document describes the shared state, axisymmetric geometry,
> and timestep coupling between the organism-growth and soil-phosphate models.
> Growth equations and geometries are detailed in the [growth model](growth-model.md);
> transport and uptake are detailed in the [phosphate model](phosphate-model.md).
> Policy optimisation and the PPO implementation are outside this scope.
>
> **Evidence note:** Statements about current behaviour are traced to source and
> tests. Scientific interpretation and empirical provenance are labelled
> separately from implemented behaviour.

## Executive summary

MycorMARL contains two coupled models on one axisymmetric cylindrical `r-z`
grid. The **growth model** converts plant and fungal carbon (C) and phosphorus
(P) allocations into realised dry biomass, then converts surviving biomass
into root and hyphal length-density fields. Photosynthetically-active biomass
is handled outside this scope. The **phosphate model** diffuses a
finite labile-P inventory, derives soil-solution concentration, computes root
and fungal uptake requests from those density fields, and credits accepted P
to organism pools.

The coupling is deliberately delayed across a numerical environment step. Resources
visible after maintenance and incoming partner trade can fund growth;
post-growth geometry then controls uptake. New soil P and photosynthate become
available for allocation at the next numerical step. Incoming trade participates
inside coupled allocation, as recorded by
[ADR-0020](adr/0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md).

## Shared state and geometry

The dynamic state contains organism dry biomass and free C/P pools, canonical
cellwise labile P, and biomass-derived root/hyphal density fields. Soil P is
stored in µmol per cell; organism P pools are stored in mg. Concentration is a
derived quantity, never a second mutable soil state.

Note that P is a finite resource and must be conserved, whereas C is currently
unbounded in its supply. P tracking is used to ensure conservation and
accountability by the dynamical update steps.

| Quantity | State field | Unit / shape | Ownership |
|---|---|---|---|
| Plant/fungal dry biomass | `plant_biomass`, `fungus_biomass` | g; one value per organism | growth |
| Free organism C | `plant_c_pool`, `fungus_c_pool` | g C | growth/trade |
| Free organism P | `plant_p_pool`, `fungus_p_pool` | mg P | growth/trade/uptake |
| Labile soil P | `soil_labile_p` | µmol P cell⁻¹; `(n_r, n_z)` | phosphate |
| Root/hyphal density | `root_length_density`, `hyphae_length_density` | cm cm⁻³; `(n_r, n_z)` | growth → phosphate |
| Mortality, irreversible free-P loss, reproduction P | cumulative diagnostic fields | mg P | system accounting |

These fields are declared in [`State`](../src/mycormarl/state.py).

The grid edges are configured directly from a maximum radius, maximum depth,
and explicit radial/depth intervals. Each cell is an annular cylindrical shell,

$$
V_{ij}=\pi\left(r_{i+1}^{2}-r_i^{2}\right)\left(z_{j+1}-z_j\right),
$$

where $V_{ij}$ is bulk-soil volume (cm³). Every interval must divide its
corresponding maximum extent into a whole number of uniform cells. An invalid
request is rejected with the nearest valid interval and cell count, making the
actual geometry explicit rather than silently shortening a boundary cell. The
same edges and volumes are used for P amount, absorber density, diffusion
faces, and uptake integration. Implemented by
[`axisymmetric_edges_from_intervals`](../src/mycormarl/soil/phosphate_grid.py)
and [`axisymmetric_cylindrical_cell_volumes`](../src/mycormarl/soil/phosphate_grid.py);
geometry and boundary behaviour are tested in
[`test_axisymmetric_geometry.py`](../tests/test_axisymmetric_geometry.py).

The default `50 cm` radius and `100 cm` depth use `0.05 cm` radial and depth
intervals, giving `1000 × 2000 = 2,000,000` cells. This is four times the cell
count of the earlier `0.1 cm` grid, with the same physical volume and initial
total P. The finer default follows the retained numerical qualification;
reduced domains remain appropriate for tests and development runs.

## Coupling pipeline

```text
start-of-step organism pools + Rate actions (per day)
  -> automatic maintenance and maintenance-deficit mortality
  -> coupled bilateral trade and stoichiometric growth/reproductive output
     (unavailable trade disabled; surplus and finite-rate residue retained)
  -> surviving biomass and mortality/export accounting
  -> photosynthetic C fixation
  -> biomass-derived root and hyphal density on the shared grid
  -> stable diffusion/uptake substeps using fixed geometry
  -> accepted soil P credited to organism pools
  -> next observation and timestep
```

The controlling entry point is
[`BaseMycorMarl.step_env`](../src/mycormarl/environments/base_mycor.py).
It pays maintenance from start pools, integrates trade and biological spending
together, then credits photosynthate. It
refreshes both density fields after biomass change and invokes the soil model
through `step_phosphorus_field`.

Ignoring trade terms for clarity, the implemented timing invariant is:

$$
\Delta G_t=f(C_t,P_t,a_t),\qquad
\lambda_t=g(G_t+\Delta G_t-\Delta G_{\mathrm{loss},t}),\qquad
P_{t+1}=P_t-P_{\mathrm{used},t}+P_{\mathrm{uptake},t}.
$$

Here $a_t$ is the allocation action, $G$ is structural dry biomass (g),
$\lambda$ is length density (cm cm⁻³), and free $P$ is in mg. Thus uptake at
$t$ responds immediately to the newly realised geometry, but it cannot fund
$\Delta G_t$. This is verified by
[`test_realised_growth_updates_geometry_before_soil_stage`](../tests/test_growth_geometry.py)
and [`test_uptake_credit_cannot_fund_growth_in_the_same_step`](../tests/test_environment_phosphate_uptake.py).

Three clocks must be distinguished. `EnvConfig.dt` is the numerical environment
step in days; the soil substep is its seconds-based diffusion/uptake subdivision;
the policy decision interval may contain several complete environment steps.
[`PolicyIntervalMycorMarl`](../src/mycormarl/environments/policy_interval.py)
holds the same Rate action over an integral number of numerical steps and
accumulates their rewards and flow diagnostics, freezing after a terminal
boundary. New soil P and photosynthate are eligible at the next numerical
step, even within one held policy interval. Allocation also has its own
outgoing-hazard subdivision: it holds commands fixed and admits incoming trade
without repeating outer maintenance, acquisition or soil transactions.
Soil subcycling holds geometry fixed and does not repeat maintenance, growth,
trade, or photosynthesis. Increasing soil substeps therefore cannot by itself
remove errors in those biological processes.

New growth supplies absorbing geometry for the full following soil transaction,
rather than a midpoint exposure. This explicit splitting can bias coarse-step
uptake and growth feedback. Numerical comparisons must preserve the physical
horizon, initial resource inventories, and policy decision schedule; holding
the rates fixed does not guarantee exact timestep independence of the split
solver. `max_steps` counts numerical steps (`14600 × 0.025 = 365 days` by
default); the wrapper has its own declared number of policy steps.

## Module boundaries

| Responsibility | Principal symbol | Location |
|---|---|---|
| Configuration and timestep/grid controls | `EnvConfig` | [`params.py`](../src/mycormarl/params.py) |
| Plant & fungal traits | `SpeciesParams` | [`params.py`](../src/mycormarl/params.py) |
| Canonical dynamic state | `State` | [`state.py`](../src/mycormarl/state.py) |
| Biological ordering and coupling | `BaseMycorMarl.step_env` | [`base_mycor.py`](../src/mycormarl/environments/base_mycor.py) |
| Shared grid construction | `axisymmetric_edges_from_intervals` | [`phosphate_grid.py`](../src/mycormarl/soil/phosphate_grid.py) |
| Plant geometry | `plant.density_field_from_biomass` | [`roots.py`](../src/mycormarl/plant/roots.py) |
| Fungal geometry | `fungus.density_field_from_biomass` | [`mycelium.py`](../src/mycormarl/fungus/mycelium.py) |
| Soil-P update | `evolve_soil_p` | [`soil.py`](../src/mycormarl/soil/soil.py) |

The important ownership rule is that geometry modules decide **where absorbing
length exists**; phosphate uptake decides **how much that length requests**;
the competition transaction decides **what the finite cell inventory can
supply**. The environment, rather than the soil package, owns allocation,
growth, trade, mortality, and reward.

## Operating modes and boundaries

`consumer_mode` may be `mixed`, `plant-only`, or `fungus-only` allowing
simulations with both or either. However, `fungus-only` makes little biological
sense due to their obligate nature. Independent modes preserve the two-agent
interface but initialise the absent partner with
zero pools and biomass and mark it dormant. Active death is absorbing: later
actions, fixation, geometry, uptake, and reward remain zero. These contracts
are exercised by
[`test_independent_consumer_mode_keeps_absent_partner_dormant`](../tests/test_review_repairs.py)
and [`test_absorbing_death_removes_real_root_geometry_and_uptake`](../tests/test_review_repairs.py).

For an active partner, an unspecified reset free C or P pool funds exactly one
following automatic-maintenance payment: `kappa_c × initial_biomass × dt` or
`kappa_p × initial_biomass × dt`, respectively. This is distinct from
structural resources implicit in biomass. An explicit numeric pool is a named
scenario override, including qualification fixtures.

The soil domain is closed to P at its top, bottom, outer radius, and symmetry
axis. Organism uptake is consequently an internal soil-to-organism transfer;
mortality and reproduction are explicit exports from the represented system.

Scalar configuration and trait validation runs at environment construction,
before allocating the grid or tracing JAX kernels. Extents, intervals, `dt`,
water content, growth costs, absorber radii and density scales must be finite
and positive; buffer power, initial concentration, maintenance coefficients
and maximum uptake flux may be zero but not negative. Impedance is in `[0,1]`,
CFL safety in `(0,1]`, and reference time and transition exponent are positive.
Invalid production grid intervals are rejected rather than rounded into a
different physical domain. The pure numerical helpers rely on their documented
physical input contracts; their limiting-case guards do not replace validation.
Biological rates use days, while soil diffusion and uptake convert each
interval to seconds using `86400 s day⁻¹`.

## Phosphorus accounting

The common-unit extended ledger, in mg P, is

$$
P_{ledger}=0.0309738\sum_i M_i
 +\sum_{o\in\{plant,fungus\}}\left(
 P_{o,free}+\gamma_{P,o}G_o
 +E_{o,mortality}+E_{o,loss}+E_{o,reproduction}\right).
$$

Here $M_i$ is soil labile P in µmol, $G_o$ is stored dry biomass in grams,
and each $E$ is a cumulative state counter in mg P. `E_loss` is the
legacy-named `cumulative_*_p_maintenance_loss_mg`: it records paid
irreversible free-P loss, not structural P and not unmet demand. The early
implementation left that paid-P pathway undiagnosed; the current counters
make it an explicit export. Its physical receiving pool remains unmodelled.

Diffusion cancels internally, accepted uptake moves soil P to free pools,
fungal trade transfers free P to the plant, and realised growth converts free
P to implicit structural P. Reproduction and maintenance-induced structural
loss leave the active system through their recorded counters. Death remains
absorbing but does not erase residual stored biomass or free pools; those
quantities must remain in the ledger even when their uptake activity is zero.
Omitting them or substituting uptake-active biomass would manufacture a loss.
There is no litter, decomposition, or remobilisation return to soil.

The ledger should remain constant up to numerical error in a closed,
unreset trajectory with fixed traits. Compare initial and final states before
autoreset, include all export counters, and do not add cumulative uptake or
trade again: those are internal transfers. Soil-only balance and non-negative
cell inventories are necessary checks but do not establish whole-system
balance. Tests cover soil-to-pool credit, realised growth at the plant guard,
loss counters, absorbing death, and the extended qualification ledger; see
[`test_environment_phosphate_uptake.py`](../tests/test_environment_phosphate_uptake.py),
[`test_base_mycor_refactor.py`](../tests/test_base_mycor_refactor.py), and
[`test_phosphate_qualification.py`](../tests/test_phosphate_qualification.py).

## Changes since the early phosphate implementation

The early implementation archive records a sequence of working slices, some
with deliberately unavailable soil updates, rather than a second supported
model. Its scientific contracts survive in these three model guides and the
executable tests. Historical phase checklists and pass counts describe those
revisions; they do not prove compatibility with current actions or parameters.

Several early assumptions have been superseded: shortened final grid cells
became explicitly uniform production grids; upper-25-cm P became uniform-domain
P with an optional depth profile; fine-root allocation, fungal P cost and
saturation density were revised; rate commands replaced fractional allocation;
and paid free-P loss gained an explicit export counter. Current equations and
provenance are given here and in the [growth](growth-model.md) and
[phosphate](phosphate-model.md) guides. Reproducing an archived result requires
its own source revision, configuration and runtime, not today's defaults.

## Relationship to the literature

| Source | Relevant contribution | Relationship to this implementation |
|---|---|---|
| [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x) | Couples root/fungal geometry with phosphate transport and uptake; analyses overlapping hyphal depletion zones. | Direct methodological precedent for the model’s coupled geometry–uptake structure, but the code uses a finite-volume axisymmetric grid and a smooth sparse/continuous closure. |
| [Jakobsen et al. (1992)](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x) | Measures external AM hyphal spread, density profiles, and P inflow into roots. | Supplies the `2,000 cm cm⁻³` estimated local saturation-density default from the lower end of observed upper profiles; it is not treated as a universal AMF constant. |
| [Oyarte Galvez et al. (2025)](https://doi.org/10.1038/s41586-025-08614-x) | Describes extraradical mycelial growth dynamics. | Supports the provisional fungal growth and hemispherical-geometry relationship, but no longer supplies the saturation-density default. |
| [Schnepf et al. (2008)](https://www.jstor.org/stable/24124123) | Examines fungal growth and uptake patterns. | Related modelling context; the present geometry is a biomass-derived spatial closure. |
| [Bisot et al. (2026)](https://doi.org/10.1073/pnas.2512182123) | Describes coupled spatial growth and plant–fungal resource exchange. | Modern conceptual support for spatially coupled trade and growth; the current model simplifies colony development to a saturated hemisphere. |

## Assumptions and limitations

### Soil

- **Assumption:** The soil is represented by an axisymmetric cylindrical
  domain. Radial and depth heterogeneity are represented; angular variation
  and separate individual-absorber ownership maps are not.
- **Assumption:** Only finite labile inorganic P is modelled. Organic P,
  occluded P, mineralisation, immobilisation, weathering, and fertiliser input
  are omitted.
- **Assumption:** Labile P is related to solution concentration by an
  instantaneous linear buffer. Sorption/desorption kinetics and nonlinear
  buffering are omitted.
- **Assumption:** Soil water content, buffer power, diffusion coefficient,
  impedance, and uptake parameters are spatially and temporally constant.
- **Assumption:** P moves through soil only by diffusion, with no convective
  transport or preferential flow.
- **Assumption:** The top, bottom, outer radial, and symmetry-axis boundaries
  are closed to P. There is no leaching or external P renewal.
- **Assumption:** Grid cells are homogenised; sub-cell depletion around roots
  and hyphae is represented only by the sparse and continuous uptake closures.
- **Assumption:** Roots and fungi compete for the same local P inventory, and
  cellwise uptake cannot exceed that inventory.
- **Limitation:** Soil properties are not altered by roots, hyphae, exudates,
  microbial activity, pH, or depletion history.

### Fungus

- **Assumption:** Fungal biomass is represented as living, extraradical,
  uptake-active hyphae. Spores, vesicles, arbuscules, and intraradical
  mycelium are omitted from the geometry.
- **Assumption:** Mycelium grows as a centred, isotropic saturated hemisphere.
  Explicit branching, network topology, anastomosis, gaps, and directional
  growth are not represented.
- **Assumption:** Hyphal radius, tissue density, structural C:P stoichiometry,
  and uptake parameters are constant.
- **Assumption:** Hyphal density is independent of fungal age, C status, P
  status, plant signalling, and local soil conditions.
- **Limitation:** Existing hyphae remain active indefinitely unless the fungal
  organism dies; local senescence, turnover, fragmentation, and abandoned
  territory are not represented.
- **Limitation:** Fungal biomass is converted directly to hyphal length,
  without a construction delay or a distinction between structural and
  metabolically active biomass.
- **Limitation:** Fungal mortality is represented as an organism-level event,
  rather than spatially or tissue-specific mortality.

### Plant

- **Assumption:** All represented fine-root length is living, uniformly
  distributed within each disc, and equally active for P uptake.
- **Assumption:** Root length is a static allometric conversion from whole-
  plant biomass and follows the prescribed depth profile throughout plant
  development.
- **Limitation:** Root age, turnover, senescence, damage, branching
  topology, root hairs, and local physiological differences are not
  represented.
- **Limitation:** Roots outside the simulated soil domain remain included in
  whole-plant biomass and maintenance costs but cannot contribute to uptake.
- **Assumption:** Plant biomass compartments, structural C:P composition,
  root allocation, and allometry are fixed.
- **Assumption:** Photosynthetic carbon input is proportional to leaf biomass
  and constant per unit leaf biomass.
- **Limitation:** Irradiance, photoperiod, temperature, CO2, water status,
  leaf age, P status, and canopy development do not affect photosynthesis.
- **Limitation:** Carbon fixation is an external input; atmospheric CO2 and
  carbon supply are not finite state variables.
- **Limitation:** Biosynthetic or growth-respiration costs are not represented
  separately from structural carbon requirements.
- **Assumption:** The plant biomass cap is a numerical guard, not a genetic
  maximum or an emergent physiological carrying capacity.

### Growth, storage, and mortality

- **Assumption:** C and P are essential growth and reproductive resources with fixed
  stoichiometric requirements. Uptake can accumulate P in a free pool, but no
  separate regulated luxury-storage compartment, resource substitution,
  remobilisation, or internal recycling is represented.
- **Assumption:** Maintenance is paid before coupled trade, growth and reproduction;
  a maintenance deficit is converted immediately into biomass loss
  according to the more severe resource deficit.
- **Limitation:** Structural P lost through mortality is exported or discarded
  rather than returned to soil.
- **Assumption:** Reproduction is an instantaneous resource export and reward,
  not offspring production with establishment, survival, or demographic
  feedback.
- **Assumption:** Retention is the remaining free pool, with no controlled
  storage hazard or distinct reserve tissue.
- **Assumption:** New soil P and photosynthate cannot finance allocation in the
  same numerical step. Incoming trade can; newly realised biomass changes
  uptake geometry immediately.
- **Limitation:** Age structure, lifespan, dormancy, developmental stage, and
  seasonal phenology are not represented.
- **Limitation:** Plant and fungal `kappa_p` are irreversible free-P losses
  rather than recyclable biological or soil pools. Both default to zero;
  positive values are sensitivity assumptions for leakage or unrecovered
  turnover rather than measured P-maintenance requirements. Structural P is
  represented separately by `gamma_p`.

### Plant-fungal interaction and agent objective

- **Assumption:** The symbiosis is pre-established. Colonisation, recognition,
  infection, arbuscule formation, compatibility, and initiation costs are
  omitted.
- **Assumption:** Trade is represented as direct resource transfer, with no
  explicit transport delay, exchange-interface capacity, or spatial separation
  between plant and fungus.
- **Assumption:** Trade does not require a minimum local contact or
  colonisation threshold beyond both organisms being operational.
- **Limitation:** Conflict over exchange rates, sanctions, cheating, and
  partner choice are represented only indirectly through learned policies.
- **Assumption:** Reproductive reward is an adequate proxy for evolutionary
  success. Policies optimise expected cumulative reward over a finite episode,
  rather than lifetime reproductive value, geometric-mean fitness, or
  long-run population persistence.
- **Interpretation:** A learned policy is a behavioural strategy produced by
  optimisation, not evidence of a globally optimal strategy or an
  evolutionarily stable equilibrium.

- **Interpretation:** The shared grid is a coupling abstraction, not a claim
  that root and fungal architectures are physically continuous at all scales.

## Verification and further reading

The source-level test map is in the [module map](module-map.md). Numerical
convergence, conservation, sensitivity, and performance results are recorded
in the [qualification report](qualification/phosphate-numerical-qualification.md).
Unresolved scientific decisions are maintained in [open questions](open-questions.md).

### Running and checking the model

From the repository root, run the small examples through the production soil
transaction:

```sh
uv run python scripts/phosphate_examples.py --mode all
```

`plant-only`, `fungus-only`, and `mixed` select individual examples. They report
consumer uptake, inventory, regime weights, cap frequency, soil substeps and
P-balance error. They are numerical demonstrations, not calibrated ecological
or learning results. To regenerate reduced-domain qualification separately
from the retained evidence:

```sh
uv run python scripts/phosphate_qualification.py \
  --skip-target-benchmark --output-dir /tmp/mycormarl-phosphate-qualification
```

Omit `--skip-target-benchmark` to include the target-grid benchmark. The runner
emits JSON configurations, diagnostics and comparison tables alongside its
Markdown report. Keep generated runs outside the tracked documentation and
retain source/configuration identities when using them as scientific evidence.

## References

1. Schnepf, A., & Roose, T. “Modelling the contribution of arbuscular mycorrhizal fungi to plant phosphate uptake.” *New Phytologist* 171 (2006). [DOI](https://doi.org/10.1111/j.1469-8137.2006.01771.x).
2. Jakobsen, I. et al,. “External hyphae of vesicular-arbuscular mycorrhizal fungi associated with *Trifolium subterraneum* L. 1. Spread of hyphae and phosphorus inflow into roots.” *New Phytologist* 120 (1992). [DOI](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x).
3. Oyarte Galvez, L. et al. “A travelling-wave strategy for plant–fungal trade.” *Nature* 639 (2025). [DOI](https://doi.org/10.1038/s41586-025-08614-x).
4. Schnepf, A., Roose, T. & Schweiger, P. "Impact of growth and uptake patterns of arbuscular mycorrhizal fungi on plant phosphorus uptake—a modelling study." _Plant and Soil_ **312**, 85–99 (2008). [DOI](https://www.jstor.org/stable/24124123)
5. Bisot, C. et al. “Carbon-phosphorus exchange rate constrains density-speed trade-off in arbuscular mycorrhizal fungal growth.” *Proceedings of the National Academy of Sciences* 123 (2026). [DOI](https://doi.org/10.1073/pnas.2512182123).
