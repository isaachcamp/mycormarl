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

The coupling is deliberately delayed across a biological timestep. Resources
visible at its start can fund growth; post-growth geometry then controls uptake;
newly acquired P becomes available for allocation at the next timestep. This
makes credit assignment explicit and prevents same-step resource recycling.

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
| Mortality/reproduction P | cumulative diagnostic fields | mg P | system accounting |

These fields are declared in [`State`](../mycormarl/mycormarl/state.py#L7-L33).

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
[`axisymmetric_edges_from_intervals`](../mycormarl/mycormarl/soil/phosphate_grid.py#L76-L95)
and [`axisymmetric_cylindrical_cell_volumes`](../mycormarl/mycormarl/soil/phosphate_grid.py#L98-L116);
geometry and boundary behaviour are tested in
[`test_axisymmetric_geometry.py`](../tests/test_axisymmetric_geometry.py#L22-L174).

## Coupling pipeline

```text
start-of-step organism pools + allocation actions
  -> trade, stoichiometric growth, maintenance and reproduction
  -> surviving biomass and mortality/export accounting
  -> photosynthetic C fixation
  -> biomass-derived root and hyphal density on the shared grid
  -> stable diffusion/uptake substeps using fixed geometry
  -> accepted soil P credited to organism pools
  -> next observation and timestep
```

The controlling entry point is
[`BaseMycorMarl.step_env`](../mycormarl/mycormarl/environments/base_mycor.py#L257-L453).
It snapshots the start pools before biological allocation
([lines 281–299](../mycormarl/mycormarl/environments/base_mycor.py#L281-L299)),
refreshes both density fields after biomass change
([lines 379–404](../mycormarl/mycormarl/environments/base_mycor.py#L379-L404)),
and then invokes the soil model through
[`step_phosphorus_field`](../mycormarl/mycormarl/environments/base_mycor.py#L455-L471).

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
[`test_realised_growth_updates_geometry_before_soil_stage`](../tests/test_growth_geometry.py#L371-L409)
and [`test_uptake_credit_cannot_fund_growth_in_the_same_step`](../tests/test_environment_phosphate_uptake.py#L241-L258).

## Module boundaries

| Responsibility | Principal symbol | Location |
|---|---|---|
| Configuration and timestep/grid controls | `EnvConfig` | [`params.py`](../mycormarl/mycormarl/params.py#L15-L46) |
| Plant & fungal traits | `SpeciesParams` | [`params.py`](../mycormarl/mycormarl/params.py#L8-L12) |
| Canonical dynamic state | `State` | [`state.py`](../mycormarl/mycormarl/state.py#L7-L33) |
| Biological ordering and coupling | `BaseMycorMarl.step_env` | [`base_mycor.py`](../mycormarl/mycormarl/environments/base_mycor.py#L257-L453) |
| Shared grid construction | `axisymmetric_edges_from_intervals` | [`phosphate_grid.py`](../mycormarl/mycormarl/soil/phosphate_grid.py#L14-L151) |
| Plant geometry | `plant.density_field_from_biomass` | [`roots.py`](../mycormarl/mycormarl/plant/roots.py#L109-L121) |
| Fungal geometry | `fungus.density_field_from_biomass` | [`mycelium.py`](../mycormarl/mycormarl/fungus/mycelium.py#L143-L155) |
| Soil-P update | `evolve_soil_p` | [`soil.py`](../mycormarl/mycormarl/soil/soil.py#L233-L295) |

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
[`test_independent_consumer_mode_keeps_absent_partner_dormant`](../tests/test_review_repairs.py#L73-L95)
and [`test_absorbing_death_removes_real_root_geometry_and_uptake`](../tests/test_review_repairs.py#L96-L121).

The soil domain is closed to P at its top, bottom, outer radius, and symmetry
axis. Organism uptake is consequently an internal soil-to-organism transfer;
mortality and reproduction are explicit exports from the represented system.

## Relationship to the literature

| Source | Relevant contribution | Relationship to this implementation |
|---|---|---|
| [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x) | Couples root/fungal geometry with phosphate transport and uptake; analyses overlapping hyphal depletion zones. | Direct methodological precedent for the model’s coupled geometry–uptake structure, but the code uses a finite-volume axisymmetric grid and a smooth sparse/continuous closure. |
| [Jakobsen et al. (1992)](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x) | Measures external AM hyphal spread and P inflow into roots. | Empirical motivation for treating external hyphae as a spatially extended P-acquisition pathway; not a direct parameter calibration here. |
| [Oyarte Galvez et al. (2025)](https://doi.org/10.1038/s41586-025-08614-x) | Describes extraradical mycelial growth dynamics. | Supplies provisional fungal biomass/geometry parameters; its reported planar saturation density is transformed into the provisional 3-D fungal density used here. |
| [Schnepf et al. (2008)](https://www.jstor.org/stable/24124123) | Describes coupled spatial growth and plant–fungal resource exchange |
| [Bisot et al. (2026)](https://doi.org/10.1073/pnas.2512182123) | Describes coupled spatial growth and plant–fungal resource exchange. | Modern conceptual support for spatially coupled trade and growth; the current model simplifies colony development to a saturated hemisphere. |

## Assumptions and limitations

- **Assumption:** The geometry is axisymmetric and does not represent explicit
  branching topology or angular heterogeneity.
- **Assumption:** Mycelium grows as saturated hemisphere.
- **Assumption:** Plant roots grow as stacked discs with uniform root density,
  and follow exponential depth distribution.
- **Assumption:** Root and fungal geometry is held fixed across all numerical
  soil substeps within one biological step.
- **Assumption:** Newly acquired P is delayed until the next allocation, while
  newly realised biomass changes uptake geometry immediately.
- **Limitation:** Every represented root or hyphal segment remains active
  indefinitely; tissue age, turnover, and growth-front-only absorption are not
  represented.
- **Limitation:** Mortality P leaves the system and reproduction P is exported;
  there is no litter, mineralisation, or recycling pool.
- **Limitation:** Maintenance P is removed from free pools but has no explicit
  destination, so whole-system P conservation must not be claimed for
  maintenance-active trajectories.
- **Interpretation:** The shared grid is a coupling abstraction, not a claim
  that root and fungal architectures are physically continuous at all scales.

## Verification and further reading

The source-level test map is in the [module map](module-map.md). Numerical
convergence, conservation, sensitivity, and performance results are recorded
in the [qualification report](qualification/phosphate-numerical-qualification.md).
Unresolved scientific decisions are maintained in [open questions](open-questions.md).

## References

1. Schnepf, A., & Roose, T. “Modelling the contribution of arbuscular mycorrhizal fungi to plant phosphate uptake.” *New Phytologist* 171 (2006). [DOI](https://doi.org/10.1111/j.1469-8137.2006.01771.x).
2. Jakobsen, I. et al,. “External hyphae of vesicular-arbuscular mycorrhizal fungi associated with *Trifolium subterraneum* L. 1. Spread of hyphae and phosphorus inflow into roots.” *New Phytologist* 120 (1992). [DOI](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x).
3. Oyarte Galvez, L. et al. “A travelling-wave strategy for plant–fungal trade.” *Nature* 639 (2025). [DOI](https://doi.org/10.1038/s41586-025-08614-x).
4. Schnepf, A., Roose, T. & Schweiger, P. "Impact of growth and uptake patterns of arbuscular mycorrhizal fungi on plant phosphorus uptake—a modelling study." _Plant and Soil_ **312**, 85–99 (2008). [DOI](https://www.jstor.org/stable/24124123)
5. Bisot, C. et al. “Carbon-phosphorus exchange rate constrains density-speed trade-off in arbuscular mycorrhizal fungal growth.” *Proceedings of the National Academy of Sciences* 123 (2026). [DOI](https://doi.org/10.1073/pnas.2512182123).
