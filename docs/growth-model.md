# Plant and fungal growth model

> **Scope:** This document covers resource-limited biomass growth, maintenance,
> reproduction, mortality, and conversion of surviving biomass into root and
> hyphal length-density fields. Soil-P diffusion and uptake are covered in the
> [phosphate model](phosphate-model.md); shared scheduling is covered in the
> [model overview](model-overview.md).

## Executive summary

Plant and fungus allocate fractions of their start-of-step free C and P pools
to trade, growth, maintenance, and reproduction. Growth treats C and P as
essential resources: the scarcer resource after conversion by organism-specific
stoichiometric costs limits new dry biomass. Maintenance shortfalls remove
biomass, reproduction exports resources and generates reward, and death does not
replenish C or P resources.

Surviving plant biomass is mapped to a stack of depth-dependent root discs with
uniform within-disc length density. Surviving fungal biomass is mapped to an
external hyphal length and then to a saturated hemisphere. These fields are
the sole geometry inputs to phosphate uptake.

## Resource-limited biomass change

For organism $o\in\{p,f\}$, the realised growth increment is

$$
\Delta G_o=\min\left(\frac{C_{o,g}}{\gamma_{C,o}},
                         \frac{P_{o,g}}{\gamma_{P,o}}\right),
$$

where $C_{o,g}$ is allocated C (g C), $P_{o,g}$ is allocated P (mg P),
$\gamma_{C,o}$ is structural C cost (g C g⁻¹ dry biomass), and
$\gamma_{P,o}$ is structural P cost (mg P g⁻¹ dry biomass). The realised costs
are $\Delta G_o\gamma_{C,o}$ and $\Delta G_o\gamma_{P,o}$; unused portions of
the limiting allocation remain in the free pools. Plant growth is additionally
clipped at its biomass cap.

```python
delta_biomass = jnp.minimum(
    (allocated_c) / grow_c_cost,
    (allocated_p) / grow_p_cost
)
```

Implemented by
[`_grow_biomass_essential_resources`](../mycormarl/mycormarl/growth.py#L8-L20)
and called for plants and fungi in
[`step_plant`](../mycormarl/mycormarl/environments/base_mycor.py#L473-L574) and
[`step_fungus`](../mycormarl/mycormarl/environments/base_mycor.py#L576-L672).
Start-pool timing and the biomass cap are protected by
[`test_newly_fixed_carbon_is_not_available_for_same_step_growth`](../tests/test_base_mycor_refactor.py#L114-L128)
and [`test_plant_growth_at_biomass_cap_charges_only_realised_structure`](../tests/test_base_mycor_refactor.py#L129-L155).

## Maintenance, mortality, reproduction, and trade

Maintenance demand is proportional to active biomass and timestep,

$$
C_{o,m}^{*}=G_o\kappa_{C,o}\Delta t,\qquad
P_{o,m}^{*}=G_o\kappa_{P,o}\Delta t,
$$

with $\kappa_C$ in g C g⁻¹ day⁻¹, $\kappa_P$ in mg P g⁻¹ day⁻¹, and
$\Delta t$ in days. Actual use is the lesser of allocated and required
resource. A deficit is translated to lost biomass using the more severe
stoichiometric deficit:

$$
\Delta G_{o,\mathrm{loss}}=
\max\left(\frac{C_{o,m}^{*}-C_{o,m}}{\gamma_{C,o}},
          \frac{P_{o,m}^{*}-P_{o,m}}{\gamma_{P,o}},0\right).
$$

The current implementation is visible for the plant at
[`base_mycor.py:508–546`](../mycormarl/mycormarl/environments/base_mycor.py#L508-L546)
and fungus at
[`base_mycor.py:606–644`](../mycormarl/mycormarl/environments/base_mycor.py#L606-L644).
Over-allocation is returned to the pools, as tested by
[`test_excess_maintenance_allocation_is_not_wasted`](../tests/test_base_mycor_refactor.py#L288-L315).

Reproduction removes allocated C and P and scores a Cobb–Douglas reward after
both resources are converted to dry-biomass equivalents. Plant C trade and
fungal P trade are calculated from the same start-of-step pools, and incoming
trade is unavailable for same-step growth. See
[`step_env`](../mycormarl/mycormarl/environments/base_mycor.py#L281-L377) and
[`test_incoming_trade_is_not_available_for_same_step_growth`](../tests/test_base_mycor_refactor.py#L209-L221).

Structural P associated with biomass lost to maintenance shortfall is added to
cumulative mortality-loss diagnostics. It is not recycled to soil. Free P used
for maintenance is deducted but currently has no destination; this is an
acknowledged accounting gap rather than a C-only maintenance assumption.

## Plant biomass to root geometry

Total plant dry biomass $G_p$ is converted to fine-root length by

$$
L_{root}=G_p k_{root}\,SRL,
$$

where $k_{root}$ is the root dry-mass fraction (dimensionless), $SRL$ is
specific root length (cm g⁻¹ root dry mass), and $L_{root}$ is cm. Implemented
by [`root_length_from_plant_biomass`](../mycormarl/mycormarl/plant/roots.py#L9-L21).

The cumulative depth distribution is $F(d)=1-\beta^d$. It is normalized over
the intended maximum rooting depth $D_{root}$, which defaults to 150 cm, rather
than over the simulated soil depth. For layer $k$ bounded by
$z_k,z_{k+1}$,

$$
w_k=\frac{F(\min(z_{k+1},D_{root}))
              -F(\min(z_k,D_{root}))}
             {F(D_{root})}.
$$

Consequently, a truncated soil domain represents only its analytical fraction
of the complete root system. Roots below the simulated boundary remain implicit:
their biomass construction and maintenance costs remain in the whole-plant
accounts, but they provide no in-domain absorbing surface. Given a prescribed
uniform root length density $\lambda_{root}$ (cm cm⁻³), each represented layer
receives its own disc radius:

$$
R_k=\sqrt{\frac{L_{root}w_k}
                    {\pi\lambda_{root}(z_{k+1}-z_k)}}.
$$

Thus density is uniform inside each disc but deeper discs expand more slowly
because their assigned length is smaller. Layers below $D_{root}$ are empty.
Partially crossed annular cells are volume averaged; radii beyond the radial soil
boundary are clipped, not redistributed.
Implemented by
[`root_disc_radii_from_biomass`](../mycormarl/mycormarl/plant/roots.py#L60-L88)
and [`axisymmetric_stacked_disc_root_density`](../mycormarl/mycormarl/plant/roots.py#L90-L107).
Tests verify depth-dependent radii, uniform density, conservation before domain
clipping, and clipping behaviour in
[`test_growth_geometry.py`](../tests/test_growth_geometry.py#L91-L193).

## Fungal biomass to hyphal geometry

Fungal dry biomass $G_f$ becomes structural C, tissue volume, and cylindrical
external-hyphal length:

$$
L_h=\frac{G_f\gamma_{C,f}}{M_C\pi r_h^2},
$$

where $\gamma_{C,f}$ is g C g⁻¹ dry biomass, $M_C$ is tissue C density
(g C cm⁻³ tissue), $r_h$ is hyphal radius (cm), and $L_h$ is cm. Spores and
intraradical structures are excluded. Implemented by
[`hyphal_length_from_fungal_biomass`](../mycormarl/mycormarl/fungus/mycelium.py#L7-L26).

The length fills a hemisphere at saturation density $\lambda_{sat}$:

$$
R_f=\left(\frac{3L_h}{2\pi\lambda_{sat}}\right)^{1/3}.
$$

The inverse transformations from colony radius to saturated length and from
length to dry biomass are owned by the same mycelium module. Their composition,
[`fungal_biomass_for_colony_radius`](../mycormarl/mycormarl/fungus/mycelium.py),
provides the radial-fill biomass. Half of that value is used as the fungal
actor-observation reference.

Each annular cell receives $\lambda_{sat}$ times its exact occupied-volume
fraction. Implemented by
[`colony_radius_from_length_axisymmetric`](../mycormarl/mycormarl/fungus/mycelium.py#L28-L30),
[`axisymmetric_hemisphere_cell_fractions`](../mycormarl/mycormarl/fungus/mycelium.py#L78-L106),
and [`axisymmetric_density_from_biomass`](../mycormarl/mycormarl/fungus/mycelium.py#L119-L141).
Partial-front conservation and domain saturation are tested in
[`test_growth_geometry.py`](../tests/test_growth_geometry.py#L195-L289).

Future work should change the assumed fungal geometry with sufficient
justification.

## Parameterisation and literature relationship

| Parameter | Default | Evidence and status |
|---|---:|---|
| Plant $\gamma_C$ | `0.402 g C g⁻¹` | Carrot-root elemental analysis from [Kaur et al. (2022)](https://doi.org/10.1038/s41598-022-20971-5); a root-dominated proxy, not whole-plant calibration. |
| Plant $\gamma_P$ | `1.92 mg P g⁻¹` | Derived dry-mass-weighted carrot value from [Kováčik et al. (2022)](https://doi.org/10.3390/agronomy12112770), an MDPI *Agronomy* paper; independent validation remains required. |
| $k_{root}$, $SRL$ | `0.62`, `25,434.3 cm g⁻¹` | Separate *Daucus carota* medians from the [GRooT database](https://doi.org/10.1111/geb.13179); not matched observations from one specimen. |
| $\beta$, $D_{root}$ | `0.96`, `150 cm` | Provisional depth profile and near-infinite rooting horizon. A shallower simulated domain retains only $F(D_{soil})/F(D_{root})$ of total roots. |
| Fungal $\gamma_C$, $M_C$ | `0.5`, `0.1155 g C cm⁻³` | Provisional values from [Bisot et al. (2026)](https://doi.org/10.1073/pnas.2512182123). |
| Fungal $\gamma_P$ | `40 mg P g⁻¹` | Upper-bound 4% mass fraction reported by the Bisot et al. literature search; underlying evidence still needs validation. |
| $\lambda_{sat}$ | `168.75 cm cm⁻³` | Converted from the 2-D density reported by [Oyarte Galvez et al. (2025)](https://doi.org/10.1038/s41586-025-08614-x) using the stipulated $\rho_{3D}=3\rho_{2D}^2/4$; not a direct 3-D soil measurement. |
| $\lambda_{root}$, $\beta$ | `1 cm cm⁻³`, `0.96` | Provisional inherited geometry choices without empirical calibration. |

Runtime defaults and units are defined in
[`PlantTraits`](../mycormarl/mycormarl/plant/traits.py#L6-L34) and
[`FungusTraits`](../mycormarl/mycormarl/fungus/traits.py#L6-L29), with
construction-time validation immediately below those definitions.

## Assumptions and limitations

- **Assumption:** Both organisms use fixed C:P structural stoichiometry and
  follow Tilman-style essential resource with scarcer resource limiting growth.
- **Assumption:** The plant has a biomass cap; the fungus currently does not,
  but is ultimately limited by geometry.
- **Assumption:** Root allocation uses a static root-mass fraction as a
  marginal conversion from whole-plant biomass; allometry is constant.
- **Assumption:** Fungal biomass is treated as external cylindrical hyphae for
  geometry; spores and intraradical biomass are omitted.
- **Assumption:** Mycelial density remains constant independent of soil
  P concentration.
- **Limitation:** Plant root fraction covers fine- and taproots yet the length
  density specified is fine-root-specific. This likely overestimates absorptive
  surface.
- **Limitation:** Both geometries are spatial closures, not explicit branching
  networks, and their saturation/density parameters require calibration.
- **Limitation:** No maximum tissue age, turnover, dormancy, remobilisation, or
  age-dependent uptake activity is represented.
- **Accounting limitation:** Maintenance P has no destination, while mortality
  P and reproduction P are explicit exports.

## References

1. Bisot, C. et al. “Carbon-phosphorus exchange rate constrains density-speed trade-off in arbuscular mycorrhizal fungal growth.” *PNAS* 123 (2026). [DOI](https://doi.org/10.1073/pnas.2512182123).
2. Oyarte Galvez, L. et al. “A travelling-wave strategy for plant–fungal trade.” *Nature* 639 (2025). [DOI](https://doi.org/10.1038/s41586-025-08614-x).
3. Guerrero-Ramírez, N. R. et al. “Global root traits (GRooT) database.” *Global Ecology and Biogeography* 30 (2021). [DOI](https://doi.org/10.1111/geb.13179).
4. Kaur, P. et al. “Green extraction of bioactive components from carrot industry waste and evaluation of spent residue as an energy source.” *Scientific Reports* (2022). [DOI](https://doi.org/10.1038/s41598-022-20971-5).
5. Kováčik, P. et al. “The Effect of Vermicompost and Earthworms (*Eisenia fetida*) Application on Phytomass and Macroelement Concentration and Tetanic Ratio in Carrot.” *Agronomy* 12 (2022). [DOI](https://doi.org/10.3390/agronomy12112770).
