# Plant and fungal growth model

> **Scope:** This document covers resource-limited biomass growth, carbon
> maintenance, irreversible free-P loss, reproduction, mortality, and
> conversion of surviving biomass into root and hyphal length-density fields. Soil-P diffusion and uptake are covered in the
> [phosphate model](phosphate-model.md); shared scheduling is covered in the
> [model overview](model-overview.md).

## Executive summary

Plant and fungus pay unavoidable carbon maintenance and irreversible free-P
loss from their start-of-step pools, then execute a Rate action
`[trade, growth, reproduction, storage]` in non-negative per-day rates.
Trade is removed first; competing growth, reproduction, and storage rates
select allocations from the remaining C and P. Growth treats C and P as
essential resources: the scarcer resource after conversion by
organism-specific stoichiometric costs limits new
dry biomass. Payment shortfalls remove biomass, reproduction exports
resources and generates reward, and death does not replenish C or P resources.

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
are $\Delta G_o\gamma_{C,o}$ and $\Delta G_o\gamma_{P,o}$; all unspent offered
resources, including surplus complementary resource, remain in the free pools.
Plant growth is limited to remaining biomass-guard capacity **before** charging
these costs, so blocked growth consumes neither C nor P.

```python
delta_biomass = jnp.minimum(
    (allocated_c) / grow_c_cost,
    (allocated_p) / grow_p_cost
)
```

Implemented by
[`_grow_biomass_essential_resources`](../mycormarl/mycormarl/growth.py)
and called for plants and fungi in
[`step_plant`](../mycormarl/mycormarl/environments/base_mycor.py) and
[`step_fungus`](../mycormarl/mycormarl/environments/base_mycor.py).
Start-pool timing and the biomass cap are protected by
[`test_newly_fixed_carbon_is_not_available_for_same_step_growth`](../tests/test_base_mycor_refactor.py)
and [`test_plant_growth_at_biomass_cap_charges_only_realised_structure`](../tests/test_base_mycor_refactor.py).

The default plant guard is `50 g DM`, selected as headroom above a
`25--35 g DM` favourable carrot trajectory rather than as a carrying-capacity
estimate. Contact invalidates a qualification trajectory. Actor scaling is
independent: `biomass_observation_reference=50 g DM` preserves the previous
policy-input scale without coupling it to growth truncation. See
[ADR-0011](adr/0011-separate-the-plant-biomass-guard-from-policy-observation-scaling.md).

## Plant carbon input

The plant receives apparent-gross carbon according to

$$
\Delta C_{fix}=G_p k_{leaf} a_{mass}\Delta t,
$$

where $G_p$ is whole-plant dry biomass (g), $k_{leaf}$ is the leaf dry-mass
fraction, $a_{mass}$ is g C g⁻¹ leaf dry mass for one reference day, and
$\Delta t$ is days. The selected `amass = 0.05` is a daily carbon budget for
the 16 h light regime used to parameterise it, not fixation under literal
24 h illumination. The current implementation spreads this budget uniformly
through numerical time.

A future diurnal implementation can preserve the parameter and its units by
multiplying by a dimensionless $f_{light}(t)$ whose one-day integral is one
day. The present behaviour is $f_{light}=1$. The selected value is an
acknowledged apparent-gross approximation: the source data do not isolate
biochemical gross photosynthesis, and photorespiration remains embedded.

## Carbon maintenance, irreversible P loss, mortality, reproduction, and trade

Carbon-maintenance demand and irreversible free-P loss are proportional to
active biomass and timestep,

$$
C_{o,m}^{*}=G_o\kappa_{C,o}\Delta t,\qquad
P_{o,m}^{*}=G_o\kappa_{P,o}\Delta t,
$$

with $\kappa_C$ in g C g⁻¹ day⁻¹, $\kappa_P$ in mg P g⁻¹ day⁻¹, and
$\Delta t$ in days. $\kappa_P$ has the same irreversible-loss interpretation
for plant and fungus; it does not include P immobilised in structure, which is
already represented by $\gamma_P$. Actual use is the lesser of the
start-of-step free pool and required resource. A deficit is translated to lost
biomass using the more severe stoichiometric deficit:

$$
\Delta G_{o,\mathrm{loss}}=
\max\left(\frac{C_{o,m}^{*}-C_{o,m}}{\gamma_{C,o}},
          \frac{P_{o,m}^{*}-P_{o,m}}{\gamma_{P,o}},0\right).
$$

The shared automatic payment transaction is implemented in
[`BaseMycorMarl._pay_maintenance`](../mycormarl/mycormarl/environments/base_mycor.py).
Allocation leaves retained resources in the free pools, as tested by
[`test_automatic_maintenance_does_not_spend_reserved_resources`](../tests/test_base_mycor_refactor.py).

For trade rate $k_t$, the outgoing fraction is $1-\exp(-k_tdt)$, applied to
post-maintenance donor C for the plant and P for the fungus. Bilateral transfer
is cancelled if either partner is non-operational after maintenance. Incoming
trade is credited after allocation and cannot fund growth in that numerical
step. Both outgoing commands are resolved before either recipient uses them.

For biological rates $k_g,k_r,k_s$, define $K=k_g+k_r+k_s$. The offered
growth/reproduction fractions are

$$
f_g=(1-e^{-Kdt})k_g/K,\qquad f_r=(1-e^{-Kdt})k_r/K,
$$

with both zero when $K=0$. Each fraction is applied separately to the remaining
C and P pools. Storage has no separate receiving compartment: only realised
growth costs and reproductive exports are subtracted. Its inclusion in $K$
does affect finite-step allocation, so refinement still needs verification.
Reproduction exports $f_rC$ and $f_rP$ independently and scores
$(f_rC/\gamma_C)^\alpha(f_rP/\gamma_P)^{1-\alpha}$; it does not require
matched C:P construction and creates no parental biomass or offspring state.
This is the implemented four-rate contract; the proposed three-rate,
stoichiometric-reproduction redesign is not implemented here. See
[`step_env`](../mycormarl/mycormarl/environments/base_mycor.py) and
[`test_incoming_trade_is_not_available_for_same_step_growth`](../tests/test_base_mycor_refactor.py).

Structural P associated with biomass lost to a payment shortfall is added to
cumulative mortality-loss diagnostics. It is not recycled to soil. Free P
removed by $\kappa_P$ is recorded separately in the legacy-named plant and
fungal maintenance-loss counters; unmet demand is recorded only as a deficit.
The deficit-induced biomass loss is capped at current active biomass.
Death is absorbing once post-maintenance biomass falls below the configured
fraction of its historical maximum. Residual biomass and free pools remain in
state for accounting but have zero activity; the threshold does not recycle
them into soil. See the [extended P ledger](model-overview.md#phosphorus-accounting).

## Plant biomass to root geometry

Total plant dry biomass $G_p$ is converted to fine-root length by

$$
L_{root}=G_p k_{root}\,SRL,
$$

where $k_{root}$ denotes `PlantTraits.kfroot`, the represented fine-root
dry-mass fraction (dimensionless), applied exactly once; $SRL$ is
specific root length (cm g⁻¹ root dry mass), and $L_{root}$ is cm. Implemented
by [`root_length_from_plant_biomass`](../mycormarl/mycormarl/plant/roots.py).

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
[`root_disc_radii_from_biomass`](../mycormarl/mycormarl/plant/roots.py)
and [`axisymmetric_stacked_disc_root_density`](../mycormarl/mycormarl/plant/roots.py).
Tests verify depth-dependent radii, uniform density, conservation before domain
clipping, and clipping behaviour in
[`test_growth_geometry.py`](../tests/test_growth_geometry.py).

The integral $\sum_i\lambda_{root,i}V_i$ equals total biomass-implied root
length only if the full rooting-depth profile and all radial discs fit inside
the domain. Otherwise it equals the analytical disc-domain intersection;
missing length is not redistributed into represented soil. At the current
`kfroot=0.18` and SRL defaults, one gram of plant biomass implies about
`4,578.2 cm` of fine roots before clipping. The early `k_root=0.62` conversion
is superseded and must not be used as the current fine-root fraction.

## Fungal biomass to hyphal geometry

Fungal dry biomass $G_f$ becomes structural C, tissue volume, and cylindrical
external-hyphal length:

$$
L_h=\frac{G_f\gamma_{C,f}}{M_C\pi r_h^2},
$$

where $\gamma_{C,f}$ is g C g⁻¹ dry biomass, $M_C$ is tissue C density
(g C cm⁻³ tissue), $r_h$ is hyphal radius (cm), and $L_h$ is cm. Spores and
intraradical structures are excluded. Implemented by
[`hyphal_length_from_fungal_biomass`](../mycormarl/mycormarl/fungus/mycelium.py).

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
[`colony_radius_from_length_axisymmetric`](../mycormarl/mycormarl/fungus/mycelium.py),
[`axisymmetric_hemisphere_cell_fractions`](../mycormarl/mycormarl/fungus/mycelium.py),
and [`axisymmetric_density_from_biomass`](../mycormarl/mycormarl/fungus/mycelium.py).
Partial-front conservation and domain saturation are tested in
[`test_growth_geometry.py`](../tests/test_growth_geometry.py).

In particular,
$\sum_i\lambda_{h,i}V_i=\lambda_{sat}\,\mathrm{Vol}(H(R_f)\cap\mathrm{soil})$.
Before boundary contact this equals $L_h$; after contact, outside structure is
clipped without raising the in-domain density above $\lambda_{sat}$. Reaching
one boundary does not saturate the entire soil domain: that occurs only once
the hemisphere covers it. Geometry clipping does not cap fungal biomass.
Current defaults imply about `5,511,859.5 cm` hyphae per gram fungal dry mass,
as a dimensional reference rather than an empirical prediction.

Both density fields are reconstructed from current living biomass at reset
and after each maintenance/allocation update. Loss retracts disc radii and the
fungal front; later regrowth can recreate the same geometry. Equal living
biomass and traits imply equal geometry regardless of history. Zero living
biomass or an absent/dead organism gives exactly zero density. There is no
persistent front, thinning/damage state, abandoned territory, or independent
absorber age; see [ADR-0004](adr/0004-derive-uptake-geometry-from-current-biomass.md).

Future work should change the assumed fungal geometry with sufficient
justification.

## Parameterisation and literature relationship

| Parameter | Default | Evidence and status |
|---|---:|---|
| Plant initial biomass and free pools | `0.01 g`; `1.75e-6 g C`; `0 mg P` | An early-established seedling fixture above the `0.7–3.3 mg` *Daucus carota* propagule means reported by [Vandelook et al. (2024)](https://doi.org/10.1017/S0960258524000230) and below later resource-exchange harvests. The default C pool funds one maintenance timestep; default P is zero because default $\kappa_P=0$. |
| Fungal initial biomass and free pools | `0.001 g`; `3.75e-7 g C`; `0 mg P` | Living external mycelium in an early-established fixture, deliberately below the approximately `3–17 mg` mature external-mycelium measurements in *R. irregularis* root-organ microcosms ([Sun et al. (2024)](https://doi.org/10.1007/s00572-024-01154-8)). This is not inoculum, colonized-root mass, or a spore mass. The default C pool funds one maintenance timestep; default P is zero because default $\kappa_P=0$. |
| Plant $\gamma_C$ | `0.402 g C g⁻¹` | Carrot-root elemental analysis from [Kaur et al. (2022)](https://doi.org/10.1038/s41598-022-20971-5); a root-dominated proxy, not whole-plant calibration. |
| Plant $\gamma_P$ | `1.92 mg P g⁻¹` | Derived dry-mass-weighted carrot value from [Kováčik et al. (2022)](https://doi.org/10.3390/agronomy12112770), an MDPI *Agronomy* paper; independent validation remains required. |
| $k_{froot}$, $SRL$, $r_{root}$ | `0.18`, `25,434.3 cm g⁻¹`, `0.01 cm` | `k_froot` is a representative value inside the directly observed `0.119–0.244` interval for six-month carrots in deep silica-sand greenhouse columns ([Westerveld 2005](https://bradford-crops.uoguelph.ca/sites/default/files/Sean%20Westerveld%20Thesis.pdf)); the SRL and effective radius are compatible but separately aggregated fine-root traits from [GRooT](https://doi.org/10.1111/geb.13179). This is not a field or wild-carrot default. |
| $\beta$, $D_{root}$ | `0.96`, `150 cm` | Provisional depth profile and near-infinite rooting horizon. A shallower simulated domain retains only $F(D_{soil})/F(D_{root})$ of total roots. |
| Fungal $\gamma_C$, $M_C$ | `0.5`, `0.1155 g C cm⁻³` | Provisional values from [Bisot et al. (2026)](https://doi.org/10.1073/pnas.2512182123). |
| Fungal $\gamma_P$ | `1.3 mg P g⁻¹` | Low-P spore concentration measured inside *Glomus intraradices*, a lineage now assigned to *Rhizophagus irregularis*, by [Olsson et al. (2008)](https://doi.org/10.1128/AEM.00376-08). This is an evidence-led low-P spore proxy used as fixed structural stoichiometry, not a maintenance measurement or a whole-extraradical-mycelium mean. |
| Plant $a_{mass}$ | `0.05 g C g⁻¹ leaf DM d⁻¹` | Apparent-gross reference-day carbon input derived from carrot light-response curves at 450 µmol photons m⁻² s⁻¹ over a 16 h photoperiod, converted with carrot SLA. It is spread uniformly through time by the current implementation; see the [research review in REDbyRBD](https://github.com/isaachcamp/REDbyRBD/blob/main/docs/research/default-biomass-stoichiometry-and-photosynthesis-parameters.md). |
| Plant $\kappa_C$ | `0.007 g C g⁻¹ whole-plant DM d⁻¹` | Full standing-biomass maintenance, rounded from the `0.00694` whole-carrot conversion of shoot and root coefficients fitted by [Reid (2019)](https://doi.org/10.1080/01140671.2019.1588134). Growth/construction efficiency remains separate and unmodelled. |
| Plant and fungal $\kappa_P$ | `0 mg P g⁻¹ d⁻¹` | Default irreversible free-P loss is disabled for both organisms. Positive $\kappa_P$ remains an explicit sensitivity for omitted leakage or unrecovered turnover; it is not physiological maintenance or structural immobilisation. Structural P is represented only by $\gamma_P$. |
| $\lambda_{sat}$ | `2,000 cm cm⁻³` | Lower end of the approximately `2,000–2,500 cm cm⁻³` upper external-hypha profiles observed by [Jakobsen et al. (1992)](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x). It is an estimated local saturation density, not a universal AMF constant. |
| $\lambda_{root}$, $\beta$ | `1 cm cm⁻³`, `0.96` | Provisional inherited geometry choices without empirical calibration. |

Runtime defaults and units are defined in
[`PlantTraits`](../mycormarl/mycormarl/plant/traits.py) and
[`FungusTraits`](../mycormarl/mycormarl/fungus/traits.py), with
construction-time validation immediately below those definitions.

The early implementation used `gamma_p,fungus=40 mg P g⁻¹` as a reported
maximum, not a representative whole-mycelium estimate, and a converted
`lambda_sat=168.75 cm cm⁻³` as a provisional density proxy. Those assumptions
were replaced by the evidence and defaults above. Archived trajectories retain
their original parameter identities; copying a current default into their
metadata would change their scientific meaning. Initial free-pool values in
the table are for `dt=0.025 day`; unspecified pools are calculated at reset as
`initial_biomass*kappa*dt`, rather than fixed independently of timestep.

## Assumptions and limitations

- **Assumption:** Both organisms use fixed C:P structural stoichiometry and
  follow Tilman-style essential resource with scarcer resource limiting growth.
- **Assumption:** The plant has a numerical biomass guard; the fungus has no
  explicit biomass cap. Soil boundaries limit its represented uptake geometry,
  not biomass directly.
- **Assumption:** Root allocation uses a static root-mass fraction as a
  marginal conversion from whole-plant biomass; allometry is constant.
- **Assumption:** Fungal biomass is treated as external cylindrical hyphae for
  geometry; spores and intraradical biomass are omitted.
- **Initial-condition assumption:** The symbiosis is pre-established. Initial
  fungal biomass represents living, external, absorptive hyphae; the model does
  not simulate inoculum establishment, dormancy or germination.
- **Assumption:** Mycelial density remains constant independent of soil
  P concentration.
- **Assumption:** Represented absorbing length is $k_{froot} \times SRL$.
  Its `0.119–0.244` uncertainty range is observed only in six-month deep-sand
  greenhouse carrots. The GRooT SRL and effective radius
  are species-level fine-root aggregates rather than values for one branch order.
  The entire `k_froot` fraction is the only represented active plant absorber:
  all of its reconstructed fine-root length is uptake-active, with no separate
  inactive, coarse-root, or active-absorber fraction. This is not a default for
  field or wild carrot.
- **Limitation:** Both geometries are spatial closures, not explicit branching
  networks, and their saturation/density parameters require calibration.
- **Limitation:** No maximum tissue age, turnover, dormancy, remobilisation, or
  age-dependent uptake activity is represented.
- **Accounting limitation:** Paid irreversible-loss P, mortality P, and
  reproduction P are recorded as exports with no represented receiving pool;
  none is recycled into soil.
- **Qualification scope:** The `2,000 cm cm⁻³` fungal-density default changes
  colony extent and the sparse/continuous uptake blend. Changing this or other
  geometry traits requires requalification rather than reuse of the early
  parameterisation's results. The canonical numerical qualification includes
  a deep-soil confinement and extended-P balance check.

## References

1. Bisot, C. et al. “Carbon-phosphorus exchange rate constrains density-speed trade-off in arbuscular mycorrhizal fungal growth.” *PNAS* 123 (2026). [DOI](https://doi.org/10.1073/pnas.2512182123).
2. Oyarte Galvez, L. et al. “A travelling-wave strategy for plant–fungal trade.” *Nature* 639 (2025). [DOI](https://doi.org/10.1038/s41586-025-08614-x).
3. Jakobsen, I., Abbott, L. K. & Robson, A. D. “External hyphae of vesicular-arbuscular mycorrhizal fungi associated with *Trifolium subterraneum* L. 1. Spread of hyphae and phosphorus inflow into roots.” *New Phytologist* 120 (1992). [DOI](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x).
4. Olsson, P. A. et al. “Elemental composition in vesicles of an arbuscular mycorrhizal fungus, as revealed by PIXE analysis.” *Applied and Environmental Microbiology* 74 (2008). [DOI](https://doi.org/10.1128/AEM.00376-08).
5. Guerrero-Ramírez, N. R. et al. “Global root traits (GRooT) database.” *Global Ecology and Biogeography* 30 (2021). [DOI](https://doi.org/10.1111/geb.13179).
6. Kaur, P. et al. “Green extraction of bioactive components from carrot industry waste and evaluation of spent residue as an energy source.” *Scientific Reports* (2022). [DOI](https://doi.org/10.1038/s41598-022-20971-5).
7. Kováčik, P. et al. “The Effect of Vermicompost and Earthworms (*Eisenia fetida*) Application on Phytomass and Macroelement Concentration and Tetanic Ratio in Carrot.” *Agronomy* 12 (2022). [DOI](https://doi.org/10.3390/agronomy12112770).
