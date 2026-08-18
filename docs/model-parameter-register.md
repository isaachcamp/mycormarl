# MycorMARL model parameter register

> **Scope:** This register covers every user-configurable field currently declared by
> `PlantTraits`, `FungusTraits`, and `EnvConfig` (49 fields in total). It excludes PPO
> optimiser and network hyperparameters, policy actions, state variables, diagnostic
> tolerances, and physical conversion constants. Those quantities do not parameterise
> the biological or soil model.
>
> **Evidence note:** “Literature-backed” does not always mean directly measured for the
> represented species. The tables distinguish direct observations, cross-study or
> cross-species derivations, numerical choices, explicit modelling assumptions, and
> values for which the repository currently has no empirical source.

## Executive summary

MycorMARL's defaults combine several evidence levels. The strongest organism-specific
anchors are carrot carbon and phosphorus measurements,
carrot photosynthesis and maintenance data, and the ordinary phosphorus concentration
measured in spores of the *Rhizophagus irregularis* lineage. Root traits come from
separate *D. carota* entries in the GRooT database rather than one matched experiment.
The initial fungal mass is a cross-species AMF spore-mass regression, and several soil
transport and uptake values are inherited provisionally from Schnepf and Roose's AMF
phosphate model.

The most important unsupported or explicitly abstract defaults are plant and fungal
death thresholds, fungal carbon maintenance, plant leaf fraction,
root absorbing radius, root spatial-density and depth-profile parameters, the uptake
transition timescale and exponent, and the reproductive reward exponent. Plant and
fungal `kappa_p` are not measured maintenance costs: both are modelled irreversible
free-P losses. Structural or “frozen” phosphorus belongs only to `gamma_p`.

For the planned 120-day cultivated-carrot study, the effective carbon-input
scale requires qualification before policy training. A favourable `Forto`
field trajectory reached `23.26 g` shoot-plus-storage-root dry mass at 120 days
after sowing, with successive 20-day RGRs declining from approximately `0.066`
to `0.042 d^-1`. The current carbon-only sustained ceiling is about
`0.0199 d^-1`. The leaf-level `amass=0.05` remains defensible for its named
reference light regime; the larger mismatch is the fixed `kleaf=0.30`, because
the same trajectory's leaf fraction declined from about `0.81` to `0.23`. See
the [growth-scale research note](research/carrot-growth-biomass-cap-and-carbon-fixation.md).
The completed static-allocation qualification provisionally selects `kleaf=0.50`
for the reference calibration. This is a qualification choice, not an
ontogenetic claim; the runtime default remains `0.30` until deliberately
migrated.

There is also one current configuration defect. `EnvConfig.topsoil_depth_cm` is declared
as `None`, although validation requires a finite number and the scientific documents
specify an intended `25 cm`. This register reports the actual runtime declaration and
the documented intent separately.

## Evidence-status key

| Status | Meaning |
|---|---|
| **Direct** | The default is a direct or rounded observation for the represented species or lineage. |
| **Derived** | Literature measurements are transformed, combined, or rounded to obtain the runtime value. |
| **Transferred** | A value is borrowed from another taxon, structure, soil, or mechanistic model. |
| **Model choice** | The value defines an abstraction or scenario and is not claimed as an empirical estimate. |
| **Numerical** | The value controls discretisation, stability, or experiment duration. |
| **Unsupported** | No literature source or documented quantitative derivation currently backs the value. |
| **Inconsistent** | Code and active scientific documentation disagree or the declared default is invalid. |

The declarations are in [`PlantTraits`](../mycormarl/mycormarl/plant/traits.py#L7-L44),
[`FungusTraits`](../mycormarl/mycormarl/fungus/traits.py#L7-L33), and
[`EnvConfig`](../mycormarl/mycormarl/params.py#L15-L45). Construction-time validation
is immediately below the two trait classes and in
[`BaseMycorMarl._validate_soil_config`](../mycormarl/mycormarl/environments/base_mycor.py#L731-L796).

## Core equations: how parameters affect the model

For either organism, allocated carbon and phosphorus produce dry biomass according to

$$
\Delta G=\min\left(\frac{C_g}{\gamma_C},\frac{P_g}{\gamma_P}\right).
$$

Larger `gamma_c` or `gamma_p` therefore reduces growth from a fixed allocation and
increases the structural resource represented by a unit of biomass. The same
stoichiometries convert maintenance deficits to biomass loss. Automatic standing costs
are

$$
C_m^*=G\kappa_C\Delta t,\qquad P_{loss}^*=G\kappa_P\Delta t.
$$

They are deducted before trade and allocation. The implementation is
[`_pay_maintenance`](../mycormarl/mycormarl/environments/base_mycor.py#L607-L644)
and [`_apply_allocation`](../mycormarl/mycormarl/environments/base_mycor.py#L646-L700).

Plant carbon input is

$$
\Delta C_{fix}=G_p k_{leaf}a_{mass}\Delta t.
$$

Here `amass` is an apparent-gross carbon budget for a 16-hour-light reference day,
spread uniformly through the current numerical time axis. It is not grams, nor a direct
area-based gas-exchange rate. This is implemented in
[`photosynthesise`](../mycormarl/mycormarl/plant/photosynthesis.py#L10-L22).

Biomass-derived absorbing lengths are

$$
L_{root}=G_p k_{root}SRL,
\qquad
L_h=\frac{G_f\gamma_{C,f}}{M_C\pi r_h^2}.
$$

Root length is distributed through stacked discs; fungal length fills a hemisphere at
`saturation_density`. See
[`root_length_from_plant_biomass`](../mycormarl/mycormarl/plant/roots.py#L9-L21)
and [`hyphal_length_from_fungal_biomass`](../mycormarl/mycormarl/fungus/mycelium.py#L7-L26).

Soil labile P and solution concentration are linked by

$$
M=C\,V(\theta+b_p),
$$

while transport uses $D_l\theta f_l$ and surface influx uses

$$
J(C_s)=\frac{J_{max}C_s}{K_m+C_s}.
$$

These equations make `theta_water`, `b_p`, the diffusion coefficient, impedance factor,
absorber radii, and uptake kinetics jointly determine P supply. The sparse and
continuous closures are blended using `uptake_reference_time_days` and
`uptake_transition_exponent`; full equations are in the
[phosphate model](phosphate-model.md).

## Plant parameters

| Runtime field | Default and units | Description and model effect | Evidence, derivation, and status |
|---|---:|---|---|
| `initial_biomass` | `0.01 g DM` | Plant dry biomass at reset; immediately determines starting root length, photosynthetic input, maintenance demand, and death reference. | Early-established seedling fixture chosen above the `0.7–3.3 mg` *Daucus carota* propagule means ([Vandelook et al. 2024](https://doi.org/10.1017/S0960258524000230)) and below later resource-exchange harvests. **Qualification-informed modelling choice.** |
| `initial_c_pool` | `0.00402 g C` | Free, allocatable carbon at reset; this is additional to structural C implicit in biomass. | One structural-biomass equivalent: `0.01 × 0.402`. **Derived model initial condition.** |
| `initial_p_pool` | `0.0192 mg P` | Free, allocatable phosphorus at reset; additional to structural P implicit in biomass. | One structural-biomass equivalent: `0.01 × 1.92`. **Derived model initial condition.** |
| `kleaf` | `0.30` runtime; `0.50` selected qualification reference | Fraction of whole-plant dry biomass contributing to photosynthesis; carbon input is directly proportional to it. | `Forto` fitted shoot and storage-root curves imply a leaf fraction declining from `0.812` at 40 DAS to `0.234` at 120 DAS ([Cecilio Filho & Peixoto 2013](https://repositorio.unesp.br/bitstream/11449/75622/1/2-s2.0-84878476833.pdf)); fine roots were excluded from that denominator. The static high-P sweep retained `0.30`, `0.45`, `0.475`, `0.50`, `0.525`, `0.55`, `0.575`, and `0.60`. `0.50` was selected provisionally because it reached `34.60 g DM` at day 120 without contacting the `50 g` guard; `0.525` and higher contacted the guard. This is a calibration reference, not an ontogenetic claim or a partition complement to `kfroot`. **Qualification-informed provisional choice; constant whole-episode fraction remains a model limitation.** |
| `kfroot` | `0.18` | Fine/fibrous-root dry mass divided by whole-plant dry mass. It is the only represented active plant-absorber fraction: the model converts all `kfroot × specific_root_length` to uptake-active fine-root length, with no inactive, coarse-root, or secondary active-fraction correction. | Representative value inside the directly observed `0.119–0.244` interval for six-month `Idaho` and `Fontana` carrots grown in 150-cm, 98%-silica-sand greenhouse columns ([Westerveld 2005](https://bradford-crops.uoguelph.ca/sites/default/files/Sean%20Westerveld%20Thesis.pdf), Table 2.18 and Appendix A2.16). Treat those endpoints as the uncertainty range for this late deep-sand regime. Mature field `Nantes Duke` observations have a different denominator—fibrous root / total root `0.0177–0.0329`—so they cannot be converted to `kfroot` without matched shoot and storage-root mass. Neither range is a field or wild-carrot default. **Derived; cultivation-regime scoped.** |
| `amass` | `0.05 g C g⁻¹ leaf DM reference-day⁻¹` | Apparent-gross daily leaf-mass carbon input. Higher values raise free C linearly; the current implementation spreads the reference-day budget uniformly. | Carrot net light-response curves were evaluated at `450 µmol photons m⁻² s⁻¹`, their fitted respiration intercept was added, and the result was integrated over a 16 h photoperiod ([Kyei-Boahen et al. 2003](https://ps.ueb.cas.cz/pdfs/phs/2003/02/30.pdf)). Conversion used independent carrot SLA `66–94 cm² g⁻¹` ([Acosta-Motos et al. 2021](https://doi.org/10.3390/agronomy11122460)), producing `0.036–0.073` with midpoint `0.051`, rounded to `0.05`. Approximate `0.06–0.07` sensitivities correspond to brighter `700–1000 µmol m⁻² s⁻¹` reference conditions; they are not corrections for missing ontogenetic leaf allocation. Photorespiration remains embedded. **Derived from two studies.** |
| `jmax` | `3.26e-6 µmol P cm⁻² s⁻¹` | Maximum root-surface P influx in both sparse and continuous uptake closures. Higher values increase kinetic demand but may intensify local diffusion limitation or inventory capping. | Tinker and Nye uptake value reported and used by [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x), not a carrot measurement. **Transferred.** |
| `km` | `5.8e-3 µmol P cm⁻³` (`5.8 µM`) | Half-saturation concentration for root P uptake; lower values maintain a larger fraction of `jmax` at low solution P. | Same source and transfer as `jmax`; not carrot-specific. **Transferred.** |
| `root_radius` | `0.01 cm` (`100 µm`) | Effective fine-root radius; converts represented fine-root length to absorbing lateral area and enters cylindrical sparse-uptake resistance. | Half the GRooT *D. carota* fine-root diameter median (`0.2036 mm`, 12 study-site entries) is `0.01018 cm`, rounded to `0.01 cm` ([Guerrero-Ramírez et al. 2021](https://doi.org/10.1111/geb.13179)). A cultivated-field fibrous-root study found length-dominant roots about `0.15 mm` diameter; use `0.0075–0.0114 cm` as a sensitivity range. Diameter, SRL, and mass pair are not a matched plant-level observation. **Derived effective fine-root trait.** |
| `specific_root_length` | `25,434.3 cm g⁻¹ fine-root DM` (`254.343 m g⁻¹`) | Converts represented fine-root dry mass to total absorbing root length; uptake geometry scales linearly with it until domain clipping. | Species-level median of ten *D. carota* SRL study-site entries in GRooT, separate from the records used for `kfroot` ([Guerrero-Ramírez et al. 2021](https://doi.org/10.1111/geb.13179)). All contributing SRL records are classified as fine-root (`FR`) measurements; two explicitly use the 0–2 mm diameter class, while the others do not report an order or diameter class in GRooT. The aggregate therefore reflects diameter and order mixtures within sampled fine-root systems, not a single root order or an explicit correction for excluded coarse/storage roots. **Derived; unmatched trait aggregation.** |
| `root_length_density` | `1 cm cm⁻³ bulk soil` | Uniform density inside each depth-specific root disc. At fixed total length, a larger value contracts disc radii; a smaller value spreads roots farther. | No empirical calibration is recorded. **Unsupported geometry choice.** |
| `beta_root_distribution` | `0.96` | Controls cumulative root depth as `F(d)=1-beta^d`; values closer to one place relatively more length at depth. | No quantitative source is recorded. **Unsupported provisional shape parameter.** |
| `max_rooting_depth_cm` | `150 cm` | Normalisation horizon and hard lower boundary for the analytical root profile. Roots assigned below the simulated soil retain biomass cost but provide no in-domain uptake. | Described as a provisional near-infinite rooting horizon rather than a measured carrot maximum. **Unsupported model choice.** |
| `gamma_c` | `0.402 g C g⁻¹ DM` | Structural C per unit plant biomass; sets C-limited growth, structural C accounting, initial C reserve, and C-deficit biomass loss. | Carbon elemental analysis of untreated dry carrot-root material ([Kaur et al. 2022](https://doi.org/10.1038/s41598-022-20971-5)). Root-dominated material is used as a whole-plant proxy. **Transferred within species.** |
| `gamma_p` | `1.92 mg P g⁻¹ DM` | Structural/frozen plant P per unit biomass; sets P-limited growth, initial P reserve, mortality P accounting, and P-deficit biomass loss. It is not a daily P loss. | Dry-mass-weighted value derived from unfertilised-soil carrot root (`1.975 mg g⁻¹`) and leaf (`1.688 mg g⁻¹`) P concentrations and their reported dry masses ([Kováčik et al. 2022](https://doi.org/10.3390/agronomy12112770)). **Derived; independent corroboration remains needed.** |
| `kappa_c` | `0.007 g C g⁻¹ whole-plant DM d⁻¹` | Unavoidable standing-biomass C maintenance paid before action; deficits remove biomass. It intentionally excludes a future construction-efficiency term. | `0.402 × (0.38×0.021 + 0.62×0.015) = 0.00694`, rounded to `0.007`, using carrot shoot/root maintenance coefficients at 20 °C ([Reid 2019](https://doi.org/10.1080/01140671.2019.1588134)). **Derived.** |
| `kappa_p` | `0.001 mg P g⁻¹ whole-plant DM d⁻¹` | Lumped irreversible free-P loss paid before action. It represents omitted herbivory, leakage, and unrecovered turnover, not biochemical P maintenance or structural immobilisation. | No carrot study directly estimates the coefficient. Carrot time courses show continued P accumulation and redistribution ([Fernández-Pérez et al. 2023](https://doi.org/10.17584/rcch.2023v17i3.16508); [Cecílio Filho et al. 2013](https://acervodigital.unesp.br/handle/11449/75622)). The positive value is an explicit small-loss abstraction; zero is the conservation sensitivity case. **Model choice, not literature-backed.** |
| `death_fraction` | `0.20` | The plant dies when post-maintenance biomass falls below 20% of its historical maximum. It is a deterministic termination threshold, not daily mortality. | No empirical derivation is recorded. **Unsupported.** |
| `biomass_cap` | `50 g DM` | Hard numerical guard on plant biomass and realised structural growth. Contact invalidates a qualification trajectory; it is not a physiological plateau. | Comparable field endpoints are approximately `22--25 g DM`; `Forto` reached `23.26 g` at 120 DAS and its unobserved fitted shoot-plus-storage-root asymptote is `35.05 g` ([Cecilio Filho & Peixoto 2013](https://repositorio.unesp.br/bitstream/11449/75622/1/2-s2.0-84878476833.pdf); [Gomes et al. 2021](https://doi.org/10.4025/actasciagron.v43i1.51831)). `50 g` leaves about 43% headroom over the fitted asymptote and roughly twofold headroom over the largest independent cultivar mean, while replacing the unsupported `100 g` scale. **Evidence-bounded numerical/model guard; not a biological maximum.** |
| `biomass_observation_reference` | `50 g DM` | Independent reference in the bounded actor feature `B / (B + reference)`. It changes policy-input scaling but does not constrain growth. | Preserves the previous observation scale, which was implicitly `0.5 * biomass_cap = 50 g` under the old `100 g` cap. It is now decoupled so cap changes do not confound model dynamics with policy inputs. **Policy-interface model choice.** |

## Fungal parameters

| Runtime field | Default and units | Description and model effect | Evidence, derivation, and status |
|---|---:|---|---|
| `initial_biomass` | `0.0001 g DM` | Fungal dry biomass at reset, interpreted as living external mycelium in an already operating symbiosis; spores and inoculum are not represented. | Early-established fixture chosen below the approximately `3–17 mg` mature external-mycelium measurements in *R. irregularis* root-organ microcosms ([Sun et al. 2024](https://doi.org/10.1007/s00572-024-01154-8)). **Qualification-informed modelling choice; not a direct initial-condition measurement.** |
| `initial_c_pool` | `0.00005 g C` | Free fungal C at reset, additional to structural C implicit in biomass. | One structural-biomass equivalent: `0.0001 × 0.5`. **Derived model initial condition.** |
| `initial_p_pool` | `0.0002 mg P` | Free fungal P at reset, additional to structural P implicit in biomass. | One structural-biomass equivalent: `0.0001 × 2`. **Derived model initial condition.** |
| `gamma_c` | `0.5 g C g⁻¹ DM` | Structural fungal C per unit biomass; controls C-limited growth, initial reserve, deficit mortality, and conversion of biomass to hyphal tissue volume. | Provisional fungal carbon fraction used by [Bisot et al. (2026)](https://doi.org/10.1073/pnas.2512182123). Its precise AMF value is acknowledged as uncertain. **Transferred/provisional.** |
| `gamma_p` | `2 mg P g⁻¹ DM` | Fixed structural/frozen P per unit fungal biomass. Lower values reduce P cost of growth but increase biomass loss per unit unmet P-loss demand. It does not specify P maintenance. | Approximate ordinary P concentration inside *Glomus intraradices* spores, from the lineage now called *R. irregularis* ([Olsson et al. 2008](https://doi.org/10.1128/AEM.00376-08)). The paper measured `1.3±0.35` under low P and `8.0±1.6 mg g⁻¹` under high P; `2` is its ordinary-spore description. **Direct lineage observation used as a whole-fungus fixed stoichiometry proxy.** |
| `kappa_c` | `0.03 g C g⁻¹ DM d⁻¹` | Unavoidable fungal carbon maintenance; reduces the free C received from the plant and can cause biomass loss when unpaid. | No source or derivation is recorded in active research or model documentation. **Unsupported.** |
| `kappa_p` | `0.003 mg P g⁻¹ DM d⁻¹` | Lumped irreversible free-P loss, excluding P transfer to the plant and structural P in biomass. | Literature constrains fine-hyphal lifetimes to roughly `5–7 d` ([Bago et al. 1998](https://doi.org/10.1046/j.1469-8137.1998.00199.x); [Olsson & Johnson 2005](https://doi.org/10.1111/j.1461-0248.2005.00831.x)), but not the non-recycled P fraction. With `gamma_p=2`, the default assumes only about `0.75–1.05%` irreversible loss per turnover: `kappa_p=(gamma_p/tau)f_irreversible`. **Derived scale with an assumed loss fraction; not directly literature-backed.** |
| `death_fraction` | `0.05` | Fungus dies when post-maintenance biomass falls below 5% of historical maximum. It is not hyphal turnover or stochastic mortality. | No empirical derivation is recorded. **Unsupported.** |
| `hyphal_radius` | `5e-4 cm` (`5 µm`) | Converts fungal tissue volume to cylindrical length and length to absorbing area; also enters sparse resistance and depletion-zone overlap. | Adopted from [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x), not identified as a direct *R. irregularis* measurement. **Transferred.** |
| `hyphal_tissue_carbon_density` | `0.1155 g C cm⁻³ tissue` | Converts structural fungal carbon to living hyphal volume. Higher density produces less length and smaller colony extent per unit biomass. | Provisional `M_C` from [Bisot et al. (2026)](https://doi.org/10.1073/pnas.2512182123); their construction combines tissue density, dry fraction, and carbon fraction and notes uncertainty for AMF. **Transferred/provisional.** |
| `saturation_density` | `2,000 cm hypha cm⁻³ bulk soil` | Local length density inside the hemispherical fungal colony. At fixed total length, higher density makes the colony more compact and changes depletion overlap and the uptake blend. | Lower end of approximately `2,000–2,500 cm cm⁻³` upper external-hypha profiles in [Jakobsen et al. (1992)](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x). The experimental AMF were not *R. irregularis*. **Transferred observed local density.** |
| `jmax` | `3.26e-6 µmol P cm⁻² s⁻¹` | Maximum fungal surface P influx. It has the same current value as the root field but remains separately configurable. | Tinker and Nye value reported by [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x); not a direct *R. irregularis* estimate. **Transferred.** |
| `km` | `5.8e-3 µmol P cm⁻³` (`5.8 µM`) | Fungal uptake half-saturation concentration; lower values sustain uptake at lower P. | Same source and limitation as fungal `jmax`. **Transferred.** |

## Environment and soil parameters

| Runtime field | Default and units | Description and model effect | Evidence, derivation, and status |
|---|---:|---|---|
| `max_steps` | `14,600 steps` | Administrative episode limit. At the default timestep it represents `14,600×0.025=365 d`; truncation is not biological death. | Arithmetic one-year scenario, not literature-derived. **Numerical/model choice.** |
| `dt` | `0.025 d` (`0.6 h`) | Biological/action step and basis for maintenance, photosynthesis, uptake, and termination timing. Diffusion is internally subcycled if required. | Operational default retained pending issue #24, which will separate the policy decision interval from numerical integration. The standard fixed-geometry soil qualification supports `0.05 d`, but coupled fixed-action timestep comparisons change action frequency and are diagnostic only. **Numerical/model choice, not a biological parameter estimate.** |
| `consumer_mode` | `"mixed"` | Activates plant and fungus together; alternatives are `plant-only` and `fungus-only`. It changes which organisms are operational and which uptake/trade processes occur. | Experimental scenario selection. **Model choice.** |
| `soil_radius_cm` | `50 cm` | Outer radius of the axisymmetric cylinder; controls cell count, soil volume, radial boundary, available P, and clipping of organism geometry. | No literature or experimental-vessel source is recorded for the default. **Unsupported scenario geometry.** |
| `soil_depth_cm` | `100 cm` | Cylinder depth; controls soil volume, vertical boundary, and the represented fraction of the analytical root profile. | No literature or experimental-vessel source is recorded for the default. **Unsupported scenario geometry.** |
| `radial_interval_cm` | `0.1 cm` | Exact uniform radial cell spacing. Smaller values increase resolution and cost; extent must be evenly divisible. | Existing spatial qualification retained `0.1 cm` against finer candidates for the reduced test problem. **Numerical choice; not universal convergence evidence.** |
| `depth_interval_cm` | `0.1 cm` | Exact uniform depth spacing with the same cost and divisibility implications. | Same qualification basis and limitation as radial spacing. **Numerical choice.** |
| `topsoil_depth_cm` | `None` in code; `25 cm` documented intent | Depth initially assigned `initial_solution_p_um`; below it starts with zero labile P. | `None` fails the current finite-number validator, whereas active phosphate documentation says `25 cm`. The `25 cm` profile is a scenario choice, not literature-derived. **Inconsistent and currently invalid default.** |
| `initial_solution_p_um` | `1 µM` | Initial soil-solution inorganic phosphate within topsoil; through buffering it sets total initial labile inventory. It is not extractable or total soil P. | Order-of-magnitude low-P condition associated with [Vance et al. (2003)](https://doi.org/10.1046/j.1469-8137.2003.00695.x), not a universal mean. **Transferred/provisional.** |
| `phosphate_diffusion_coefficient_cm2_s` | `1e-5 cm² s⁻¹` | Solution diffusion coefficient in finite-volume transport; with water, impedance, and buffering it determines amount conductance and apparent propagation. | Provisional parameterisation from [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x). **Transferred.** |
| `b_p` | `239 cm³ cm⁻³` | Linear volumetric P buffer power. Larger values increase labile inventory at fixed solution concentration and slow apparent diffusion. | Reported by Schnepf and Roose and attributed there to Barber; it is soil-specific ([Schnepf & Roose 2006](https://doi.org/10.1111/j.1469-8137.2006.01771.x)). **Transferred.** |
| `phosphate_impedance_factor` | `0.308` | Dimensionless tortuosity/impedance multiplier on diffusive amount flux. Lower values slow diffusion and increase sparse diffusion resistance. | Provisional Schnepf–Roose parameterisation. **Transferred.** |
| `diffusion_cfl_safety` | `0.8` | Multiplies the explicit finite-volume CFL ceiling when computing soil substeps; lower values increase stability margin and computational cost. | Standard conservative numerical margin selected by the project, not literature-calibrated biology. **Numerical choice.** |
| `uptake_reference_time_days` | `1 d` | Limits sparse depletion-zone propagation and sets the exposure scale in the sparse-to-continuous transition. It is neither the timestep nor a measured absorber lifetime. | Explicit provisional modelling choice. **Not literature-backed.** |
| `uptake_transition_exponent` | `2` | Sharpness of `w=1/[1+(t_diff/T_ref)^p]`; larger values make the sparse/continuous switch steeper. | Explicit smooth-transition choice; sensitivity values `1` and `4` were proposed. **Not literature-backed.** |
| `theta_water` | `0.3 cm³ water cm⁻³ bulk soil` | Volumetric water content; raises dissolved capacity and flux conductance and contributes with `b_p` to total labile capacity. It is static in the current model. | Provisional value used by [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x). **Transferred; soil-specific calibration required.** |
| `alpha` | `0.5` | Carbon exponent in Cobb–Douglas reproductive fitness, `R=(C_rep/gamma_c)^alpha(P_rep/gamma_p)^(1-alpha)`. At `0.5`, scaled C and P contribute symmetrically. Reward is a fitness index, not conserved offspring biomass. | No empirical reproductive-allocation calibration is recorded. **Model choice.** |

## Parameters that are not currently literature-backed

The following defaults should not be presented as measurements:

- **Unsupported organism values:** plant constant whole-episode `kleaf`, `root_radius`,
  `root_length_density`, `beta_root_distribution`, `max_rooting_depth_cm`,
  `death_fraction`, and any interpretation of `biomass_cap` as a biological
  maximum; fungal `kappa_c`
  and `death_fraction`.
- **Explicit biological abstractions:** plant and fungal `kappa_p`. Literature
  informs their scale and interpretation, but not the selected irreversible fraction.
- **Explicit uptake/reward choices:** `uptake_reference_time_days`,
  `uptake_transition_exponent`, and `alpha`.
- **Scenario geometry:** `soil_radius_cm`, `soil_depth_cm`, and the intended
  `topsoil_depth_cm=25`.
- **Numerical/experimental controls rather than empirical traits:** `max_steps`, `dt`,
  `consumer_mode`, both grid intervals, and `diffusion_cfl_safety`.

Several literature-linked parameters also remain weakly transferred: root and fungal
uptake kinetics are not species-specific; root radius is derived from unmatched
species-level and cultivated-field evidence; fungal tissue C density is acknowledged
as uncertain; root fraction and SRL are unmatched GRooT
aggregates; plant `gamma_c` is root-dominated; plant `gamma_p` needs independent
corroboration; and spore-derived fungal `gamma_p` is used as fixed whole-fungus
stoichiometry.

## Known gaps and recommended parameter work

1. Qualify the 120-day plant carbon and growth scale before the P-response
   pilot. Test `kleaf=0.30/0.45/0.60`, retain `amass=0.05` for the named
   450-PAR reference day, and compare windowed RGR and biomass against the
   `Forto` trajectory. Use the independent `50 g` numerical guard and reject
   any guard-contacting trajectory. Prefer an explicit ontogenetic leaf-mass
   or leaf-area state before final inference.
2. Fix the `topsoil_depth_cm` declaration/documentation mismatch before calling the
   complete `EnvConfig()` set a runnable default.
3. Obtain a direct mean seed-mass estimate for the intended *D. carota* population;
   the current `1 mg` value is only bounded by a multi-population range.
4. Prioritise fungal `kappa_c`, because it is a large direct standing sink with no
   recorded evidence and controls fungal dependence on traded plant C.
5. Replace the plant root radius and spatial-density/depth parameters with matched
   *D. carota* fine-root measurements; these directly determine absorbing area and
   overlap geometry.
6. Calibrate soil geometry, `theta_water`, `b_p`, diffusion, impedance, and uptake
   kinetics to a named experiment rather than treating the Schnepf–Roose set as generic.
7. Obtain a direct *R. irregularis* dry mass per spore and whole-mycelium C:P data.
8. Retain zero-loss sensitivity cases for both `kappa_p` values and replace the lumped
   sinks if explicit turnover, resorption, necromass, or recycling is implemented.
9. Rerun phosphate qualification after the fungal saturation-density change and resolve
   the existing timestep-convergence gap before interpreting long-horizon outputs.

## Verification and traceability

The accepted plant/fungal default values and key derived invariants are locked by
[`test_accepted_growth_geometry_trait_defaults`](../tests/test_growth_geometry.py#L27-L63).
Growth, maintenance, delayed resource availability, biomass caps, and P-loss accounting
are exercised in [`test_base_mycor_refactor.py`](../tests/test_base_mycor_refactor.py).
Grid, buffer, diffusion, sparse/continuous uptake, and cell-inventory conservation have
focused tests listed in the [phosphate model](phosphate-model.md#numerical-qualification-and-limitations).

This register is a consolidation of the runtime declarations and the existing
[growth model](growth-model.md), [phosphate model](phosphate-model.md), and
[parameter research review](research/default-biomass-stoichiometry-and-photosynthesis-parameters.md).
It does not turn a repository assumption into literature evidence merely because that
assumption has already been documented.

## References

1. Bago, B., Azcón-Aguilar, C., Goulet, A. & Piché, Y. “Branched absorbing structures (BAS): a feature of the extraradical mycelium of symbiotic arbuscular mycorrhizal fungi.” *New Phytologist* 139 (1998). [DOI](https://doi.org/10.1046/j.1469-8137.1998.00199.x).
2. Bisot, C. et al. “Carbon-phosphorus exchange rate constrains density-speed trade-off in arbuscular mycorrhizal fungal growth.” *PNAS* 123 (2026). [DOI](https://doi.org/10.1073/pnas.2512182123).
3. Cecílio Filho, A. B. et al. “Growth and accumulation of nutrients in carrot cultivar Forto.” (2013 repository record). [Stable record](https://acervodigital.unesp.br/handle/11449/75622).
4. Fernández-Pérez, M. et al. “Nutrients absorption curves in carrot.” (2023). [DOI](https://doi.org/10.17584/rcch.2023v17i3.16508).
5. Guerrero-Ramírez, N. R. et al. “Global root traits (GRooT) database.” *Global Ecology and Biogeography* 30 (2021). [DOI](https://doi.org/10.1111/geb.13179).
6. Jakobsen, I., Abbott, L. K. & Robson, A. D. “External hyphae of vesicular-arbuscular mycorrhizal fungi associated with *Trifolium subterraneum* L. 1. Spread of hyphae and phosphorus inflow into roots.” *New Phytologist* 120 (1992). [DOI](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x).
7. Kaur, P., Subramanian, J. & Singh, A. “Green extraction of bioactive components from carrot industry waste and evaluation of spent residue as an energy source.” *Scientific Reports* 12 (2022). [DOI](https://doi.org/10.1038/s41598-022-20971-5).
8. Kováčik, P. et al. “The Effect of Vermicompost and Earthworms (*Eisenia fetida*) Application on Phytomass and Macroelement Concentration and Tetanic Ratio in Carrot.” *Agronomy* 12 (2022). [DOI](https://doi.org/10.3390/agronomy12112770).
9. Kyei-Boahen, S., Lada, R., Astatkie, T., Gordon, R. & Caldwell, C. “Photosynthetic response of carrots to varying irradiances.” *Photosynthetica* 41 (2003). [Primary PDF](https://ps.ueb.cas.cz/pdfs/phs/2003/02/30.pdf).
10. Acosta-Motos, J. R. et al. “Comparative Characterization of Eastern Carrot Accessions for Some Main Agricultural Traits.” *Agronomy* 11 (2021). [DOI](https://doi.org/10.3390/agronomy11122460).
11. Olsson, P. A. & Johnson, N. C. “Tracking carbon from the atmosphere to the rhizosphere.” *Ecology Letters* 8 (2005). [DOI](https://doi.org/10.1111/j.1461-0248.2005.00831.x).
12. Olsson, P. A., Hammer, E. C., Wallander, H. & Pallon, J. “Phosphorus availability influences elemental uptake in the mycorrhizal fungus *Glomus intraradices*, as revealed by particle-induced X-ray emission analysis.” *Applied and Environmental Microbiology* 74 (2008). [DOI](https://doi.org/10.1128/AEM.00376-08).
13. Reid, J. B. “Modelling growth and dry matter partitioning in root crops: a case study with carrot (*Daucus carota* L.).” *New Zealand Journal of Crop and Horticultural Science* 47 (2019). [DOI](https://doi.org/10.1080/01140671.2019.1588134).
14. Schnepf, A. & Roose, T. “Modelling the contribution of arbuscular mycorrhizal fungi to plant phosphate uptake.” *New Phytologist* 171 (2006). [DOI](https://doi.org/10.1111/j.1469-8137.2006.01771.x).
15. Sieverding, E., Toro, S. T. & Mosquera, O. “Biomass production and nutrient concentrations in spores of VA mycorrhizal fungi.” *Soil Biology and Biochemistry* 21 (1989). [DOI](https://doi.org/10.1016/0038-0717(89)90013-8).
16. Vance, C. P., Uhde-Stone, C. & Allan, D. L. “Phosphorus acquisition and use: critical adaptations by plants for securing a nonrenewable resource.” *New Phytologist* 157 (2003). [DOI](https://doi.org/10.1046/j.1469-8137.2003.00695.x).
17. Vandelook, F. et al. “Intra-specific variation in relative embryo length and germination of wild *Daucus carota* across climate gradients in North America and Europe.” *Seed Science Research* 34 (2024). [DOI](https://doi.org/10.1017/S0960258524000230).
