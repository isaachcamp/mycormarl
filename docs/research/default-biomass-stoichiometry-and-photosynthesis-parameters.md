# Evidence review for biomass, stoichiometry, hyphal density, and photosynthesis defaults

**Research date:** 4 August 2026
**Scope:** Candidate evidence for fungal `gamma_p`, AMF saturation density,
initial plant and fungal biomass, initial resource pools, and plant `amass`.
This note records evidence and the resulting parameter decisions; runtime
changes are implemented in the corresponding trait definitions.

## Accepted decisions

### Fungal `gamma_p`: use the ordinary spore value

Use **`gamma_p = 2 mg P g⁻¹ dry biomass`** as the provisional fungal
default. Olsson et al. describe approximately `2,000 µg P g⁻¹`
(`2 mg P g⁻¹`) as an ordinary concentration inside *Glomus intraradices*
spores. The isolate lineage was subsequently called *G. irregulare* and is now
*Rhizophagus irregularis* ([Olsson et al. 2008](https://doi.org/10.1128/AEM.00376-08)).
The value is therefore an approximate spore P-content measurement, not a direct
measurement of whole-mycelium structural stoichiometry or maintenance
expenditure.

The selection uses the paper's ordinary spore-level concentration rather than
the lower `1.3 ± 0.35 mg P g⁻¹` treatment mean. In the current model,
`gamma_p` is the structural P cost used for growth and for converting an unmet
maintenance-P deficit into lost biomass. It does **not** set maintenance
demand; that rate is `kappa_p`. Consequently, lowering `gamma_p` reduces the P
cost of growth but increases the biomass lost for a given unmet P-maintenance
deficit.

### Fungal saturation density: use the lower observed upper-profile estimate

Use **`saturation_density = 2,000 cm cm⁻³`**, sourced from the lower end of
the approximately `2,000–2,500 cm cm⁻³` upper external-hypha profiles in
[Jakobsen et al. (1992)](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x).
This is an estimated local saturation density, not a universal AMF constant.
Because it changes colony extent and the sparse/continuous uptake blend, its
numerical qualification will be rerun with the deep-soil work tracked by
[issue #18](https://github.com/isaachcamp/mycormarl/issues/18).

### Plant initial biomass and pools: use a rounded one-milligram propagule

Use **`initial_biomass = 0.001 g`** as a representative low-end propagule
within the `0.7–3.3 mg` *Daucus carota* population-mean range reported by
Vandelook et al. (2024). Give each free resource pool one structural-biomass
equivalent:

- `initial_c_pool = 0.001 * 0.402 = 0.000402 g C`;
- `initial_p_pool = 0.001 * 1.92 = 0.00192 mg P`.

These are additional free reserves; structural C and P remain implicit in
biomass through `gamma_c` and `gamma_p`.

## Accepted recommendation

### Plant `amass`: apparent-gross daily leaf-mass carbon input

Use **`amass = 0.05 g C g⁻¹ leaf dry mass d⁻¹`** as the reference
default. This is the rounded midpoint (`0.051`) obtained by adding the fitted
respiration intercept magnitude to the carrot net light-response curves,
evaluating them at the experiment's 450 µmol photons m⁻² s⁻¹ irradiance,
integrating over its 16-h photoperiod, and converting with independently
measured carrot SLA. The resulting `0.036–0.073` range is an **apparent-gross**
estimate: photorespiration remains embedded, and the fitted dark intercept is
used as a proxy for mitochondrial respiration in light.

Interpret `amass` as the carbon input for one stated **reference day**, not as
literal fixation under 24 h of illumination. The current implementation
spreads that daily budget uniformly through numerical time. A future diurnal
extension should multiply it by a dimensionless, daily-normalised
`f_light(t)` without changing `amass` or its units.

### Plant `kappa_c`: charge full standing-biomass maintenance

Use **`kappa_c = 0.007 g C g⁻¹ whole-plant dry mass d⁻¹`** as the reference
default, rounded from the full-day carrot estimate of `0.00694`.
Reid's carrot growth model fitted shoot and root maintenance coefficients of
`q_s.maint = 0.021 d⁻¹` and `q_r.maint = 0.015 d⁻¹` at 20 °C, with `Q10 = 2`
([Reid 2019](https://doi.org/10.1080/01140671.2019.1588134)). Applying those
rates for a full day at 20 °C to the repository's `0.38` shoot and `0.62` root
fractions, then converting dry biomass with `gamma_c = 0.402`, gives
`0.00694 g C g⁻¹ whole-plant DM d⁻¹`. Because respiration has been removed
from the apparent-gross `amass` input, all fitted standing-biomass maintenance
is assigned to `kappa_c` rather than partitioned between the two parameters.

At the accepted `amass = 0.05` and `kleaf = 0.30`, carbon supply is `0.015 g C
g⁻¹ whole-plant DM d⁻¹`. Subtracting `kappa_c = 0.007` leaves
`0.008 g C g⁻¹ whole-plant DM d⁻¹` before growth and fungal trade.

## Executive findings

| Model quantity | Closest primary evidence | Recommendation for planning |
|---|---|---|
| Fungal `gamma_p` | *Glomus intraradices* (the isolate was subsequently called *G. irregulare*, now *Rhizophagus irregularis*) spores contained **1.3 ± 0.35 mg P g⁻¹** under low P and **8.0 ± 1.6 mg P g⁻¹** under high P; four selected young-hypha regions contained **2.4–9.6 mg P g⁻¹** ([Olsson et al. 2008](https://doi.org/10.1128/AEM.00376-08)). | **Selected provisionally: `2 mg P g⁻¹`**, the paper's approximate ordinary spore concentration. It is not an invariant whole-fungus stoichiometry or a direct maintenance-rate measurement. |
| Fungal saturation density | Jakobsen et al.'s external-hypha profiles reach approximately **20–25 m hypha cm⁻³ soil**, i.e. **2,000–2,500 cm cm⁻³**, while later mechanistic work summarizes the measurements as order `10³ cm cm⁻³` ([Jakobsen et al. 1992](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x); [Schnepf & Roose 2006](https://doi.org/10.1111/j.1469-8137.2006.01771.x)). | **Selected: `2,000 cm cm⁻³`**, the lower observed upper-profile estimate. It is not a universal AMF constant. |
| Plant initial biomass | A 40-population study found *Daucus carota* population means spanning **0.7–3.3 mg** ([Vandelook et al. 2024](https://doi.org/10.1017/S0960258524000230)). | **Selected: `0.001 g dry mass`**, a representative low-end one-milligram propagule, with one structural-biomass equivalent in each free pool. This is literature-bounded rather than a directly reported species mean. |
| Fungal initial biomass | No direct numerical *R. irregularis* dry mass per spore was recovered. The strongest fallback is a counted-and-dried five-species AMF calibration, `M = 0.4458e-5 d^2.5372`, where `M` is µg dry mass spore⁻¹ and `d` is mean diameter in µm ([Sieverding et al. 1989](https://doi.org/10.1016/0038-0717(89)90013-8)). Applying it to the reported *R. irregularis* diameter limits gives **0.214–1.885 µg spore⁻¹**. | **Selected: `7.97e-7 g`**, the regression evaluated at the 117.5-µm range midpoint. This is an empirical cross-species estimate, not a measured *R. irregularis* mean. |
| Plant `amass` | Carrot net light-response curves plus their fitted respiration intercepts give an apparent-gross **7.79–11.17 µmol CO₂ m⁻² s⁻¹** at the study's 450 µmol m⁻² s⁻¹ irradiance and 16-h photoperiod ([Kyei-Boahen et al. 2003](https://ps.ueb.cas.cz/pdfs/phs/2003/02/30.pdf)). Independent carrot accessions had SLA **66–94 cm² g⁻¹ leaf DM** ([Acosta-Motos et al. 2021](https://doi.org/10.3390/agronomy11122460)). | **Selected: `0.05 g C g⁻¹ leaf DM d⁻¹`**, the rounded midpoint of the `0.036–0.073` apparent-gross reference-day range. This is not true biochemical gross photosynthesis. |
| Plant `kappa_c` | A carrot growth model fitted `q_s.maint = 0.021 d⁻¹` and `q_r.maint = 0.015 d⁻¹` at 20 °C, with `Q10 = 2` ([Reid 2019](https://doi.org/10.1080/01140671.2019.1588134)). A full-day, 20 °C whole-plant conversion is `0.00694 g C g⁻¹ DM d⁻¹`. | **Selected: `0.007 g C g⁻¹ whole-plant DM d⁻¹`**, charging all fitted standing-biomass maintenance separately from apparent-gross carbon input. |
| Plant `kappa_p` | Carrot time courses show continued P accumulation and redistribution to the storage root, while no primary carrot study recovered a sustained irreversible P-loss rate ([Fernández-Pérez et al. 2023](https://doi.org/10.17584/rcch.2023v17i3.16508); [Cecílio Filho et al. 2013](https://acervodigital.unesp.br/handle/11449/75622)). | **Selected: `0.001 mg P g⁻¹ whole-plant DM d⁻¹`** as a small positive abstraction for herbivory, leakage, and unrecovered turnover—not measured carrot maintenance. Sensitivity: `0`, `0.001`, `0.002`; exploratory high-turnover case `0.005`. |

## 1. Preserve the repository contracts

The runtime trait definitions state that biomass is grams dry mass, fungal
`gamma_p` is mg P per g dry biomass, and fungal `saturation_density` is cm
hypha per cm³ bulk soil. Growth consumes `growth * gamma_p`, so `gamma_p` is a
dry-biomass P requirement/concentration, not phosphate uptake rate, internal
phosphate molarity, polyphosphate fraction, or P-transfer efficiency.

The photosynthesis implementation is

`C fixed = kleaf * plant biomass * amass * dt`,

where biomass is g dry mass and `dt` is days. Therefore dimensional closure
requires `amass` to be **g C g⁻¹ leaf dry mass day⁻¹**. The code comment
"photosynthetic rate per unit leaf dry mass" omits these units, but the current
default `1.0` necessarily means `1 g C g⁻¹ leaf DM d⁻¹`. It is not a value in
grams alone and cannot be directly compared with µmol CO₂ m⁻² s⁻¹.

## 2. Fungal phosphorus requirement (`gamma_p`)

Olsson et al. analyzed spores and young hyphae from monoxenic cultures of the
AM fungus then identified as *Glomus intraradices*. The isolate used in this
line of work was later renamed *G. irregulare*, corresponding to present-day
*Rhizophagus irregularis*. PIXE/STIM gave these dry-mass concentrations:

- spores: `1,300 ± 350 µg P g⁻¹` at 25 µM external P and
  `8,000 ± 1,600 µg P g⁻¹` after high-P enrichment;
- four selected young-hypha regions: `2,400`, `3,400`, `6,700`, and
  `9,600 µg P g⁻¹`.

Thus the direct species-priority evidence is **1.3–9.6 mg P g⁻¹**, far below
the former `40 mg P g⁻¹` default. However, P concentration was plastic with external
P, spores are storage structures, the hyphal measurements were four selected
microscopic regions rather than a whole-mycelium mean, and the measurements do
not establish fixed structural P demand. The paper itself describes about
`2,000 µg g⁻¹` (2 mg g⁻¹) as the usual spore-level concentration
([Olsson et al. 2008](https://doi.org/10.1128/AEM.00376-08)).

**Decision:** use `2 mg P g⁻¹`, the paper's approximate ordinary spore
concentration, as the fixed P-content assumption. The low-P treatment mean
(`1.3 mg P g⁻¹`) and high-P treatment mean (`8 mg P g⁻¹`) should remain
sensitivity cases because the model treats `gamma_p` as fixed stoichiometry.

## 3. Saturated external-hyphal density

Jakobsen, Abbott, and Robson grew subterranean clover with *Acaulospora
laevis*, an unidentified *Glomus* sp., or *Scutellospora calospora*. Roots
were excluded from a hyphal compartment by mesh; soil cores were sampled at
different distances after 0, 7, 14, 28, and 47 days. The original profiles are
reported in **m hypha cm⁻³ soil** and rise to roughly 20–25 m cm⁻³, equivalent
to 2,000–2,500 cm cm⁻³. Schnepf and Roose independently summarize these
observations as order `10³ cm cm⁻³`.

Important limitations are:

- the values are spatial profiles, not a fitted universal saturation constant;
- the upper values depend on fungal species, distance from the root boundary,
  and harvest time;
- none of the fungi was identified as *R. irregularis*;
- extraction/grid-intersection measurements quantify visible external hyphae
  and do not establish that all measured length was alive or uptake-active;
- **2,000–2,500**, not 3,000, is the closest reading of the plotted upper
  observations. A `3,000 cm cm⁻³` default would be a modelling margin.

Sources: [Jakobsen et al. (1992)](https://doi.org/10.1111/j.1469-8137.1992.tb01077.x),
[Schnepf et al. (2008)](https://doi.org/10.1098/rspb.2007.1380), and
[Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x).

## 4. Initial organism biomasses and resource pools

### *Daucus carota* seed mass

Vandelook et al.'s 40-population experiment weighed usually about 100
dry-stored seeds per population and found population means of **0.7–3.3 mg**.
Carrot "seeds" in normal agricultural/ecological usage are mericarps (dry
one-seeded fruit), so the chosen state should be documented as propagule mass
rather than isolated embryo mass
([Vandelook et al. 2024](https://doi.org/10.1017/S0960258524000230)).

The accepted default selects a representative low-end 1 mg propagule from
this range. It is a literature-bounded model choice, not a directly reported
species mean. Full stoichiometric free pools are:

- `initial_c_pool = biomass * gamma_c = 0.001 * 0.402 = 0.000402 g C`;
- `initial_p_pool = biomass * gamma_p = 0.001 * 1.92 = 0.00192 mg P`.

### *Rhizophagus irregularis* spore mass

**Accepted representation:** reset begins with an already established
plant–fungus symbiosis. One *R. irregularis* spore's dry mass is used only as a
proxy for initial fungal dry biomass; the model immediately treats all of that
mass as living external hyphae capable of phosphate absorption. It does not
represent a dormant spore, germination delay, retained spore tissue,
intraradical biomass, or establishment failure.

No published numerical **mean dry mass per *R. irregularis* spore** was
located. Saito's
primary dataset is unusually close: DAOM197198 spores were purified, counted,
fungal pellets were dried at 70 °C for 48 h and weighed on a microbalance, and
the equation explicitly subtracts `DW_p`, the dry weight of one parent spore.
But neither the archived CSV files nor its methods expose `DW_p`. The dataset
therefore proves that the quantity was used, not what its value was.

The strongest other-AMF evidence is Sieverding, Toro, and Mosquera's direct
determination of spore counts and dry matter for five AMF species. Across their
observations, the best-fit relationship was

`M = (0.4458e-5) * d^2.5372`,

where the published response is g dry matter per `10^6` spores and `d` is mean
spore diameter in µm. Numerically, g per `10^6` spores equals **µg per spore**.
Dry matter was only weakly affected by the tested soils and host plants
([Sieverding et al. 1989](https://doi.org/10.1016/0038-0717(89)90013-8)).

| Species or application | Diameter used (µm) | Dry mass (µg spore⁻¹) | Evidence basis and caveat |
|---|---:|---:|---|
| Five-species AMF calibration: *Acaulospora appendicula*, *Entrophospora colombiana*, *Glomus manihotis*, *G. occultum*, and *Scutellospora heterogama* | Species mean diameters | Underlying measurements not separately recoverable from the accessible paper record | Spores were counted and dry matter was determined directly; the published regression above is the transferable result. |
| *E. colombiana* | 121 mean (n=83) | 0.858 predicted | Regression prediction, not the species' measured table value; diameter from [INVAM](https://invam.ku.edu/colombiana). |
| *G. manihotis* | 182 mean (n=120) | 2.418 predicted | Regression prediction, not the species' measured table value; diameter from [INVAM](https://invam.ku.edu/manihotis). |
| *G. occultum* | 71.5 mean (n=120) | 0.226 predicted | Regression prediction, not the species' measured table value; diameter from [INVAM](https://invam.ku.edu/occultum). |
| *S. heterogama* | 159 mean (n=95) | 1.716 predicted | Regression prediction, not the species' measured table value; diameter from [INVAM](https://invam.ku.edu/heterogama). |
| *R. irregularis*, lower reported limit | 70 | 0.214 predicted | Cross-species extrapolation using the lower limit of the [INVAM range](https://invam.ku.edu/irregularis), not a mean. |
| *R. irregularis*, range midpoint | 117.5 | **0.797 predicted** | Cross-species interpolation at the midpoint of endpoints; **117.5 µm is not a measured mean**. |
| *R. irregularis*, upper reported limit | 165 | 1.885 predicted | Cross-species extrapolation using the upper limit, not a mean. |

Thus the empirical AMF estimate for the documented *R. irregularis* size
range is `2.14e-7–1.89e-6 g` per spore. The accepted estimate is the
midpoint-diameter result, **`7.97e-7 g` per spore**. This is
better grounded than using the repository's hyphal dry-tissue density, but it
is still not a measured *R. irregularis* mean: the calibration contains five
other taxa, the diameter endpoints may combine isolates and maturity states,
and wall thickness, contents, shape, hydration history, and spore maturity all
affect mass.

This relationship can calibrate **spore biomass scaling with diameter**. It
should not calibrate the model's general fungal-tissue density: treating the
spores as spheres gives implied dry bulk densities of about `0.80–1.19 g
cm⁻³` across the *R. irregularis* diameter limits, much higher than the
repository's provisional `0.231 g cm⁻³` for hyphal tissue. Wall-rich resting
spores and metabolically active hyphae are not density-equivalent. Commercial
inoculum mass per spore, fresh mass, and inoculum containing carrier, roots, or
attached hyphae were therefore excluded.

For comparison, the old geometry-and-repository-density calculation gives
`4.1e-8–5.4e-7 g` per spore and `2.0e-7 g` at the diameter midpoint. It is a
model-derived estimate rather than literature evidence and is about fourfold
lower than the empirical AMF regression at the midpoint.

Once a fungal mass `M_f` is chosen, pools follow without ambiguity:

- `initial_c_pool = M_f * gamma_c` in g C;
- `initial_p_pool = M_f * gamma_p` in mg P.

At the accepted `M_f = 7.97e-7 g`, `gamma_c = 0.5` gives exactly
`3.985e-7 g C`, and the selected `gamma_p = 2 mg g⁻¹` gives exactly
`1.594e-6 mg P`.

## 5. `amass`: area-based observations versus the model's mass-based rate

[Kyei-Boahen et al. (2003)](https://ps.ueb.cas.cz/pdfs/phs/2003/02/30.pdf)
measured leaf **net** photosynthesis, not gross carbon fixation, across PAR
100–1,000 µmol photons m⁻² s⁻¹ at 20 °C, 350 µmol CO₂ mol⁻¹ air, and 65%
relative humidity. Four cultivar curve fits gave predicted
photon-saturated `P_Nmax` of 16.40, 18.11, 19.78, and 19.79 µmol CO₂ m⁻² s⁻¹.
The plants did not actually reach saturation at the highest available PAR.
Dark respiration estimates were 0.85–2.66 µmol CO₂ m⁻² s⁻¹. These are
instantaneous controlled-condition, one-sided-area gas-exchange rates, not
daily carbon production.

[Acosta-Motos et al. (2021)](https://doi.org/10.3390/agronomy11122460)
independently measured carrot SLA of **66–94 cm² g⁻¹ leaf DM** across
accessions. I found no primary carrot study pairing the above gas-exchange
observations with leaf dry mass/SLA on the same sampled leaves. Combining
independent studies is therefore provisional.

The saturation parameters are poor daily defaults. The plants were grown at
approximately **450 ± 20 µmol photons m⁻² s⁻¹** under a **16-h day**, and none
of the four cultivars actually saturated even at 1,000 µmol m⁻² s⁻¹. Using the
paper's rectangular-hyperbola model,

`P_N(I) = alpha * I * P_Nmax / (alpha * I + P_Nmax) - R_D`,

with `R_D` expressed as a positive respiration magnitude, gives the following
rates at the actual 450-µmol growth irradiance. The mass-based ranges use the
independent carrot SLA envelope of 66–94 cm² g⁻¹.

| Cultivar | Predicted light-period net `P_N(450)` (µmol CO₂ m⁻² s⁻¹) | 16-h light-period gain (g C g⁻¹ leaf DM d⁻¹) | After 8-h dark respiration at reported 20 °C `R_D` (g C g⁻¹ leaf DM d⁻¹) |
|---|---:|---:|---:|
| Cascade | 8.51 | 0.039–0.055 | 0.033–0.047 |
| Caro Choice | 6.68 | 0.030–0.043 | 0.028–0.040 |
| Oranza | 8.02 | 0.037–0.052 | 0.033–0.046 |
| Red Core Chantenay | 7.57 | 0.035–0.049 | 0.033–0.046 |

Across cultivars and SLA, the reference envelope is therefore **0.030–0.055
g C g⁻¹ leaf DM d⁻¹** for the 16-h light period. At midpoint SLA
`80 cm² g⁻¹` and the four-cultivar mean predicted assimilation, it is `0.0426
g C g⁻¹ leaf DM d⁻¹`. This supports a rounded provisional default of
**`0.04 g C g⁻¹ leaf DM d⁻¹`**.

For area-based net assimilation `A_n`, SLA, and light duration `h`, the
mass-based light-period carbon gain is

`amass_light = A_n * (SLA * 1e-4 m² cm⁻²) * 12e-6 g C µmol⁻¹ * (3600 h)`.

Combining the extrema `A_n = 16.40–19.79` and `SLA = 66–94 cm² g⁻¹` gives:

- **0.075–0.129 g C g⁻¹ leaf DM d⁻¹** for a 16-hour light period;
- about **0.066–0.127 g C g⁻¹ leaf DM d⁻¹** after also applying the reported
  dark-respiration extrema over an 8-hour night (extrema are not paired).

The user's 4.5–15 µmol CO₂ m⁻² s⁻¹ range is likewise area-based; with the same
SLA envelope and 16 h light it converts to approximately
**0.021–0.097 g C g⁻¹ leaf DM d⁻¹**, before nighttime respiration.

The field study of Topcut and Sugarsnax provides a useful lower-performance
check: its highest reported midday `P_N` was only `4.5 µmol m⁻² s⁻¹` under
field timing treatments ([Thiagarajan et al. 2012](https://doi.org/10.1007/s11099-012-0034-6)).
Holding that instantaneous maximum for 16 h—which likely overstates a real
field day—and applying the same SLA envelope gives `0.021–0.029 g C g⁻¹ leaf
DM d⁻¹` before night respiration. This supports retaining a lower sensitivity
case near `0.02–0.03`, not replacing the controlled-condition reference with
an unintegrated midday observation.

Nighttime correction requires a semantic choice. Applying the Kyei-Boahen
dark-respiration estimates (`0.85–2.66 µmol CO₂ m⁻² s⁻¹`) for an 8-h night,
matched by cultivar, gives `0.028–0.047 g C g⁻¹ leaf DM d⁻¹`. This is a
conservative correction because those rates were fitted at a 20 °C leaf
temperature whereas the plants were grown at 10 °C at night. More
importantly, the model separately removes carbon through `kappa_c`; putting
night respiration into a net `amass` can double-count maintenance. The earlier
`amass = 0.04`, `kappa_c = 0.004` mapping avoided that overlap, but it has been
superseded by the apparent-gross boundary in Section 7.

The current photosynthesis function has no irradiance or photoperiod state and
applies `amass` throughout elapsed days. A calibrated `amass` must therefore
be a day-integrated rate; inserting `4.5–15` directly would be dimensionally
wrong by area, molar-mass, and time conversions. No primary carrot study was
found that paired leaf-area gas exchange with SLA/LMA on the same sampled
leaves, cultivar, age, and environment. Kyei-Boahen measured area but not leaf
dry mass; Acosta-Motos et al. measured SLA but not gas exchange, used different
accessions, and grew them in a warmer glasshouse. This cross-study mismatch is
the main remaining evidence limitation.

## 6. `kappa_c`: whole-plant standing-biomass maintenance

[Reid (2019)](https://doi.org/10.1080/01140671.2019.1588134) fitted a carrot
growth model with maintenance coefficients `q_s.maint = 0.021 d⁻¹` for shoots
and `q_r.maint = 0.015 d⁻¹` for roots at 20 °C, and scaled them with `Q10 = 2`.
Mapping those fitted biomass-rate coefficients to the repository's carbon
currency requires multiplication by `gamma_c = 0.402 g C g⁻¹ DM`.

For comparison, charging all shoot and root biomass for a full day at 20 °C
would give

`kappa_c,20 = 0.402 * (0.38 * 0.021 + 0.62 * 0.015) = 0.00694 g C g⁻¹ DM d⁻¹`.

That full-day conversion is the accepted mapping under the apparent-gross
carbon-input boundary adopted in Section 7. For comparison, the earlier net
mapping used the repository fractions `kleaf = 0.30`, `kroot = 0.62`, and
hence nonphotosynthetic shoot fraction `1 - kleaf - kroot = 0.08`:

`kappa_c = 0.402 * [0.62*0.015*(16/24 + 0.5*8/24) + 0.08*0.021*(16/24 + 0.5*8/24) + 0.30*0.021*(0.5*8/24)] = 0.00410 g C g⁻¹ DM d⁻¹`.

Here the factor `0.5` is the `Q10 = 2` scaling from 20 °C to the 10 °C night
used by Kyei-Boahen et al. The root and nonphotosynthetic shoot incur
maintenance over both temperature periods, while the leaf incurs a separate
maintenance charge only at night. This time-and-organ partition is an
**inference**, not a coefficient directly estimated by Reid. It supported the
now-superseded `kappa_c = 0.004` net-assimilation mapping. The selected
apparent-gross formulation instead rounds the full-day value to `0.007`.

At `amass = 0.05`, daily supply per whole-plant dry mass is
`kleaf * amass = 0.30 * 0.05 = 0.015 g C g⁻¹ DM d⁻¹`; after the selected
`0.007` maintenance charge the balance is `0.008 g C g⁻¹ DM d⁻¹` before growth and
fungal exchange. This does not account for **growth respiration or biosynthetic
costs associated with producing new biomass**. Those costs belong to growth,
not maintenance `kappa_c`, and remain a separate unresolved model term rather
than a reason to inflate the maintenance coefficient.

## 7. Accepted apparent-gross boundary and future diurnal light contract

Separating gross fixation from respiration is physiologically possible, but
the carrot evidence used here does not make that separation. Kyei-Boahen et
al. measured ordinary IRGA **net CO₂ exchange** at PAR values from 100 to
1,000 µmol m⁻² s⁻¹; they did not measure at zero light. Their rectangular
hyperbola can be written using a signed respiration intercept as

`P_N(I) = alpha * I * P_Nmax / (alpha * I + P_Nmax) + R_D`.

Consequently, `R_D` is a jointly fitted, extrapolated zero-light intercept,
not a direct dark-respiration observation, and its standard errors are large.
The light-driven term is mathematically asymptotic at `P_Nmax`; net exchange
is that term plus the negative `R_D`, notwithstanding the paper's looser
description of `P_Nmax` as photon-saturated net photosynthesis.

For a C3 leaf, the biochemical gas-exchange identity is
`A = V_c - 0.5 V_o - R_d`: net assimilation is Rubisco carboxylation minus
photorespiratory CO₂ release and mitochondrial respiration in the light
([Farquhar et al. 1980](https://doi.org/10.1007/BF00386231)). Adding
`|R_D|` to Kyei-Boahen's `P_N` therefore yields only an **apparent gross
light-response term**. It leaves photorespiration embedded and assumes that
mitochondrial respiration in light equals the extrapolated dark intercept.
The carrot study used neither low O₂, isotopes, chlorophyll fluorescence, nor
a Kok/Laisk protocol, so it cannot identify `V_c`, photorespiration, and
respiration in light separately.

The practical numerical alternatives are materially different:

- At 450 µmol m⁻² s⁻¹, `P_N + |R_D|` is `7.79–11.17 µmol CO₂ m⁻² s⁻¹`
  across the four carrot cultivars.
- If the current light-independent implementation is interpreted as **literal
  24-h constant illumination**, that apparent-gross range and carrot SLA
  `66–94 cm² g⁻¹` convert to `0.053–0.109 g C g⁻¹ leaf DM d⁻¹`; the
  cultivar-mean rate and midpoint SLA give `0.077`, suggesting a rounded
  candidate `amass = 0.08` under that artificial regime.
- If `amass` instead stores a **16-h daily light budget smeared uniformly over
  24 h** by the numerical implementation, the same apparent-gross conversion
  is `0.036–0.073`, with midpoint `0.051`, suggesting a rounded candidate
  `0.05`.

Neither candidate is a measured biochemical gross-fixation rate. Methods that
can separate relevant fluxes include combined gas exchange and chlorophyll
fluorescence for respiration in light
([Yin et al. 2011](https://doi.org/10.1093/jxb/err038)), `¹³CO₂` isotope flux
measurements of gross assimilation and CO₂ evolution
([Haupt-Herting et al. 2001](https://doi.org/10.1104/pp.126.1.388)), and
`H₂¹⁸O`/O₂ isotope exchange for gross and net O₂ production
([Gauthier et al. 2018](https://doi.org/10.1104/pp.16.00741)). The latter
study also illustrates why substituting dark respiration is unsafe: in French
bean at ambient O₂, inferred respiration in light was about 36% below dark
respiration. These are method demonstrations in other species, not carrot
parameter estimates.

The accepted operational boundary is therefore: **`amass` is an acknowledged
apparent-gross reference-day carbon input (`0.05`), while `kappa_c` contains
the full standing-biomass maintenance charge (`0.007`).** This is a cleaner
accounting separation than embedding some respiration in each parameter, even
though the carrot data do not identify true biochemical gross fixation. The
uncertainty belongs explicitly to the apparent-gross approximation rather than
being hidden in a net-assimilation/maintenance partition.

For a future diurnal implementation without changing parameter units, retain
`amass` in `g C g⁻¹ leaf DM d⁻¹` as a reference daily carbon budget and apply
a dimensionless light-response multiplier `f_light(t)`:

`dC_fix/dt = kleaf * biomass * amass * f_light(t)`.

Set `f_light = 1` for backward-compatible runs. A diurnal profile should be
normalised so that its one-day integral is one day when preserving the same
daily budget; a dark interval can then have `f_light = 0` without silently
changing total fixation. A later mechanistic PAR response may instead define
the multiplier relative to a stated reference irradiance. Maintenance can
remain in the existing `g C g⁻¹ whole-plant DM d⁻¹` units and gain separate
dimensionless temperature and organ multipliers.

Finally, changing from net to apparent-gross `amass` and charging all maintenance in
`kappa_c` would **not remove growth respiration**. Carrot-specific support is
especially direct: Reid fitted separate growth-respiration coefficients of
`0.26` for shoots and `0.46` for storage roots in addition to maintenance.
More generally, biosynthesis consumes substrate for ATP and reducing power,
so carbon input exceeds the carbon retained in new biomass
([Penning de Vries et al. 1974](https://doi.org/10.1016/0022-5193(74)90119-2);
[Thornley 1970](https://doi.org/10.1038/227304b0)). The current model's
`gamma_c` is structural carbon and growth consumes exactly
`gamma_c * new biomass`, implying 100% conversion of allocated substrate C
into structural C. Growth cost should therefore later become a separate
dimensionless growth yield/conversion efficiency (potentially organ-specific),
or an explicit substrate-C cost per unit biomass. It should not be folded into
maintenance `kappa_c`. That extension is deliberately deferred; the present
model continues to imply 100% conversion of allocated substrate C into
structural C.

## Decision gaps before changing defaults

1. Obtain the numerical DAOM197198 parent-spore dry mass or measure the target
   isolate to replace or validate the accepted cross-species estimate.
2. Later design and parameterise growth respiration or biosynthetic costs as a
   term separate from standing-biomass maintenance `kappa_c`.
3. Replace the cross-study SLA conversion if paired carrot gas exchange and
   leaf dry mass are measured for the modeled cultivar and growth regime.
