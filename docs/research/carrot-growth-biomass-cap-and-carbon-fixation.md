# Carrot growth, the biomass cap, and carbon-fixation scale

**Research date:** 13 August 2026  
**Scope:** Primary evidence for calibrating the plant biomass trajectory,
windowed relative growth rate, leaf-mass carbon input, and biomass cap for a
roughly 120-day cultivated-carrot scenario. This note evaluates the active
MycorMARL defaults but does not change them.

## Executive conclusion

The current carbon-only growth ceiling is too low for a favourable cultivated
carrot crop, but the strongest evidence does **not** support doubling `amass`
in isolation.

- A high-yield field time course for carrot cv. Forto reached **23.26 g dry
  matter plant^-1 at 120 days after sowing (DAS)**. Its fitted shoot and storage-
  root curves imply whole-plant RGRs of approximately `0.066`, `0.065`, `0.060`,
  and `0.042 d^-1` in successive 20-day windows from 40 to 120 DAS
  ([Cecilio Filho & Peixoto 2013](https://repositorio.unesp.br/bitstream/11449/75622/1/2-s2.0-84878476833.pdf)).
- MycorMARL's sustained carbon-only ceiling is `0.0199 d^-1`, and its idealized
  upper-bound biomass after spending the initial free pools and growing for
  120 days is only about **0.22 g**. That is about two orders of magnitude below
  the Forto endpoint.
- The active `amass = 0.05 g C g^-1 leaf DM reference-day^-1` remains a
  defensible rounded midpoint for the explicitly stated 16-hour,
  450-micromol-PAR reference day. The underlying carrot measurements were made
  at 20 degrees C and 350 micromol CO2 mol^-1, and the plants were raised under
  a 16-hour photoperiod at `450 +/- 20 micromol PAR m^-2 s^-1`
  ([Kyei-Boahen et al. 2003](https://ps.ueb.cas.cz/pdfs/phs/2003/02/30.pdf)).
- The larger inconsistency is `kleaf = 0.30`. Forto's fitted curves imply leaf
  mass fractions declining from about **0.81 at 40 DAS** to **0.23 at 120 DAS**.
  Thus `0.30` is representative of a mature plant, not the early and middle
  growth phases. Independent glasshouse carrot measurements also show that
  leaf area, leaf mass, and partitioning are strongly accession- and age-
  dependent ([Acosta-Motos et al. 2021](https://doi.org/10.3390/agronomy11122460)).
- The former `100 g DM` cap is not supported as a 120-day physiological
  maximum. The project therefore selected `50 g DM` as a provisional numerical
  guard: it is about `1.43` times the fitted `Forto` asymptote and roughly twice
  the largest independent cultivar-mean endpoint. It is not an observed
  maximum, and qualification runs must not contact it.
- The Forto `35.05 g` asymptote is a useful **scenario reference**, but not a
  defensible hard biological carrying capacity. Independent field endpoints
  for ten cultivars were `21.63--24.65 g whole-plant DM` around 120 DAS, while
  two controlled-pot studies ended near `13--14 g whole-plant DM`. None
  observed a `35.05 g` plant. If a smaller numerical guard is needed, `50 g DM`
  is more defensible than `35.05 g`: it leaves headroom above the fitted Forto
  trajectory, but remains an engineering choice that runs should not contact.

The most defensible immediate qualification is therefore to vary the effective
whole-plant fixation term `kleaf * amass`, prioritising a higher or dynamic
`kleaf` over an unsupported increase in leaf-level `amass`. A fixed `kleaf`
near `0.60` is a useful **interim sensitivity**, not a final ontogenetic model:
with the current `amass`, maintenance, and structural-C coefficient it gives a
carbon-only ceiling of about `0.057 d^-1`, close to the Forto 40--120 DAS mean.

## 1. What the model currently implies

MycorMARL fixes plant carbon according to

```text
C_fixed = biomass * kleaf * amass * dt
```

and pays standing maintenance before allocation. With the active defaults,

```text
kleaf * amass = 0.30 * 0.05 = 0.015 g C g^-1 plant DM d^-1
net C before growth = 0.015 - 0.007 = 0.008 g C g^-1 DM d^-1
maximum sustained RGR = 0.008 / 0.402 = 0.0199 d^-1
```

This is an upper bound before reproduction or fungal transfer, and assumes P
is sufficient. Each initial free pool contains one structural-biomass
equivalent, so an idealized growth-only plant can first double from `0.01` to
`0.02 g`, then reach

```text
0.02 * exp(0.0199 * 120) = 0.218 g DM.
```

The calculation follows the runtime traits in
[`plant/traits.py`](../../mycormarl/mycormarl/plant/traits.py) and the fixation
implementation in
[`plant/photosynthesis.py`](../../mycormarl/mycormarl/plant/photosynthesis.py).
It deliberately omits P limitation and policy allocation, so a realized model
trajectory can only be lower unless another pool or transfer supplies carbon.

## 2. A directly usable 120-day biomass trajectory

Cecilio Filho and Peixoto sampled carrot cv. Forto at 10-day intervals from 40
to 120 DAS in a high-yield irrigated field experiment in Sao Gotardo, Brazil.
Plants were separated into leaves and storage roots and dried at 65 degrees C
to constant mass. The crop received substantial NPK fertilization and reached
`72 t ha^-1` commercial fresh-root yield, so it is best treated as a favourable
crop-growth benchmark rather than a nutrient-limited control
([primary paper](https://repositorio.unesp.br/bitstream/11449/75622/1/2-s2.0-84878476833.pdf)).

The fitted dry-mass equations were

```text
leaf DM = 6.84 / (1 + exp[-0.062 * (DAS - 98.00)])
storage-root DM = 28.21 / (1 + exp[-0.088 * (DAS - 113.91)])
```

with reported `R^2 = 0.99` and `0.989`, respectively. Evaluating their sum gives:

| DAS | Fitted leaf DM (g) | Fitted storage-root DM (g) | Fitted total DM (g) | Leaf mass fraction | RGR over following 20 d (d^-1) |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.183 | 0.042 | 0.225 | 0.812 | 0.0656 |
| 60 | 0.592 | 0.243 | 0.836 | 0.709 | 0.0647 |
| 80 | 1.688 | 1.358 | 3.046 | 0.554 | 0.0596 |
| 100 | 3.632 | 6.410 | 10.042 | 0.362 | 0.0420 |
| 120 | 5.447 | 17.797 | 23.244 | 0.234 | -- |

The interval calculation is `ln(M2 / M1) / (t2 - t1)`. The observed values
reported in the paper were `0.18 g` leaf and `0.04 g` storage-root DM at 40 DAS,
and `5.45 g` and `17.81 g` at 120 DAS, corroborating the fitted endpoints.

Important limits are:

- the reported total excludes unharvested fine-root biomass;
- the reported leaf fraction includes harvested foliar/support tissue and is
  therefore a proxy for photosynthetic mass, not proof that every gram is
  equally photosynthetically active;
- it is one winter cultivar in one well-fertilized field environment;
- crop competition and canopy light interception differ from an isolated
  model plant;
- the logistic asymptotes describe the fitted crop trajectory and are not a
  species-wide maximum.

Reid's independent carrot model is consistent with treating canopy development
and source limitation as dynamic. It was fitted and tested against roughly
fortnightly shoot, storage-root, and leaf-area measurements from two field
experiments, and calculated photosynthesis from intercepted radiation rather
than a constant fraction of whole-plant mass. Its simulated net radiation-use
efficiency was `1.340 +/- 0.0079 g DM MJ^-1`, within the authors' cited carrot
range of `1.25--1.85 g DM MJ^-1`
([Reid 2019](https://doi.org/10.1080/01140671.2019.1588134)).

## 3. Is the leaf-level `amass` too low?

Kyei-Boahen et al. measured net photosynthesis on the youngest fully expanded
leaves of four 30-day-post-emergence carrot cultivars. Plants were grown at a
16-hour photoperiod, `20/10 degrees C` day/night, and about
`450 micromol PAR m^-2 s^-1`; response curves were measured at 20 degrees C
over `100--1000 micromol PAR m^-2 s^-1`. The fitted photon-saturated net rates
were `16.40--19.79 micromol CO2 m^-2 s^-1`, and none of the cultivars saturated
at 1000 PAR ([Kyei-Boahen et al. 2003](https://ps.ueb.cas.cz/pdfs/phs/2003/02/30.pdf)).

At 450 PAR, adding the magnitude of each fitted dark intercept to the fitted net
curve gives the existing note's **apparent-gross** range of
`7.79--11.17 micromol CO2 m^-2 s^-1`. Integrating for 16 hours and applying the
independently observed carrot SLA range of `66--94 cm^2 g^-1 leaf DM` gives
`0.036--0.073 g C g^-1 leaf DM reference-day^-1`, whose approximate midpoint
is the active `amass = 0.05`
([Acosta-Motos et al. 2021](https://doi.org/10.3390/agronomy11122460)).

The conversion is useful but not a direct whole-plant carbon balance:

- it combines different cultivars and experiments;
- a youngest fully expanded leaf is not the canopy mean;
- the light-response measurement is instantaneous, whereas `amass` represents
  an integrated reference day;
- adding the dark intercept is only an apparent-gross correction and does not
  remove photorespiration;
- SLA and leaf mass fraction change with genotype, environment, and age.

At 700 and 1000 PAR, the same calculation gives approximate midrange `amass`
values of `0.062` and `0.071`, with broad cross-study envelopes extending to
about `0.086` and `0.095`. Thus `0.05` is defensible for the named 450-PAR day,
but should not be treated as invariant across irradiance regimes. Conversely,
an `amass` near `0.10` would be difficult to justify for that 450-PAR reference
condition.

## 4. The fixed leaf fraction is the larger mismatch

The active `kleaf = 0.30` produces the same whole-plant fixation term at every
age. Forto's fitted leaf mass fraction falls from `0.81` at 40 DAS to `0.23` at
120 DAS. With `amass = 0.05`, those fractions imply the following unshaded
whole-plant apparent-gross inputs:

| DAS | Forto fitted leaf fraction | `leaf fraction * amass` (g C g^-1 plant DM d^-1) |
|---:|---:|---:|
| 40 | 0.812 | 0.0406 |
| 60 | 0.709 | 0.0354 |
| 80 | 0.554 | 0.0277 |
| 100 | 0.362 | 0.0181 |
| 120 | 0.234 | 0.0117 |

The current constant is `0.015`. It therefore strongly understates the early
and middle source mass, approximates the mature crop, and slightly overstates
the late fitted Forto value. This pattern can explain both the very low
cumulative model upper bound and why simply increasing `amass` would be risky:
a larger constant `amass` corrects early growth by maintaining excessive late
fixation, preserving exponential growth where observed crop growth decelerates.
Because harvested foliar dry matter includes support tissue, these products
are deliberately interpreted as scale diagnostics rather than direct canopy
carbon-flux estimates.

The effective fixation term required to sustain a specified carbon-only RGR is

```text
kleaf * amass = kappa_c + gamma_c * RGR.
```

Matching the Forto 40--120 DAS average RGR of `0.0580 d^-1` would require
`kleaf * amass = 0.0303`. This can be produced by either `kleaf ~= 0.61` at the
current `amass`, or `amass ~= 0.101` at the current `kleaf`. The former is
better aligned with the measured age-varying biomass allocation; the latter
conflicts with the stated 450-PAR leaf-level reference.

This does **not** establish `kleaf = 0.60` as a universal carrot trait. It is an
age-averaged sensitivity that can test whether the low biomass is primarily a
source-mass scaling problem. A biologically stronger solution would make leaf
mass or leaf area an explicit, age- and allocation-dependent state and include
canopy/light limitation.

## 5. Is a 100-g dry-mass cap plausible?

No primary source recovered here establishes `100 g DM plant^-1` as a plausible
120-day cultivated-carrot maximum.

- Forto reached `23.26 g DM` at 120 DAS under a high-yield field regime. The
  sum of its fitted leaf and storage-root asymptotes is `35.05 g`, though this
  excludes fine roots and is not a species maximum.
- Twelve-week glasshouse carrots grown by Acosta-Motos et al. were much smaller:
  the reference cultivar had a `21.9 g` **fresh** taproot with `12.6%` root dry
  matter, or about `2.76 g` taproot DM. The experiment used 7-L pots, half-
  strength Hoagland fertigation, and a warm `17--33 degrees C` regime
  ([Acosta-Motos et al. 2021](https://doi.org/10.3390/agronomy11122460)).
- Reid's field experiments continued to 170 and 203 DAS and demonstrate that
  cultivar, plant density, radiation interception, water, and nitrogen alter
  the attainable trajectory; they do not identify a universal per-plant cap
  ([Reid et al. 2017](https://doi.org/10.1080/01140671.2017.1402790)).

The selected `50 g` value is an evidence-bounded numerical guard, not a
biological carrying capacity. The implementation now represents the actor's
biomass observation reference independently as `50 g`. This preserves the
former policy-input scale (`0.5 * 100 g`) while preventing future cap changes
from also changing observation normalization.

### What `35.05 g` does and does not mean

The Forto value is the arithmetic sum of two independently fitted logistic
asymptotes: `6.84 g` leaves and `28.21 g` storage root. It is therefore the
limit of the **equations as DAS tends to infinity**, not a harvested plant.
The last observations at 120 DAS summed to `23.26 g`; the fitted sum reaches
approximately `34.4 g` only at 160 DAS. Moreover, the reported harvest omitted
fine roots, and an organ can senesce while another continues accumulating.
Summing two organ asymptotes is consequently not equivalent to estimating a
whole-plant carrying capacity.

Independent primary endpoints put that interpretation in context:

| Primary experiment | Age and setting | Reported biomass | Comparable whole-plant DM | What it establishes |
|---|---|---:|---:|---|
| Forto field time course | 120 DAS; irrigated, heavily fertilized field | `5.45 g` leaf + `17.81 g` storage-root DM | `23.26 g` observed; `35.05 g` fitted asymptote | A favourable cultivar-specific trajectory; the asymptote was not observed |
| Ten-cultivar Brazilian semiarid field trial | generally about 120 DAS; August planting harvested at 150 DAS | cultivar mean dry plant weights `21.63--24.65 g`; season means `22.69--23.54 g` | `21.63--24.65 g` | A replicated whole-plant harvest range across cultivar and sowing environment, not individual maxima ([Gomes et al. 2021](https://doi.org/10.4025/actasciagron.v43i1.51831)) |
| DLBA and DH1 controlled pots | pot study; control and saline treatments | non-saline cultivar-treatment means `9.5--14.4 g` total dry weight | `9.5--14.4 g` | Genotype and stress can move endpoints substantially below the Forto benchmark ([Smolen et al. 2020](https://doi.org/10.3390/agronomy10050659)) |
| Vermicompost/earthworm pots | 137 days; two annual runs, ten plants per treatment | maximum treatment means `64.933 g` fresh root and `15.400 g` fresh leaves per plant; dry matter `16.50%` and `15.03%` | approximately `13.03 g`, calculated within the same treatment | A directly convertible endpoint under an unusual nutrient-rich pot substrate, still well below `35.05 g` ([Kovacik et al. 2022](https://doi.org/10.3390/agronomy12112770)) |

The Gomes values are treatment means from ten harvested plants, not the
largest individual plants. They therefore cannot prove a species maximum.
They do show that `35.05 g` is about **42% above** the largest reported
cultivar mean (`24.65 g`) in that independent experiment. Conversely, fresh
storage-root weights cannot be compared directly with the model's dry-mass
state: Gomes et al. reported cultivar mean fresh root weights of approximately
`80--122 g`, and other primary experiments report similarly large fresh roots,
but water comprises most of that mass. A fresh root endpoint must not be used
as a whole-plant dry-mass cap without organ-specific dry fractions from the
same treatment.

Thus `35.05 g` is defensible as a **Forto-like upper reference scale** for a
first-year vegetative scenario, or as one member of a cap-sensitivity set. It
is not defensible as an observed maximum, a species-wide maximum, or a
mechanistic carrying capacity. Making it a hard cap would also risk building
the desired late sigmoid shape into the model and then treating that shape as
a validation result.

For the pilot, report distance to the cap and reject runs that contact it. Do
not lower the cap merely to force observed sigmoid growth or a P-response
threshold. A practical hierarchy is:

1. use approximately `25--35 g DM` as the biological **reference range** for a
   favourable 120-day cultivated-carrot trajectory;
2. use the selected `50 g DM` hard numerical guard, because it is `1.43` times
   the Forto fitted asymptote and about twice the largest independent cultivar
   mean;
3. retain the independent `50 g DM` actor-observation reference so the guard
   and policy-input scale can be varied separately; and
4. treat any run contacting the guard as a failed qualification, not evidence
   that the guard is a biological limit.

## 6. Recommended qualification and parameter-register updates

### Parameter decisions

1. **Do not change `amass = 0.05` solely because the current whole-plant growth
   ceiling is low.** Preserve it as the 450-PAR, 16-hour reference-day value.
   Add `0.06` and `0.07` as irradiance sensitivities if the intended reference
   day is closer to 700--1000 PAR.
2. **Treat `kleaf = 0.30` as mature-stage, not whole-episode, evidence.** Run a
   pre-pilot plant-only, high-P growth qualification at `kleaf = 0.30`, `0.45`,
   and `0.60`. The `0.60` case is the evidence-informed age-averaged diagnostic.
3. **Prefer an ontogenetic leaf-mass or leaf-area state before final inference.**
   A constant `kleaf` cannot reproduce both rapid early growth and late
   deceleration without conflating partitioning with leaf physiology.
4. **Keep biological reference and numerical guard conceptually separate.**
   Use `25--35 g` as a favourable first-year trajectory reference and the
   selected `biomass_cap = 50 g` as a provisional numerical guard. It is not a
   measured carrying capacity. Do not use the exact Forto asymptote (`35.05 g`)
   as a hard ceiling, and reject guard-contacting qualification trajectories.

### Acceptance checks

The plant-only high-P vegetative control should report:

- fitted and realized RGR over at least `40--60`, `60--80`, `80--100`, and
  `100--120 d` windows, with explicit alignment between simulation age and DAS;
- biomass at the same time points as the Forto trajectory;
- gross C fixation, maintenance, structural-growth C, and free-pool change;
- the fraction of endpoint biomass attributable to initial free pools;
- whether P, C, soil inventory, domain contact, or the hard cap limited growth;
- sensitivity to `kleaf`, `amass`, and the cap/observation reference separately.

A parameterisation should not be accepted merely because its 120-day endpoint
matches `23.26 g`: it should also reproduce the declining windowed RGR and a
credible carbon balance. Conversely, the Forto curve should be treated as one
favourable field benchmark, not a mandatory universal target.

### Changes needed in `docs/model-parameter-register.md`

The register should be updated after the project chooses a qualification
outcome:

- replace the statement that `kleaf` has no quantitative source with the
  stage-specific Forto evidence, while retaining its status as unsupported as
  a **constant** whole-episode value;
- retain the current `amass` derivation but state explicitly that higher-light
  sensitivities are approximately `0.06--0.07` and that increasing `amass`
  cannot substitute for ontogenetic leaf allocation;
- record the selected `50 g` numerical guard with the Forto endpoint,
  asymptote, independent cultivar endpoints, and the guard-contact rejection
  rule; and
- record `biomass_observation_reference = 50 g` as an independent
  policy-interface parameter preserving the former input scale.

## Primary sources

1. Cecilio Filho, A. B. & Peixoto, F. C. (2013). Accumulation and exportation
   of nutrients by carrot 'Forto'. *Revista Caatinga* 26, 64--70.
   [Primary PDF](https://repositorio.unesp.br/bitstream/11449/75622/1/2-s2.0-84878476833.pdf).
2. Kyei-Boahen, S., Lada, R., Astatkie, T., Gordon, R. & Caldwell, C. (2003).
   Photosynthetic response of carrots to varying irradiances.
   *Photosynthetica* 41, 301--305.
   [Primary PDF](https://ps.ueb.cas.cz/pdfs/phs/2003/02/30.pdf).
3. Acosta-Motos, J. R. et al. (2021). Comparative characterization of Eastern
   carrot accessions for some main agricultural traits. *Agronomy* 11, 2460.
   [DOI](https://doi.org/10.3390/agronomy11122460).
4. Reid, J. B. (2019). Modelling growth and dry matter partitioning in root
   crops: a case study with carrot (*Daucus carota* L.). *New Zealand Journal
   of Crop and Horticultural Science* 47, 99--124.
   [DOI](https://doi.org/10.1080/01140671.2019.1588134).
5. Reid, J. B., Hunt, A. G., Johnstone, P. R., Searle, B. P. & Jesson, L. K.
   (2017). On the responses of carrots (*Daucus carota* L.) to nitrogen supply.
   *New Zealand Journal of Crop and Horticultural Science* 46, 298--318.
   [DOI](https://doi.org/10.1080/01140671.2017.1402790).
6. Gomes, V. E. V., Grangeiro, L. C., Ferreira, N. M., Lacerda, R. R. A.,
   Almeida, A. F. A. & Silva, J. L. A. (2021). Effect of the planting season on
   carrot cultivars growth and yield in the Brazilian semiarid region. *Acta
   Scientiarum. Agronomy* 43, e51831.
   [DOI](https://doi.org/10.4025/actasciagron.v43i1.51831).
7. Smolen, S., Lukasiewicz, A., Klimek-Chodacka, M. & Baranski, R. (2020).
   Effect of soil salinity and foliar application of jasmonic acid on mineral
   balance of carrot plants tolerant and sensitive to salt stress. *Agronomy*
   10, 659. [DOI](https://doi.org/10.3390/agronomy10050659).
8. Kovacik, P., Simansky, V., Smolen, S. & Neupauer, J. (2022). The effect of
   vermicompost and earthworms (*Eisenia fetida*) application on phytomass and
   macroelement concentration and tetanic ratio in carrot. *Agronomy* 12,
   2770. [DOI](https://doi.org/10.3390/agronomy12112770).
