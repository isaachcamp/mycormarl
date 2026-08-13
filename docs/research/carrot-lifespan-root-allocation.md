# Cultivated carrot allocation over the crop lifespan

**Research date:** 13 August 2026
**Question:** How does cultivated *Daucus carota* change dry-mass allocation
between shoot and root over its usual first-season crop lifespan, and is there
evidence for allocation between fine/fibrous and storage roots?

## Decision-relevant answer

Cultivated carrot does **not** have a time-invariant root:shoot allocation.
Across independent primary studies, its first-season crop trajectory is
shoot-favoured early, followed by a pronounced increase in the storage-root
share. The clearest directly reported crossover is for cultivar `Forto`: root
dry mass was lower than leaf dry mass through 80 days after sowing (DAS), then
overtook leaf dry mass at 88 DAS. This is a standing-biomass result, not a
marginal allocation flux.

Fine/fibrous-root evidence is much thinner. The available direct measurements
establish that fibrous roots are a material but strongly regime-dependent
component of whole-plant mass: **11.9–24.4%** across late deep-sand greenhouse
treatment means (with pooled-cultivar values of 14.8–17.7%), but about
**1.8–3.3% of total root dry mass** in a mature field experiment. Neither study sampled the fine-root fraction
repeatedly across a single crop lifespan. Therefore the evidence supports a
time-dependent *storage-root versus shoot* representation, but does **not**
identify a defensible age-dependent `k_froot` curve.

Here, `k_froot` means the standing fine/fibrous-root dry-mass fraction of the
whole plant:

```text
k_froot = fibrous-root DM / (shoot DM + storage-root DM + fibrous-root DM)
```

It is not a developmental allocation flux, a fraction of root length, or an
active absorbing-tissue fraction.

## Primary observations by crop age

| Source and regime | Crop age / size | Direct observation | Interpretation and limit |
|---|---|---|---|
| [Benjamin & Wren 1978](https://doi.org/10.1093/jxb/29.2.425), carrot growth analysis and `14CO2` feeding | Within 9 weeks (63 d) after sowing | The developing storage organ had accumulated **40% of plant dry matter produced**. | Storage-root sink importance increases rapidly. The paper's wording is cumulative produced DM, so this is not necessarily the instantaneous standing root fraction. |
| [Peixoto 2011](http://hdl.handle.net/11449/88323), field-grown `Forto`, São Gotardo, Brazil, May–September 2004 | 40 DAS | Leaves: **0.18 g DM plant⁻¹**; root: **0.04 g DM plant⁻¹**. Root/(leaf + root) = **0.182** and root:leaf = **0.222**, calculated from the reported values. | Root DM is initially much smaller. The thesis records "root" rather than separating storage and fibrous roots. |
| Same Peixoto experiment | Through 80 DAS; 88 DAS | Dry-matter partitioning was shoot-favoured through **80 DAS**; root DM then increased strongly and exceeded leaf DM at **88 DAS**. | Strong crossover evidence for this cultivar and planting season. It does not yield a complete numerical time series from the accessible record. |
| [Hole et al. 1983](https://doi.org/10.1093/oxfordjournals.aob.a086456), glasshouse and field studies, cultivars plus wild type | Glasshouse 10–600 mg mean plant DM; field 48–115 DAS and 180 mg–6 g plant DM | After early development, the intercept in the log shoot–storage-root DM relation became progressively more negative with time; the authors fitted `ln(shoot) = alpha - eta*time + gamma*ln(storage root)`. | At comparable plant size, a progressively lower shoot mass relative to storage root is consistent with increasing storage-root share. Cultivar maturity changes the intercept; this is not one universal curve. |
| [Hole & Dearman 1991](https://academic.oup.com/aob/article/68/5/427/165076), controlled 20 °C study of two cultivars | Storage-root initiation, after the storage root became morphologically identifiable | Of net photosynthesis, **64%** was allocated to shoot and **36%** exported to the root system; **19%** was used in root growth and **17%** in root respiration. Storage-root allocation was **4.6%** versus **7.5%** for cultivars with high versus low mature shoot:storage-root ratio. | Direct evidence that the storage/fibrous-root split diverges early and differs by genotype, even where shoot:total-root partition does not explain the difference. These are short initiation-period fluxes, not standing mass fractions. |
| [Reid 2019](https://doi.org/10.1080/01140671.2019.1588134), two commercial New Zealand field crops | Destructive harvests about every two weeks: **24–170 DAS** and **44–203 DAS** | The underlying experiment measured live and senesced shoot mass and storage-root mass over the crop. The resulting model treats the shoot partition factor as time-dependent and explicitly warns that fine roots were omitted. | Confirms multi-month first-season change and late leaf senescence, but its root pool is storage root only; it cannot quantify `k_froot`. |

## Fine/fibrous versus storage-root evidence

| Source and regime | Age / sample scope | Reported or derived value | What it can and cannot support |
|---|---|---:|---|
| [Westerveld 2005, Table 2.18 and Appendix Table A2.16](https://bradford-crops.uoguelph.ca/sites/default/files/Sean%20Westerveld%20Thesis.pdf), `Idaho` and `Fontana`, one plant per 10-cm-diameter, 150-cm-deep PVC column, 98% silica sand, N treatments | Six-month plants | From separately measured shoot, storage-root and fibrous-root DM, `k_froot` is **0.119–0.244** across treatment means (calculated from Table 2.18). Pooled cultivar means are `Idaho`: top 1.59 g, storage 7.25 g, fibrous 1.53 g (**0.148**); `Fontana`: 0.95, 5.49, 1.38 g (**0.177**). | Direct separated-mass evidence with the requested whole-plant denominator. It is a single late harvest in an unusually deep, low-resistance sand regime, not a lifespan trajectory or field default. Use the observed treatment rows, rather than combining independent extrema, if sensitivity is confined to this regime. |
| [Sakamoto & Suzuki 2015, Table 2](https://www.scirp.org/pdf/AS_2015080611205559.pdf), hydroponic carrot under three root-zone temperatures, `n=5` | Final measured plants; age is not extracted here | At 20 °C, shoot/taproot/fibrous-root DM = **1.70 ± 0.24 / 1.56 ± 0.20 / 0.36 ± 0.07 g**, giving `k_froot = 0.099`; corresponding calculated values are **0.130** at 25 °C and **0.138** at 29 °C. | Independent direct separated-mass corroboration that `k_froot` can be about 0.10–0.14. A hydroponic temperature treatment is not a cultivation-lifespan trajectory and should not be pooled mechanically with Westerveld. |
| [Pietola 1995](https://doi.org/10.23986/afsci.72611), mature field-grown `Nantes Duke` in four fine-sand/compaction treatments | Mature plants; total plant DM denominator not reported in the available primary record | Fibrous-root/total-root DM = **0.0177–0.0329**. | Establishes a much smaller mature field fine-root share *within total root mass*, but cannot be converted honestly to `k_froot` without the matched shoot and storage-root DM. |
| [Pietola & Smucker 1998](https://doi.org/10.1023/A:1004294330427), field-grown carrots | Small carrots, storage-root DM < **1.1 g plant⁻¹** | Fine-root DM was reported as up to about **7% of storage-root DM**; **75–90%** of fibrous-root length was in approximately **0.15-mm** roots. | Supports a small fine-root mass relative to storage root in this field experiment, and establishes the root-length system as very fine. It does not provide whole-plant DM or repeated ages. |
| [Benjamin & Wren 1978](https://doi.org/10.1093/jxb/29.2.425), root-pruning experiment | Pruned at 35 DAS | Loss of fibrous roots reduced subsequent leaf growth; increased fibrous-root relative growth soon re-established the normal fibrous-root:shoot ratio. | Direct functional evidence that the fibrous-root system is regulated and important early. No numerical ratio or intact-plant time series is reported in the accessible article record. |

## What follows for `k_froot`

The six-month Westerveld range is the only located observation that directly
matches the requested whole-plant fine-root fraction across an N-treatment
series. It can delimit an **uncertainty interval for a late, deep-sand
greenhouse carrot**, `k_froot = 0.119–0.244`, but not a universal
cultivated-carrot interval. It is inappropriate to treat its upper
and lower bounds as independently varying fractions of total root and whole
plant, or to use it as evidence that fine-root allocation stays constant from
seedling to harvest.

The field result points in the opposite direction for mature plants: fine roots
are only 1.77–3.29% of total root DM under its conditions. Because it lacks a
whole-plant denominator, it should be retained as an external validity warning
rather than transformed into `k_froot` by assuming a storage-root fraction.

## Unsupported gaps and next measurement

- No located primary study repeatedly measures **shoot, storage-root, and
  fibrous-root dry mass in the same cultivated plants** from emergence to
  harvest. This is the measurement needed for an age-dependent `k_froot(t)`.
- The cited root:shoot studies usually define “root” as the storage root and
  discard or omit fibrous roots. Their values cannot be used to infer fine-root
  mass by subtraction.
- No evidence here identifies marginal carbon allocation, root turnover, or
  the fraction of fibrous-root DM that remains physiologically active. Standing
  dry-mass fractions should not be substituted for those quantities.
- Cultivar, soil impedance/compaction, nitrogen supply, plant density,
  temperature, and harvest age visibly affect partitioning. A model should
  label any selected range with its cultivation regime.

The discriminating experiment is a repeated destructive harvest of the same
cultivar and soil regime (for example every 10–14 d), recording dry mass of
leaves/petioles, storage root, and washed fine/fibrous roots separately. Report
the joint rows and uncertainty, then calculate `k_froot(t)` directly rather
than mixing denominators across experiments.

## Source notes

All sources above are original experiments or the original thesis/dissertation
records; no review is used as evidence for the numerical claims. The 2019 Reid
paper is included because it reports and models its own two field experiments,
but it explicitly omits fine-root mass. Values labelled “calculated” are simple
ratios of source-reported masses and show the calculation in the table.
