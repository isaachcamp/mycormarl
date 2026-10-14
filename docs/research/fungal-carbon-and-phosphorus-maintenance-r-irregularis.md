# Evidence review for *Rhizophagus irregularis* carbon and phosphorus loss

**Research date:** 19 August 2026
**Question:** Improve the MycorMARL fungal `kappa_c` (standing carbon loss) and
`kappa_p` (irreversible free-phosphorus loss) estimates for *Rhizophagus
irregularis*, without silently treating growth, trade, or structural stocks as
maintenance.

Older primary studies call this lineage *Glomus intraradices* (and sometimes
*G. irregulare*). Those lineage measurements are used below only where the
measured quantity and fungal structure match the model. The model represents
living external mycelium, not spores or all intraradical structures.

## Recommendation

| Parameter | Earlier default | Evidence-led recommendation | Confidence and decision |
|---|---:|---|---|
| `kappa_c` | `0.030 g C g^-1 fungal DM d^-1` | Use **`0.015 g C g^-1 fungal DM d^-1`** as the evidence-led reference, with a qualification/sensitivity envelope of **`0.010–0.025`**. Retain `0.030` only as an intentionally high sensitivity point. | A direct AM-fungal external-mycelium respiration experiment, normalized below, gives this range for *Funneliformis mosseae*, not *R. irregularis*. There is no direct biomass-normalised *R. irregularis* maintenance-respiration measurement. The earlier value was high relative to the closest direct evidence, so the implementation-default change is scientifically supported. |
| `kappa_p` | `0.003 mg P g^-1 fungal DM d^-1` | Use **`0.001`**, matching the plant value, as a shared small irreversible-free-P-loss model assumption. Use `0`, `0.001`, `0.003`, `0.015`, and `0.030 mg P g^-1 DM d^-1` as zero, about-0.3%, about-1%, 5%, and 10% non-recycling scenarios under the deliberately conservative turnover mapping below. | No primary source reports a biomass-normalised daily irreversible-P loss for *R. irregularis* or another AM fungus. Equality with the plant is a **model-policy choice**, not evidence of identical fungal and plant P loss. |

`kappa_c` is a flux of C ultimately emitted as CO2 or otherwise catabolised,
whereas P is not respired. The latter must therefore remain an *irreversible
free-P export/loss* abstraction, not a biochemical maintenance analogue.

## What the coefficients mean in this model

For active fungal dry biomass `B` over timestep `dt`, the environment charges

`C_loss = kappa_c * B * dt`

and

`P_loss = kappa_p * B * dt`.

The model pays these from free pools before allocation, and converts an unpaid
demand to biomass loss using `gamma_c` or `gamma_p`. Consequently:

- `kappa_c` has units `g C g^-1 fungal DM d^-1` and must approximate an
  unavoidable flux per standing living external mycelium;
- `kappa_p` has units `mg P g^-1 fungal DM d^-1` and must exclude structural P
  embodied in biomass (`gamma_p`) and fungal-to-plant P trade; and
- a respiration measurement during rapid growth is not pure maintenance. It
  can still bound the current lumped *standing carbon-loss* term, but should
  not be claimed to isolate basal maintenance.

The environment calculates `required_p = kappa_p * active_biomass * dt`, then
removes the amount available from the free fungal-P pool. It converts any
shortfall to biomass loss using `gamma_p`; paid P is recorded as external loss,
while P released with mortality is accounted separately
([`base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py#L607-L644),
[`base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py#L436-L454)).
The code has no age-dependent hyphal turnover, remobilisation, necromass, or
soil-return process, and `death_fraction` is a resource-deficit termination
threshold, not a daily mortality rate
([`base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py#L322-L345)).
Thus `kappa_p` is a **lumped irreversible free-P loss coefficient**:
phosphorus is not respired, and internal polyphosphate cycling or
fungal-to-plant transfer is not loss.

## Carbon: direct AM external-mycelium respiration gives a usable transferred scale

Heinemeyer, Ineson, Ostle and Fitter directly separated a root-excluding hyphal
compartment and measured respiration attributable to the external mycelium of
the AM fungus *Glomus mosseae* (now *Funneliformis mosseae*). They reported
external-mycelium (ERM) length densities, compartment mass, a hyphal dry-mass
conversion, and respiration. This is the closest primary measurement found
that closes from AM-hyphal CO2 flux to fungal dry mass. It is a congen­eric
Glomeraceae transfer, **not** a direct *R. irregularis* measurement.

Their stated conversion is `3.6 ug fungal DM m^-1 ERM`; their reported
length-normalised respiration values are `1.5–3.8 ng C m^-1 h^-1` (ambient
`2.4`, initially warmed `3.8`, and warmed after acclimation `1.5`). The direct
dimensional conversion is `kappa_C,resp = r_length * 24 h d^-1 / (3,600 ng DM
m^-1)`, where `3,600 ng DM m^-1 = 3.6 ug DM m^-1`.

| Source condition | ERM respiration (ng C m^-1 h^-1) | Calculation | `kappa_C,resp` (g C g^-1 DM d^-1) |
|---|---:|---:|---:|
| Ambient | 2.4 | `2.4 * 24 / 3600` | 0.0160 |
| Warmed before acclimation | 3.8 | `3.8 * 24 / 3600` | 0.0253 |
| Warmed after 2-week acclimation | 1.5 | `1.5 * 24 / 3600` | 0.0100 |

The arithmetic gives a compact external-mycelium respiration range of
`0.010–0.025 g C g^-1 DM d^-1`; the ambient measurement, `0.016`, rounds to
the recommended `0.015`. Importantly,
the authors found a strong dependence on recently supplied photosynthate and
temperature acclimation. Their flux therefore combines basal and
activity/growth-associated respiration, and may overcharge a carbon-starved or
quiescent fungal state. Conversely, excluding it entirely omits a directly
observed standing external-mycelium carbon demand.

Two *G. intraradices* / *R. irregularis*-lineage results strengthen the
interpretation but do **not** supply a second coefficient:

- Bücking et al. measured CO2 release from labelled acetate and glucose in
  germinating spores. About 46% of taken-up glucose and roughly half of
  acetate were respired under controls, with root exudates increasing
  respiration. This establishes active and condition-dependent fungal
  respiration, but the published result is a fraction of supplied substrate,
  not a standing-DM daily rate.
- Olsson and Johnson observed that fine absorbing structures lasted `5.3 +/−
  0.52 d`, whereas runner hyphae/spores persisted and most labelled AM-fungal
  C remained after 32 d. A 5-d turnover rate therefore cannot be applied to
  all represented fungal DM as an additional carbon sink.
- Besserer et al. measured a 35% increase in O2 consumption of germinating
  *G. intraradices* spores within 5 h of strigolactone treatment—too soon for
  added hyphal growth. This is direct lineage evidence that standing fungal
  material respires and responds to host signals, but the polarographic data
  lack a dry-mass-calibrated absolute rate.
- Andrino et al. used root-excluding fungal compartments in a tomato ×
  *R. irregularis* mesocosm and isotopic CO2 measurements to identify rapid
  plant-derived C respiration. Its fungal PLFA/NLFA measurements are biomass
  proxies rather than a conversion to fungal DM, so it corroborates the exact
  species and compartment but cannot supply `kappa_c`.

### Carbon uncertainty and use

The `0.010–0.025` envelope is conditional on: active ERM; the paper's
`3.6 ug m^-1` dry-mass conversion; plant carbon supply;
and a 14–25 C glasshouse regime. It excludes intraradical fungal biomass and
spores, which the current model does not purport to represent. It also should
not be added to an explicit growth-respiration or hyphal-turnover module later
without re-partitioning: some of the measured flux will overlap such modules.

For the present lumped model, **`0.015` is the recommended evidence-led
reference** and `0.010`, `0.015`, and `0.025` are the appropriate primary
sensitivity points. `0.030` is a deliberately high sensitivity point, above
the closest direct AM-ERM observations. If `kappa_c` is to mean strict basal
maintenance rather than the current broader standing carbon-loss term, `0.010`
is more defensible: all available direct fluxes retain activity/growth
respiration. No value should be described as a direct *R. irregularis*
maintenance measurement.

## Phosphorus: no direct irreversible-loss estimate exists

No primary study located measures both (1) P leaving *R. irregularis* or
another AM fungus to an irreversible destination and (2) standing fungal dry
mass, as required to report `mg P g^-1 DM d^-1`. The evidence instead
constrains structure-specific turnover, P stock, or transfer.

The relevant evidence has different dimensions:

| Primary observation | What it supports | Why it cannot set `kappa_p` |
|---|---|---|
| *G. intraradices* branched absorbing structures developed for an average 7 d then degenerated (Bago et al. 1998); fine absorbing hyphae lasted `5.3 +/− 0.52 d`, while runner hyphae and spores persisted and most labelled C remained after 32 d (Olsson & Johnson 2005). | A `5–7 d` **fine-structure** turnover timescale and evidence against applying it to all fungal biomass. | Whole-mycelium turnover; P mass balance; remobilisation fraction; irreversible P loss. |
| Field AM external hyphae had mean radiocarbon ages interpreted as `5–6 d` (Staddon et al. 2003). | A broad AMF C-turnover timescale of `0.167–0.20 d^-1`. | Species-specific *R. irregularis* behaviour, P turnover, or whether label replacement is death, redistribution, or metabolic replacement. |
| *G. intraradices* lineage spores had `1.3 +/− 0.35 mg P g^-1` under low P and `8.0 +/− 1.6 mg P g^-1` under high P; `1.3 mg P g^-1` is the low-P spore value used by this model as `gamma_p` (Olsson et al. 2008). | A provisional structural P stock scale. | A stock concentration is not a daily loss flux and differs strongly by P supply and structure. |
| Isolated intraradical *Gigaspora margarita* hyphae released phosphate as poly-P declined; glucose-enhanced efflux was about `3.5 ug P g^-1 fresh hyphae h^-1` (Solaiman & Saito 2001). | Mechanistic evidence for fungal-to-host P transfer / internal poly-P hydrolysis. | Different species, fresh-mass denominator, fragmented isolated tissue, and—most decisively—the destination is P transfer, which MycorMARL already represents as trade rather than loss. |

With the model's provisional `gamma_p = 1.3 mg P g^-1 DM`, it is useful to make
the unmeasured non-recycling fraction explicit:

`kappa_p = (gamma_p / tau) * f_irreversible`.

If, unrealistically, every unit of represented fungal biomass had the fine
structure lifetime `tau = 5–7 d`, complete non-recycled loss would be
`1.3/7–1.3/5 = 0.186–0.260 mg P g^-1 DM d^-1`. This is an **upper-envelope
scenario**, not a measurement or a recommended rate. The shared default
`0.001` instead implies

`f_irreversible = kappa_p * tau / gamma_p = 0.38–0.54%`.

| Assumed irreversible fraction of the fine-structure P stock | Derived `kappa_p` (mg P g^-1 DM d^-1) |
|---:|---:|
| 0% | 0 |
| 1% | 0.0019–0.0026 |
| 5% | 0.0093–0.013 |
| 10% | 0.019–0.026 |
| 100% | 0.186–0.260 |

This calculation makes the shared `0.001` interpretable as a deliberately
small non-recycling assumption, rather than an apparent measurement. It does
not provide evidence to replace it with a more precise number or to infer that
fungal and plant loss fractions are biologically identical.

### Scope and risks of the lumped P coefficient

The coefficient is a useful coarse closure only when the experiment requires a
small standing P drain and its mechanism is stated honestly. It should not be
presented as measured P maintenance.

1. **Mechanism mismatch.** The model drains free P while retaining biomass
   when the demand is paid. Actual hyphal turnover removes structure, can
   remobilise cytoplasm, creates necromass, and requires regrowth.
2. **Double counting.** Structural P belongs to `gamma_p`. A future explicit
   turnover/mortality-and-regrowth module must reduce or remove this lumped
   loss. Fungal-to-plant phosphate efflux is already represented as trade.
3. **Scale mismatch.** Fine absorbing branches turn over quickly, whereas
   runner hyphae and spores persist; applying the fine-branch lifetime to all
   represented external-mycelium dry mass would overstate loss.

If turnover becomes a material research mechanism, replace `kappa_p` with
explicit structure-class turnover, remobilisation, and necromass/soil-return
destinations.

## Parameter-register disposition

The fungal default now uses `kappa_p = 0` as the conservation-preserving
baseline; positive values remain explicit sensitivity cases. Its
`kappa_c = 0.015` is the transferred AM-ERM respiration reference. Neither
parameter should be presented as a direct *R. irregularis* maintenance
measurement.

## Primary sources

1. Heinemeyer, A., Ineson, P., Ostle, N. & Fitter, A. H. (2006). Respiration of the
   external mycelium in the arbuscular mycorrhizal symbiosis shows strong
   dependence on recent photosynthates and acclimation to temperature. *New
   Phytologist* 171, 159–170. [DOI](https://doi.org/10.1111/j.1469-8137.2006.01730.x).
   Direct ERM respiration and the length-to-DM assumptions used in the carbon
   calculation.
2. Bücking, H. et al. (2008). Root exudates stimulate the uptake and metabolism
   of organic carbon in germinating spores of *Glomus intraradices*. *New
   Phytologist* 180, 684–695. [DOI](https://doi.org/10.1111/j.1469-8137.2008.02590.x).
3. Olsson, P. A. & Johnson, N. C. (2005). Tracking carbon from the atmosphere
   to the rhizosphere. *Ecology Letters* 8, 1264–1270.
   [DOI](https://doi.org/10.1111/j.1461-0248.2005.00831.x).
4. Bago, B., Azcón-Aguilar, C., Goulet, A. & Piché, Y. (1998). Branched
   absorbing structures: a feature of the extraradical mycelium of symbiotic
   arbuscular mycorrhizal fungi. *New Phytologist* 139, 375–388.
   [DOI](https://doi.org/10.1046/j.1469-8137.1998.00199.x).
5. Olsson, P. A. et al. (2008). Elemental composition in vesicles of an
   arbuscular mycorrhizal fungus, as revealed by PIXE analysis. *Applied and
   Environmental Microbiology* 74, 4151–4158.
   [DOI](https://doi.org/10.1128/AEM.00376-08).
6. Solaiman, M. Z. & Saito, M. (2001). Phosphate efflux from intraradical
   hyphae of *Gigaspora margarita* in vitro and its implication for phosphorus
   translocation. *New Phytologist* 151, 525–533.
   [DOI](https://doi.org/10.1046/j.0028-646x.2001.00182.x).
7. Besserer, A. et al. (2006). Strigolactones stimulate arbuscular mycorrhizal
   fungi by activating mitochondria. *PLOS Biology* 4, e226.
   [DOI](https://doi.org/10.1371/journal.pbio.0040226).
8. Andrino, A., Guggenberger, G., Sauheitl, L., Burkart, S. & Boy, J. (2021).
   Carbon investment into mobilization of mineral and organic phosphorus by
   arbuscular mycorrhiza. *Biology and Fertility of Soils* 57, 47–64.
   [DOI](https://doi.org/10.1007/s00374-020-01505-5).
9. Staddon, P. L., Ramsey, C. B., Ostle, N., Ineson, P. & Fitter, A. H. (2003).
   Rapid turnover of hyphae of mycorrhizal fungi determined by AMS
   microanalysis of carbon-14. *Science* 300, 1138–1140.
   [DOI](https://doi.org/10.1126/science.1084269).
