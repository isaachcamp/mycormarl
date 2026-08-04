# Evidence review for fungal phosphorus-loss rate (`kappa_p`)

**Research date:** 4 August 2026
**Scope:** A fungal free-phosphorus loss rate in mg P g⁻¹ fungal dry biomass
d⁻¹, prioritising the *Rhizophagus irregularis* lineage, then other
arbuscular mycorrhizal (AM) fungi, then fungi generally. This review does not
change any model defaults.

## Accepted decision

Retain **`kappa_p = 0.003 mg P g⁻¹ fungal dry biomass d⁻¹`** as a small
standing sink. This is an explicit modelling assumption, not a directly
measured literature value. Under the observed 5–7 d fine-hyphal turnover
envelope and the selected `gamma_p = 2 mg P g⁻¹`, it represents approximately
**0.75–1.05% irreversible P loss per turnover**.

In model documentation, interpret `kappa_p` as an **irreversible free-P loss
rate**, not physiological P maintenance. Qualify `0`, `0.003`, `0.015`, and
`0.03 mg P g⁻¹ d⁻¹` as approximate 0%, 1%, 5%, and 10% irreversible-loss
cases under the same turnover envelope.

P incorporated into structural biomass belongs to the structural
stock/construction parameter `gamma_p`; it must not also be charged through
`kappa_p`. P that is merely immobilised or "frozen" in fungal biomass is not an
irreversible free-P loss. Counting it under both parameters would double-count
the same biological requirement. If future work needs variable structural
sequestration beyond the fixed `gamma_p` stock, it requires an explicit
structural-P state rather than the current external-loss ledger.

The plant uses the same conceptual split: plant `gamma_p` represents its fixed
structural P stock/construction cost, while plant `kappa_p` is an irreversible
free-P loss rate. This shared interpretation does not imply that the fungal
5–7 d turnover evidence or approximately 1% non-recycling assumption applies
quantitatively to the plant.

## Conclusion

No primary study located reports an irreversible, biomass-normalised daily P
loss for *R. irregularis*, another AM fungus, or fungi generally that can be
inserted directly as `kappa_p`.

The most defensible provisional formulation is

`kappa_p = gamma_p * r_turnover * f_irreversible`,

where `r_turnover` is the fraction of fungal biomass or P stock turned over per
day and `f_irreversible` is the unknown fraction of that P which is neither
remobilised within the mycelium nor transferred to the plant. With the selected
`gamma_p = 2 mg P g⁻¹`, observed lifetimes of fine absorbing structures give
an upper-envelope, deliberately unrealistic **complete-loss** flux of
approximately **0.29–0.40 mg P g⁻¹ d⁻¹**. It is an upper envelope, not a
recommended point value, because the observations do not measure P loss and
do not imply that all fungal biomass turns over at the fine-branch rate.

For a minimal-loss model, retaining **`kappa_p = 0.003 mg P g⁻¹ d⁻¹`** is a
reasonable *explicit modelling assumption*, but not a literature estimate. It
corresponds to assuming that only about **0.75–1.05%** of the P associated with
the observed 5–7 d fine-hyphal turnover is irreversibly lost. Sensitivity tests
should vary `f_irreversible`, rather than imply false precision in `kappa_p`.
If defaults must be evidence-led rather than assumption-led, a **zero-loss
baseline plus nonzero sensitivity cases is scientifically preferable**, because
the sign of a possible loss is plausible but its magnitude is unconstrained.

## What the parameter currently does

The fungal trait default is `0.003`, while `gamma_p` is a P stock per dry-mass
unit ([`fungus/traits.py`](../../mycormarl/mycormarl/fungus/traits.py#L8-L28)).
Each step computes

`required_p = kappa_p * active_biomass * dt`,

so dimensional closure requires **mg P g⁻¹ fungal dry biomass d⁻¹**. The
amount available is removed from the free fungal P pool. Any shortfall is
converted into biomass loss by dividing by `gamma_p`
([`base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py#L607-L644)).
Paid free P is accumulated as an external maintenance loss, while structural P
associated with resulting biomass mortality is accumulated separately
([`base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py#L436-L454)).

The code has no age-dependent or background hyphal turnover, remobilisation,
necromass, or soil-return process. `death_fraction` is only a termination
threshold after resource-deficit biomass loss; it is not a daily mortality
rate ([`base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py#L322-L345)).

Accordingly, `kappa_p` is best named and interpreted as a **lumped irreversible
free-P loss coefficient**, not physiological P maintenance. Phosphorus itself
is not respired for energy, and internal polyphosphate cycling is not a loss.

## Evidence, separated by dimensional type

| Evidence | Quantity actually measured | What it can support | What it cannot support |
|---|---|---|---|
| *Glomus intraradices* (the *R. irregularis* lineage) branched absorbing structures developed for an average **7 d**, then degenerated into empty septate structures ([Bago et al. 1998](https://doi.org/10.1046/j.1469-8137.1998.00199.x)). | Lifetime of one fine extraradical structure class (days). | A turnover timescale of `1/7 = 0.143 d⁻¹` for that structure class. | Whole-mycelium turnover; P stock; the fraction remobilised; irreversible P loss. |
| In another *G. intraradices* experiment, runner hyphae and spores persisted throughout observation, fine absorbing hyphae lasted **5.3 ± 0.52 d**, and most labelled fungal C remained after 32 d ([Olsson & Johnson 2005](https://doi.org/10.1111/j.1461-0248.2005.00831.x)). | Structure-specific longevity plus fungal C-label retention. | `1/5.3 = 0.189 d⁻¹` for fine absorbing hyphae and strong evidence against applying it to all fungal biomass. | P loss or a whole-fungus P turnover coefficient. |
| Field AM fungal extraradical hyphae had mean radiocarbon ages interpreted as **5–6 d** ([Staddon et al. 2003](https://doi.org/10.1126/science.1084269)). | Turnover of recently assimilated C in collected external hyphae. | A broad AMF C-turnover timescale of `0.167–0.20 d⁻¹`. | Species-specific *R. irregularis* behaviour; P turnover; whether C-label replacement was death, redistribution, or metabolic replacement; irreversible P loss. |
| In *Gigaspora margarita*, estimated polyphosphate-pool turnover was **0.74 d⁻¹** in extraradical and **0.32 d⁻¹** in intraradical hyphae; isolated intraradical hyphae released P as polyphosphate declined ([Solaiman & Saito 2001](https://doi.org/10.1046/j.0028-646x.2001.00182.x)). | Internal polyphosphate-pool turnover and in-vitro phosphate efflux. | Rapid P translocation through the fungus toward the host. | Irreversible environmental loss: this is the symbiotic P-transfer pathway, which the model already represents as fungal-to-plant trade; fragmented isolated hyphae may also leak. |
| Low-P *G. intraradices* spores contained `1.3 ± 0.35 mg P g⁻¹`, high-P spores `8.0 ± 1.6 mg P g⁻¹`, and selected young-hypha regions `2.4–9.6 mg P g⁻¹`; the paper describes about `2 mg P g⁻¹` as ordinary spore concentration ([Olsson et al. 2008](https://doi.org/10.1128/AEM.00376-08)). | P stock concentration (mg P g⁻¹ dry mass). | The chosen provisional `gamma_p = 2 mg P g⁻¹`. | Any per-day loss or maintenance flux without an independent turnover and non-recycling fraction. |

No quantitative primary evidence was found for AMF P leakage, P-containing
exudation, or senescence loss that both identifies the external destination and
closes to mg P g⁻¹ dry biomass d⁻¹. The absence of such a result is why
the derivation below keeps the irreversible fraction explicit.

## Dimensional derivation and uncertainty

For a P stock concentration `gamma_p`, turnover time `tau`, and irreversible
fraction `f_irreversible`:

`kappa_p = gamma_p / tau * f_irreversible`.

Using the selected `gamma_p = 2 mg P g⁻¹` and only the directly observed
fine-structure lifetimes:

- `tau = 7 d`: `kappa_p = 2 / 7 * f = 0.286 f mg P g⁻¹ d⁻¹`;
- `tau = 5.3 d`: `kappa_p = 2 / 5.3 * f = 0.377 f mg P g⁻¹ d⁻¹`;
- including the field 5 d bound: `2 / 5 * f = 0.400 f mg P g⁻¹ d⁻¹`.

Examples are scenarios, not measurements:

| Assumed irreversible fraction `f` | `kappa_p` for 5–7 d turnover (mg P g⁻¹ d⁻¹) |
|---:|---:|
| 0% | 0 |
| 1% | 0.0029–0.0040 |
| 5% | 0.014–0.020 |
| 10% | 0.029–0.040 |
| 100% | 0.29–0.40 |

The current `0.003` therefore encodes `f = kappa_p * tau / gamma_p`, or about
`0.003 * 5 / 2 = 0.75%` to `0.003 * 7 / 2 = 1.05%`. This is a transparent,
minimal irreversible-loss assumption; the literature constrains `tau`, but not
`f`.

## Is one lumped coefficient scientifically sensible?

It is sensible as a deliberately coarse closure if the research question only
needs a small standing P drain and the documentation states that it combines
unresolved leakage, unrecovered material from local senescence, and other
exports. It should not be presented as a measured maintenance requirement.

There are three important risks:

1. **Mechanism mismatch.** The code removes P from the free pool while leaving
   biomass intact when payment succeeds. True hyphal turnover removes old
   structure, may remobilise its cytoplasm, creates necromass, and requires
   regrowth. A turnover model should therefore act on biomass/structure and
   route its structural P, rather than only drain free P.
2. **Double counting.** There is no current double counting merely between
   `gamma_p` and `kappa_p`: `gamma_p` is a stock/construction cost, while
   `kappa_p` is a flux. However, interpreting `kappa_p` as full structural
   turnover and later adding explicit mortality/regrowth would count that
   process twice unless the lumped term were reduced or removed. Fungal-to-
   plant phosphate efflux must also remain excluded because trade is explicit.
3. **Scale mismatch.** Fine absorbing branches turn over rapidly, but runner
   hyphae and spores persist. Applying the fastest observed lifetime to all
   modelled fungal dry biomass overstates turnover and loss.

## Accepted default-parameter decision

- Record `gamma_p = 2 mg P g⁻¹ dry biomass` as an approximate ordinary
  *R. irregularis*-lineage spore P concentration from Olsson et al. (2008), not
  as a maintenance measurement.
- Retain `kappa_p = 0.003 mg P g⁻¹ d⁻¹` as an explicit minimal-loss
  assumption equivalent to approximately 1% non-recycling under the observed
  fine-hyphal turnover envelope; do not describe it as a measured value.
- Keep `kappa_p` conceptually separate and change its documentation to
  "irreversible free-P loss rate".
- Qualify at least `0`, `0.003`, `0.015`, and `0.03 mg P g⁻¹ d⁻¹`,
  corresponding roughly to 0%, 1%, 5%, and 10% irreversible loss under the
  observed fine-branch lifetime envelope.
- Keep P immobilised in structural biomass under `gamma_p`; do not include it
  in `kappa_p`.
- Apply the same stock-versus-loss interpretation to the plant, without
  transferring the fungus-specific turnover estimate to plant `kappa_p`.
- If turnover becomes biologically important to the research question, replace
  the lumped sink with explicit structure-class turnover, remobilisation, and
  soil/necromass destinations.

## Primary sources

1. Bago, B., Azcón-Aguilar, C., Goulet, A. & Piché, Y. (1998). Branched
   absorbing structures (BAS): a feature of the extraradical mycelium of
   symbiotic arbuscular mycorrhizal fungi. *New Phytologist* 139, 375–388.
   [DOI](https://doi.org/10.1046/j.1469-8137.1998.00199.x).
2. Olsson, P. A. & Johnson, N. C. (2005). Tracking carbon from the atmosphere
   to the rhizosphere. *Ecology Letters* 8, 1264–1270.
   [DOI](https://doi.org/10.1111/j.1461-0248.2005.00831.x).
3. Staddon, P. L., Ramsey, C. B., Ostle, N., Ineson, P. & Fitter, A. H. (2003).
   Rapid turnover of hyphae of mycorrhizal fungi determined by AMS
   microanalysis of carbon-14. *Science* 300, 1138–1140.
   [DOI](https://doi.org/10.1126/science.1084269).
4. Solaiman, M. Z. & Saito, M. (2001). Phosphate efflux from intraradical
   hyphae of *Gigaspora margarita* in vitro and its implication for phosphorus
   translocation. *New Phytologist* 151, 525–533.
   [DOI](https://doi.org/10.1046/j.0028-646x.2001.00182.x).
5. Olsson, P. A. et al. (2008). Elemental composition in vesicles of an
   arbuscular mycorrhizal fungus, as revealed by PIXE analysis.
   *Applied and Environmental Microbiology* 74, 4151–4158.
   [DOI](https://doi.org/10.1128/AEM.00376-08).
