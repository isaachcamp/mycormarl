# Plant `kappa_c`: respiration accounting and recommendation

**Research date:** 23 August 2026
**Question:** What should the plant standing-carbon-maintenance parameter
`kappa_c` be in MycorMARL, given the derivation of `amass = 0.05` from carrot
gas exchange? Does the current model double-count respiration?

## Decision

**Retain `kappa_c = 0.007 g C g^-1 whole-plant dry mass d^-1` at the
current `amass = 0.05` semantic.** It is the rounded, full-day 20 °C
whole-plant maintenance estimate from the carrot model of Reid (2019):

\[
\kappa_C = \gamma_C\,[f_s q_{s,\mathrm{maint}} + f_r q_{r,\mathrm{maint}}]
= 0.402\,[0.38(0.021) + 0.62(0.015)]
= 0.00694.
\]

Here `gamma_c = 0.402 g C g^-1 DM` is the model structural-C fraction,
`q_s,maint = 0.021 d^-1` and `q_r,maint = 0.015 d^-1` are Reid's fitted
shoot and storage-root maintenance coefficients at 20 °C, and `0.38`/`0.62`
are the shoot/root fractions used in the prior model conversion. Rounding to
`0.007` is appropriate for the present temperature-independent model.

There is **not an intended arithmetic double count of leaf mitochondrial
maintenance respiration** under the current parameter boundary. The apparent
paradox arises because the input was deliberately converted from measured net
exchange toward a pre-maintenance carbon input before maintenance is charged.
There is, however, an important *empirical approximation*: the fitted dark
intercept was used as a proxy for respiration in the light, and it is not a
direct measurement of either respiration in the light or whole-plant
respiration. This is uncertainty in the split, not a reason to silently halve
`kappa_c`.

## What the two carrot studies actually measured or fitted

### Kyei-Boahen et al. (2003): leaf net gas exchange

Kyei-Boahen et al. measured `P_N`, explicitly described as **leaf net
photosynthetic rate**, on the youngest fully expanded intact leaves of
30-day-old carrot plants. An IRGA leaf chamber measured area-based exchange at
20 °C, 350 ± 10 µmol CO2 mol^-1, 65% RH, and PAR from 100 to
1,000 µmol m^-2 s^-1. The plants themselves had grown at a 16-h photoperiod,
20/10 °C day/night, and roughly 450 µmol photons m^-2 s^-1 PAR.
([Kyei-Boahen et al. 2003](https://ps.ueb.cas.cz/pdfs/phs/2003/02/30.pdf),
pp. 301--303.)

Their rectangular-hyperbola fits had an extrapolated zero-light intercept
called `R_D`, with magnitudes 0.85--2.66 µmol CO2 m^-2 s^-1. This
is a **fitted intercept**: the response curves started at 100 µmol photons
m^-2 s^-1, so it is not a separately measured dark-respiration treatment.
The paper's fitted equation identifies `Y` as `P_N` and `R_D` as its
dark-respiration term. Consequently, the experiment does not identify true
biochemical gross assimilation or respiration in the light.

For a C3 leaf, the gas-exchange accounting identity is

\[
A_n = V_c - \tfrac12 V_o - R_l,
\]

where `A_n` is net CO2 assimilation, `V_c` is carboxylation, `V_o` represents
oxygenation/photorespiratory loss, and `R_l` is mitochondrial respiration in
the light ([Farquhar, von Caemmerer & Berry 1980](https://doi.org/10.1007/BF00386231)).
Thus ordinary IRGA `P_N` already excludes both photorespiratory CO2 release and
respiration in the light.

### Reid (2019): separate standing and construction costs

Reid's carrot growth model separately fitted **maintenance** coefficients of
`0.021 d^-1` for shoots and `0.015 d^-1` for storage roots at 20 °C, with
`Q10 = 2`. It also fitted separate **growth-respiration** coefficients of
`0.26` for shoots and `0.46` for storage roots.
([Reid 2019](https://doi.org/10.1080/01140671.2019.1588134).)

The numerator of `q_maint` is not a separately assayed leaf dark- or
light-respiration flux. It is a **fitted aggregate maintenance loss per unit
organ dry mass per day** in Reid's crop carbon-balance model, which first
calculates gross daily photosynthesis and then deducts shoot maintenance and
senescence to obtain labile growth dry matter. It therefore represents the
model's standing respiratory/metabolic requirement for each organ class,
rather than a direct measurement that can prove `R_l = R_D` for the
Kyei-Boahen leaves. This is why the conversion is a defensible whole-plant
reference but not a precise organ-respiration partition.

The first pair is nevertheless the closest carrot-specific evidence for a
standing-biomass maintenance flux, so it belongs in `kappa_c`. The second pair
is a construction cost incurred when biomass is made. It cannot validly be
folded into `kappa_c`: doing so would make the model pay a growth cost even
when no growth occurs.

## Carbon ledger under the current model

The active implementation fixes

\[
C_{\mathrm{input}} = B\,k_{\mathrm{leaf}}\,a_{\mathrm{mass}}\,\Delta t
\]

and then pays

\[
C_{\mathrm{maintenance}} = B\,\kappa_C\,\Delta t.
\]

`amass = 0.05 g C g^-1 leaf DM d^-1` was not set to raw measured `P_N`.
At the 450-µmol-PAR, 16-h reference day it was derived as

\[
P_N(450) + |R_D| = 7.79\text{--}11.17\ \mathrm{\mu mol\ CO_2\ m^{-2}\ s^{-1}},
\]

then converted with carrot SLA 66--94 cm2 g^-1 leaf DM:

\[
7.79\text{--}11.17\ \mathrm{\mu mol\ m^{-2}\ s^{-1}}
\times 57{,}600\ \mathrm{s}
\times 12\times10^{-6}\ \mathrm{g\ C\ \mu mol^{-1}}
\times 0.0066\text{--}0.0094\ \mathrm{m^2\ g^{-1}}
= 0.036\text{--}0.073\ \mathrm{g\ C\ g^{-1}\ leaf\ DM\ d^{-1}}.
\]

Its midpoint was rounded to `0.05`. The conversion joins the gas-exchange
study to an independent carrot SLA dataset, so it is a stated reference-day
approximation rather than a direct whole-plant carbon balance.

| Flux or cost | Is it represented now? | Correct location | Double-count status |
|---|---|---|---|
| Leaf net IRGA assimilation, `P_N` | Indirectly | Starting observation for `amass` | Not itself an input in the active parameterisation |
| Leaf respiration in light, approximated as `|R_D|` | Added back when constructing `amass`; then removed by `kappa_c` | Input/maintenance boundary | **Charged once** if `R_l ≈ |R_D|` |
| Photorespiratory CO2 loss | Remains inside the apparent-gross `amass` approximation | Not separated by current data | Not separately charged |
| Shoot and root standing maintenance | Yes | `kappa_c` | Intended charge |
| Night respiration | Yes, within full-day `kappa_c` | `kappa_c` | Intended charge; 20 °C all-day assumption is approximate |
| Growth respiration / biosynthetic inefficiency | No | Future growth-yield or explicit construction term | **Missing**, not double-counted |
| Carbon sent to AM fungus | Yes, when the plant trades | Policy-controlled trade | Separate from maintenance |
| Root exudation, rhizosphere respiration, herbivory losses | No explicit carbon term | Future process, if required | Neither included nor calibrated by `kappa_c` |

In compact form, setting the model input to the apparent-light term gives

\[
C_{\mathrm{input}} \approx A_n + R_l
\quad\Longrightarrow\quad
C_{\mathrm{after\ maintenance}} \approx A_n - R_{\mathrm{other}},
\]

after `kappa_c` removes leaf and nonleaf standing maintenance. Adding back
`R_l` and subtracting it once is precisely what changes a net leaf-exchange
measurement into a pre-maintenance input. **A double count would instead occur
if `amass` were raw net `A_n` while the full leaf maintenance component were
also retained in `kappa_c`.**

The three possible accounting choices make the distinction explicit, with
`R_other` denoting standing non-leaf maintenance:

| `amass` semantic | Input to free-C pool | Maintenance debit | Resulting balance | Status |
|---|---|---|---|---|
| **Current apparent-gross proxy** | `A_n + R_l` | `R_l + R_other` | `A_n - R_other` | Retain `kappa_c = 0.007` |
| Raw net light-period `P_N` | `A_n` | `R_l + R_other` | `A_n - R_l - R_other` | **Double-debits leaf light respiration** |
| Net daily leaf carbon gain | `A_n - R_night` | `R_other` only | `A_n - R_night - R_other` | Requires a jointly re-estimated, lower `kappa_c` |

This table is an accounting identity, not a claim that the Kyei-Boahen
intercept exactly equals `R_l`. That empirical equality is the main
uncertainty of the current approximation.

## What would change if `amass` were redefined as net daily carbon gain?

This would be a different, internally consistent model, but it needs an
explicit new calibration rather than a one-number swap.

1. Define `amass` from measured light-period `P_N`, subtract a separately
   measured night respiratory loss, and call the result **net daily leaf carbon
   gain**. The Kyei-Boahen experiment alone cannot provide this: its `R_D` is
   extrapolated and measured at 20 °C, while plants experienced 10 °C nights.
2. Remove from `kappa_c` the leaf respiration already included in net daily
   `amass`; retain only non-leaf maintenance not embedded in that input.
3. Recompute the retained coefficient using actual organ fractions, organ
   temperatures, and a clear definition of the photosynthetic leaf fraction.
   The former `0.00410` calculation in earlier project notes was one such
   *conditional* mapping. It used `k_leaf=0.30`, root fraction `0.62`, an
   8-h 10 °C night, `Q10=2`, and charged leaf maintenance only at night. It is
   not transferable to the current `k_leaf=0.68` / `kfroot=0.18` trait set and
   should not become the default merely because it is smaller.

Therefore, under a future **net daily** `amass` semantic, the correct
recommendation is **re-estimate `kappa_c` jointly**, not retain 0.007 and not
assume 0.0041 is universal. Under the current **apparent-gross reference-day**
semantic, retain 0.007.

## Remaining uncertainty and next measurement

The main uncertainty is not a demonstrated double count; it is that `R_D` is
an extrapolated leaf-dark intercept used as a stand-in for `R_l`, whereas
Reid's fitted coefficients arise from whole-plant crop dynamics. Respiratory
partitioning varies with temperature, organ, age, nitrogen status, and growth
state. A stronger calibration would measure, in the same carrot cultivar and
stage: (1) leaf gas exchange with a method that estimates respiration in the
light, (2) night leaf, root and storage-root respiration at the realised
temperatures, (3) leaf dry mass/SLA, and (4) whole-plant growth to identify
construction efficiency. Until then, report `kappa_c=0.007` as a
20 °C, full-standing-biomass maintenance reference, and treat a future
growth-respiration term as a separate model extension.

## Sources

1. Kyei-Boahen, S., Lada, R., Astatkie, T., Gordon, R. & Caldwell, C. (2003).
   [Photosynthetic response of carrots to varying irradiances](https://ps.ueb.cas.cz/pdfs/phs/2003/02/30.pdf).
   *Photosynthetica* 41, 301--305. DOI:
   [10.1023/B:PHOT.0000011967.74465.cc](https://doi.org/10.1023/B:PHOT.0000011967.74465.cc).
2. Reid, J. B. (2019). [Modelling growth and dry matter partitioning in root
   crops: a case study with carrot (*Daucus carota* L.)](https://doi.org/10.1080/01140671.2019.1588134).
   *New Zealand Journal of Crop and Horticultural Science* 47, 99--124.
3. Farquhar, G. D., von Caemmerer, S. & Berry, J. A. (1980).
   [A biochemical model of photosynthetic CO2 assimilation in leaves of C3
   species](https://doi.org/10.1007/BF00386231). *Planta* 149, 78--90.
4. Acosta-Motos, J. R. et al. (2021). [Morphological and physiological
   characteristics of carrot under salt stress](https://doi.org/10.3390/agronomy11122460).
   *Agronomy* 11, 2460. Used only for the independent SLA conversion.
