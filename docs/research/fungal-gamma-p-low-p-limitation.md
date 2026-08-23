# Fungal `gamma_p` under low phosphorus: evidence and interpretation

**Research date:** 23 August 2026
**Question:** Does primary literature support lowering fungal `gamma_p` from `2 mg P g^-1` dry biomass, and is it credible that low-P runs change nonlinearly because the fungus becomes phosphate limited?

## Short answer

Yes, with a clear boundary. The strongest directly relevant experiment used the *Glomus intraradices* lineage now assigned to *Rhizophagus irregularis* in carrot root-organ culture. At `25 µM` external P, spores contained `1.30 ± 0.35 mg P g^-1` dry mass—below the model's fixed `2 mg P g^-1` default. This directly supports a low-P reference or sensitivity value near **`1.3 mg P g^-1`**.

It does **not** establish `0.5`, `0.75`, or `1.0 mg P g^-1` as whole-fungus structural constants: the measurement is for spores, which also store polyphosphate, and fungal P concentration was strongly P-plastic. Values below `1.3` remain useful sensitivity probes for locating the model's regime boundary, but not literature-calibrated stoichiometries.

Independent primary experiments and a field fertilisation test support the proposed mechanism: AM fungi can be directly P-limited at very low P. They support a resource-limitation regime transition, but not the exact numerical threshold or curve produced by this model.

## Model meaning

The model uses:

$$
\Delta G=\min(C_g/\gamma_C, P_g/\gamma_P).
$$

Thus fungal `gamma_p` is **mg structural P per g dry fungal biomass**, not P uptake rate, P-transfer fraction, or the irreversible P-loss coefficient `kappa_p`. Lowering it makes a fixed P allocation build more fungal biomass and delays growth-P limitation. In the paired low-P screen, `kappa_p = 0`; its limitation trace therefore concerns P available for realised fungal growth, not maintenance-P loss.

## Direct basis for lowering `2.0`

Olsson, Hammer, Wallander and Pallon measured elemental concentrations of freeze-dried spores and hyphae in monoxenic *G. intraradices* cultures using PIXE/STIM, with concentrations calculated per dry weight. P treatments were `25 µM` (low P) and `2.5 mM` (high P).

| Structure and condition | Measured P concentration | Interpretation |
|---|---:|---|
| Spores, low P | `1.30 ± 0.35 mg P g^-1` | Direct support for lowering `2.0` toward `~1.3` in a low-P reference/sensitivity run. |
| Spores, high P | `8.0 ± 1.6 mg P g^-1` | Shows total P concentration is environmentally plastic and includes storage. |
| Four selected young-hypha regions, low P | `2.4–9.6 mg P g^-1` | Shows strong spatial heterogeneity; not a whole-mycelium structural mean. |

The low-to-high spore P increase was significant (`P = 0.02`), and the authors identify spore P with polyphosphate storage. This refutes treating `2 mg g^-1` as invariant. It does not imply that all active external hyphae require `1.3 mg g^-1`, because spores are survival/storage organs and the hyphal observations were selected micro-regions.

**Calibrated recommendation:** use `1.3 mg P g^-1` as the evidence-led default for this low-P model. It is a low-P spore proxy, not a whole-extraradical-mycelium calibration. Describe `1.0`, `0.75`, and `0.5 mg P g^-1` as exploratory sensitivity values only.

The study does not support saying that spores are usually more nutritionally or P-dense than extraradical mycelium. Although spores are storage/survival structures and their P concentration is P-plastic, its selected young-hypha regions contained `2.4–9.6 mg P g^-1`, above the low-P spore mean. The samples are not a matched whole-structure comparison, so no general ordering is justified.

Primary source: [Olsson et al. (2008), *Applied and Environmental Microbiology*, DOI: 10.1128/AEM.00376-08](https://doi.org/10.1128/AEM.00376-08).

## Evidence for fungal P limitation under low P

### Controlled lineage experiment

In another carrot root-organ culture experiment with *G. intraradices*, the fungal compartment received either `25 µM` or `2.5 mM` P. In the low-P treatment, solution P fell to `0.54 µM` by day 33 in one experiment and `1.1 µM` by day 21 in the other. Near the 50-day harvests, low-P mycelial dry mass was `3.5` and `2.1 mg`, versus `5.6` and `7.7 mg` under high P. The fungal high-affinity phosphate-transporter transcript was higher under low P.

This is controlled, same-lineage evidence that depletion of a small external P pool coincides with reduced fungal biomass and a changed fungal P-acquisition state. It supports fungal phosphate limitation as a credible low-P mechanism, although it does not estimate the model's `gamma_p` transition.

Primary source: [Olsson, Hammer & Pallon (2006), *Applied and Environmental Microbiology*, DOI: 10.1128/AEM.02154-05](https://doi.org/10.1128/AEM.02154-05).

### Field fertilisation test

Treseder and Allen measured extraradical AM-hyphal length and root colonisation across a Hawaiian soil-fertility gradient. AM biomass increased after P addition at the P-limited site, but P fertilisation reduced it at a fertile site. Their field manipulation supports interacting controls: at the lowest nutrient availability fungi as well as plants can be nutrient limited; at higher fertility reduced host carbon allocation can instead limit fungi.

This reversal supports a limitation-regime interpretation and cautions against claiming that more P universally increases fungal biomass.

Primary source: [Treseder & Allen (2002), *New Phytologist*, DOI: 10.1046/j.1469-8137.2002.00470.x](https://doi.org/10.1046/j.1469-8137.2002.00470.x).

## How to interpret the paired low-P nonlinearity

Within a matched five-run block, only fungal `gamma_p` changes; common random numbers, initial P, carbon losses, initial fungal biomass, and both trade fractions are held fixed. The screen identifies the fungal state from realised-growth timesteps labelled phosphate limited.

The internally supported causal account is:

1. Raising `gamma_p` increases the P needed per unit fungal growth.
2. With a finite low-P pool, fungal uptake and incoming P eventually fail to match that requirement in some matched conditions.
3. Crossing the `min(C_g/gamma_C, P_g/gamma_P)` boundary changes realised fungal growth; coupled fungal P export and plant outcomes can therefore shift sharply rather than proportionally.

The literature supports steps 1–2 qualitatively: AM-fungal P stocks are P-plastic, external P can be rapidly depleted, and AM biomass can be directly P limited. Step 3's exact nonlinear curve and transition point are **model-specific results**, not empirically calibrated dose-response relationships from these papers.

## Biological interpretation: fungal P requirement is not all fungal P

Fungal P supports structural biomass and growth machinery, including nucleic
acids/ribosomes, phospholipids, ATP, and other phosphorylated metabolites. AM
fungi also use a substantial **mobile** P pool: extraradical hyphae take up
orthophosphate, convert it to polyphosphate (polyP), translocate it to
intraradical hyphae/arbuscules, then hydrolyse and export orthophosphate to the
plant. PolyP is consequently a transport and storage pool, rather than P
irreversibly immobilised in new fungal biomass ([Tani et al. 2000,
*Applied and Environmental Microbiology*](https://pmc.ncbi.nlm.nih.gov/articles/PMC91766/)).

The model's fixed `gamma_p` should therefore be interpreted narrowly as the P
that is effectively immobilised per unit new fungal biomass. It should not be
read as the total P physically present in an active mycelium, nor as the P
available for fungal-to-plant transfer. With only a free-P pool and a frozen
`gamma_p` pool, lowering `gamma_p` is a coarse proxy for lower fungal P
retention; it is not a direct calibration of the size or kinetics of the
polyP/metabolic pool. This strengthens the motivation for an evidence-led
low-P default of `1.3 mg P g^-1`, but does not directly calibrate the
exploratory `0.5–0.75 mg P g^-1` cases.

### Why young hyphae may have higher measured P

Higher P concentrations in young hyphae are biologically plausible because
actively growing tips need high biosynthetic and energy turnover, and because
foraging hyphae can encounter P-rich microsites before locally depleting them.
The Olsson et al. (2008) measurements do not establish either mechanism:
their young-hypha samples are not a matched whole-extraradical-mycelium
comparison and do not provide the hyphae's local soil-P history. They are thus
useful reasons not to identify spore total P with fungal structural P, but not
evidence for a universal spore-versus-extraradical-mycelium P ordering.

### Implication for the high-trade screen cells

The displayed `fungus→plant trade fraction` is the action-controlled fraction
of the fungal **free-P pool** offered for trade in each step. It is not the
fraction of fungal soil-P uptake ultimately transferred to the plant. At lower
`gamma_p`, less P is immobilised by growth, so high trade actions can remain
feasible; at higher `gamma_p`, growth competes more strongly with trade and
the fungal P-limitation boundary appears. This is the appropriate internal
interpretation of the interaction surface.

Recent direct work is consistent with treating P export as an important,
context-dependent flux: Bisot et al. found plant-to-fungus C transfer and
fungus-to-plant P transfer were proportional on average, with the ratio
strongly affected by plant host genotype ([Bisot et al. 2026,
*PNAS*](https://www.pnas.org/doi/10.1073/pnas.2512182123)). It does not by
itself establish a universal claim that most fungal P uptake is transferred.
The appropriate follow-up diagnostic is
`cumulative fungal→plant P transfer / cumulative fungal soil-P uptake`,
stratified by `gamma_p` and trade action.

## Reporting wording

> The fungal `gamma_p = 1.3 mg P g^-1` default is motivated by direct evidence that the *R. irregularis* lineage reaches about `1.3 mg P g^-1` dry spores under low external P, and by experiments demonstrating direct AM-fungal P limitation. It is a low-P spore proxy for a fixed fungal P requirement, not a measurement of whole extraradical mycelium. The nonlinear transition is an internally diagnosed model result that is mechanistically supported, but not quantitatively calibrated, by the cited studies.

## Scope limits

- The culture experiments used `25 µM` low P; the screen starts at `0.1–0.3 µM` solution P in a finite soil model. They establish direction and mechanism, not numeric calibration of that screen.
- The articles retain the historical name *G. intraradices*; this repository already uses the lineage as the *R. irregularis* reference.
- A state-dependent fungal P-storage/stoichiometry model would need separate conservation accounting. It should not be inferred solely from this fixed-`gamma_p` sensitivity experiment.
