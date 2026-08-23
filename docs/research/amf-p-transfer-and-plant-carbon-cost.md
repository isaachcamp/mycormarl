# AMF phosphorus transfer and plant carbon cost

## Question

What fraction of phosphorus acquired by an arbuscular mycorrhizal fungus (AMF) is typically transferred to the host plant, and is a plant-to-fungus carbon allocation of 5% reasonable as a static model parameter?

## Short answer

There is no single universal transfer fraction. Primary measurements support **a majority transfer under effective symbiosis**, with a useful modelling range of roughly **0.6–0.9 of fungal-acquired P**, and a provisional central value of **0.75–0.8**. Values near 0.9 are defensible for a highly functional, low-P association, but should not be treated as universal.

An allocation of **5% of plant carbon fixation/photosynthate to the fungus is reasonable**, particularly as a conservative central value for a well-nourished or moderately colonised plant. Direct whole-plant estimates include approximately **3–8% of daily gross photosynthesis** in perennial ryegrass, while older syntheses and measurements span approximately **5–20%**. A practical sensitivity range is **0.03–0.10**, with **0.15–0.20** reserved for high fungal demand or strong colonisation.

These are not mirror-image parameters: fungal P transfer is a fraction of P acquired by the fungus, whereas plant C allocation is a fraction of plant carbon fixation or assimilated carbon. Both denominators must be specified explicitly in the model.

## Evidence for P transfer

* **Rufyikiri, Declerck & Thiry (2004)** used a two-compartment root-organ culture with `33P`. They reported 16% of initial isotope supply taken up by extraradical AM mycelium and 72% of initial supply detected in roots after two weeks. The experiment demonstrates strong translocation, but the initial-supply denominator is not directly equivalent to a fraction of fungal uptake. [Mycorrhiza, DOI: 10.1007/s00572-003-0258-1](https://doi.org/10.1007/s00572-003-0258-1)

* **Bücking & Shachar-Hill (2005)** quantified `33P` uptake and transfer in *Glomus intraradices* with `14C` carbohydrate tracing. The fungus took up approximately 41% of supplied `33P` in controls and approximately 63–65% with sucrose; higher carbon availability stimulated P uptake and transfer to roots. Transfer efficiency is therefore conditional on carbon supply. [New Phytologist 165:899–912, DOI: 10.1111/j.1469-8137.2004.01274.x](https://doi.org/10.1111/j.1469-8137.2004.01274.x)

* **Phospho-imaging work (2021)** found that virtually all labelled phosphate was absorbed by AM hyphae, but transfer to roots was incomplete and bidirectional flux occurred. This cautions against assuming that all fungal uptake is immediately exported. [Mycorrhiza, DOI: 10.1007/s00572-021-01028-5](https://doi.org/10.1007/s00572-021-01028-5)

* **A recent PNAS quantitative growth experiment (2026)** inferred that at least approximately 85% of absorbed P had been transferred to the host root after 100 h, using measured fungal P concentrations to bound P retained in fungal biomass. This is a useful upper-end benchmark from one controlled system, not a universal value. [PNAS 123:e2512182123, DOI: 10.1073/pnas.2512182123](https://doi.org/10.1073/pnas.2512182123)

### Do not confuse transfer efficiency with the AM contribution to plant uptake

Studies asking what fraction of *plant total P uptake* came through the AM pathway measure a different quantity. A `33P` field study in wheat estimated 6.5–21% of shoot P uptake through indigenous AM fungi, while a low-P crop study reported AM-pathway contributions up to 81.8% in maize and 75.8% in wheat. These show that AMF can dominate plant P supply under some conditions, but do not measure the fraction of fungal uptake exported. [Wheat field study, DOI: 10.1016/j.apsoil.2015.07.002](https://doi.org/10.1016/j.apsoil.2015.07.002); [32P crop study, DOI: 10.1016/j.apsoil.2022.104624](https://doi.org/10.1016/j.apsoil.2022.104624)

### Recommended static P-transfer values

For a parameter representing **fungal P acquired and paid to the plant**:

| Use | Suggested value |
|---|---:|
| Conservative exploratory range | 0.60–0.90 |
| Central/default trial | 0.75 or 0.80 |
| High-efficiency scenario | 0.90–0.95 |
| Stress/storage scenario | 0.40–0.60 |

## Evidence for plant C allocation to AMF

* **Grimoldi et al. (2006)** compared mycorrhizal and nonmycorrhizal perennial ryegrass using steady-state `13CO2` labelling and gas exchange. AMF increased below-ground respiratory demand by about 3% of daily gross photosynthesis, and total C flow into AMF growth plus respiration was estimated at **less than 8% of daily gross photosynthesis**. This is the strongest direct basis for 5% as a central value. [New Phytologist 172:544–553, DOI: 10.1111/j.1469-8137.2006.01853.x](https://doi.org/10.1111/j.1469-8137.2006.01853.x)

* **Pearson & Jakobsen (1993)** and related isotope studies are commonly cited for a broader **5–20%** photosynthate cost to AM fungi. Their work also shows that fungal identity can substantially alter the balance between root and hyphal P uptake. [New Phytologist 124:489–494, DOI: 10.1111/j.1469-8137.1993.tb03840.x](https://doi.org/10.1111/j.1469-8137.1993.tb03840.x)

* **Nottingham et al. (2010)** summarised field AMF respiration work and reported the commonly used 5–20% range, while noting host and environmental dependence. This is secondary context rather than a new estimate. [New Phytologist, DOI: 10.1111/j.1469-8137.2010.03226.x](https://doi.org/10.1111/j.1469-8137.2010.03226.x)

The relevant quantity is definition-sensitive: a short `14C` pulse into extraradical hyphae, C incorporated into fungal biomass, fungal respiration, and total host C cost are different endpoints.

### Recommended static C-trade values

For a parameter representing **plant C paid to the fungus as a fraction of plant carbon fixation/available assimilate**:

| Use | Suggested value |
|---|---:|
| Conservative exploratory range | 0.03–0.10 |
| Central/default trial | 0.05 |
| High-demand sensitivity | 0.10–0.20 |

Thus 5% is biologically reasonable and supported by direct isotope/gas-exchange estimates. It is not necessarily a ceiling.

## Implications for the current model

1. If fungal trade means **fraction of fungal P uptake transferred to the plant**, test `0.6`, `0.75`, `0.8`, and `0.9`; retain `0.5` as an inefficient comparison.
2. If plant trade means **fraction of available plant C sent to AMF**, `0.05` is a defensible default. Test `0.03`, `0.05`, `0.08`, and `0.12` for the main sensitivity analysis and extend to `0.2` as a high-cost scenario.
3. Do not interpret high AM-pathway contribution to plant P uptake as proof that the same percentage of fungal-acquired P is transferred. The denominators differ.
4. Keep transfer fractions separate from fungal retention, maintenance, and growth. A fungus can acquire most P, retain some temporarily, and still transfer most net available P over a longer interval.

## Bottom line

The literature supports the qualitative claim that AMF commonly transfer a majority of acquired P to the host, but not one universal percentage. A static fungal-to-plant P fraction of **0.75–0.8** is a reasonable starting point, with **0.6–0.9** as a defensible exploratory range. A plant-to-fungus C fraction of **0.05** is also reasonable as a moderate-demand baseline; use **0.03–0.10** for the main sensitivity analysis and extend to **0.2** only as a high-cost scenario.
