# Phosphorus content of fungal cells

**Research date:** 15 September 2026  
**Question:** How much phosphorus is contained within the structure of a fungal cell, prioritising arbuscular mycorrhizal fungi (AMF), then aseptate fungi, septate filamentous fungi, unicellular fungi, and finally general eukaryotic cells? What phosphorus-containing biomolecules contribute to the total?

## Short answer

There is no single universal fungal-cell phosphorus content. It changes strongly with phosphorus supply, growth rate, cell type, and storage-polyphosphate accumulation.

For a living fungal biomass model, a defensible working range is **0.4–3% of dry biomass as elemental P**, with a central value near **1% P dry mass**. The broadest fungal compilation found **0.040–5.5% P of dry mass**, with median molar C:N:P of **250:16:1**. If fungal dry biomass is approximately 50% C, that median converts to about **0.52% P dry mass**:

`P mass fraction = (0.50 / 12) / (250 / 31) = 0.0052`.

This central estimate is not AMF-specific. For AMF, direct measurements show substantial, spatially concentrated P pools, but do not provide a reliable whole-cell dry-mass percentage. Intraradical AMF hyphae contained approximately **60–170 mM P**, with observations up to **600 mM**; young arbuscule branches contained **30–50 mM P**. These are hydrated, local concentrations and should not be converted to dry-mass P without water fraction and compartment volume.

## Evidence in the requested priority order

| Fungal group / material | Quantitative result | Interpretation and limitation |
|---|---:|---|
| **AMF: intraradical hyphae and arbuscules** | Intraradical hyphae generally **60–170 mM P**, occasionally **600 mM**; young arbuscules **30–50 mM P** | Direct in situ cryo-analytical measurement in soil- and glasshouse-grown AM; local hydrated concentration, not whole-fungus dry-mass P. |
| **AMF: *Glomus/Rhizophagus intraradices* hyphal P pools** | P was detected in vacuolar polyphosphate granules; increased carbon supply shifted labelling among soluble phosphate, phospholipids, nucleic-acid/protein-phosphate pools, and polyP | Direct AMF evidence that polyP and structural/functional P pools are dynamic. The study did not close total P to fungal dry mass. |
| **Aseptate Mucoromycota** | Total P was **0.64–6.24% of cell dry weight** across *Umbelopsis vinacea*, *Rhizopus stolonifer*, *Mucor circinelloides*, *M. racemosus*, and *Amylomyces rouxii* | Strong direct fallback for aseptate/coenocytic fungi. Values were growth-medium dependent; high values are plausibly dominated by polyP and should not be treated as structural minimums. |
| **Fungi across phyla, including Glomeromycota and Zygomycota** | Fungal biomass P **0.040–5.5% dry mass**; median molar C:N:P **250:16:1** | Best broad benchmark, but it is a meta-analysis of heterogeneous source data and includes fruiting bodies and other materials; mycelial data are comparatively sparse. |
| **Septate filamentous fungi** | Included in the broad fungal range above; no robust, general whole-mycelium P percentage superior to the fungal compilation was located | Fruiting-body measurements are not automatically transferable to living external hyphae. Use the broad range with an explicit morphology and growth-condition uncertainty. |
| **Unicellular fungi: *Saccharomyces cerevisiae*** | Single-cell ICP-TOF-MS measured **12.4–890 fg P cell⁻¹** under standard conditions | Direct per-cell result showing large cell-to-cell variation; conversion to P percentage needs the matching cell dry mass, which was not supplied by that measurement. |
| **Unicellular fungi: *Candida utilis*** | Total cell-bound P **2.85% w/w** in the reported sample. Under potassium limitation: 0.4% inorganic-pool P, 0.7% nucleic-acid P, 1.9% polyP, and about 0.8% residual/other P. Under glucose limitation: 0.23% inorganic P, 0.7% nucleic-acid P, 0.85% polyP, and 1.07% unidentified P | Direct biochemical fractionation. It demonstrates that storage polyP can be a large and condition-sensitive fraction of fungal P. |
| **General eukaryotes** | A comparative model and data set found whole-organism P declines with size in unicellular eukaryotes, associated particularly with phosphorus-rich RNA | Supports RNA as a major variable P pool, but is not a fungal-specific cell-composition estimate. |

## What “inside the structure” includes

The total cellular P should be partitioned conceptually rather than treated as one fixed structural constant:

1. **Nucleic acids.** DNA and RNA contain phosphodiester backbone P. RNA can be a major variable pool because ribosome abundance rises with biosynthetic demand. The *Candida utilis* fractionation above found about **0.7% dry mass as nucleic-acid P** in two nutrient-limitation regimes.
2. **Polyphosphate (polyP).** PolyP is a major fungal P reserve, often stored in vacuoles/acidocalcisome-like compartments and complexed with cations. It is especially important for AMF, where electron-dense vacuolar granules are P-rich, and for Mucoromycota and yeasts under P excess or particular nutrient limitations.
3. **Phospholipids.** Membrane phospholipids contribute P through one phosphate group per molecule; phosphatidylcholine, for example, is approximately 4.1% P by molecular mass before accounting for its abundance in total biomass. This is a chemical conversion, not a measured whole-cell fraction.
4. **Phosphorylated proteins and carbohydrates.** Protein phosphorylation, phosphomannans, and other phosphate esters contribute smaller but functionally important pools. The *Candida* study assigned part of its residual P to phosphomannans and other polymers, including phospholipids.
5. **Soluble phosphate and phosphorylated metabolites.** Orthophosphate, ATP/ADP/AMP, sugar phosphates, and related intermediates are usually a smaller standing pool than polyP or RNA, but they turn over rapidly and are central to metabolism. A triphosphate nucleotide such as ATP contains three P atoms; its P fraction is about 17% by molecular mass, but ATP concentration alone cannot determine whole-cell P content.

## Recommended value for modelling

If the model needs one structural phosphorus quota for living fungal dry mass and no AMF-specific dry-mass measurement is available, use:

- **reference:** `1% P of fungal dry mass`;
- **ordinary sensitivity range:** `0.4–3% P of dry mass`;
- **high-P/polyP sensitivity:** `3–6% P of dry mass`;
- **P-starved/low-P sensitivity:** approximately `0.04–0.4% P of dry mass`, recognising that the lower end comes from the broad fungal compilation and is not an AMF-specific measurement.

For a phosphorus stock in mass units, `1% P dry mass` means **10 mg P per g fungal dry mass**. The central meta-analytic C:N:P conversion gives about **5.2 mg P per g dry mass**, so a narrower central prior could be `5–10 mg P g⁻¹ fungal DM`.

Do not use the 60–170 mM AMF hyphal measurements as if they were 60–170 mmol P per litre of dry fungal biomass. Those measurements are local concentrations in fully hydrated tissues and are valuable primarily for showing compartmentalisation and transfer/storage, not for setting a whole-biomass quota.

## Sources considered

1. Ryan, M. H., McCully, M. E. & Huang, C. X. (2003). “Location and quantification of phosphorus and other elements in fully hydrated, soil-grown arbuscular mycorrhizas: a cryo-analytical scanning electron microscopy study.” *New Phytologist* 160, 429–441. [DOI](https://doi.org/10.1046/j.1469-8137.2003.00884.x). Direct AMF hyphal and arbuscule P concentrations.
2. Viereck, N. et al. (2004). “Phosphate pool dynamics in the arbuscular mycorrhizal fungus *Glomus intraradices* studied by in vivo 31P NMR spectroscopy.” *New Phytologist*. [DOI](https://doi.org/10.1111/j.1469-8137.2004.01048.x). AMF phosphate-pool dynamics.
3. Bücking, H. et al. (2005). “Phosphate uptake, transport and transfer by the arbuscular mycorrhizal fungus *Glomus intraradices* is stimulated by increased carbohydrate availability.” *New Phytologist*. [DOI](https://doi.org/10.1111/j.1469-8137.2004.01274.x). AMF polyP, soluble P, phospholipid, nucleic-acid/protein-phosphate pools.
4. Zhang, J. & Elser, J. J. (2017). “Carbon:Nitrogen:Phosphorus Stoichiometry in Fungi: A Meta-Analysis.” *Frontiers in Microbiology* 8:1281. [DOI](https://doi.org/10.3389/fmicb.2017.01281). Broad fungal P range and median C:N:P.
5. Dzurendova, S. et al. (2021). “Calcium Affects Polyphosphate and Lipid Accumulation in Mucoromycota Fungi.” *Journal of Fungi* 7:300. [DOI](https://doi.org/10.3390/jof7040300). Direct total-P dry-weight ranges for aseptate Mucoromycota and polyP/lipid dependence.
6. Aiking, H., Sterkenburg, A. & Tempest, D. W. (1977). “The occurrence of polyphosphates in *Candida utilis* NCYC 321, grown in chemostat cultures under conditions of potassium- and glucose-limitation.” *FEMS Microbiology Letters* 1, 251–254. [DOI](https://doi.org/10.1111/j.1574-6968.1977.tb00626.x). Total yeast P and biochemical P fractions.
7. Katchman, B. J. & Fetty, W. O. (1955). “Phosphorus metabolism in growing cultures of *Saccharomyces cerevisiae*.” *Journal of Bacteriology* 69, 607–615. [DOI](https://doi.org/10.1128/jb.69.6.607-615.1955). Yeast phosphorus pools, including acid-soluble and DNA-associated P as indexed in the source record.
8. Ramsay, A. M. & Douglas, L. J. (1979). “Effects of phosphate limitation of growth on the cell-wall and lipid composition of *Saccharomyces cerevisiae*.” *Microbiology* 110, 185–190. [DOI](https://doi.org/10.1099/00221287-110-1-185). Direct phosphate-limitation effect on total P and P-containing wall/lipid composition.
9. Zhou, X. et al. (2022). “Exploring the Extent of Phosphorus and Heavy Metal Uptake by Single Cells of *Saccharomyces cerevisiae* and Their Effects on Intrinsic Elements by SC-ICP-TOF-MS.” [PubMed record](https://pubmed.ncbi.nlm.nih.gov/35547146/). Direct single-cell P range.
10. Gillooly, J. F. et al. (2005). “The metabolic basis of whole-organism RNA and phosphorus content.” *PNAS* 102, 11923–11927. [DOI](https://doi.org/10.1073/pnas.0504756102). General eukaryotic RNA/P relationship.

## Caveats

The literature mixes hydrated local concentrations, whole-cell dry-mass elemental analysis, isolated structures, spores, hyphae, and fruiting bodies. These are not interchangeable. AMF are particularly difficult because they are obligate symbionts, coenocytic, and spatially heterogeneous; the best direct AMF studies often quantify P pools or concentration in specific hyphal compartments rather than total P per fungal dry mass. PolyP accumulation can move a cell from roughly 1% P to several percent P without implying a corresponding increase in indispensable structural material.
