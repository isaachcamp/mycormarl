# Initial biomass for an established resource-exchange fixture

**Research date:** 6 August 2026
**Scope:** A biologically defensible plant and arbuscular-mycorrhizal-fungus
starting biomass for the small coupled phosphate qualification domain. The aim
is to start after the seed/spore establishment bottleneck, while avoiding a
plant or fungal colony so large that most of its biomass-implied uptake
geometry lies outside the 2 cm-radius, 2 cm-deep qualification domain.

## Recommendation

Use the following values in the **coupled qualification fixture**:

| Partner | Recommended initial biomass | Plausible fixture range | Interpretation |
|---|---:|---:|---|
| Plant | **`0.01 g` dry biomass (10 mg)** | `0.005–0.02 g` | A small established seedling, not a seed or a later-harvest greenhouse plant. |
| AM fungus | **`0.0001 g` dry biomass (0.1 mg)** | `0.00003–0.0003 g` | Living external mycelium already supported by a colonized root, not inoculum, carrier, colonized-root, or spore mass. |

This is a **qualification-fixture recommendation**, not a claim that 10 mg is
the typical biomass of a plant when a mature greenhouse exchange experiment is
harvested. Published exchange systems commonly contain plants hundreds of
milligrams in size by the measurement stage. The lower fixture value represents
an earlier established symbiosis and keeps the model-generated root geometry
inside the intentionally small numerical-test domain.

If the fixture continues to give each free resource pool one structural-biomass
equivalent, the corresponding initial pools are:

- plant C: `0.01 × 0.402 = 0.00402 g C`;
- plant P: `0.01 × 1.92 = 0.0192 mg P`;
- fungal C: `0.0001 × 0.5 = 0.00005 g C`;
- fungal P: `0.0001 × 2 = 0.0002 mg P`.

## Why 10 mg is reasonable for the plant fixture

The current plant default is anchored to a carrot propagule. Population means
for *Daucus carota* propagule mass span **0.7–3.3 mg**, based on weighed seed
lots from 40 populations ([Vandelook et al. 2024](https://doi.org/10.1017/S0960258524000230)).
A 10 mg plant is therefore about 3–14 propagule masses: it is no longer a bare
seed state, but remains a deliberately small seedling.

That size is conservative relative to resource-exchange experiments at their
measurement stage. In a whole-plant, compartmented *Medicago truncatula*–
*Rhizophagus irregularis* system, plants with a profuse external mycelium had
combined shoot-plus-root dry weights of roughly **0.70–0.83 g** across
treatments (the control mycorrhizal treatment was `410 + 420 = 830 mg`)
([Le Pioufle & Declerck 2018](https://doi.org/10.3389/fmicb.2018.01254)). In a
greenhouse compartment experiment designed to manipulate fungal nutrient
signals and host carbon supply, mycorrhizal wheat produced **0.56–0.73 g shoot
dry mass per pot** and had 22–47% root colonization
([Tian et al. 2017](https://doi.org/10.1371/journal.pone.0172154)). Those are
later experimental states, and the wheat values exclude roots, so they should
not be copied into the 2 cm qualification domain. They establish that 10 mg is
a low, early-established fixture rather than an unrealistically large seed.

Under the repository's current traits, the 10 mg plant maps to approximately
158 cm (1.58 m) total root length and a maximum root-disc radius of about **1.43 cm** in
the 2 cm qualification profile. This is large enough to span many 0.025–0.1 cm
cells but small enough not to make domain clipping the dominant initial
condition. This mapping follows the repository's biomass-derived uptake
geometry contract; it is not a root measurement taken from the cited papers.

## Why 0.1 mg is reasonable for the fungal fixture

The strongest direct biomass evidence comes from a primary experiment that
physically recovered *R. irregularis* external mycelium from root-free zones,
dried it, and weighed it. After 64 days of pre-growth plus 54 days of the
resource-patch experiment, seven fungal genotypes produced approximately
**3–17 mg mycelium dry biomass per microcosm**; the published figure reports
genotype means and the methods specify membrane recovery followed by two days
of vacuum-centrifuge drying
([Sun et al. 2024](https://doi.org/10.1007/s00572-024-01154-8)). This is direct
living-system mycelium evidence, although it comes from transformed chicory
root-organ cultures rather than soil pots and represents a mature network.

The proposed **0.1 mg** is deliberately well below that measured mature range. It is
best interpreted as an early established external network. It is also
consistent in scale with a whole-plant *R. irregularis* system in which
external hyphae had already crossed into a separate compartment, plants were
pre-mycorrhized, and four weeks of hyphal growth yielded measured hyphal
lengths of roughly **7.1–14.8 m in the sampled regions**
([Le Pioufle & Declerck 2018](https://doi.org/10.3389/fmicb.2018.01254)). Those
lengths cannot be converted into total fungal biomass because they do not
represent the entire mycelium and include no tissue-diameter measurement; they
serve only as evidence that metre-scale external networks are normal once the
symbiosis is operating.

With the repository's fungal tissue-carbon density, carbon fraction, hyphal
radius, and saturation density, 0.1 mg dry biomass maps to about **5.5 m of
external hyphae** and a saturated hemispherical radius of about **0.51 cm**.
Again, that is a model conversion, not a value measured in the papers. It puts
the initial fungal front across many cells at all tested grid intervals while
remaining inside the 2 cm radial and depth bounds.

## Numerical qualification probe

An exploratory run of the existing two-day coupled qualification trajectory
used the proposed `0.01 g` plant and `0.0001 g` fungus, biomass-consistent free
resource pools, and the existing fixed Physical actions. The worst reported
change was **0.59%** between `0.025` and `0.05 cm`, and **2.56%** between
`0.05` and `0.1 cm`; both are below the qualification's 5% threshold. This
probe did not change the canonical qualification artifacts. It shows that the
proposed pair addresses the observed grid-front quantisation in this fixture;
it is numerical evidence, not empirical validation of the biomasses.

## Important interpretation limits

- **Do not use grams of commercial inoculum as fungal biomass.** Inoculum is
  commonly a mixture of carrier substrate, colonized roots, spores, microbes,
  and external hyphae. For example, Tian et al. varied additions of `20–200 g`
  inoculum, but those values are treatment material, not living fungal dry
  mass ([Tian et al. 2017](https://doi.org/10.1371/journal.pone.0172154)).
- **Do not use colonized-root mass as external fungal biomass.** It combines
  plant root tissue with intraradical fungal structures, whereas the current
  model maps all fungal biomass to living external hyphae.
- The direct Sun et al. fungal measurements are mature root-organ cultures;
  the 0.1 mg recommendation is a transparent lower, early-established modelling
  choice, not a directly observed initial condition.
- Fungal genotype, host, phosphorus supply, carbon supply, growth time, and
  culture geometry all alter external-mycelium production. The proposed
  `0.00003–0.0003 g` range should therefore be retained for numerical
  sensitivity tests; `0.001 g` remains a useful mature-network sensitivity case.
- For a full greenhouse-scale biological scenario, a plant biomass of roughly
  `0.1–1 g` and a fungal external-mycelium biomass of order `0.003–0.02 g`
  would be closer to the cited measurement-stage evidence. Those values need a
  larger soil domain and should not be substituted into the current small-grid
  qualification without checking boundary clipping and soil-resource scaling.

## Decision rationale

The pair **10 mg plant + 0.1 mg living external AM fungus** is a rounded,
asymmetric pair that is both biologically interpretable as an already operating
symbiosis and numerically well resolved in the present qualification domain. It
removes seed germination and single-spore establishment from a model that does
not represent those processes. At the same time, it avoids pretending that the
2 cm qualification domain is a complete later-stage greenhouse pot.

The qualification report should label these values as an **early-established
coupled fixture**. Production defaults and greenhouse-scale scenarios should
remain separate decisions.
