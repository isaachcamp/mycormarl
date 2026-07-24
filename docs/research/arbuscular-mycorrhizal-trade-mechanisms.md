# Mechanisms governing trade in arbuscular mycorrhizae

**Research date:** 24 July 2026  
**Scope:** Carbon supplied by plants in exchange for phosphorus (P), nitrogen
(N), and other services from arbuscular mycorrhizal (AM) fungi. The emphasis is
on primary experimental and theoretical sources. “Control” below does not imply
conscious choice: it includes active signalling, transporter regulation,
metabolic source–sink effects, and network transport.

## Executive summary

There is no single accepted rule governing AM trade. The best-supported view is
a **layered, spatially distributed control system**:

1. Plants and fungi possess the physical machinery for bidirectional exchange
   at short-lived arbuscules. Plants supply both lipids and sugars; fungi deliver
   P and, conditionally, N.
2. Whole-organism nutrient status regulates entry into symbiosis, construction
   of the exchange interface, transporter abundance, and arbuscule lifespan.
3. Both partners can redirect resources toward better-supplying partners or
   compartments. This produces the reciprocal-reward patterns predicted by
   biological-market theory, especially when partners are spatially separable,
   alternatives exist, and the traded nutrient is scarce.
4. The same allocation patterns can also arise from source–sink physiology,
   transporter kinetics, growth feedbacks, or fungal storage. Reciprocal
   allocation is therefore real, but it is not by itself proof that either
   partner compares partners or enforces a negotiated exchange rate.
5. Fungal storage, long-distance transport, changing network architecture, and
   common mycorrhizal networks decouple uptake from delivery in space and time.
   Exchange can consequently be delayed, strongly unequal, or beneficial at the
   network level without appearing “fair” in each bilateral interaction.

The strongest unknowns are the fungal P exporter and lipid importer, the
relative quantitative roles of lipid and sugar currencies, the signal that
links nutrient flux to arbuscule maintenance, and how local exchange control
scales to mixed, multispecies field networks.

## A mechanism map

| Scale | Supported mechanism | What the evidence does not establish |
|---|---|---|
| Whole plant | PHR–SPX nutrient-status signalling and strigolactones regulate fungal recruitment; nutrient status also influences arbuscule turnover. | A single scalar “need” signal or a universal response across plant taxa. |
| Arbusculated cell | RAM1/WRI transcriptional programmes build the periarbuscular membrane (PAM), activate lipid provision, and coordinate exchange genes. | That the programme computes a partner-specific exchange rate. |
| Plant-to-fungus carbon | Host fatty-acid synthesis and STR/STR2-associated export are necessary; fungal RiMST2 and plant SWEET-family proteins support a parallel sugar route. | The STR/STR2 substrate, fungal lipid importer, dominant sugar exporter, or stable lipid:sugar ratio. |
| Fungus-to-plant P | Fungi acquire, store, and transport P, including as polyphosphate; PAM H+-coupled plant transporters such as MtPT4/PT11 take it up. | The dedicated fungal arbuscular P-efflux mechanism. |
| Fungus-to-plant N | Isotope tracing supports an extraradical arginine shuttle followed by intraradical catabolism and transfer to the plant. | The chemical species and transporter responsible for crossing the fungal membrane. |
| Partner and network | Carbon availability alters fungal nutrient acquisition/allocation; nutrient delivery alters plant carbon allocation; fungi store and reroute resources across networks. | Whether a given response is active partner comparison, local source–sink coupling, or both. |

## What is mechanistically well established

### The exchange interface and its currencies

Arbuscules are not passive plumbing. They are transient fungal structures
enveloped by a plant-derived PAM whose transport capacity and lifetime are
regulated. Genetic and isotope experiments independently show that plants
supply fatty acids to AM fungi. Disrupting host `FatM`, `RAM2`, or `STR`
impairs lipid transfer, arbuscule development, and colonisation
([Keymer et al. 2017](https://doi.org/10.7554/eLife.29107);
[Luginbuehl et al. 2017](https://doi.org/10.1126/science.aan0081);
[Jiang et al. 2017](https://doi.org/10.1126/science.aam9970)).
WRI5a regulates lipid-production and PAM genes
([Jiang et al. 2018](https://doi.org/10.1016/j.molp.2018.09.006)).
These are strong causal results, but biochemical demonstration that STR/STR2
exports the proposed 16:0 beta-monoacylglycerol is still lacking.

Sugar is a parallel carbon route rather than a discarded older hypothesis.
The fungal monosaccharide transporter RiMST2 is expressed at arbuscules and
intercellular hyphae, and host-induced silencing strongly reduces arbuscule
formation ([Helber et al. 2011](https://doi.org/10.1105/tpc.111.089813)).
MtSWEET1b localises to the PAM and transports glucose in yeast, although two
loss-of-function alleles had no clear symbiotic phenotype, implying redundancy
([An et al. 2019](https://doi.org/10.1111/nph.15975)). Thus the field no longer
treats “sugar or lipid?” as an exclusive choice; the unresolved question is
their relative contribution to fungal respiration, storage, and biomass under
different conditions.

On the return path, MtPT4 localises to the PAM. Null mutants lose
fungus-delivered P and their arbuscules die prematurely
([Harrison et al. 2002](https://doi.org/10.1105/tpc.004861);
[Javot et al. 2007](https://doi.org/10.1073/pnas.0608136104)).
The PAM proton pump MtHA1 supplies the gradient needed for uptake; its loss
alkalinises the interface, blocks symbiotic P uptake, and impairs arbuscule
branching ([Wang et al. 2014](https://doi.org/10.1105/tpc.113.120436)).
Live imaging in rice now shows that OsPT11 abundance varies with P while
arbuscules of similar visible form have different transport capacity. Arbuscule
presence is therefore not equivalent to exchange activity
([McGaley et al. 2026](https://doi.org/10.1038/s41467-026-71496-8)).

Fungal P is acquired extraradically, stored and transported through
vacuolar/polyphosphate pools, and released intraradically. Water flow contributes
to long-distance polyphosphate movement
([Kikuchi et al. 2016](https://doi.org/10.1111/nph.14016)).
RiPT7 can transport P bidirectionally in yeast, and silencing it reduces P
delivery and arbuscule development at low-to-medium P
([Xie et al. 2022](https://doi.org/10.1111/nph.17973)); this may be a
homeostasis regulator rather than the missing principal effluxer.

For N, split-compartment carbon-13/nitrogen-15 experiments support an arginine
shuttle: inorganic N is assimilated into arginine extraradically, transported
inward, then catabolised so that N reaches the root without the arginine carbon
skeleton ([Govindarajulu et al. 2005](https://doi.org/10.1038/nature03610);
[Jin et al. 2005](https://doi.org/10.1111/j.1469-8137.2005.01536.x)).
Coordinated expression and enzyme activity support this spatial division
([Tian et al. 2010](https://doi.org/10.1104/pp.110.156430)).

### Control by nutrient status and interface lifespan

Plant P status regulates colonisation and exchange at several points. In rice,
PHR2 directly regulates genes for strigolactone production, fungal perception,
entry, and nutrient transport; `phr2` plants show reduced colonisation and
PT11-mediated P uptake ([Das et al. 2022](https://doi.org/10.1038/s41467-022-27976-8)).
In *Medicago*, SPX1/SPX3 affect strigolactones, P homeostasis, and
MYB1-associated arbuscule degradation
([Wang et al. 2021](https://doi.org/10.1093/plcell/koab206)).
This is not a simple low-P switch: *Medicago* PHR2 promotes colonisation while
antagonising arbuscule maintenance
([Wang et al. 2024](https://doi.org/10.1111/nph.19869)).

Premature degeneration of `mtpt4` arbuscules resembles a local sanction, but
low plant N suppresses degeneration even when the known PAM P uptake route is
disabled ([Breuillin-Sessoms et al. 2015](https://doi.org/10.1105/tpc.114.131144)).
This shows that the checkpoint integrates multiple currencies or whole-plant
needs; it does not simply detect fungal “honesty.” AMT2;3 is required for the
low-N rescue but does not behave as a conventional bulk ammonium transporter,
making a sensing or “transceptor” role plausible.

## Competing hypotheses for allocation

### 1. Reciprocal rewards and biological markets

The market hypothesis predicts that each partner allocates more of its commodity
to partners giving a better return, and that alternatives create supplier
competition. Controlled split-root and root-organ experiments provide causal
support: plants sent more carbon to fungi delivering more P, while fungi sent
more P to higher-carbon roots
([Kiers et al. 2011](https://doi.org/10.1126/science.1208473)).
Providing an alternative fungal species caused a poorer supplier to increase P
delivery and lowered the plant's carbon cost per P
([Argüello et al. 2016](https://doi.org/10.1111/ele.12601)).
Preferential allocation weakens as soil P increases, as expected when fungal P
loses value ([Ji & Bever 2016](https://doi.org/10.1002/ecs2.1256)).

This hypothesis explains conditional cooperation and responses to partner
options. It does not, by itself, identify a sensing mechanism, require an
instantaneous exchange, or predict equal benefits. “Reciprocal rewards” is a
better description of the evidence than one-step tit-for-tat.

### 2. Host sanctions, partner choice, and spatial resolution

Plants may screen partners before colonisation, preferentially feed them
afterwards, or terminate underperforming arbuscules. Experiments show
preferential allocation to beneficial fungi when symbionts occupy separable
root regions, but low-quality fungi proliferate when infections are intermixed
([Bever et al. 2009](https://doi.org/10.1111/j.1461-0248.2008.01254.x);
[Hopkins et al. 2023](https://doi.org/10.1086/722532)).
Sanctions therefore require a seam at which return can be attributed to a
partner. Natural co-colonisation and hyphal mixing may make that attribution
noisy, helping less beneficial fungi persist.

### 3. Source–sink physiology and competition for surplus resources

An alternative is that apparent rewards emerge without partner comparison.
Carbon-rich roots are stronger fungal sinks; active fungus creates a nutrient
sink; plant nutrient demand changes transporter expression and carbon export.
Increasing carbon can stimulate fungal N uptake and transfer
([Fellbaum et al. 2012](https://doi.org/10.1073/pnas.1118650109)), and common
networks send more P and N to stronger carbon-source hosts
([Fellbaum et al. 2014](https://doi.org/10.1111/nph.12827)).
These results are compatible with both strategic allocation and mass-action
physiology.

Source–sink control explains why shading need not reduce fungal carbon:
P addition sharply reduced recent plant C allocation when fungal benefit fell,
whereas shading did not produce the carbon-budget response expected by a simple
cost-minimiser ([Olsson et al. 2010](https://doi.org/10.1111/j.1574-6941.2009.00833.x)).
Its weakness is that physiology alone does not easily explain poorer fungi
changing delivery specifically when a competitor is introduced.

### 4. Fungal storage, transport, and spatial arbitrage

Under low host carbon, *Rhizophagus irregularis* accumulated far more P in
hyphae and spores while arbuscule density declined
([Hammer et al. 2011](https://doi.org/10.1111/j.1574-6941.2011.01043.x)).
This is compatible with withholding, but reduced root sink strength is an
alternative. When exposed to unequal P patches, fungi increased total P
delivery, reduced storage, and moved P from rich to poor patches where inferred
returns were higher
([Whiteside et al. 2019](https://doi.org/10.1016/j.cub.2019.04.061)).
Storage and routing mean low delivery now may be future inventory management,
not low cooperation.

Network form is itself part of resource economics. High-throughput imaging
shows *R. irregularis* building self-regulating travelling waves, loops, wider
trunk hyphae, and faster flows that preserve transport efficiency while the
network expands
([Oyarte Galvez et al. 2025](https://doi.org/10.1038/s41586-025-08614-x)).
This demonstrates adaptive network architecture and flow, not direct reciprocal
exchange control. Recent controlled work links the carbon:P exchange rate to a
density-versus-expansion-speed trade-off
([Bisot et al. 2026](https://doi.org/10.1073/pnas.2512182123)); field and
multispecies generality remain open.

### 5. Fixed traits, partner matching, and fungal enforcement

Fungal taxa differ in exploration, intraradical/extraradical allocation,
nutrient acquisition, and timing. Plants differ in demand and root traits.
Apparent choice can therefore be a growth feedback between matched traits.
Across many pairings, fungi delivering more P received more C and offering two
fungi raised plant shoot P, yet differential allocation within individual
split-root plants was weak or equivocal
([Weber et al. 2024](https://doi.org/10.1371/journal.pone.0292811)).

A distinct theoretical model proposes that fungi restrict direct plant P
uptake, forcing dependence on fungal trade
([Wyatt et al. 2016](https://doi.org/10.1038/ncomms10322)).
The model shows conditions under which enforcement could evolve, but direct
evidence that AM fungi use this mechanism remains much weaker than evidence for
conditional allocation.

## Patterns no current hypothesis explains alone

- **Extreme unequal terms in common networks.** Flax supplied little C but
  received up to 94% of network-delivered N and P, while sorghum supplied much
  C for little return and suffered little growth penalty
  ([Walder et al. 2012](https://doi.org/10.1104/pp.112.195727)). Trait and
  source–sink accounts explain the direction better than simple markets, but
  not the quantitative ratios.
- **Weak local choice alongside strong cross-species associations.** Better
  fungal suppliers often receive more C across treatments, while the same
  plant shows little preference between root halves. Active choice, fungal sink
  strength, and accumulated growth feedback remain confounded.
- **Persistence of poor partners.** Spatial mixing lets low-quality fungi
  escape local discrimination; the amount of partner-level resolution in
  natural roots is poorly quantified.
- **Changing quality through time.** Short isotope pulses and season-long plant
  benefit can rank fungi differently. Neither market nor source–sink models
  normally specify the integration window used by either organism.
- **Context reversals.** P addition, shade, herbivory, N status, host genotype,
  and fungal identity can change both the magnitude and direction of exchange.
  No general model yet predicts these interactions.
- **Heterogeneous arbuscules.** Transporter imaging shows that morphologically
  present arbuscules differ in exchange capacity, but what generates and
  coordinates this cell-to-cell variability is unknown.
- **Multiple currencies.** Most decisive experiments reduce exchange to C for
  P, while fungi may supply N, water, micronutrients, or protection. A poor P
  supplier may be valuable on another axis.
- **Local-to-network scaling.** It is unknown how millions of nutrient-sensitive,
  short-lived arbuscules aggregate into allocation across multiple hosts and
  genetically mixed fungal networks.
- **Field generality.** Strong causal evidence comes mainly from carrot
  root-organ cultures, split roots, and a small set of *Medicago*, *Lotus*,
  rice, and *Rhizophagus* combinations. Soil heterogeneity, microbes,
  herbivores, and multispecies networks may alter the governing mechanisms.

## Implications for MycorMARL

The current two-agent allocation abstraction is a defensible hypothesis-testing
baseline, but it should not be labelled as a settled biological-market
mechanism. Biologically informative extensions would separate:

1. carbon into lipid and sugar currencies;
2. fungal P uptake, storage, transport, and release;
3. local arbuscule exchange capacity from whole-organism resource pools;
4. source–sink coupling from explicit partner-contingent allocation;
5. nutrient-status control from learned actions;
6. immediate exchange from delayed network delivery; and
7. bilateral trade from multiple hosts, fungi, and currencies.

The strongest discriminating simulations would compare several mechanisms
against the same patterns: reciprocal allocation under partner choice,
collapse of preference at high P, persistence of low-quality fungi under
mixing, unequal common-network exchange, delayed delivery after storage, and
within-root arbuscule heterogeneity. A learned policy reproducing one pattern
would not identify the mechanism unless competing physiological models fail on
the remaining patterns.

## Confidence and limitations

Confidence is high for the existence of parallel lipid/sugar supply, PAM
H+-coupled plant P uptake, the fungal arginine-N shuttle, nutrient-sensitive
arbuscule turnover, and bidirectional allocation responses in controlled
systems. Confidence is moderate that biological-market competition commonly
governs allocation beyond those systems. Confidence is low for a universal
exchange rule, direct “intent” sensing, fungal enforcement of plant dependence,
or quantitative transfer of laboratory exchange rates to field communities.

The literature also has a measurement problem: fungal biomass is sometimes used
as a proxy for received carbon; short tracer pulses can miss storage and delayed
delivery; colonisation is not exchange capacity; and plant growth is an
integrated outcome rather than a direct flux. Future tests need simultaneous,
time-resolved measurement of both currencies at local interfaces and across
whole networks.
