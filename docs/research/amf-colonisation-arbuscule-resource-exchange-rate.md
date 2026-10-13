# AM fungal colonisation, arbuscules, and resource-exchange rates

## Question

How much can we say that the rate or extent of AM fungal root colonisation, and
the number/abundance of arbuscules, affects the rate of resource exchange in
arbuscular mycorrhizal (AM) symbioses? Studies are classified here as **field**,
**glasshouse**, or **lab/controlled-environment** according to where the
plant--fungus experiment was performed. A study can be mechanistically strong
without directly measuring an exchange rate; those cases are labelled as
proxies.

## Short answer

The defensible conclusion is **not** that exchange rate is proportional to
percentage root colonisation or arbuscule frequency. Colonisation and arbuscule
counts tell us how much fungal interface is present, but they do not uniquely
specify how much C, P, or N crosses that interface per unit time.

There is nevertheless a causal connection:

* More functional interface can provide more capacity for exchange, and
  arbuscules are the main cellular site of C/P exchange.
* The best direct evidence shows that C and P fluxes can be approximately
  proportional in a controlled in-vitro system, but the exchange ratio depends
  strongly on host genotype and context.
* Field evidence directly testing the usual inference is cautionary: in wheat,
  the amount of AM-delivered P was not significantly related to percentage root
  length colonised.
* Arbuscule abundance is a better mechanistic proxy than total colonisation,
  but still an incomplete one. Live imaging shows that arbuscules differ in
  developmental stage and in phosphate-transporter abundance; arbuscule
  presence is therefore uncoupled from arbuscule function.

For modelling, it is reasonable to let colonisation or arbuscule abundance
raise a potential exchange capacity, but the realised exchange rate should
also depend on host nutrient demand, carbon supply, fungal identity, soil P,
arbuscule age/function, and transporter regulation. A saturating or
stage-weighted relationship is better supported than a linear one.

## What the primary studies show

| Primary study | Setting | What was measured | Result relevant to colonisation/arbuscules |
|---|---|---|---|
| Smith et al. (2015), wheat | **Field**, with a separate glasshouse comparison | `33P` supplied to AM-accessible compartments; AM-pathway contribution to plant P uptake; root colonisation | In the field, AM fungi supplied **6.5–21% of shoot P uptake** by day 67 despite low colonisation. There was **no significant relationship** between percentage root length colonised and AM-delivered P. The same null relationship occurred in the glasshouse experiment. This is the clearest direct warning against using colonisation percentage as an exchange-rate proxy. [Applied Soil Ecology, DOI: 10.1016/j.apsoil.2015.07.002](https://doi.org/10.1016/j.apsoil.2015.07.002) |
| McGaley et al. (2026), rice–*Rhizophagus irregularis* | **Lab/controlled environment**: sterile sand, controlled rooms, live confocal imaging and AM-slide chambers | Arbuscule development/lifespan and OsPT11 abundance, an AM-specific phosphate-uptake transporter | Arbuscules were transient and heterogeneous. Phosphate-responsive PT11 abundance varied among arbuscules, so arbuscules of similar visible presence did not have identical P-uptake capacity. The authors explicitly conclude that arbuscule presence and function are uncoupled. This supports counting functional/stage-weighted arbuscules, not treating every arbuscule as one equal exchange unit. [Nature Communications, DOI: 10.1038/s41467-026-71496-8](https://doi.org/10.1038/s41467-026-71496-8) |
| Bisot et al. (2026), AM fungal network growth | **Lab/controlled environment**: two-compartment in-vitro root-organ culture with quantitative imaging | Estimated plant-to-fungus C flux and fungus-to-plant P flux from network morphology and P measurements | C and P transfer rates were approximately proportional on average. The C:P exchange ratio was nearly invariant across fungal strains in one host but strongly changed when the host genotype changed. The experiment did not manipulate root colonisation percentage or arbuscule number directly, so it supports a context-dependent exchange-rate parameter, not a colonisation-to-flux slope. [PNAS, DOI: 10.1073/pnas.2512182123](https://doi.org/10.1073/pnas.2512182123) |
| Bücking & Shachar-Hill (2005), *Glomus intraradices* | **Lab**: two-compartment root-organ culture | `33P` uptake/transfer and `14C` carbohydrate tracing under different sucrose supplies | Greater carbon availability increased fungal P uptake and transfer: the study reports roughly **41% of supplied `33P`** taken up in controls versus **about 63–65% with sucrose**. This shows that exchange rate can change without requiring more colonisation, through carbon-dependent physiology. [New Phytologist, DOI: 10.1111/j.1469-8137.2004.01274.x](https://doi.org/10.1111/j.1469-8137.2004.01274.x) |
| Hammer et al. (2011), *G. intraradices* | **Lab**: monoxenic in-vitro root-organ culture | Fungal P and other nutrient accumulation after halving carbon available to roots | Halving the plant carbon pool led to up to **7-fold more nutrient accumulation in spores** and **9-fold more in hyphae**, with the strongest effect for P. This is evidence that low carbon changes fungal retention/storage and therefore decouples fungal nutrient content from immediate plant delivery. It is not a direct measurement of arbuscule-specific transfer rate. [FEMS Microbiology Ecology, DOI: 10.1111/j.1574-6941.2011.01043.x](https://doi.org/10.1111/j.1574-6941.2011.01043.x) |
| Argüello et al. (2016), split-root partner-choice experiment | **Glasshouse/pot** experiment | `33P` and `14C` isotope allocation with one plant exposed to two AM fungi | With an alternative fungal partner available, plants received more P from the less cooperative fungus and paid a lower C cost per unit P from it. This demonstrates that realised exchange rate is altered by partner context; it does not show that the effect was caused by more root colonisation or more arbuscules. [Ecology Letters, DOI: 10.1111/ele.12601](https://doi.org/10.1111/ele.12601) |
| Jiang et al. (2017), lipid-transfer mutants | **Lab/controlled environment**: genetic and inoculation experiments | Host lipid synthesis/export, arbuscule formation, fungal growth and colonisation | Disrupting host lipid-transfer machinery impaired arbuscule development and fungal growth. This is causal evidence that arbuscule formation is required for a functional carbon supply pathway, but it does not establish that each additional arbuscule transfers a fixed amount of C. [Science, DOI: 10.1126/science.aam9970](https://doi.org/10.1126/science.aam9970) |
| Charters et al. (2020), aphid herbivory and wheat AM symbiosis | **Glasshouse** | Dual-isotope C/P exchange, root colonisation, arbuscule/vesicle abundance | Herbivory changed the C:P exchange asymmetry, but the authors found **no correlation between root colonisation and mycorrhizal-acquired shoot `33P`**, and no correlation between plant C outlay and shoot `33P`. This is another controlled example in which fungal abundance did not predict exchange function. [Current Biology, DOI: 10.1016/j.cub.2020.02.087](https://doi.org/10.1016/j.cub.2020.02.087) |

## Interpretation by variable

### Percentage root colonisation

Percentage colonisation usually combines several structures and developmental
states: intraradical hyphae, arbuscules, coils, and sometimes vesicles. It is a
measure of occupancy, not a flux. The field wheat study is particularly useful
because it measured AM-delivered P directly and found no significant
colonisation--P-delivery relationship. Low colonisation can still deliver a
substantial amount of P if the external hyphal network is effective, and high
colonisation can have little net benefit if P is retained, the plant is not a
strong carbon sink, or the interface is poorly active.

### Arbuscule frequency or number

Arbuscules are closer to the relevant mechanism: they are the specialised
interface at which fungal mineral delivery and host carbon acquisition are
coordinated. Thus, holding other factors constant, more mature, active
arbuscules should increase potential exchange capacity. However, frequency is
still only a proxy because arbuscules differ in size, age, lifespan, transporter
abundance, and developmental stage. The rice live-imaging study provides direct
evidence for this heterogeneity and is especially important for avoiding the
assumption that “one arbuscule = one fixed unit of exchange.”

### Exchange-rate parameters

The strongest quantitative rate result among the studies above is not a
colonisation slope; it is the approximately proportional C/P flux relationship
in the in-vitro network study, with a host-genotype-dependent exchange ratio.
The isotope studies also show order-of-magnitude context effects: sucrose
increased fungal `33P` uptake from roughly 41% to 63–65% of supplied tracer,
whereas carbon limitation caused large increases in fungal P retention. These
effects show why a model should not infer a universal rate from root occupancy
alone.

## Practical modelling implication

If a single state variable is needed, use colonisation/arbuscule abundance as a
**capacity term**, not as the exchange rate itself. A defensible structure is

```text
realised exchange rate
  = capacity(colonisation, mature-arbuscule abundance, arbuscule size)
    × activity(host C supply, plant P demand, fungal P supply,
               transporter state, partner identity, developmental stage)
```

For a minimal model, a saturating capacity term such as `1 - exp(-k A)` is more
defensible than `rate = k A`, where `A` is active arbuscule abundance. If only
total colonisation is available, treat its coefficient as uncertain and expect
substantial residual variation. If data permit, distinguish total colonisation
from arbuscule frequency and weight arbuscules by developmental or transporter
activity state.

## Bottom line

We can say that arbuscules are necessary exchange interfaces and that increasing
the amount of active interface can increase potential exchange. We cannot say
that percentage colonisation, or even raw arbuscule counts, determines resource
exchange rate in a fixed or generally linear way. The most robust empirical
statement is that **functional arbuscule abundance is one determinant of
exchange capacity, while realised C/P/N flux is jointly controlled by interface
amount, interface activity, host carbon and nutrient status, fungal identity,
and environmental context**.
