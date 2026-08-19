# Natural soil-solution phosphate with depth

**Research date:** 19 August 2026
**Question:** What topsoil-to-subsoil inorganic-phosphate (Pi) concentration
profiles are measured in soils without agricultural nutrient enrichment, and
can one serve as a representative initial condition for the model?

## Decision-relevant answer

There is no defensible universal concentration profile for an unspecified
“wild-type soil.” Natural forest and grassland soils differ by orders of
magnitude according to parent material, pH, Fe/Al or Ca mineralogy, organic
matter, moisture, and biological cycling. More importantly, total inorganic P,
extractable P, the amount of phosphate presently in soil solution, and the
solid-phase capacity to replenish that solution are distinct measurements.

The model's `initial_solution_p_um` is a **soil-solution phosphate
concentration**. It should be initialized from measured solution Pi (or a
carefully matched equilibrium concentration), not from total P or a routine
soil-test extraction.

For a first non-agriculturally-enriched, depth-resolved sensitivity, the best
available direct primary measurement supports a *declining mineral-horizon
solution-P profile*: approximately 8--21 micromolar P in surface mineral soil
and 0.8--1.4 micromolar P in deeper E/B horizons. This is a candidate
**Siberian temperate-forest profile**, not a global default. It is compatible
with the present 0.1, 1, and 3 micromolar study range in the subsoil, but it
also shows why initializing all subsoil at zero is a specific treatment rather
than a generic natural-soil condition.

## Measurement distinction

| Measurement | What it means | Suitable for the current reset concentration? |
|---|---|---|
| Soil-solution `C_P` | phosphate ions in liquid phase, usually mass P per volume solution | Yes, after conversion to micromolar P |
| Solution-pool `Q_w` | amount presently in solution per dry-soil mass | Only with that soil's solution-volume convention |
| Diffusive/isotopically exchangeable P | solid-phase P that replenishes solution over stated time | No: informs buffering, not instantaneous concentration |
| Total/inorganic soil P | all P in an extracted solid sample | No |

The current model represents a solution concentration plus a single linear
buffer factor (`b_p`). A profile of solution Pi can therefore be introduced
without claiming to represent full natural P speciation, but its associated
buffering should not be treated as automatically calibrated.

## Direct natural-soil evidence

### Siberian forest mineral horizons: directly measured solution Pi

Achat et al. measured phosphate-ion concentration in soil solution (`C_P`) by
isotopic dilution across four contrasting Siberian forest plots. The authors
report 0.24--0.65 micrograms P per millilitre in surface mineral horizons
(A+AE or A) and 0.026--0.044 micrograms P per millilitre in deeper E, EBt,
and Bt horizons.

Using the atomic mass of P (30.9738 mg mmol-1), these values are:

| Horizon group | Measured `C_P` (µg P mL-1) | Equivalent µM P |
|---|---:|---:|
| Surface mineral (A+AE or A) | 0.24--0.65 | 7.7--21.0 |
| Deeper E / EBt / Bt | 0.026--0.044 | 0.84--1.42 |

The measured decline is about one order of magnitude. However, these soils had
relatively high P status compared with other permanently vegetated ecosystems;
they are useful as a complete, non-fertilizer forest case, not as a claim about
all natural soils.

### Forest and grassland profiles: solution-P pools decline, but subsoil is not empty

Brédoire et al. sampled forest and grassland sites in south-western Siberia,
separating about 0--20 cm topsoil from about 20--120 cm subsoil. Their solution
pool `Q_w` decreased in all studied 1-m profiles, from 2--22 µg P g-1 soil at
about 5 cm to 0.1--0.4 µg P g-1 at about 100 cm. This confirms a strongly
depth-stratified available-P pool in those non-cropland ecosystems, but the
units are per dry soil mass rather than per solution volume, so they must not
be passed directly to the model as micromolar concentration.

The same study found no systematic depth trend for *total inorganic P*;
subsoil can contain substantial inorganic P while maintaining much lower
solution Pi because the solid phase is more reactive. This is precisely the
distinction the new initial-condition design must preserve.

### Natural beech forests: replenishment persists below the topsoil

Rinderer et al. measured total P in depth-specific subsurface flow during
large sprinkling experiments at three German beech forests. Mineral-soil and
saprolite P concentrations were lower than forest-floor concentrations but
were mostly chemostatic during flow, indicating fast replenishment from mineral
or organic sources. This does not provide an initial solution-P profile for a
carrot simulation—the analyte is total P in moving water—but it rules out the
interpretation that lower natural horizons are inert zero-P space.

## Global labile-P evidence and additional natural biomes

### The global depth signal: labile Pi declines through the upper 50 cm

He et al. assembled 1,857 Hedley-fractionation measurements from 729
geographically distinct sites on six continents, spanning all major biomes and
12 USDA soil orders. The source samples span 0.5--450 cm, although 83 % are
from 0--30 cm. Their random-forest analysis predicts a sharp decline of both
labile and moderately labile Pi and Po through the upper 50 cm, followed by a
relatively stable 50--100 cm interval. In contrast, primary-mineral and
occluded P generally increase with depth.

This is the strongest available global analysis specifically including depth.
It corroborates the *direction* of the Siberian forest and grassland profiles:
surface enrichment is normal, but the subsoil is not generically P-free. It is
not a direct solution-P profile: its ``labile Pi`` is an operational
Hedley-fraction pool (resin/water Pi plus bicarbonate Pi), reported per mass of
soil. It cannot set `initial_solution_p_um`, and the global model should not be
read as a universal concentration-versus-depth curve.

The same analysis identifies a strong biome contrast at 0--30 cm: predicted
P-pool concentrations are lower in warm, humid **tropical forest** and
**savanna** than in northern cold **boreal forest** and **tundra**. Across
those biomes, labile Pi varies less as a *proportion* of total P than as an
absolute concentration, while occluded-P proportion increases from tundra and
boreal forest towards tropical forest and savanna. Thus a depth shape derived
from one grassland cannot honestly be called a global biome-independent
profile; mineralogy, weathering, pH, and total P remain essential covariates.

### Semi-arid temperate savanna: an independent shallow-profile observation

Zhao and Zeng measured P fractions in a temperate savanna on P-deficient
semi-arid sandy soil in the Horqin Sandy Land, China, comparing
0--5 with 5--20 cm. Every reported P fraction and phosphomonoesterase activity
declined with depth in the savanna. This is an independent **semi-arid
temperate-savanna** observation consistent with surface enrichment, but it is
only a 20-cm profile and uses chemical fractions rather than soil solution; it
therefore supplies no defensible numeric value for the model's 20--150 cm
domain.

### What these additions change

Together, the global synthesis and the savanna field study expand the evidence
beyond the original temperate forest and grassland cases to tropical forest,
savanna, boreal forest, and tundra. They strengthen the qualitative modelling
decision to permit an explicit declining depth profile and to reject a
zero-P-by-default subsoil. They **do not** change the selected grassland
assay-equivalent shape or justify replacing it with a global numeric default:
none supplies matched native pore-water Pi and buffering measurements by
horizon.

## Implication for the proposed model change

The reset condition now has two explicit choices: no depth profile means a
uniform field throughout the represented domain; a provenance-recorded depth
profile means a heterogeneous field. Do not use one scalar “representative Pi”
to imply an unrecorded vertical gradient.

A suitable first declaration has named depth bands and solution concentrations,
for example a *Siberian-forest sensitivity profile*:

```text
0--20 cm: choose within 7.7--21.0 µM P
20--150 cm: choose within 0.84--1.42 µM P
```

That profile has two unavoidable modelling choices:

1. the value selected within each observed range; and
2. whether the horizon boundary is fixed at 20 cm or represented as a smooth
   transition.

Neither choice can be settled by calling the profile “representative.” They
should be recorded as a primary scenario plus bounded sensitivity cases.

## Selected natural-grassland reference: south-western Siberia

Brédoire et al. provide the best fit to the requested reference case. They
sampled five nearby grassland sites developed on loess in south-western Siberia
at 5, 15, 30, 60, and 100 cm, with **no active management for the preceding
few decades**. The grasslands were temperate forest-steppe to sub-taiga
communities dominated by species including *Bromopsis inermis*, *Alopecurus
pratensis*, and *Calamagrostis epigeios*.

This is not an in-situ pore-water sample: air-dried, sieved soil was equilibrated
at 20 °C in 10 mL deionized water per gram soil before isotope dilution. It is
therefore an equilibrium soil-solution Pi profile under one explicit laboratory
convention. That convention is closer to the model's reset concentration than
an extraction pool, but it does not calibrate the model's buffer power.

The paper reports solution-pool `Qw` in µg P g-1 soil. Its method gives
`Qw = Cp × 10 mL g-1`, so `Cp = Qw / 10` in µg P mL-1. Converting `Cp` with
the atomic mass of P gives the following site-median profile:

| Sample depth (cm) | Site range (µM P) | Median (µM P) |
|---:|---:|---:|
| 5 | 5.5--44.2 | 9.4 |
| 15 | 0.32--4.2 | 3.2 |
| 30 | 0.32--2.6 | 1.6 |
| 60 | 0.65--2.3 | 1.0 |
| 100 | 0.32--1.3 | 0.65 |

The monotone median profile is the selected **representative semi-natural
Siberian-grassland depth profile** for this model. It supplies Pi throughout
the first metre but declines sharply from the surface. It also brackets the
present 1 µM default in the 60--100 cm subsoil, while its shallow value remains
above the current 3 µM high-P condition. "Representative" here means a named
assay-equivalent sensitivity shape, not a global or calibrated field default.

### Selected modelling use: relative, linearly interpolated profile

Use the grassland measurements as a **relative depth shape**, not as an
absolute calibration. Let `surface_solution_p_um` denote the model-defined
concentration at 0 cm. The unmeasured 0-cm value is set equal to the 5-cm
median, then linearly interpolate the following dimensionless knots from 0 to
100 cm:

| Depth (cm) | Relative solution-P factor | Derivation |
|---:|---:|---|
| 0 | 1.000 | Assumed equal to the 5-cm median |
| 5 | 1.000 | 9.4 / 9.4 |
| 15 | 0.345 | 3.2 / 9.4 |
| 30 | 0.170 | 1.6 / 9.4 |
| 60 | 0.103 | 1.0 / 9.4 |
| 100 | 0.069 | 0.65 / 9.4 |

The configured profile is therefore `C(z) = surface_solution_p_um * f(z)`.
Linearly interpolate every pair of knots from 0 to 100 cm; because the first
two factors are equal, this has the same value over 0--5 cm as the previous
surface treatment. Do not silently define it below 100 cm. The median decline
from 5 to 100 cm is 14.5-fold; individual sampled grasslands exhibited
approximately 15--34-fold declines. Qualification candidates should
consequently stop at 100 cm unless a separately evidenced sub-100-cm
extrapolation is later adopted.

## Mapping an assay profile to the model's P convention

There is no scientifically defensible scalar correction from the Brédoire
water-equilibration concentrations to `initial_solution_p_um`.  The model has
two separately consequential quantities.  For a bulk-soil cell of volume
`V`, it stores reversible labile P as

\[
M_{labile}=V(\theta+b_p)C,
\]

where `C` is the solution concentration seen by uptake kinetics, `theta` is
volumetric water content, and `b_p` is the linear, short-timescale buffer
power.  Changing `C` changes immediate root influx; changing `b_p` changes
the inventory that can replenish that solution and slows its apparent
diffusion.  Thus dividing the assay concentration by an arbitrary factor
would alter both the biological and transport interpretation incorrectly.

A calibration for a selected grassland soil should use intact or minimally
disturbed samples from each modelled horizon and proceed as follows:

1. Measure **native soil-solution orthophosphate** at field moisture and
   temperature (for example, sampled pore water), at the intended depth bands.
   Convert that concentration of elemental P directly to micromolar and set
   `C(z)` to it.  A 1:10 deionized-water equilibration is retained only as a
   useful prior/sensitivity bound when no native solution measurement exists.
2. At concentrations bracketing that `C(z)`, measure the exchangeable P
   released to solution over the time window the model treats as reversible.
   Isotopic-exchange/dilution data or a small-perturbation sorption--desorption
   experiment can provide the local slope
   `b_p(z) = dS/dC`, after expressing solid-associated `S` per bulk soil
   volume and `C` per water volume.  The chosen exchange duration must be
   reported: slow mineralisation and irreversible desorption are not the
   instantaneous linear buffer represented by the current model.
3. Measure `theta(z)` at the same water potential and fit the transport
   combination `D_l * theta * f_l / (theta + b_p)` to a depth-specific P
   tracer or depletion experiment.  This avoids trying to identify buffer
   power and impedance from plant uptake alone.
4. Validate the assembled profile against an independent carrot uptake or
   rhizosphere-depletion experiment.  Hold plant geometry and uptake traits
   fixed where independently known; otherwise report a joint fit and its
   uncertainty rather than calling it a soil calibration.

The current `b_p=239`, water content `theta=0.3`, and kinetic traits are a
literature-derived provisional parameter set, not measurements for the
Siberian grassland.  Until steps 1--4 are available, the honest use of the
five-band profile is as a named **assay-equivalent sensitivity scenario**,
not a calibrated field-soil default.  A useful interim experiment is to hold
the profile shape fixed and sweep a common concentration multiplier and
buffer power, then compare direct root uptake and depletion times.  That
identifies whether the apparent C limitation is robust to the measurement
convention, but does not itself calibrate it.

## DOI verification

Each DOI below was checked against the publisher's canonical article page or
publisher PDF metadata on 19 August 2026; title, authors, journal, year, and
DOI matched.

1. Achat, D. L. et al. (2013). [Phosphorus status of Siberian forest soils:
   effects of microbiological and physicochemical
   properties](https://doi.org/10.5194/bg-10-733-2013). *Biogeosciences*, 10,
   733--752. Primary depth-horizon soil-solution Pi measurements.
2. Brédoire, F. et al. (2016). [What is the P value of Siberian soils? Soil
   phosphorus status in south-western Siberia and comparison with a global data
   set](https://doi.org/10.5194/bg-13-2493-2016). *Biogeosciences*, 13,
   2493--2509. Primary forest/grassland profile measurements and global
   comparison.
3. Rinderer, M. et al. (2021). [Subsurface flow and phosphorus dynamics in
   beech forest hillslopes during sprinkling experiments: how fast is
   phosphorus replenished?](https://doi.org/10.5194/bg-18-1009-2021).
   *Biogeosciences*, 18, 1009--1027. Primary depth-specific forest flow and P
   replenishment experiment.
4. He, X. et al. (2023). [Global patterns and drivers of phosphorus fractions
   in natural soils](https://doi.org/10.5194/bg-20-4147-2023).
   *Biogeosciences*, 20, 4147--4163. Global, depth-aware model of Hedley P
   fractions for natural soils; it is evidence about labile pools, not
   soil-solution concentration.
5. Zhao, Q. & Zeng, D.-H. (2006). [Phosphorus fractions and
   phosphomonoesterase activities in sandy soils under a temperate savanna and
   a neighboring Mongolian pine plantation](https://doi.org/10.1007/s11676-006-0006-4).
   *Journal of Forestry Research*, 17, 25--30. Primary 0--20 cm natural
   semi-arid savanna depth comparison of P fractions.
6. Hou, E. et al. (2018). [A global dataset of plant available and unavailable
   phosphorus in natural soils derived by Hedley method](https://doi.org/10.1038/sdata.2018.166).
   *Scientific Data*, 5, 180166. The primary 802-soil, 99-study natural-soil
   compilation underpinning the earlier global Hedley-fraction data resource.

## Scope limits

- These are natural forest and grassland data, not carrot-field data. They
  define a plausible non-agricultural P environment, not a cultivar-specific
  validation target.
- The model presently uses a uniform linear P buffer. Natural horizon-specific
  sorption and replenishment are not represented by changing the initial
  solution concentration alone.
- A 5% qualification criterion for direct plant Pi uptake must compare the
  **integrated direct root uptake flux**, excluding fungal uptake and transfer,
  at matched policy, P profile, seed, and horizon. Final plant P pool or total
  soil depletion is not an equivalent endpoint.
