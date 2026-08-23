# Carbon limitation, plant growth rates, and high-phosphorus AM response

**Research date:** 13 August 2026  
**Scope:** Whether the surplus-carbon hypothesis implies that plant carbon is
never limiting; empirical growth and respiration anchors for auditing the
MycorMARL plant; evidence for a phosphorus-sufficient region in which an
established arbuscular mycorrhizal (AM) association is neutral or costly; and
whether an arbitrary external limiting-factor threshold is defensible.

## Executive answer

The surplus-carbon hypothesis does **not** establish that plants are never
carbon limited. It proposes that, in many environments, water, nutrients, or
temperature restrict growth more strongly or sooner than they restrict
photosynthesis, leaving recently fixed carbon that cannot immediately be used
for biomass production. The paper that formalized this framing presents it as
a hypothesis, not a universal physiological law
([Prescott et al. 2020](https://doi.org/10.1016/j.tree.2020.08.007)). A recent
review of carbon transfer to mycorrhizal fungi also concludes that existing
experiments do not cleanly distinguish surplus-carbon from alternative
exchange hypotheses and that deciding whether carbon is genuinely surplus
depends on which plant sinks have been measured
([Bunn et al. 2024](https://doi.org/10.1111/nph.20145)).

There is good evidence that plant growth eventually becomes insensitive to
additional P and that AM growth responses can become neutral or negative under
P-sufficient conditions. There is not, however, a universal soil-solution Pi
concentration at which this occurs. The transition depends on plant and fungal
genotype, plant age, tissue P demand, light, other nutrients, soil buffering
and sorption, rooting volume, and the duration and form of P supply.

The defensible sequence for MycorMARL is therefore:

1. Audit the model's carbon fixation, respiration, and windowed relative growth
   rate before trying to estimate an AM-favourability threshold.
2. Determine whether the present dynamics already produce a growth plateau
   and a sign change in the complete-mode contrast.
3. If growth remains implausibly exponential, add a named physiological
   constraint calibrated independently of the desired sign change.
4. Do not choose a hard external threshold merely to manufacture
   \(P_\mathrm{thresh}\). If a phenomenological ceiling is needed for an early
   sensitivity analysis, make it smooth, sweep a justified range, and do not
   interpret the resulting P threshold as empirically validated.

## What “surplus carbon” does and does not mean

Prescott et al. define surplus carbon as photosynthate that cannot be used for
growth because another resource or process constrains sink activity. That
claim is compatible with carbon being nonlimiting under a particular mild
nutrient or water limitation; it is not equivalent to carbon being nonlimiting
at every light level, developmental stage, or sink demand
([Prescott et al. 2020](https://doi.org/10.1016/j.tree.2020.08.007)).

Primary experiments show both source- and sink-limited regimes. In alfalfa,
CO2 enrichment and defoliation experiments indicated source limitation in
seedlings but a developmental shift toward sink limitation in older plants
([Baysdorfer & Bassham 1985](https://doi.org/10.1104/pp.77.2.313)). In Arabidopsis,
well-watered rosettes fixed about 300 mg carbohydrate equivalent (CHO-eq)
g\(^{-1}\) dry weight d\(^{-1}\); dark respiration used about 20% of daily
photosynthesis, estimated shoot-growth demand was 182 mg CHO-eq g\(^{-1}\)
d\(^{-1}\), and the residual carbon balance was about 15% of fixation. Water
deficit reduced growth demand more than fixation and increased that residual
to 24--30%
([Muller et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC2938159/)).
This is strong evidence for context-dependent surplus carbon, not evidence that
carbon supply can never constrain growth.

Light also changes the costs and benefits of AM. In onion, lowering photon
irradiance from roughly 550--600 to 250 µmol m\(^{-2}\) s\(^{-1}\) reduced AM
growth responses, while growth depressions occurred under both low irradiance
and the combination of high P with high irradiance
([Smith & Gianinazzi-Pearson 1988](https://pubmed.ncbi.nlm.nih.gov/33873939/)).
Thus a model in which carbon availability can never influence AM outcomes
would omit a documented interaction, even if fixed carbon is often in surplus
within the intended experimental regime.

## Growth-rate anchors

Relative growth rate (RGR) should be calculated on dry biomass over explicit
time windows:

\[
\mathrm{RGR}_{t_1,t_2}
=\frac{\ln B(t_2)-\ln B(t_1)}{t_2-t_1}
\quad [\mathrm{g\ g^{-1}\ d^{-1}}].
\]

Under controlled conditions with nonlimiting nutrient supply, 24 wild
herbaceous species had ontogeny-corrected mean RGRs of 113--356 mg g\(^{-1}\)
d\(^{-1}\), or 0.113--0.356 g g\(^{-1}\) d\(^{-1}\)
([Poorter & Remkes 1990](https://doi.org/10.1007/BF00317209)). These are useful
orders of magnitude for early vegetative growth under favourable conditions,
not rates that a plant should sustain for an entire 120-day life. RGR normally
declines with size and development, so comparing a 120-day average alone can
hide an unrealistic early acceleration or late exponential runaway.

For a model audit, report at least:

- daily or multi-day windowed RGR for total living plant biomass and each
  major plant partition;
- gross fixed C, plant respiration, C transferred to the fungus, biomass
  construction demand, and change in plant C pools in common C units;
- the fraction of gross fixed C spent on respiration and AM transfer;
- RGR by early, middle, and late life, rather than only final biomass; and
- doubling time, \(\ln(2)/\mathrm{RGR}\), wherever RGR is positive.

The Poorter--Remkes range cannot by itself validate a particular model plant.
A final benchmark should match the intended species, growth stage, temperature,
photoperiod, and dry-mass definition. It is nevertheless a useful warning
bound: indefinitely maintaining even 0.1 g g\(^{-1}\) d\(^{-1}\) would multiply
biomass by \(e^{12}\) over 120 days. Observed high juvenile RGR is therefore not
a justification for unconstrained whole-life exponential growth.

## Respiration anchors

Respiration measurements must not be mixed without unit conversion. Leaf gas
exchange is often reported as µmol CO2 m\(^{-2}\) s\(^{-1}\), tissue respiration
as µmol CO2 g\(^{-1}\) fresh or dry mass per unit time, and model maintenance
coefficients as glucose or C mass g\(^{-1}\) dry mass d\(^{-1}\).

A foundational synthesis of plant-cell maintenance costs reported estimated
components of roughly 7--13 mg glucose g\(^{-1}\) dry weight d\(^{-1}\) for
protein turnover and 6--10 mg glucose g\(^{-1}\) dry weight d\(^{-1}\) for
maintaining ion gradients. Measurements assembled by the same paper spanned
approximately 8--60 mg glucose g\(^{-1}\) dry weight d\(^{-1}\) at 25 °C,
depending on tissue, species, prior assimilation, and estimation method
([Penning de Vries 1975](https://doi.org/10.1093/oxfordjournals.aob.a084919)).
These values justify testing an explicit biomass-proportional maintenance sink,
but not selecting one universal coefficient without species and temperature
qualification.

In an integrated Arabidopsis carbon budget, daily dark respiration was about
20% of photosynthesis
([Muller et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC2938159/)). In a
separate Arabidopsis study, estimates of maintenance respiration varied with
day/night temperature from about 0.010 to 0.034 g C g\(^{-1}\) dry weight per
night across accessions and treatments
([Pyl et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3406903/)). The
variation reinforces the need to reproduce the experimental temperature and
time basis before comparing these numbers with a model parameter.

## Evidence for a high-P plateau and neutral or negative AM response

### Plant growth can plateau with internal P status

In wheat supplied with 11 soil-P rates, shoot biomass versus shoot P
concentration followed a linear-plateau model (\(R^2=0.96\)). Shoot biomass
reached its fitted plateau at 4.63 mg P g\(^{-1}\) shoot dry mass, while root
biomass levelled off at 3.00 mg P g\(^{-1}\)
([Shen et al. 2018](https://doi.org/10.1093/aobpla/ply054)). This demonstrates a
physiological precedent for estimating a critical internal P status from a
response curve. It does not provide a directly transferable external Pi
concentration: the experiment used soil additions of 0--1200 mg P kg\(^{-1}\),
and the fitted critical tissue concentration was growth-stage and genotype
specific.

In *Medicago truncatula*, nutrient solutions containing 0.0075 and 3.75 mM Pi
produced sharply different colonization (about 62% versus less than 2% of root
length), yet total biomass did not differ significantly between the P supplies
(\(P=0.606\)). High P therefore strongly suppressed the plant's capacity to
host AM while adding little total growth in that experimental system
([Balzergue et al. 2013](https://doi.org/10.3389/fpls.2013.00426)).

### An established AM association can be neutral or costly

In high-P sour orange, AM colonization was associated with 37% greater
root-plus-soil respiration; the nonmycorrhizal plants had about 20% greater
daily specific C gain. The authors attributed portions of the additional cost
to root construction, greater root biomass allocation, and maintenance or
growth of fungal tissue and extraradical hyphae
([Peng et al. 1993](https://doi.org/10.1104/pp.101.3.1063)). This is direct
evidence that a negative high-P AM growth response can arise through carbon
costs.

However, absence of a positive growth response does not mean that the AM
pathway is inactive. In wheat grown in calcareous, P-fixing soil, AM plants had
no positive and sometimes negative growth responses, while isotope tracing
showed that more than half of plant P uptake could still pass through the AM
pathway
([Li et al. 2006](https://doi.org/10.1111/j.1469-8137.2006.01846.x)). This
supports the phase-one qualification that the current sigmoid policy may
continue nonzero trade even where the complete AM association has no biomass
advantage.

The numerical P supplies from these experiments must not be treated as a prior
for MycorMARL's initial soil-solution µM threshold. Nutrient-solution
concentration, fertilizer added per soil mass, extractable soil P, tissue P,
and instantaneous soil-solution Pi are different quantities. Buffering,
sorption, replenishment, rooting volume, depletion, and exposure duration all
affect their relationship.

## Current MycorMARL scale audit

The default plant fixes carbon at

\[
k_{leaf}a_{mass}=0.30\times0.05=0.015
\quad\mathrm{g\ C\ g^{-1}\ plant\ DM\ d^{-1}},
\]

and pays standing maintenance of `0.007 g C g^-1 DM d^-1`. The remaining
`0.008 g C g^-1 DM d^-1`, divided by the structural carbon cost
`gamma_c = 0.402`, gives a maximum **sustained carbon-only** RGR of about
`0.0199 d^-1`, or a 34.8-day doubling time, before reproduction or fungal
transfer. These values follow directly from
[`PlantTraits`](../../mycormarl/mycormarl/plant/traits.py) and
[`photosynthesise`](../../mycormarl/mycormarl/plant/photosynthesis.py).

That ceiling is below the `0.113--0.356 d^-1` favourable-condition juvenile
herb range above, so the default fixation rate alone does not indicate
excessively rapid sustained growth. It is not a validation of the complete
trajectory. Fixation, maintenance, and root uptake all scale with current
biomass, the model lacks growth respiration and ontogenetic decline, and each
initial free-resource pool contains one structural-biomass equivalent. A
growth-heavy policy can therefore produce a large start-up pulse before
settling into biomass-proportional exponential growth. Windowed RGR and the
initial-pool contribution must be reported separately.

A subsequent species-matched audit shows that this ceiling is nevertheless
too low for a favourable 120-day cultivated-carrot scenario. Fitted `Forto`
whole-plant RGR declined from approximately `0.066` to `0.042 d^-1` across
successive 20-day windows between 40 and 120 DAS, reaching `23.26 g` measured
shoot-plus-storage-root DM. The historical mismatch was associated primarily with the
fixed `kleaf=0.30`, not evidence that the leaf-level `amass=0.05` should be
doubled. See
[`carrot-growth-biomass-cap-and-carbon-fixation.md`](carrot-growth-biomass-cap-and-carbon-fixation.md).

The selected `50 g` biomass guard is not a physiological maximum, but it is
unlikely to create the 120-day transition from P limitation in the default
initial condition. Even
an idealized plant that converts both initial resource pools immediately into
one additional biomass equivalent and then devotes all net fixed carbon to
growth reaches only about `0.22 g` after 120 days under the calculation above.
The cap must still be checked against realized trajectories: contact fails
qualification rather than establishing a physiological plateau.

As a local diagnostic, the production uptake equations were evaluated at the
default initial `0.01 g` plant biomass and full `50 x 100 cm` P-bearing domain,
with no fungus and no preceding depletion. Direct root uptake was approximately
`0.0058`, `0.0175`, `0.0580`, and `0.1729 mg P g^-1 DM d^-1` at `0.1`, `0.3`,
`1.0`, and `3.0 micromolar` initial solution P. Carbon-balanced maximum growth
requires

\[
0.008\,\gamma_P/\gamma_C + \kappa_P
\simeq 0.0392\ \mathrm{mg\ P\ g^{-1}\ DM\ d^{-1}}.
\]

The initial direct-uptake crossover is therefore near `0.7 micromolar`, inside
the pilot grid. This is evidence that the implemented dynamics can move from
P-limited toward C-limited growth without an added resource. It is **not** an
estimate of the association-favourability threshold: reserves, depletion,
reproduction, fungal competition and transfer, geometry, and learned actions
all affect the episode-level contrast.

## Implications for the two experimental phases

### Phase 1: association response across uniform P

Phase 1 can still ask whether the learned benefit of an established AM
association increases with P limitation. It should also diagnose whether the
plant-only response itself saturates. A credible transition to phase 2 would
require more than a noisy sign change between two pilot points:

- the P grid spans a region in which plant-only growth or fitness becomes
  weakly responsive to further P;
- the mixed-minus-plant-only contrast changes sign, with uncertainty small
  enough to localize a bracket;
- the direction is reproduced across paired training seeds and is not caused
  by one failed or unconverged run;
- carbon balance and windowed RGR remain physiologically plausible on both
  sides of the bracket; and
- the sign change is stable to reasonable evaluation and domain-size checks.

Failure to find a bracket would be ambiguous if neither mode reaches a
P-sufficient growth plateau. It could mean that the candidate grid is too low,
that the finite soil inventory is depleted over the season, or that the plant
dynamics lack a non-P growth constraint. Those alternatives should be
distinguished before concluding that no AM-favourability threshold exists.

### Phase 2: refine an association-favourability threshold

If phase 1 produces a credible bracket, phase 2 can refine the zero of

\[
\Delta_{\mathrm{AM}}(P)
=\mathbb{E}[Y_{\mathrm{mixed}}(P)]
-\mathbb{E}[Y_{\mathrm{plant-only}}(P)].
\]

This remains a threshold for the favourability of the **complete consumer
mode**, not proof that a mixed-mode policy learns exact zero trade. The
literature above also cautions that the inferred threshold belongs to the
specified plant--fungus--environment model; it is not a universal soil Pi
threshold.

## Should an arbitrary external limiting-factor threshold be imposed?

Not as the primary solution. Choosing a ceiling specifically so that AM becomes
unfavourable above a desired P level would make the conclusion circular. A
hard switch would also introduce a discontinuity into both the biological
response and the policy-learning landscape.

If the carbon audit shows implausible growth, the preferred response is to add
the smallest independently defensible mechanism that is actually missing. In
increasing order of conceptual expansion, candidates include:

1. refine the existing biomass-proportional maintenance respiration and add
   explicit growth-respiration costs;
2. saturating carbon fixation based on finite photosynthetic area, light, or
   canopy/self-shading capacity;
3. ontogenetic decline, tissue turnover, senescence, or a finite seasonal sink;
4. explicit nitrogen or water limitation, only if those resources are within
   the scientific scope and can be parameterized.

Parameterization should use data that were not selected to obtain a particular
AM sign crossing. The revised model should then be checked against growth and
respiration observations before rerunning the P experiment, allowing
\(P_\mathrm{thresh}\) to emerge.

If a temporary phenomenological ceiling is unavoidable, it should be a smooth
saturating function rather than a hard threshold, predeclared as a sensitivity
scenario, and varied over a literature-bracketed range. Results from that
scenario can show whether the research conclusion depends on a plausible
growth ceiling; they cannot validate the ceiling or the resulting numerical
P threshold.
