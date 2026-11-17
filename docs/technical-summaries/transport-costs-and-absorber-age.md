# Internal transport costs, absorber density, and tissue age

> Research note, 27 October 2026. Primary literature reviewed for a possible model extension; the equations below are proposals, not fitted biological laws. Current-code integration is discussed separately below. This note does not implement a change.

## Recommendation from the literature

Represent three different processes: soil-to-absorber uptake, movement through living transport infrastructure, and carbon expenditure supporting that infrastructure and its fluxes. Length density can increase both absorbing surface and axial transport capacity, but these are different consequences with different dependencies. Distance should refer to a transport route: root-to-stem for plant transport, and hypha-to-connected-colonised-root for fungal transport. Nearest geometric root distance is a useful approximation only under an explicit attachment assumption.

The literature supports these distinctions more strongly than any particular universal equation. A mechanistic AM uptake model already separates soil removal, internal mycelial transport, and transfer into root cells; under its parameters internal transport was fast relative to uptake, so an additional strong distance restriction would change that model regime rather than merely complete it. Its internal diffusion, streaming and interface coefficients were uncertain. [Schnepf & Roose (2006), model and parameter discussion](https://doi.org/10.1111/j.1469-8137.2006.01771.x).

An explicit carbon cost and a flux-capacity limit answer different questions. The first asks how much carbon is spent when a given flux is realised; the second asks how much can arrive through the retained infrastructure. Treating them separately allows oldest-first maintenance mortality to create the proposed feedback: less retained living infrastructure can save maintenance while constraining delivery from distal tissue. Whether that trade-off is important is a model hypothesis to test.

## Primary evidence and its limits

| Primary source | Relevant observation or result | What it supports here; limits |
|---|---|---|
| [Schnepf & Roose 2006, New Phytologist 171:669–682](https://doi.org/10.1111/j.1469-8137.2006.01771.x) | Separates uptake by external hyphae, internal transport and root transfer; represents mycelial volume using hyphal radius and length density. | Density can enter a continuum transport representation; this does not identify energetic expenditure per unit distance. The fast-transport conclusion is conditional on its parameters. |
| [Oyarte Galvez et al. 2025, Nature 639:172–180](https://doi.org/10.1038/s41586-025-08614-x) | Root-organ cultures revealed bidirectional cytoplasmic streams, wider and faster central transport routes, and approximately maintained normalised graph-theoretical efficiency of routes to roots as networks expanded. This is a shortest-path network metric, not measured carbon efficiency or nutrient throughput. | Architecture and flow adjustment matter; total length or Euclidean distance alone cannot determine capacity or cost. These were constrained, effectively two-dimensional cultures, not field soil measurements. |
| [Kikuchi et al. 2016, New Phytologist 211:1202–1208](https://doi.org/10.1111/nph.14016) | Suppressing host transpiration and silencing a fungal aquaporin slowed polyphosphate translocation in *Rhizophagus clarus*. | Supports a water-flow contribution to directional transport. It does not establish that all nutrient movement is paid for through fungal active pumping proportional to route length. |
| [Uetake et al. 2002, New Phytologist 154:761–768](https://doi.org/10.1046/j.1469-8137.2002.00425.x) | Describes an extensive motile tubular vacuole system in *Gigaspora margarita*. | Identifies relevant intracellular structure, without independently calibrating nutrient flux or energetic cost along it. |
| [Olsson et al. 2002, Plant Physiology, “Phosphorus Effects on Metabolic Processes in Monoxenic Arbuscular Mycorrhiza Cultures”](https://pmc.ncbi.nlm.nih.gov/articles/PMC166637/) | Tubular vacuoles occurred in live, streaming hyphae; streaming was bidirectional. High-P treatment included more empty hyphae and slower streaming. | Live length and physiological state matter, not merely visible wall length. Culture treatment also affects physiology, so the result cannot isolate age effects. |
| [Kameoka et al. 2019, Plant and Cell Physiology 60:2272–2281](https://doi.org/10.1093/pcp/pcz122) | Structure-specific transcriptomes distinguish runner hyphae, branched absorbing structures (BAS), spores and other structures, including nutrient-related expression. | A common “all living hyphae have identical uptake and conduction” approximation needs acknowledgement. Transcript expression is not a measured transport conductance. |
| [Bago et al. 1998, New Phytologist 139:375–388](https://doi.org/10.1046/j.1469-8137.1998.00199.x) | BAS in monoxenic *Glomus intraradices*–tomato cultures developed over about seven days and subsequently became empty, septate structures. | Supports turnover of a particular fine structure, not a universal seven-day death timer for runner hyphae or an entire network. |
| [Staddon et al. 2003, Science 300:1138–1140](https://doi.org/10.1126/science.1084269) | Radiocarbon microanalysis was interpreted as average turnover of most sampled external hyphae over five to six days. | Useful turnover evidence in the studied system. Carbon-label turnover is not a direct age-specific hazard or measurement of loss of axial conducting capacity. |
| [Volder et al. 2005, New Phytologist, “Rapid decline in nitrate uptake and respiration with age in fine lateral roots of grape”](https://doi.org/10.1111/j.1469-8137.2004.01222.x) | Nitrate uptake and respiration declined rapidly with the age of intact grape lateral roots. | Living root function can vary with age before death. This is nitrate and a particular grape rootstock; it does not supply a phosphate-age curve or show that old axial conduits are nonfunctional. |
| [Doussan, Vercambre & Pagès 1998, Annals of Botany 81:225–232](https://doi.org/10.1006/anbo.1997.0541) | A hydraulic architecture model required spatially varying axial/radial conductances to reproduce maize flow profiles. | Absorption into roots and conduction along roots are distinct processes. Hydraulic results do not directly quantify a carbon cost for axial phosphate transport. |
| [Meunier et al. 2017, Journal of Mathematical Biology, “Towards quantitative root hydraulic phenotyping”](https://doi.org/10.1007/s00285-017-1111-z) | Derives whole-root conductance and uptake distributions from radial and axial properties; adding branch length has diminishing conductance returns. | More total length need not translate into proportionately greater whole-system throughput; topology and axial bottlenecks matter. |

The 2025 Nature results also show that local flow is bidirectional rather than a one-way conveyor. A capacity model based on **net** P delivery is an aggregate closure, not a literal description of every internal stream. Carbon provision, fungal growth and storage also make transport demands. Fluxes of different resources should not simply be added without unit conversion and a definition of shared capacity. [Oyarte Galvez et al. 2025](https://doi.org/10.1038/s41586-025-08614-x).

## What distance and density mean

Soil diffusion distance to an absorbing membrane is different from the distance travelled internally after uptake. The former is already part of a rhizosphere/depletion closure. Increasing density reduces typical soil access distances and increases membrane area; neither effect alone estimates the ability to move the absorbed material back to a host. [Schnepf & Roose 2006](https://doi.org/10.1111/j.1469-8137.2006.01771.x).

For an ideal population of cylindrical absorbers of radius \(r\), length density \(\lambda\) [length per soil volume] gives lateral membrane area per soil volume \(2\pi r\lambda\) and cylinder volume fraction \(\pi r^2\lambda\). These are geometric identities, not evidence that uptake transporters, cytoplasmic streaming area and xylem conduits are proportional in the same way. The latter additionally depends on alignment, living connectivity, radius distribution, maturation and the fraction of tissue serving transport. In particular, membrane transporter capacity should be attached to membrane area and kinetics, whereas axial capacity belongs to the through-path.

For hyphae, use distance to a root segment that belongs to the same established association, rather than indiscriminately choosing any nearby root. For plant delivery, attach the root-to-stem leg to that receiving root. A longer fungal leg might therefore permit a shorter plant leg, or vice versa. These are proposed bookkeeping rules; the available experiments do not justify a universal inverse-density nearest-root distance law for this model's geometry.

An illustrative continuum path resistance is

\[
\mathcal R_p=\int_p\frac{ds}{k(s)A_{\rm eff}(s)},\qquad
F_p=\frac{\Delta\Psi_p}{\mathcal R_p}.
\]

Here \(F_p\) is resource delivery [amount/time], \(\Psi\) is a specified driving potential, and \(kA_{\rm eff}\) must have units [amount × length / (time × potential)]. This is a proposed analogy, with linear response assumed; it is not an AM-fungal transport equation established by the cited work. It explains why an interior minimum can dominate: averaging local density arithmetically can hide a narrow conducting bottleneck. Parallel connected routes must be combined as parallel capacities, not as one sequence of resistances. A graph-free model needs to say explicitly how it approximates this.

For an economical carbon-cost approximation, a candidate is

\[
\dot C_{\rm transport}
=\sum_iF_i\left[\epsilon_{0,i}+\epsilon_{d,i}\bar d_i\right],
\]

where realised \(F_i\) is [resource amount/time], \(\epsilon_{0,i}\) is [carbon/resource amount], and \(\epsilon_{d,i}\) is [carbon/(resource amount × length)]. Root and hyphal coefficients can differ. This is a transparent phenomenological baseline, **not** a fitted law; it should be compared with zero-distance-cost and capacity-only alternatives. It has no congestion effect unless one is added deliberately. Hydraulic resistance or slower arrival does not by itself identify how much extra respiratory carbon must be debited.

## Consequences for oldest-first maintenance mortality

The agreed project concept is **oldest-first maintenance mortality**, not intrinsic age-dependent senescence: maintenance deficits choose the oldest eligible living tissue first. Retained density floors remain living and uptake-active, with front-only absorber growth. Research evidence about tissue age should inform future extensions without silently changing that agreement.

Several distinct mechanisms are possible:

1. **Loss of uptake surface:** death reduces live length and therefore the represented membrane area. Existing uptake calculations should use the post-mortality live-density field at the model's chosen process boundary.
2. **Loss of through-capacity:** interior thinning can constrain material acquired by younger distal absorbers even if their uptake remains high. This is the main additional transport hypothesis worth testing.
3. **Physiological ageing without death:** uptake activity and conducting capacity could have different age weights. The root nitrate evidence and fungal structure differences motivate distinguishing them; neither provides the coefficients for the present P model.
4. **Connection loss:** a root or hyphal corridor may become disconnected. A positive density floor can be a continuity assumption, but is not evidence of adequate throughput or of an AM attachment at its root endpoint.

Thus it is defensible initially to use live density for both functions while naming that assumption, and test separate uptake-active and conducting fractions later. It would be unjustified to equate old tissue with dead tissue, or to make every old root/hypha equally poor at uptake and axial movement. Likewise, empty fungal wall remnants should not retain active cytoplasmic transport capacity merely because their geometric length remains visible; the living-hypha observations are the relevant physiological qualification. [Olsson et al. 2002](https://pmc.ncbi.nlm.nih.gov/articles/PMC166637/).

Oldest-first thinning can produce a tension between preserved distal exploration and declining interior supply. However, a model with no replacement inside the historical front can also accumulate permanent bottlenecks; that is a consequence of the chosen growth rule, not a finding that all biological root–fungal systems must behave that way.

## Calibration and falsifiers

The papers above do not provide a transferable pair of coefficients for carbon per P per centimetre in this organism configuration. Required evidence would ideally jointly measure resource uptake, host delivery, retained live geometry and additional respiration while varying path distance and transport density independently. Without this separation, growth respiration, tissue maintenance, membrane uptake and transport can be confounded.

Before assigning a new operating-carbon debit, audit the existing maintenance parameters: a respiration-derived operating coefficient may already contain active nutrient transport. Adding the full new estimate on top risks double counting. A defensible comparison either repartitions the existing operating budget or explicitly treats the added cost as a sensitivity experiment.

Useful model experiments are distance-only cost; capacity-only limitation; combined cost and capacity; and the same alternatives under memoryless versus oldest-first mortality. Strong support for the transport extension would require robust biological conclusions across plausible coefficients, rather than a particular hand-picked resistance. Reject or revise a closure if it predicts delivery from zero conducting density, creates or destroys P during a capacity cap, rewards disconnected absorbers, or changes total energetic burden when a uniform route is subdivided into more numerical cells.

## Evidence-access qualification

The source search used publisher articles, author institutional repositories and primary-paper records in PubMed/PMC. Some publisher pages were intermittently inaccessible; table claims for those sources are restricted to the primary paper's indexed abstract or accessible text. Unverified secondary summaries were used only to locate primary sources and are not evidence for the proposed coefficients. No universally applicable distance–density–age transport-cost law was identified in this search.

## Repository-specific integration assessment

This section is a proposed modelling design, derived for this repository; its equations are **not fitted physiological laws**.

### Current model versus agreed mortality extension

The inspected production [environment](../../src/mycormarl/environments/base_mycor.py) pays maintenance, integrates coupled trade/growth/reproduction, adds photosynthate, rebuilds biomass-derived geometry, then advances soil diffusion/uptake. Its [state](../../src/mycormarl/state.py) has organism-wide free C/P pools and root/hyphal density grids, with no local internal transport inventory, attachment graph, or cohort state. The [allocation integrator](../../src/mycormarl/allocation.py) makes received trade immediately available during allocation; soil uptake becomes eligible next numerical step. The [soil uptake closure](../../src/mycormarl/soil/phosphate_uptake.py) handles soil supply and membrane entry, not axial transport to the stem or a root partner.

The earlier discussion was retrieved from the chat “Assess time-varying uptake and age” and checked against [specification #89](https://github.com/isaachcamp/mycormarl/issues/89). It agrees front-only growth, oldest-first **maintenance-deficit** mortality, no spontaneous age death, live uptake-active interior density floors, no interior repair, and provisional front retreat after thinning capacity is exhausted. Cohort order is enough for that mortality rule; absolute age would be needed for a separate age-dependent physiological law. This is planned work, not present production behaviour. [ADR-0004](../adr/0004-derive-uptake-geometry-from-current-biomass.md) and [ADR-0017](../adr/0017-represent-absorbers-as-fine-root-discs-and-an-external-hyphal-hemisphere.md) still describe current memoryless geometry. A transport extension using persistent thinning must coordinate with #89 and explicitly update those decisions when implemented.

### Two distances, with explicit attachment assumptions

Let the stem/root collar be at radial/depth coordinates `(0,0)`. A coarse root path to a soil location is

\[
\ell_p(r,z)=\tau_p\sqrt{r^2+z^2},\qquad \tau_p\ge1.
\]

This is a tortuosity-scaled lower-bound geometry, not a reconstructed root trajectory. `tau_p(r+z)` is an alternative vertical-then-lateral route to test. A depth-only distance misses radial exploration; a radius-only distance misses deep transport. The model's fine-root discs do not resolve the axial root backbone, and a collar-to-cell distance excludes any aboveground stem path.

For fungus, the relevant endpoint is an **exchange-capable root connected to that fungal network**, not automatically the nearest root. Root and fungal overlap does not establish colonisation. One possible coarse closure is

\[
\ell_h(x)=\tau_h\left[\delta_{\rm root}(x)
+\frac{b}{\sqrt{\lambda_{\rm attached}(x_*)}}\right],
\qquad
\lambda_{\rm attached}=\chi\lambda_{\rm root,live},
\]

where `delta_root` is the shortest geometric gap to an eligible occupied root envelope, `x_*` is a chosen endpoint/neighbourhood within that envelope, `chi` is an assumed attached fraction, and `b` is a dimensionless spacing coefficient. The inverse square-root scaling follows dimensional reasoning for a spatial line density (cm/cm³ = cm⁻²); its coefficient and additivity are hypotheses, not a measured mean distance. Dense occupied cells still contain gaps between individual roots; treating the entire root envelope as a continuous exchange surface would assign spurious zero distances. Conversely, a partially occupied grid cell must not acquire long paths just because its volume-averaged density is low. Use within-occupied-region density and front fractions separately, and test grid refinement.

No eligible connected roots means no host delivery through this pathway; do not substitute an arbitrary small positive density that creates a functioning link. In fungus-only experiments, host delivery is absent, but fungal transport toward its own growing front may still exist. Instant reassignment to the nearest surviving root after death would be a strong rewiring assumption; fixed attachment or finite reconnection time is a separate design choice. A gap across empty soil also cannot be crossed by intracellular transport without a surviving hyphal route.

For a mean distance diagnostic, weight by the relevant realised flux:

\[
\bar\ell_{o,P}=\frac{\sum_i F_{o,P,i}\ell_{o,i}}{\sum_i F_{o,P,i}}.
\]

This exactly aggregates a linear per-unit-distance cost; nonlinear costs require summing local costs before averaging. Zero flux has zero cost and an undefined (masked) mean distance. Carbon delivery and P return can have different spatial weights. Fungal P reaches the plant at a root endpoint, then follows that endpoint's root-to-collar path; charging the full stem distance to both partners would duplicate a route unless explicitly intended.

### Length density as a capacity proxy

For fixed conduit radius `a_o`, a useful first approximation is the tissue cross-sectional fraction

\[
\phi_o=\pi a_o^2\lambda_{o,\rm live},\qquad
K_o=D_{\parallel,o}\,\phi_o\,m_o,
\]

where `phi` is dimensionless, `D_parallel` has cm²/day, and `m` is a dimensionless orientation/connectivity factor. For an internal concentration `c` in mg P/cm³ tissue, a diffusion-like homogenised closure has bulk flux density `j=-K grad(c)` in mg P/cm²/day. This is an effective conductance model; AM cytoplasmic streaming and root xylem/phloem require different constitutive laws and parameters if represented mechanistically. The root absorbing radius is not its xylem lumen radius. Substituting the former into a Poiseuille formula would be unjustified.

A route with cross-sectional area `A(s)` has effective resistance

\[
\mathcal R_o=\int_{\rm route}\frac{ds}{K_o(s)A(s)},
\qquad
F_o=\frac{\Delta c_o}{\mathcal R_o}.
\]

Here `R` has day/cm³ and `F` mg P/day. Specify a finite exchange/collar boundary rather than integrating an ideal radial point singularity. This equation makes the desired dependencies explicit: longer routes increase resistance, while greater surviving conducting density decreases it. It is a proposed effective law, not evidence that phosphate actually moves by molecular diffusion over the full path. A transport-capacity interpretation also needs an admissible concentration gradient or other driving-force limit; resistance alone is not a maximum flux.

Most importantly, **resources from many distal cells share the interior route**. Independent local limits can let every cell spend the same capacity. For a coarse radial implementation, impose a summed distal demand across each shared shell/cut and compare it with that cut's capacity. Hemispherical fungal shells and per-depth root disc shells offer compact approximations, but root exchange is distributed across space rather than concentrated at the hemispherical centre. A single central fungal sink is an explicit alternative geometry, not equivalent to nearest-root delivery. Root-layer convergence toward the collar also needs a common axial capacity or a declared unlimited-backbone approximation.

### A deliberately simple energetic hypothesis

If an explicit carbon charge is required before detailed physiological calibration, test

\[
\dot C_{\rm transport,o}
=\sum_i F_{P,o,i}
\left(e_{0,o}+e_{\ell,o}\ell_{o,i}\right),
\]

with `F_P` in mg P/day, `e_0` in g C/mg P and `e_ell` in g C/(mg P cm). Treat the distance term as a **sensitivity hypothesis**. `e_0` must identify which membrane loading/unloading costs it covers. Add a separate term for actual C delivery if modelled: `F_C * e_C`, with `e_C` in g C respired per g C delivered, optionally distance-dependent. Do not equate C delivery with contemporaneous P export: the fungus also needs C for its own maintenance and front growth when it sends no P.

Put density dependence primarily into capacity/resistance. Multiplying the energetic cost by inverse density as well introduces another physiological assumption; it does not follow automatically from having fewer conduits. At unchanged flux, thinning might reduce throughput, increase delay, require greater driving forces, or force rerouting; which of these consumes extra C depends on mechanism. Compare a capacity-only extension, an energy-only extension, and their combination to identify which assumption drives outcomes.

### Mortality implications and what a floor does not guarantee

Within the fixed-radius linear-capacity proxy, thinning a shared route to fraction `f` gives capacity `f` times baseline and resistance `1/f` times baseline at unchanged geometry. Thus #89's root floor at 20% implies fivefold resistance, and the fungal floor at 5% implies twentyfold resistance. These are algebraic consequences of the proxy, **not empirical predictions**. Under the root-spacing distance hypothesis, thinning attached root density to 20% increases the spacing term by `sqrt(5)` (about 2.24), even where the root envelope has not retreated.

Oldest-first interior thinning can therefore preserve a distant young uptake front while reducing the capacity needed to supply it and return its nutrients. Maintenance demand falls with lost tissue, so total C cost need not increase monotonically: savings in standing respiration compete with transport limits. Retreat can shorten paths and reduce distal demand, but it also removes uptake and growth territory. Front-only regrowth preserves the thinned bottleneck because it does not repair the interior.

An extra multiplier such as `g(age)=exp(-age/tau)` on conductivity should **not** be added merely because old tissue dies first. That would be an additional ageing mechanism with its own evidence and calibration. For roots especially, the model's living absorbing length is not a measurement of vascular conducting capacity. Losing absorptive function is not automatically loss of the root backbone. Keep that coupling explicit or separate functional absorbing and conducting fractions in a later extension.

### Conservation and update timing required before implementation

Two coherent levels of approximation are available:

1. **No local storage:** accept only uptake that can be transported and energetically funded, and deduct only that accepted amount from soil. This treats uptake and delivery as one transaction; it cannot represent nutrient stored locally while waiting for transport.
2. **Local internal storage:** remove membrane uptake from soil into a local fungal/root inventory, then transport it conservatively into the organism-wide available pool. Include local stocks in the P ledger and declare their fate on tissue death. This supports delay and backlog but introduces materially more state.

Do not remove full potential uptake from soil, clip delivered uptake, and lose the difference. Likewise do not allow global pools to fund growth before a resource is delivered. Preserve or explicitly revise the current next-step eligibility of new soil P. Transport C expenditure must be limited by available C before flux is committed; an after-the-fact negative-pool correction is not a conservative transaction. Trade and biology already compete within allocation, so adding trade-related transport C changes that integrator's resource balances. Uptake-related transport C paid during the later soil stage has different timing. These choices require an explicit scheduling decision rather than a second call to the existing maintenance routine.

Current fungal `kappa_c` is based on **total external-mycelium respiration**, described as a broader operating-respiration proxy in the [parameter register](../model-parameter-register.md). Adding explicit uptake/transport respiration on top can double count that measured budget. Repartition or recalibrate the baseline, or report the new term as an intentionally additional sensitivity. Construction carbon, standing maintenance, and flux-dependent transport should have distinct diagnostics.

### Recommended next modelling experiment

Use #89's surviving spatial densities, then compare capacity-only, cost-only and combined closures under matched growth–stress–recovery trajectories. Include equal total biomass with different interior thinning; identical local front density but different interior bottlenecks; root thinning without envelope retreat; and shorter versus longer routes at fixed represented tissue investment. Keep construction/maintenance economics visible when changing length density, because density also changes exploration volume and soil uptake.

Acceptance checks should include exact P accounting; nonnegative C; no delivery across a broken pathway; summed demand at shared cuts; zero transport cost at zero flux; appropriate behaviour without a host; the unconstrained/zero-extra-cost baseline limit; physical-timestep and grid refinement; and no cohort-by-timestep output history. Parameters without matched measurements should be sensitivity ranges, with results described as conditional model predictions. This research does not establish fitted cost coefficients or a validated replacement for the provisional density floors.
