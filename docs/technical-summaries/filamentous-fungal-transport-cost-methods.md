# Methods for modelling transport costs in filamentous fungi

Research note, 16 November 2026. Scope: internal movement through fungal tissue, especially carbon delivery to growth. Soil diffusion, membrane uptake and plant–fungus exchange are separate processes. This is a comparison of alternatives, not an implementation decision or a systematic review.

The strongest direct starting points for an **energetic transport charge** are hydraulic dissipation on an explicit network and the reduced energy-budget model of Heaton et al. A simple distance charge is easier to use in MycorMARL, but its coefficient and functional form remain hypotheses. Diffusion/advection models answer where resources go and how quickly; they need an additional energetic law to predict respiratory expenditure. The pros, cons and recommended uses below are modelling assessments, distinguished from the cited experimental findings.

## What “cost” means

Keep four outputs separate:

1. **Operating expenditure:** energy or respired carbon used while moving material.
2. **Infrastructure expenditure:** carbon invested in constructing and maintaining hyphae, cords, membranes and transport machinery.
3. **Delivery constraints:** finite throughput, transit time, losses and inaccessible sinks.
4. **Architectural proxies:** path length, resistance or graph efficiency used to compare networks.

A long path can score poorly in a graph metric without demonstrating a particular ATP expenditure. Likewise, a slow delivery process need not consume more carbon per unit delivered. These distinctions matter when coupling transport to resource allocation.

## Comparison at a glance

| Method | Quantity calculated | Principal advantage | Principal limitation | Best use |
|---|---|---|---|---|
| 1. Effective distance charge | Carbon spent per amount delivered and route distance | Very small state and computational burden | Uncalibrated constitutive law; shared-flow effects absent | Sensitivity analysis of distance profitability |
| 2. Reduced growth-energy budget | Growth-dependent energetic transport term | Published fungal cost model; nonlinear cost of rapid growth | Strong aggregate geometry assumptions; saprotrophic setting | Comparing growth–transport tradeoffs without a network |
| 3. Hydraulic resistor network | Flows, pressure drops and viscous dissipation | Represents shared routes, parallel paths and conducting anatomy | Requires topology, hydraulic parameters and energy conversion | Mechanistic energetic effects of network architecture |
| 4. Advection–diffusion–delivery model | Resource concentrations, arrival and sink delivery | Separates fluid movement from nutrient delivery | Transport equation alone supplies no metabolic cost | Lag, storage, retention, uptake versus delivery |
| 5. Vacuolar compartment diffusion | Bidirectional redistribution through connected vacuoles | Measurable compartment connectivity and diffusion | Marker-specific evidence; organelle operating costs unresolved | Shorter-scale solute redistribution and phosphorus hypotheses |
| 6. Motor/cargo kinetics | Cargo runs, reversals, binding and ATP budget if added | Connects costs to a named molecular mechanism | Many parameters and uncertain mapping to bulk nutrient flux | Vesicles, endosomes and local delivery near tips |
| 7. Network material/efficiency tradeoff | Construction volume, least-resistance paths, robustness | Useful with image-derived network geometry | Proxies do not establish operating respiration | Comparing fungal architectures and investment strategies |

Sources and equations for these categories follow. Categories overlap: a hydraulic model can supply velocities to a delivery model, and either can be coupled to construction costs.

## 1. Effective distance-dependent carbon charge

A proposed coarse-grained closure is

\[
\dot C_{\mathrm{tr}}=\sum_i \dot m_i\,[a+b\,d_i^p].
\]

Here `m` is the amount actually delivered, `d` is its source-to-sink route distance, `a` is an optional loading/unloading charge and `b` prices distance. With carbon cargo, `m` and expenditure can both be g C; for `p=1`, `b` has units cm⁻¹. This equation is an illustrative effective model, **not an empirically established fungal law identified by this search**. Setting `a=0, p=1` makes expenditure proportional to amount-distance. A quadratic distance dependence is a different hypothesis, not automatically equivalent to Method 2.

**Pros:** transparent; cheap; readily separates distances for different cargo and sinks; compatible with a non-limiting-capacity assumption. **Cons:** cannot infer pressure, throughput or arrival time; nearest-root Euclidean distance can understate a tortuous living route; a fitted coefficient can absorb several mechanisms and fail outside the fitted conditions. If nonlinear distance is used, evaluate it route by route: `mean(d²)` differs from `mean(d)²`.

**Closest published precedent (follow-up search):** Boswell's *Modelling mycelial networks in structured environments* charges substrate proportional to the amount actively translocated between neighbouring lattice elements (`c4`, p. 1018). This supports local movement-proportional expenditure, but does not directly establish the whole-route law `b m L`. Deriving that law requires additional assumptions about repeated equal-length transfers, approximately conserved cargo and constant cost per transfer. Its separately distance-proportional tip-extension cost concerns growth, not internal cargo travel. [Boswell 2008](https://doi.org/10.1016/j.mycres.2008.02.006), [full text, pp. 1017–1018](https://www.davidmoore.org.uk/21st_Century_Guidebook_to_Fungi_PLATINUM/REPRINT_collection/Boswell_modelling_mycelial_networks2008.pdf).

**Study selection, 17 November 2026:** use the linear form `a=0, p=1` initially,
assuming constant effective velocity across routes, times and delivery rates.
Velocity is unresolved and absorbed into the coefficient. Constant expenditure
per unit amount-distance is a separate assumption; fixed velocity alone does
not prove it. The coefficient remains uncalibrated. Keep the energetic law
replaceable independently of geometry and shared resource accounting; see
[ADR-0021](../adr/0021-treat-internal-transport-capacity-as-non-limiting.md).

**Calibration:** hold cargo demand, growth, biomass and environmental conditions as comparable as possible while varying the length of connected routes. Estimate the incremental respiratory carbon associated with delivery. Until such measurements exist, report a coefficient sweep including zero. A hypothesis test should also compare linear and nonlinear forms, not only different coefficient values.

## 2. Reduced growth-dependent energetic model

Heaton, Jones & Fricker's *Energetic Constraints on Fungal Growth* explicitly includes transport in a saprotrophic energy budget. Their Eq. 1 contains the normalized power term

\[
P_{\mathrm{tr}}=V\beta g^2\left[\frac{\lambda^2}{f}+\frac{(1-\lambda)^2}{1-f}\right].
\]

`V` is fungal volume, `g` gross specific growth rate, `f` the fraction contacting resource, and `λ` the fraction of new growth encountering resource. Energy is normalized to fungal energy per unit volume; physical power requires that energy-density factor. `β` has units of time and scales as hydraulic-resistance factor times mean transport distance squared. This is a two-region bundle approximation, not a resolved network. The authors chose a transport parameter giving roughly 5% of the energy budget; that percentage is **not a direct transport-respiration measurement**. [Heaton et al. 2016, Eq. 1 and appendices A–B](https://www.journals.uchicago.edu/doi/full/10.1086/684392).

**Pros:** compact; explicitly couples operating costs to growth speed. **Cons:** transferring it to AM fungi requires new physiological interpretation. The `f` variable must not be substituted by IRM fraction or local living density. Its geometry-dependent thinning effects conflict with this repository's current non-limiting-capacity scope. The formula is singular at the endpoints; it must be used within its derivation's domain.

## 3. Explicit hydraulic network and dissipation

Represent hyphae/cords as edges with hydraulic conductances `G_e`, joined at nodes:

\[
Q_e=G_e(p_u-p_v),\qquad
P_{\mathrm{diss}}=\sum_e Q_e(p_u-p_v)=\sum_e Q_e^2/G_e.
\]

`Q` is volume/time, `G` is volume/(pressure × time), and power is pressure × volume/time. Heaton et al. inferred flows in image-derived *Phanerochaete velutina* networks from water conservation and observed growth, minimizing viscous work. For a single cylindrical lumen, `G=πr⁴/(8μℓ)`. For a cord modelled as parallel tubes of fixed lumen radius and packing density, `G` instead scales with cord cross-sectional area divided by length. Treating cord radius as one lumen radius therefore changes the model materially. [Heaton et al. 2010, sections 2–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC2981926/).

**Pros:** shared-trunk demand and bypass routes emerge from conservation; flow speed and dissipation can be tested separately. **Cons:** pore/septal resistance, effective lumen area and sources/sinks need measurement. Growth-induced flows are a minimum compatible with specified volume changes; additional circulation need not be captured. Mechanical dissipation is not itself a measured ATP or respiratory-carbon bill. Converting it requires an explicit efficiency and carbon-to-usable-energy assumption, including how osmotic gradients are maintained.

**Calibration:** image network topology and conducting anatomy; measure flows and growth; test pressure/viscosity assumptions; then independently constrain energetic conversion. Specify whether pressure or demand is prescribed: under fixed demand, thinning raises dissipation; under fixed pressure, it may instead reduce flow. A compensatory carbon surcharge is an extra biological assumption.

## 4. Advection–diffusion–delivery, on a network or continuum

A schematic conservative edge equation is

\[
\partial_t(Ac)+\partial_x(Qc-AD\,\partial_xc)=S-Akc.
\]

`c` is cargo concentration, `A` conducting area, `D` diffusivity, `S` loading per unit length/time and `k` a first-order delivery/loss rate. Changing geometry needs consistent volume accounting. Heaton et al. developed a network algorithm incorporating advection, diffusion and delivery and compared fungal predictions with radiolabel data. The framework separates resource distribution from the underlying fluid currents. [Heaton et al. 2012](https://doi.org/10.1103/PhysRevE.86.021905).

Continuum fungal-growth models offer another spatial resolution: Boswell et al. combined diffusive and metabolically driven internal translocation in *Rhizoctonia solani*. Their calibrated model supported diffusion-dominated translocation in homogeneous conditions, with energetic investment in translocation improving exploitation of localized resources. These results concern that model and organism, not a universal prohibition on active transport in uniform environments. [Boswell et al. 2002](https://pubmed.ncbi.nlm.nih.gov/12234753/).

**Pros:** predicts lag, concentration gradients, accumulation and sink competition; permits distinct uptake and delivery endpoints. **Cons:** requires internal stocks, source/sink placement and stable integration; different mechanisms may fit the same tracer pattern. An imposed advection velocity does not identify its energy source. Add a separate hydraulic or active-transport energetic law if respiration is the endpoint.

**Calibration:** time-resolved pulse–chase tracers and local inventories, plus anatomy/velocity measurements. Use `Pe=uL/D` to assess relative advection and diffusion; characteristic times scale as `L/u` and `L²/D`. These physical comparisons select temporal resolution, not carbon expenditure coefficients.

## 5. Diffusion through a dynamic vacuolar network

Darrah et al. combined fluorescence recovery after photobleaching with simulations of linked vacuoles in *P. velutina*. Their model identified a bidirectional diffusive pathway over millimetre-to-centimetre scales, with tubular connectivity regulating flux. This is direct evidence about fluorescent solute movement and a model-based extrapolation of its consequences, rather than a measured ATP cost for transporting native polyphosphate. [Darrah et al. 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC1489287/).

An illustrative compartment implementation is `dM_i/dt=Σ_j K_ij(c_j−c_i)+loading_i−unloading_i`, where `K` is volume/time and `M=Vc`. Diffusion is downhill; establishing concentrations, membrane transport and maintaining/rearranging organelles can still carry metabolic costs.

**Pros:** resolves storage and bidirectional exchange without assuming directed motors for every solute. **Cons:** effective diffusion depends on compartment sizes and changing links; native cargo chemistry and binding can differ from the marker; it cannot justify zero whole-pathway expenditure or be transferred uncritically to AM fungi. **Calibration:** FRAP plus vacuolar geometry and cargo-specific diffusion/loading measurements.

## 6. Motor-driven vesicle/endosome transport and mechanochemical accounting

Cargo can be represented by moving, paused and unbound states with attachment, detachment and reversal rates. A fungal example is the continuum model *Mathematical model with spatially uniform regulation explains long-range bidirectional transport of early endosomes in fungal hyphae*. It couples microtubule organization, motor distributions and endosome state transitions in *Ustilago maydis*, with quantitative comparison to experimental data and mutants. It is a transport-kinetics model, not a calibrated whole-colony respiration law. [Primary paper, 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142613/).

An added mechanochemical cost could use `ATP/time=Σ(number of active motors × stepping rate × ATP/step)`, plus track/motor maintenance. This is a proposed accounting extension requiring independent stoichiometric calibration, not a result of the cited model.

**Pros:** ties an energetic hypothesis to identifiable machinery and cargo; distinguishes gross bidirectional motion from net delivery. **Cons:** requires motor occupancy, cargo loading and futile-cycle information; a single ATP-per-distance coefficient can conceal empty returns and reversals. Endosome movement is not evidence that all carbon or phosphorus travels in that cargo class. **Calibration:** single-particle tracking, run statistics, cargo content, motor numbers and ATP turnover.

## 7. Construction and network-efficiency objectives

Bebber et al. represented *P. velutina* as weighted graphs, assigning material cost `Σ_e ℓ_e A_e` and functional edge resistance proportional to `ℓ_e/A_e`. They compared least-resistance paths, efficiency and robustness with alternative network architectures. These are useful structural/functional proxies, not direct respiratory transport prices. [Bebber et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2288531/).

AM-specific imaging by Oyarte Galvez et al. found wider hyphae and faster flows on central trunk routes. Its normalized graph efficiency compares network path structure against benchmark graphs; constancy of that metric does not demonstrate constant energy per nutrient delivered. [Oyarte Galvez et al. 2025](https://www.nature.com/articles/s41586-025-08614-x).

**Pros:** supports cross-network comparisons even without local pressures or respiration measurements; explicitly exposes material–coverage–robustness tradeoffs. **Cons:** shortest paths omit simultaneous shared flows; the same geometry can support different operating fluxes. Converting standing volume to construction carbon requires tissue carbon density, while maintenance requires a separate time-dependent cost.

For a composite optimization, one could minimize `construction + maintenance + operating dissipation` under delivery constraints. The weights, time horizon and biological objective are additional hypotheses. Optimizing an assumed objective does not show that the organism achieves its global optimum.

## What the evidence implies for MycorMARL

[ADR-0021](../adr/0021-treat-internal-transport-capacity-as-non-limiting.md) currently assumes **non-limiting internal transport capacity** and sets explicit fungal polyphosphate transport expenditure to zero. The proposed extra expenditure concerns carbon delivery to new growth, using the **transport-front shell** described in [the shell-geometry note](transport-shell-geometries/README.md). The full hydraulic, dynamic-compartment and motor models are therefore research alternatives, not approved extensions.

For the current scientific scope, my recommendation is:

1. Use Method 1 as an explicitly provisional sensitivity model of carbon delivery to new growth. Specify the source, sink, living route approximation, payer and transported carbon definition before selecting a coefficient.
2. Compare a linear distance charge with a physically motivated nonlinear alternative. Method 2 provides published precedent for nonlinear growth/distance costs, but transplanting its full resource-contact factor would add assumptions outside the accepted scope.
3. Revisit Method 3 if the question becomes network architecture and energetic flow, Method 4 if it becomes arrival lag or retention, and Method 6 if it becomes a particular cargo/motor mechanism.

Two evidence boundaries are especially relevant. Bago et al. traced carbon exported from AM roots as carbohydrate as well as lipid; one soluble-carbon diffusivity is therefore not automatically a model of all fungal carbon delivery. [Bago et al. 2003](https://pmc.ncbi.nlm.nih.gov/articles/PMC166909/). Heinemeyer et al. measured external-mycelium respiration, demonstrating dependence on recent photosynthate and temperature acclimation, but not isolating a distance-specific transport component. [Heinemeyer et al. 2006](https://pubmed.ncbi.nlm.nih.gov/16771991/). The repository already uses that respiration evidence for its broader operating-respiration proxy `kappa_c`; an additional transport charge must identify which expenditure is newly represented to avoid claiming independent calibration of overlapping costs. See the [parameter register](../model-parameter-register.md).

Bisot et al.'s AM fungal density–speed study is a useful adjacent carbon-investment model; its morphology-based C/P exchange analysis does not by itself calibrate an internal movement surcharge. [Bisot et al. 2026](https://doi.org/10.1073/pnas.2512182123).

No universal, transferable carbon-per-unit-resource-per-centimetre coefficient was established by the primary sources examined here. That is a limitation of this targeted search and the cited measurements, not proof that no relevant measurements exist. The defensible modelling choice depends on whether the intended endpoint is expenditure, construction, delivery or network architecture.
