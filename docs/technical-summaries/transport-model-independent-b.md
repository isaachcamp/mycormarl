# Independent research B: transport as shared service capacity and carbon expenditure

Research date: 27 October 2026. Scope: research and implementation design; no production changes. This independent assessment favours a different first comparison from a universal inverse-distance resistance law. The user clarified that implementation comparison means design and pseudocode, rather than runnable prototypes. A three-case scalar arithmetic check was executed before that clarification; it is identified below and is not an implementation or benchmark.

## Recommendation and evidence boundary

Use a **coarse routing forest with shared throughput limits**, with an optional explicitly labelled carbon charge proportional to delivered resource and route length. Treat this as a quasi-steady, no-local-storage model. Compare it against a simpler distance-cost-only model and a genuine pressure/concentration-network model. Length density should scale an explicitly assumed conducting area; it should not silently be counted a second time as additional uptake transporter capacity.

A longer route need not have lower steady throughput if active flow compensates. Root axial conducting capacity also need not decline monotonically with tissue age. Thus distinguish:

1. uptake membrane capacity, already represented by surface area and kinetics;
2. conducting infrastructure and its shared bottlenecks;
3. time to delivery;
4. respiratory carbon expenditure.

They cannot be inferred from a single mean-distance statistic. This distinction is supported by an AM-fungal transport model separating soil uptake, internal translocation and interface transfer, although its parameterization does not establish a universal carbon law. [Schnepf & Roose 2006](https://doi.org/10.1111/j.1469-8137.2006.01771.x).

## Primary sources actually consulted

| Primary source | Relevant finding | What it does not establish |
|---|---|---|
| [Oyarte Galvez et al. 2025, Nature](https://doi.org/10.1038/s41586-025-08614-x) | Imaging of AM fungal networks in root organ culture documented bidirectional cytoplasmic motion, wider trunk routes and faster flows along routes with high network centrality. Networks maintained roughly constant normalized graph-theoretical transport efficiency as they expanded; this was a shortest-path topology statistic, not measured C/P delivery efficiency or fixed transit time. | These are two-dimensional culture observations, not a calibration of P throughput or respiratory C expenditure in three-dimensional soil. Particle speed is not P delivery rate. |
| [Boursiac et al. 2022, Plant Physiology](https://doi.org/10.1093/plphys/kiac281) | Measured water flow in branched Arabidopsis roots and inferred conductivities with HydroRoot; axial and radial conductances can co-limit flow. Their anatomical estimate of axial conductance rose strongly with maturation away from the tip; theoretical conduit estimates overestimated measured transport. | Water conductance is not P-specific transport capacity or carbon expenditure. Fine-root length density does not determine xylem anatomy. |
| [Schnepf & Roose 2006, New Phytologist](https://doi.org/10.1111/j.1469-8137.2006.01771.x) | A continuum AM model explicitly separates translocation within fungal tissue and transfer at the root interface. For its chosen parameters, internal solute equilibration was fast compared with soil replenishment. | Fast equilibration is parameter dependent. Its effective diffusion/convection treatment does not demonstrate a universal inverse-distance delivery penalty. |
| [Hammer et al. 2011, FEMS Microbiology Ecology](https://doi.org/10.1111/j.1574-6941.2011.01043.x) | In root organ culture, lowering host C availability increased nutrient accumulation in fungal structures, especially P. | Global immediately available fungal P cannot resolve where accumulated P resides or its delivery delay. The experiment does not identify the mechanism or a transport coefficient for this model. |
| [Bago et al. 2003, Plant Physiology](https://doi.org/10.1104/pp.102.007765) | Carbon-label analysis supported carbohydrate as well as lipid movement from intraradical to external mycelium. | P flux alone cannot represent the fungal C-delivery workload. This study does not calibrate distance-dependent transport respiration. |

These findings support separating capacities and routes, not a quantitative choice among the candidate closures. All equations below are proposed modelling assumptions unless marked as identities. In particular, no transferable fitted carbon-per-P-per-centimetre coefficient was identified in this search.

## Repository and agreed mortality constraints

Repository evidence inspected: [domain vocabulary](../../CONTEXT.md), [ADR-0016](../adr/0016-use-finite-inventory-buffered-phosphate-transport-and-uptake.md), [ADR-0017](../adr/0017-represent-absorbers-as-fine-root-discs-and-an-external-hyphal-hemisphere.md), [ADR-0018](../adr/0018-account-for-structural-phosphorus-and-explicit-biological-exports.md), [environment](../../src/mycormarl/environments/base_mycor.py), [soil integration](../../src/mycormarl/soil/soil.py), [uptake transaction](../../src/mycormarl/soil/phosphate_uptake.py), and [state](../../src/mycormarl/state.py). The mortality specification was read directly using `gh issue view 89 --json number,title,body,comments,labels`.

Current production geometry is reconstructed from biomass; it does not yet retain chronological mortality cohorts. [Issue #89](https://github.com/isaachcamp/mycormarl/issues/89) specifies oldest-first maintenance mortality, front-only absorber growth, living uptake-active floors (default roots 20%, fungus 5%), exact preallocated chronological cohorts, and provisional outer retreat once above-floor thinning is exhausted. It explicitly excludes transport and attachment modelling. Cohort floors preserve coarse radial continuity, not measured throughput or colonisation.

`BaseMycorMarl.step_env` pays maintenance and determines death, allocates trade/growth/reproduction, adds photosynthate, projects geometry, then runs soil evolution. `evolve_soil_p` fixes geometry for its `fori_loop`; each substep diffuses soil, computes both uptake requests from the same concentration, proportionally caps their sum against cell inventory, and credits accepted P immediately to global organism pools. This new P enters biological allocation in the next numerical step. [Code](../../src/mycormarl/soil/soil.py).

A transport extension would revise ADR-0016's transaction and ADR-0017's omitted routing assumptions; it should be a new declared biology revision. Do not represent #89 as already implemented, or add background senescence merely because its name refers to age. [Issue #89](https://github.com/isaachcamp/mycormarl/issues/89).

## Proposed model B: finite service capacity on a coarse routing forest

### Geometry, units and density

Let `lambda_o` be surviving length density [cm/cm³], `a_o` absorber radius [cm], and `A_corr,e` the cross-sectional area [cm²] of a coarse transport corridor. Define

\[
 A_{\mathrm{eff},o,e}=\chi_{o,e}\,\pi a_o^2\lambda_{o,e}A_{\mathrm{corr},e}.
\]

`pi*a²*lambda` is tissue volume fraction (a geometric identity). Interpreting it as a fraction of corridor area requires the proposed factor `chi`, representing orientation, connected fraction and conducting tissue fraction. It is not a geometric identity or a measurement. Use a conservative overlap/harmonic rule for heterogeneous corridor density; do not use an arithmetic mean that hides a narrow gap. A vanishing connected segment gives zero capacity. Partial soil boundary cells require occupied-volume density, because volume-averaged density can confuse partial occupancy with thinning.

A streaming-inspired service rate is

\[
 Q_{P,e}^{\max}=v_e\,A_{\mathrm{eff},e}\,c_{P,e}^*,
\]

where `v` is cm/day and `c_P*` is mg P/cm³ of moving conducting material, giving mg P/day. This is an effective net-delivery ceiling, not an assertion that observed bidirectional streams all carry P at `c_P*`. Fit the product `v*c_P*chi` before attempting to identify all three from biomass and uptake. Do not convert imaged particle speeds directly into whole-colony P transport capacity. The primary microscopy motivates active flow and varying trunk capacity, not these numerical coefficients. [Oyarte Galvez et al. 2025](https://doi.org/10.1038/s41586-025-08614-x).

For roots, the same expression is only a service proxy: this model does not resolve transpiration, xylem sap concentration or vascular anatomy. More defensible hydraulic calibration would need those quantities. Root axial maturity can improve conduction, so an arbitrary `exp(-age/tau)` multiplier is especially questionable. [Boursiac et al. 2022](https://doi.org/10.1093/plphys/kiac281).

### Distance and attachment

Define actual coarse route length as the sum of edge lengths, not the straight-line norm by default. Root paths go from root uptake locations to the stem collar; a first fixed topology can use radial layer corridors followed by a common axial backbone. This creates a declared shared central bottleneck. The backbone is absent from current fine-root geometry: either explicitly represent its parameterized conductance independently, or state that it has unlimited capacity. Inferring its biomass or maintenance from fine-root length would change existing allometry.

Fungal routes terminate at assigned exchange sites on connected roots. A nearest-root distance is a lower-bound geometric proxy: the closest root need not belong to the colonised connection. Recommend a small fixed set of connected attachment basins; route within each basin and aggregate at its root endpoint. This reduces state while making the association hypothesis inspectable. All live roots being eligible attachments is an alternative sensitivity, not an observed fact.

Do not send all hyphal flow to the hemisphere centre and describe the result as nearest-root transport. Distributed fungal sources and root sinks generally require a forest or continuum. A `1/sqrt(lambda_root)` spacing proxy cannot establish a path, shared trunk or AM interface capacity on its own.

A flux-weighted diagnostic is

\[
 \bar d_o=\frac{\sum_i x_{o,i}d_{o,i}}{\sum_i x_{o,i}},
\]

where `x` is actual delivery [mg P/substep], and report zero plus a no-flux flag when the denominator vanishes. Fungal distance ends at its assigned root; the plant root-to-stem leg must be accounted separately if its delivery is explicitly represented.

### Carbon costs

For accepted delivery `x_i`:

\[
 C_{\mathrm{resp},P,o}=\sum_i x_{o,i}
 [\epsilon_{0,o}+\epsilon_{d,o}d_{o,i}],
\]

with `epsilon_0` [g C/mg P], `epsilon_d` [g C/(mg P cm)], and result g C. Equivalently the distance part is `sum_e epsilon_d*ell_e*F_e*dt`; shared edges carry the sum of distal deliveries. This explicitly charges the work along shared routes without charging an edge once per local source irrespective of actual flow. It is a sensitivity law, not an empirical physiological conclusion.

At fixed speed, thinning reduces capacity but need not increase C per unit delivered; if the model lets speed increase to compensate, an energy law must relate extra speed or pressure to expenditure. Applying both inverse-density C cost and density-dependent capacity without justification would count two different hypotheses.

The fungus must also transport C to support local maintenance, growth and storage. An additional C-delivery workload `F_C,e` [g C/day] could compete through `F_P/Q_Pmax + F_C/Q_Cmax <= 1`, but that is another assumed service rule. Opposing flows need not share capacity this way; compare independent directional capacities. In the present global-pool model, C allocation is not spatially attributed, so a P-only implementation cannot claim to represent full fungal transport metabolism. [Bago et al. 2003](https://doi.org/10.1104/pp.102.007765).

Current fungal standing maintenance is calibrated as a broader external-mycelium operating-respiration proxy, documented in the [parameter register](../model-parameter-register.md). An extra respiratory term requires repartitioning that baseline or labelling the addition as deliberately extra expenditure. Do not debit C sent in trade twice as transport respiration.

### Age-dependent death

Apply #89's mortality first, then derive corridor capacities from surviving tissue. Interior thinning can leave uptake-active young tissue behind a weak trunk. Front-only growth does not repair the interior. On the stated linear proxy, fungal thinning to 5% makes a shared trunk's maximum throughput 5% of baseline; roots at 20% make their proxy capacity 20%. These are conditional arithmetic consequences, not measured thresholds.

Mortality simultaneously reduces standing maintenance, membrane area and transport capacity. Consequently absolute transport respiration need not rise after thinning: transport may cease, reducing realised C spending while also reducing P supply. Age must enter only via surviving geometry initially. Do not infer a direct physiological age penalty from the ordering of maintenance loss. The specification's reset cohort is uniformly aged and can thin uniformly, so it cannot initially generate a uniquely old inner bottleneck. [Issue #89](https://github.com/isaachcamp/mycormarl/issues/89).

For roots, losing absorbing tissue is not evidence that an axial conducting backbone disappears proportionally; compare coupled live-density and independent-backbone capacities before making that coupling a production assumption. [Boursiac et al. 2022](https://doi.org/10.1093/plphys/kiac281).

## Concrete implementation comparison

| Design | New state and pure-function seam | Advantages | Main limitation |
|---|---|---|---|
| A: distance-dependent energetic cost only | Cached route-distance grid or coarse mapping; scalar cumulative C expenditure; cost-funded uptake transaction | Small conceptual change; directly tests whether paying for distance matters | No shared throughput bottleneck, no travel delay; requires uncalibrated C cost |
| B: proposed coarse routing forest with capped throughput and optional C cost | Fixed coarse topology, edge lengths and cell-to-basin map; per-step capacities; temporary subtree demand and scaling arrays; no persistent local P stocks | Represents shared interior thinning and explicit attachment; fixed shapes; separates capacity from spending | Assumed routes and fairness; no accumulation or transit delay; global P pool remains immediately fungible after delivery |
| C: quasi-steady conductance network | Coarse node concentrations/potentials as solve variables, edge conductance, sink boundary conditions; usually no new dynamic P inventory | Physically consistent sharing of driving potential and serial/parallel resistance | Iterative or tree solve; nonlinear uptake coupling; needs potential boundary conditions and calibrated conductance |
| D: conservative internal P transport dynamics | Root/fungal local P stocks, transport edges and transfer coefficients; possibly C stocks and interface states | Accumulation, transit delay and decoupled uptake/transfer | Extra state, stiffness, boundary/mortality rules and substantial runtime/memory |

### B algorithm: shared constraints with no local P storage

All quantities in the following pseudocode are amounts per current soil substep, after explicit conversion of existing µmol requests to mg. `E` is the number of coarse edges, much smaller than soil cell count `G`. The policy is hierarchical proportional rationing, not an optimal allocation claim.

```text
At biological boundary after mortality and growth:
    project surviving uptake fields
    project coarse conducting densities and fixed-route lengths
    capacity_rate[e] = conducting_area[e] * fitted_service_coefficient[e]
    retain scalar C pool after the declared earlier biological spending

For each existing soil substep of duration h:
    diffuse canonical soil inventory
    compute root and fungal membrane requests from same new concentration
    provisional_x = proportional shared-soil inventory cap(requests)
        # Preview only; do not yet remove soil or credit organism pools.

    For each organism's fixed routing forest:
        aggregate provisional_x to coarse source nodes
        bottom-up:
            W[node] = local_request[node] + sum(admissible[child])
            admissible[node] = min(W[node], capacity_rate[parent_edge] * h)
        top-down:
            share[node] = ancestor_share * admissible[node] / W[node]
            apply share[node] to local requests and pass to descendants
            # Safe zero-demand ratios; unlimited sink edge if declared.
        map final coarse rationing to source-cell uptake candidates

    compute candidate carbon charge from delivered amounts and path lengths
    organism_scale = min(1, available_C / candidate_charge)
        # zero cost means scale=1; only scale down, never overdraw.
    final_x = candidate_x * organism_scale
    debit exactly final_x from soil
    credit exactly final_x to organism global P pools
    debit exactly recomputed final C charge; accumulate respiration diagnostic
    carry reduced C pool into next soil substep
```

The edge budget is `capacity_rate*h`, not the full outer-step budget every substep. The C pool is carried across the loop. This avoids multiplying available capacity or energy by soil-substep count.

This conservative transaction first assigns competing organisms provisional shares of scarce soil P, then releases rejected amounts back to soil. It leaves a blocked organism's rejected share available on later substeps, without immediate reallocation. Therefore substep refinement can alter competition. This is an explicit first algorithm, not a converged simultaneous allocation solution. An implementation seeking within-substep redistribution must solve or iterate joint cell-inventory, shared-route and C-budget constraints; declare a fixed convergence tolerance and test organism-order invariance. Merely switching organism loop order is unacceptable.

Global C scaling is conservative but can unfairly reduce cheap sources alongside expensive ones. It is a declared proportional allocation assumption. Solving a cost-efficient allocation would introduce a new physiological preference, especially in a policy model, so compare it explicitly rather than silently maximising P gained.

A fungal-to-root transfer bottleneck belongs in the bilateral allocation stage when fungal P is traded; direct uptake into fungal available P is not automatically host delivery. Routing all fungal uptake immediately into the plant would erase the existing trade decision. To account for root-to-stem transport of traded P, either retain attachment-local trade supply and route it then, or make a declared aggregate approximation using lagged source weights. The global fungal P pool has lost source provenance, so an exact distance for its later trade cannot be reconstructed. A first B model can constrain local-to-global fungal collection only and leave the existing global trade abstraction; it must not claim complete host-path transport.

If transport respiration is charged during soil evolution, current photosynthate can fund plant uptake after biological allocation. This is a meaningful process-order assumption. Costs of bilateral trade must instead enter `integrate_allocation` so transfer, growth, reproduction and transport compete for the evolving pools. Adding them to maintenance at the next step would permit an unfinanced current transfer and alter mortality semantics. [Environment](../../src/mycormarl/environments/base_mycor.py), [allocation integrator](../../src/mycormarl/allocation.py).

### C algorithm: genuine resistance comparison

Use edge flow `F_e=g_e*(psi_up-psi_down)` and impose node conservation with sources and sink boundaries. For diffusive solute transport, `g_e=D_eff*A_eff/ell_e` [cm³/day], `psi` is mg P/cm³ and flow is mg P/day. This is a defined linear-response mechanism, unlike an unspecified potential. A common-node concentration/potential solve shares one driving budget along all segments; independently giving every edge the full source-to-sink concentration difference would overestimate serial flow.

A tree permits elimination and back-substitution in O(E) for fixed linear coefficients; loops require a sparse iterative solve or equivalent. Coupling membrane uptake to local concentration or saturation adds nonlinearity. Carbon respiration still requires its own constitutive law. This alternative is attractive if passive/effective resistance is the scientific hypothesis; active fungal streaming means it should be compared rather than assumed universal. [Schnepf & Roose 2006](https://doi.org/10.1111/j.1469-8137.2006.01771.x).

### D algorithm: dynamics, storage and death

Advance `S_o,i` [mg P] by accepted membrane uptake minus conservative face transports, plus any declared interface exchange. Only delivery from those stocks enters the existing available global pool. Use upwind advection and optional diffusion, or a conservative implicit scheme. Explicit stability depends on internal transport velocity, cell spacing and diffusivity, not just the existing soil diffusion CFL. Store neither duplicate amounts nor independently mutable concentration fields.

Include all `S` in the system P ledger. When a cohort thins or a corridor becomes inactive, specify whether its local P is exported with mortality, retained in surviving local tissue, remobilised or stranded. Each choice changes biology. Floors and continuity alone do not answer this. D can represent fungal retention indicated by culture experiments, but spatially resolved measurements are needed to constrain where retention occurs. [Hammer et al. 2011](https://doi.org/10.1111/j.1574-6941.2011.01043.x).

## Compute and memory in compiled JAX rollouts

Measured code facts: the [default configuration](../../src/mycormarl/params.py) uses 50 cm radius, 100 cm depth and 0.05 cm spacing, giving `1000*2000 = 2,000,000` cells. Current soil subcycling uses `jax.lax.fori_loop` with fixed geometry. #89 plans 14,601 chronological cohort slots per organism at the default horizon. These are separate spatial and chronological dimensions.

Analytical estimates, not benchmarks:

- A float32 grid field is 8,000,000 bytes = 7.63 MiB/environment. Two additional local internal P stocks in D add 15.26 MiB/environment, or about 3.81 GiB across 256 environments, before intermediates, rollout storage, allocator overhead or compiler optimizations.
- A adds O(G) work per soil substep and potentially O(G) cached distance storage. Compute distance from coarse geometry on demand if caching hurts memory.
- B adds O(G) source aggregation/scaling plus O(E) forest operations per soil substep. An int32 grid-to-basin map is another 7.63 MiB/environment if duplicated; static mappings may be shared across batches, but geometry-dependent attachment maps may not. No G×cohort array is needed.
- C requires O(E) tree work or approximately O(I*E) sparse iterative work per substep, with I solver iterations. It has convergence diagnostics and temporary solver storage.
- D adds O(G) state for a full-grid scheme and O(internal_substeps*G) work; coarse internal compartments reduce cost but introduce another resolution parameter.
- Any design must keep cohort vectors and large transport fields out of per-timestep `scan` outputs; retain in live carry only when necessary. Saving them for T timesteps creates an avoidable T multiplier. Pseudocode complexity is not evidence of acceptable compiled throughput.

Benchmark warm rollout throughput, compilation time and peak memory separately at the actual batch size and mortality-cohort horizon. Do not infer speed from the scalar example or asymptotic counts.

## Calibration and decisive comparisons

Fit simple, identifiable effective coefficients first. For B, tracer arrival/throughput across known distances and density cuts constrains aggregate service capacity; spatial respiration with controlled resource flux is needed for `epsilon_d`. Root vascular observations are needed to distinguish fine-root density from axial area. Distinguish colonised attachment density from all live root density. [Boursiac et al. 2022](https://doi.org/10.1093/plphys/kiac281); [Oyarte Galvez et al. 2025](https://doi.org/10.1038/s41586-025-08614-x).

Compare matched physical growth–deficit–recovery trajectories with: capacity only; energy only; both; independent root backbone; and no transport extension. Add active-flow capacity B versus effective-resistance C with matched baseline throughput before interpreting differences. Calibrating every model to baseline growth alone would confound uptake, maintenance and transport.

Required behavioural/numerical checks, proposed and unexecuted:

1. Full P ledger closes across soil, available pools, structure and exports; D adds local stocks. C budgets never overdraw; charged C equals realised, not requested, work.
2. Zero demand, zero density, absent/dead partner, disconnected attachment, zero-cost and zero-capacity limits remain finite and conservative.
3. Two distal sources sharing one trunk cannot each claim its full capacity. For routes sharing no conducting edge, sink ceiling or C budget, increasing one route capacity must not lower another route's delivery. With a shared trunk or common C budget, proportional rationing can lower a competing branch's delivery when another branch capacity increases; test the declared fairness rule instead of assuming branchwise monotonicity.
4. A narrow interior thinning event suppresses distal flow despite unchanged distal uptake infrastructure. Front regrowth without repair preserves that limitation. Reset's uniform cohort should not produce an invented old-inner/young-outer pattern.
5. Route subdivision at fixed physical properties preserves cost and steady capacity for B when no extra bottleneck is introduced; resistance C adds segment resistances in series. Grid refinement does not change attachment identities accidentally.
6. Matched-duration numerical, soil-substep and coarse-network refinement assesses accepted uptake, transport respiration, competition shares, mortality and reproductive output. Keep physical actions and initial pools identical.
7. Scalar, compiled and batched results agree; checkpoint state versioning and restart preserve transport state. No hidden transport vectors appear in evaluation histories.
8. No-P-export scenarios still expose the limitation of a P-only cost approximation: fungal maintenance/growth may require C movement even with zero P delivery.

Before the user's pseudocode-only clarification, an isolated standard-library arithmetic check evaluated two distal demands `[8,8]`, branch caps `[10,10]` and shared trunk cap `10`, obtaining `[5,5]`; a trunk cap of `0.5` gave `[0.25,0.25]`; branch caps `[1,10]` and trunk `10` gave `[1,8]`. All three assertions passed. These unitless examples check elementary cap arithmetic only. No production test, physiological calibration, runnable model comparison or JAX performance experiment was performed.

## Decision to make before implementation

For the proposed B baseline, the fungal collection sink is an **abstract connected exchange-region inventory**, not a claim that all fungal retained P physically resides there. Defining the existing global fungal P pool this way forces locally retained growth P through collection; retaining its current completely location-free meaning instead makes collection distance an effective allocation proxy. This remains unresolved structural uncertainty and must be settled before implementation.

First choose the endpoint of each available pool: whether fungal free P means globally fungible inventory or P already collected at an exchange region, and whether plant free P is locally usable root P or P delivered to the stem. The present code has no location for these pools. An explicit choice determines whether B's no-storage model is adequate or D's compartmental transport is necessary.

For the first tractable comparison, I recommend B for **fungal collection capacity plus explicit root direct-uptake delivery**, with a separate declared treatment of bilateral transfer. Keep the carbon-distance coefficient at zero in the capacity baseline; add it as an independently varied energetic hypothesis. Report that this is a partial route model until fungal trade provenance and tipward C supply have defined representations.

## Cross-review of independent investigation A

Reviewed [A's draft](transport-model-independent-a.md) after writing this note. The independent recommendations converge: coarse shared routing, separate capacity and distance-price experiments, and dynamic internal stocks only when storage or delay matters. This is convergence of modelling judgement, not two independent experimental validations.

**Substantive flaw in the reviewed draft:** A's design B initially defines each edge ceiling as `D*phi*A/ell * Delta_cmax` with the same concentration allowance on every edge. Splitting one edge into two equal series segments doubles each segment ceiling; their minimum therefore doubles the allowed end-to-end flow. The result violates A's own subdivision-invariance check. The text acknowledges that edge caps do not solve concentration fields, but that warning does not repair mesh-dependent capacity. Use a segment-length-independent service ceiling for a throughput network (as proposed in B here), or solve shared node potentials so the total driving drop is distributed across the series path. If each path had its own full driving budget, overlapping paths would still need a consistent shared-node treatment. This issue was sent to A and the parent for correction.

**Pool-location limitation:** A correctly discusses fungal storage and trade provenance in several places, but the headline recommendation should explicitly declare the endpoint of globally available fungal P. Sending every fungal uptake packet to a nearby exchange root can impose a host-directed journey on P retained for fungal growth. Charging only later bilateral trade instead loses the acquisition location from the current global pool. Neither implementation can claim the whole symbiotic transport budget without a pool-location assumption or new provenance/local state. The same concern applies to root-to-stem delivery of traded fungal P and to tipward C supply at zero P export. B here therefore recommends a deliberately partial collection/delivery model until those endpoints are agreed.

**Change accepted from A:** A's segment-weighted resistance example is more precise than whole-network floor ratios. If only 10% of baseline series resistance lies in a segment thinned to fraction `f`, total resistance changes by `0.9 + 0.1/f`, not `1/f`. Likewise a service-ceiling network is affected only if the thinned edge becomes limiting; a bypass may preserve throughput. Statements about roots at 20% and hyphae at 5% must always be local, conditional proxies rather than universal network multipliers.

**Implementation detail accepted from A:** Keep canonical uptake requests and inventory arbitration in existing µmol units until the biological accounting boundary where feasible; the pseudocode above converts requests to mg for clear common budget arithmetic, but doing so is not intrinsically necessary. Conversion discipline and a single canonical amount matter more than the particular working unit.

**Remaining model difference:** A initially uses effective diffusive conductance to motivate capacity, whereas B starts with a service-capacity proxy motivated by active movement. Neither is physiologically calibrated. Compare them with identical baseline end-to-end throughput, conducting geometry and source/sink assumptions; otherwise a difference in outcomes may merely reflect different baseline parameter scales.
