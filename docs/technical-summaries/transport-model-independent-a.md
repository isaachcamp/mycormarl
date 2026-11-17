# Independent transport-model investigation A

Research date: 27 October 2026. Scope: research and implementation comparison; no production implementation or benchmarks. All proposed equations below are hypotheses or numerical designs, not experimentally calibrated laws. This note was developed independently from primary literature and the code, then checked against the earlier synthesis. The requested question is how root-to-stem and hypha-to-root distance, surviving length density and oldest-first maintenance mortality can jointly affect transport.

## Recommendation

Start with a **coarse, shared-route capacity model**, and use a separately switchable linear distance carbon charge as a sensitivity treatment. Retain explicit internal stocks for a later experiment if arrival delay or storage is a scientific endpoint. Length density should initially govern capacity; it need not also multiply the cost per delivered unit. Use distinct root and fungal coefficients. The strongest difference from a simple average-distance penalty is that distal consumers must share the same depleted interior conduits.

This recommendation is a modelling judgement. It follows from the model's whole-organism resource budgets, very large soil grid, planned persistent thinning and the distinction between uptake and delivery described below. It does **not** establish that effective capacity scales linearly with living fine-root length in real roots.

## Evidence and physiological boundaries

| Primary source | What it supports | What it does not calibrate |
|---|---|---|
| [Schnepf & Roose (2006), Modelling the contribution of arbuscular mycorrhizal fungi to plant phosphate uptake](https://doi.org/10.1111/j.1469-8137.2006.01771.x) | A primary mathematical model separates soil P uptake from internal mycelial diffusion/convection and root transfer. Its uptake is not fed back by internal concentration because suitable parameterisation data were unavailable. | A universal carbon/P/distance coefficient or an identified feedback law. |
| [Oyarte Galvez et al. (2025), A travelling-wave strategy for plant–fungal trade](https://www.nature.com/articles/s41586-025-08614-x) | Imaging resolves simultaneous bidirectional flows; wider, faster-flowing trunk routes have higher network centrality. Mean speeds remain stable while network length expands. Its reported root-transport efficiency is a normalised shortest-path graph metric, not respiratory efficiency, P delivery or transit time. | That every hypha has identical capacity or every extra centimetre increases respiration by a fixed amount. |
| [Bago et al. (2002), Translocation and Utilization of Fungal Storage Lipid](https://pmc.ncbi.nlm.nih.gov/articles/PMC148950/) | Imaging and isotope labelling support bidirectional internal lipid transport, with net root-to-external-mycelium movement. Carrier cross-section, transported fraction and velocity matter for flux. | A P-specific energetic cost or the identification of lipid particle speed with net phosphate speed. |
| [Hammer et al. (2011), Tit for tat?](https://pubmed.ncbi.nlm.nih.gov/21223336/) | The experiment concerns fungal P accumulation under low host carbon availability: uptake/storage and host delivery need not coincide. | A universal law relating internal concentration to uptake, or a distance-dependent respiratory debit. |
| [Melchior & Steudle (1993), Water Transport in Onion Roots](https://pubmed.ncbi.nlm.nih.gov/12231786/) | Direct measurements distinguish axial and radial conductivity, with axial conductivity increasing substantially during xylem development. | A phosphate-specific transport coefficient; a rule that older roots conduct less. |
| [Boursiac et al. (2022), Phenotyping and modeling of root hydraulic architecture](https://academic.oup.com/plphys/article/190/2/1289/6609183) | Measured root flow and graph-based inference show co-limitation by radial and axial conductances; anatomy-only estimates can overestimate axial conductance. | That absorbing root radius can be substituted for xylem lumen radius. |
| [Volder et al. (2005), Rapid decline in nitrate uptake and respiration with age](https://doi.org/10.1111/j.1469-8137.2004.01222.x) | Age-dependent uptake/respiration in grape fine lateral roots motivates separating functions of old tissue. | Equivalent age responses of phosphate transport, axial xylem flow or fungal transport. |

These studies motivate distinct representations of membrane entry, internal movement, storage, exchange and operating carbon. Adding a resistance is not itself adding an energetic cost. A hydraulic analogy gives driving-force/flow relationships; converting its dissipation into respiration would need an explicit energetic mechanism and conversion efficiency. Neither the papers above nor this search supply transferable fitted values for our plant/fungus combination.

## Current implementation and agreed future state

Inspected local source is the primary evidence for these statements:

- [BaseMycorMarl.step_env](../../src/mycormarl/environments/base_mycor.py) pays maintenance and determines biological death, integrates coupled trade/growth/reproduction, adds photosynthate, rebuilds living density fields and then advances soil uptake. New soil P becomes usable on the next numerical environment step.
- [State](../../src/mycormarl/state.py) stores organism-wide free C/P pools and two soil-cell length-density fields, but no local internal P/C, root–fungus attachment, conduit topology or cohort history.
- [soil.py](../../src/mycormarl/soil/soil.py) uses fixed post-growth geometry over a `jax.lax.fori_loop` of soil substeps. Each substep diffuses soil P, obtains uptake requests and immediately credits accepted cellwise P to global pools.
- [phosphate_uptake.py](../../src/mycormarl/soil/phosphate_uptake.py) blends sparse/continuous soil-to-membrane requests and proportionally allocates the two organisms' competing requests against each shared soil inventory. Its `sparse_uptake_resistance` is not axial resistance; the same word must not hide a second process.
- [allocation.py](../../src/mycormarl/allocation.py) couples trade, growth and reproduction with a positivity-controlled SSPRK3 integration; incoming trade is eligible during that integration. Transport charges must compete for finite C rather than be deducted after all spending.

[Issue #89](https://github.com/isaachcamp/mycormarl/issues/89) is the agreed planned mortality specification. Maintenance deficit determines total loss; oldest eligible cohorts thin toward living-density floors (root default 20%, fungus 5%), followed provisionally by front retreat when thinning is exhausted. Growth extends the front without interior repair; retained floor tissue still pays maintenance and takes up P. It introduces no spontaneous age death. Its fixed storage can contain 14,601 chronological cohorts per organism at the default effective episode horizon. Current code remains memoryless; proposals depending on persistent thinning require that work or its equivalent first. [CONTEXT.md](../../CONTEXT.md), [ADR-0004](../adr/0004-derive-uptake-geometry-from-current-biomass.md), [ADR-0017](../adr/0017-represent-absorbers-as-fine-root-discs-and-an-external-hyphal-hemisphere.md).

The transport extension would revise the geometry/transport assumptions of ADR-0017 and the source timing of [ADR-0019](../adr/0019-separate-policy-numerical-and-soil-clocks-and-qualify-resolution.md) where necessary. It must preserve [ADR-0018's](../adr/0018-account-for-structural-phosphorus-and-explicit-biological-exports.md) structural-P and biological-export ledger. These are future changes to document, not changes made by this note.

## Geometry before coefficients

Write length density as `lambda` [cm length/cm³ soil = cm⁻²], biological path length as `ell` [cm], P amount as `q` [mg P], and elapsed time as `t` [day]. Keep soil transactions in their existing µmol units and convert exactly once at biological boundaries. Root distance is to the collar/stem base, not distance to a root tip:

\[
\ell_p(r,z)=\tau_p\sqrt{r^2+z^2},\quad \tau_p\geq1.
\]

This is a route-length proxy, not a root reconstruction. Compare it with `tau_p(r+z)` for vertical-backbone/lateral architecture. A stem segment above the collar would require a separate declared length. Fine-root discs in this model do not resolve the root backbone, so either give that backbone an explicit shared capacity or state that it is unlimited.

For hyphae the correct destination is an attached exchange-capable root. A density field does not identify one. A possible unresolved-spacing proxy inside eligible occupied root territory is `b/sqrt(chi*lambda_root)`; outside it add the gap to that territory. Here `b` and attached fraction `chi` are hypotheses, not measured constants. Evaluate within-occupied density and occupancy separately: a tiny boundary-cell overlap must not create a huge biological distance merely through volume averaging. A continuous line density cannot provide the actual nearest root distance or an actual fungal route.

Specify a fixed coarse attachment map or a stated reassignment rule. Nearest-live-root reassignment is instantaneous rewiring; it cannot be inferred from overlapping biomass fields. No eligible root means zero host exchange, although a fungus can still need internal C/P movement for its own growth. A nearest-root route must follow connected fungal tissue, not jump across an unoccupied gap. The spherical fungal closure about the stem is not equivalent to nearest-root transport.

For reporting, use flux-weighted mean distance `sum(q_i ell_i)/sum(q_i)`; mask its value when total flux is zero. This exactly aggregates a linear distance cost, but cannot replace a spatial resistance calculation or a nonlinear cost. Fungal uptake, fungal host delivery and plant delivery need different weights and endpoints.

## Three implementations compared

### A. Linear energetic closure without transport stocks

For accepted, centrally delivered P in a soil substep,

\[
\Delta C_o=\sum_i\Delta q_{o,i}(e_{0,o}+e_{\ell,o}\ell_{o,i}),
\]

where `e0` [g C/mg P] and `eell` [g C/(mg P cm)] differ for plant/fungus. A nonzero `e0` must name membrane/handling operations already included in maintenance; otherwise default it to zero to isolate distance. This is an experiment, not a physiological law. Length density already affects uptake area; it does not independently constrain axial throughput in this model.

**State and integration.** Add trait coefficients, a cumulative transport-respiration counter [g C], and compact realised charge diagnostics. Route arrays can be transient. At each post-diffusion substep, compute requests; simultaneously enforce cell P inventories and the organism's available C budget before debiting soil and C and crediting P. Because photosynthate is currently credited before soil processing, this ordering permits new photosynthate to fund transport while P remains next-step eligible: name and qualify this choice. Do not retrospectively remove P from a pool already used by growth/trade.

A simple conservative allocation is: first apportion scarce soil among original requests, then scale each organism's provisional uptake by its available C/cost, finally debit soil only by the scaled realised amounts. This leaves unclaimed P in soil for later substeps and **does not** redistribute it instantly to the competitor. It is a defined one-pass arbitration rule; if instantaneous redistribution is wanted, solve a constrained allocation with a declared fairness rule and test it. Lowering requests before cell competition is another different rule, not an algebraically interchangeable implementation.

**Timing and budget.** The C pool is carried through every soil substep; never give every iteration the full outer-step carbon budget. Compute rates times the actual substep duration. Zero-cost routes need no division by zero. If charges depend on fungal P trade rather than soil uptake, put the corresponding debit in `integrate_allocation` with a spatial provenance assumption: a global fungal P pool has forgotten where its P was acquired. Trade receipt is not equivalent to current external delivery of C to fungal tips.

**What it predicts.** Long routes reduce profitability and, with finite C, can reduce uptake. Mortality changes cost by changing where uptake happens. It cannot produce transport lag, local storage or an interior bottleneck while distal uptake stays constant. Applying inverse density to `eell` would add a second, separately testable assumption, not repair that omission mechanistically.

### B. Shared-route capacity without local storage (recommended first)

Build a fixed-shape coarse directed routing tree toward each sink. Each edge represents many parallel conduits. For a streaming-inspired **service capacity** choose

\[
\phi_e=\eta_e\pi a_o^2\lambda_{o,e},\qquad
T_e=v_o\phi_e A_e c_o^*.
\]

Here `v` [cm/day], `A` [cm² bulk cut], `phi` dimensionless, and moving P concentration `c*` [mg P/cm³ conducting tissue] give `T` [mg P/day]. Fit the product `v*c*eta` first; components are not separately identifiable from whole-organism growth. This is a net delivery ceiling, not a claim that observed particles all carry P. For roots, use a fitted service coefficient: absorbing root radius is not xylem lumen radius and cannot justify Poiseuille flow. Distance enters the separately testable energetic term; finite transit time is omitted here.

A **different**, genuine resistance variant uses `G_e=D_parallel*phi_e*A_e/ell_e` [cm³/day] and `F_e=G_e(c_up-c_down)` with shared node concentrations and node mass conservation. One source-to-sink driving difference must be distributed across all serial edges. Never set every edge cap to `G_e*same_Delta_cmax`: subdividing an edge would then spuriously double the series route's capacity. A tree concentration solve or properly assembled series/parallel equivalent resistances resolves that issue. For this note B means service capacity; the resistance variant requires a potential solve and is a separate implementation comparison.

For each edge require the **sum** of distal delivered requests to satisfy

\[
\sum_{i:e\in path(i)}\Delta q_i\leq T_e\Delta t.
\]

Use a postorder tree pass to aggregate local and distal demand, reduce overflowing subtrees proportionally, then a preorder pass to propagate reductions to uptake cells. This is a feasible capacity allocation with a specified fairness rule, not a general network optimum. For genuine steady diffusion, instead solve node concentrations with Kirchhoff conservation and sink boundary conditions; series resistance is `sum(1/G_e)` and parallel routes add conductance. Independent edge caps approximate throughput and do not by themselves solve that concentration field.

**State and integration.** No internal P stock. Add static parent/destination indices on a *coarse transport mesh*, dynamic capacity derived from post-mortality/growth live density, and compact diagnostics. No new chronological cohorts are needed beyond #89. Map the uptake grid onto coarse transport sources, jointly limit requests before final soil removal, then credit only deliverable P to the existing organism pools. Undelivered potential uptake stays in soil: this is instantaneous uptake feedback, not stored intracellular P.

**Competition.** Soil availability and route constraints interact. A sequential fixed allocation may conservatively leave some P unclaimed even when a competitor has capacity. Either explicitly choose that operator splitting or solve a deterministic joint constrained allocation; do not quietly cap after soil removal. Share a root-collar backbone between root layers if it is intended to be finite. Fungal paths terminate at declared attached root nodes, not automatically at the stem. Do not reserve independent capacities for each uptake cell when they cross the same cut. Within each soil substep capacity is a rate times that duration, not the full outer-step allowance.

**Mortality.** All through-path capacities use surviving density. With otherwise unchanged coefficients, thinning an interior segment to fraction `f` multiplies its service capacity by `f`. For the resistance variant it multiplies conductance by `f` and segment resistance by `1/f`; it does not prove a whole-network resistance increases by `1/f` in the separate resistance variant. A segment initially contributing 10% of series resistance contributes a total-resistance factor `0.9+0.1/f`: 1.4 at `f=0.2`, 2.9 at `f=0.05`. Parallel bypasses can weaken the effect. No numerical calculation was executed for this algebraic example.

**What it predicts.** A long or thinned shared interior restricts benefit from young distal uptake tissue. It has no arrival time, storage or post-disconnection release. A central-shell fungal routing approximation is cheap, but changes the requested nearest-root geometry; a coarse tree with distributed root sinks is the better comparison. Compare both as structural uncertainty.

### C. Conservative dynamic internal transport

Retain P amounts `M_o,k` [mg] in fixed coarse transport nodes, distinct from the spendable central pool. Internal concentration is `c_k=M_k/V_tissue,k`, with `V_tissue=phi_k V_bulk,k` [cm³]. A conservative graph discretisation is

\[
\dot M_k=U_k+\sum_jF_{jk}-\sum_jF_{kj}-E_k-L_k,
\]

\[
F_{kj}=G_{kj}(c_k-c_j)+v_{kj}A_{t,kj}c_{upwind}.
\]

`U` is membrane entry, `E` sink delivery, `L` an explicitly accounted export; `v` [cm/day] and `A_t` [cm²] describe effective advective transport. Flux is signed once per edge; equal/opposite node updates ensure conservation. Root and fungal transport need not share the same coefficients or advective direction. Antiparallel observed particle movement does not mean one constant velocity reproduces net P flux (Oyarte Galvez; Bago above).

**State and integration.** Add internal P node vectors and explicit delivery counters; add internal C vectors only if testing local fungal maintenance/front supply. Soil accepted P enters local `M`, not global spendable P. Only delivery at collar or exchange interface increments the corresponding central pool. Preserve fungal P trade control: transport toward an exchange-ready fungal pool does not itself grant P to the plant. Plant incoming trade can remain allocation-eligible only after it has arrived at its declared exchange location; changing that location/timing requires a new boundary decision.

Use a transport graph smaller than the soil grid, with conservative cell-to-node mapping. Advance it inside compatible soil/transport substeps. Explicit positivity requires outgoing-rate/CFL limits based on `V_tissue/G` as well as advection; existing soil CFL alone is insufficient. Severe thinning makes low storage volume a stiffness problem. An implicit solve trades that restriction for solver work, positivity/convergence qualifications and fixed-shape sparse infrastructure. Empty tissue volume must be an inactive node, never division by an epsilon that creates transport through missing tissue.

**Mortality and moving geometry.** After loss, amount must not silently disappear with lost volume. Declare whether local stored P is retained in surviving tissue, remobilised to a connected neighbour, or exported. Under proportional export, remove the corresponding fraction from `M` and add it to a **new free-internal-P mortality export**, separate from structural P mortality already counted via `gamma_p`. If all living tissue vanishes, retaining its P still requires an accounted inert pool; it cannot remain a conducting node. Growth adds tissue volume and zero new P stock unless construction accounting explicitly supplies it. Coarse remapping on retreat/regrowth must preserve amounts, not interpolate concentrations.

**What it predicts.** Delays, storage, uptake–delivery separation, transient congestion and depletion after changed attachment. Uptake needs a finite-storage/backpressure hypothesis if indefinite local accumulation is implausible; otherwise even a disconnected patch can continue soil depletion. A strict no-storage capacity model and a storage model therefore ask different biological questions.


### Explicit endpoints and transaction pseudocode

A global fungal P pool has no physical location in current code. For A/B, one explicit candidate is a **root-interface mixing pool owned by the fungus**, funding both its own biological allocation and its separately controlled plant trade. Newly accepted fungal P must then return rootward even when subsequently used for fungal growth. That excludes autonomous distal P use and is a falsifiable approximation. C can permit local fungal use before export. Collection at a root interface does not transfer ownership to the plant; the fungal trade action still controls that crossing. Charging subsequent root-to-stem transport of traded P also needs an attachment provenance state or a declared aggregate route assumption; the existing global pool has lost that information.

A/B without distributed C demand is explicitly a **P-transport sensitivity model**. Full fungal transport must also supply C to standing tissue and the growth front when P export is zero. If transport is to constrain maintenance itself, integrate that supply limitation before maintenance demand executes; appending a final soil-phase cost cannot produce local maintenance starvation correctly.

The following is unexecuted design pseudocode. It specifies one-pass proportional arbitration; rejected P remains in soil until later substeps rather than being reauctioned immediately.

```text
At numerical biological boundary:
  pay maintenance; determine biological death and total biomass loss
  apply #89 oldest-first spatial loss and existing structural-P exports
  integrate coupled central-pool trade, growth and reproduction
  append front-only growth; add photosynthate as in existing ordering
  project live uptake density and coarse route density
  derive edge capacity rates and cell path prices
For each existing soil substep of physical duration h:
  diffuse soil; obtain both organisms' membrane requests
  provisional = proportional cell-inventory cap(requests)  # preview only
  for each fixed routing forest:
    aggregate provisional requests to source nodes
    bottom-up:
      W[node] = local_request[node] + sum(admissible[child])
      admissible[node] = min(W[node], capacity_rate[parent_edge]*h)
    top-down:
      scale[node] = ancestor_scale * admissible[node]/W[node]
      scale local requests and pass scale to descendants  # safe zero W
    map scaled coarse amounts back to original uptake cells
  required_C[o] = sum(candidate_P[o,i] * route_price[o,i])
  energy_scale[o] = min(1, current_C[o]/required_C[o])  # zero price: 1
  final_P = candidate_P * energy_scale
  atomically debit soil by final_P; credit matching central P pools
  debit recomputed realised C cost; increment respiration counters
  carry reduced C pools into next substep
```

Capacity-only B skips energy scaling; A skips forest rationing. `T*h` is the current substep allowance; never reuse the full outer-step allowance each iteration. Uniform energy scaling after capacity rationing cannot violate a capacity ceiling. Available photosynthate can fund transport at this soil phase; it is a declared timing assumption. A new transport debit on bilateral trade belongs inside the coupled allocation integration so it competes with growth/reproduction, rather than as an after-the-fact charge.

```text
C internal transport substep:
  diffuse soil; compute membrane requests with declared storage feedback
  debit accepted soil P and add it to mapped local internal P amounts
  derive edge fluxes from current concentrations, volumes and driving forces
  select qualified duration/solver using internal transport stability criteria
  jointly limit every donor's outgoing flows by its stock and available C
  update each edge's two endpoints with equal/opposite realised amounts
  move sink delivery only into central/exchange-ready pools
  record realised respiratory C; preserve separate fungal trade ownership
```

The donor limiter must constrain all outgoing paths together. Charging/removing requested donor P and crediting only C-affordable receiver P would destroy mass. Internal delivery and existing continuous allocation require a newly declared splitting and timestep qualification; this pseudocode is not evidence that that splitting is accurate.

## Cost, memory and implementation comparison

The following are complexity estimates, not measured performance. Let `N` soil cells, `H` chronological cohorts, `K` coarse transport nodes, `E` edges and `S` soil substeps.

| Design | New live state | Leading extra work | Main implementation risk |
|---|---|---|---|
| A: energetic | O(1) resource/respiration diagnostics; optional O(N) temporary path/cost arrays | O(SN) elementwise work/reductions | Finite C paid twice, transport competing with biology in an undefined order, mean distance hiding anatomy |
| B: coarse route capacity | O(K+E) indices/capacities (mostly static), scalar counters; no P stocks | O(N) projection plus O(S(N+K+E)) simple tree passes; concentration solve costs extra | Shared-path demand or attachment misrepresented; two-phase arbitration changes soil competition |
| C: dynamic graph | O(K) for each internal stock, O(E) routing, geometry maps | O(SN) projection plus transport solves/substeps; solver work depends on conditioning | Stiffness under thinning, storage/death ledgers, arrival/source timing |

Do not form an `N × H` array to connect each cohort to every soil cell. #89's compact shell geometry can project surviving density or coarse cut occupancy using existing/compact geometry algorithms; benchmark that projection separately. A coarse transport node may combine many soil cells but must retain enough routing structure to expose depleted interior cuts. It need not coincide with a chronological cohort. The initial cohort is equal age over its full geometry; age order is not the same as a spatial transport node index.

[ADR-0019](../adr/0019-separate-policy-numerical-and-soil-clocks-and-qualify-resolution.md) fixes a default 1000×2000 soil grid: 2,000,000 cells. A single float32 full-grid state array is 8,000,000 bytes = 7.63 MiB; two internal-P fields are 15.26 MiB per environment or 3.81 GiB for batch 256, before intermediates, checkpointing, policy state or compilation. This is arithmetic, not observed peak memory. With coarse `K=1000`, two amount vectors instead occupy about 7.8 KiB per environment; maps and operators also cost memory.

All proposals can use fixed-shape JAX arrays, masked absent edges/nodes and `lax.scan`/`fori_loop` passes, with static shape and tree topology allowing `vmap`. Dynamic topology requires padded bounded arrays and explicit overflow checks. Never save internal vectors/cohort vectors in every rollout diagnostic unless intentionally requested; resumable states must preserve actual live stock/attachment history. Throughput must be measured after compilation on the intended training batch; grid-array arithmetic is not such evidence.

## Calibration, identifiability and discriminating experiments

1. Hold geometry/standing maintenance fixed; vary source-to-sink path while measuring delivered tracer P, local retained P, delivery time and respiration. Distance-only steady endpoints cannot distinguish weaker capacity, slower transport, exchange regulation and extra respiration.
2. Independently thin an interior corridor and distal uptake tissue at equal total living length. A local-average-density or uptake-only model predicts differently from a shared-path model. Compare predicted distal delivery, not just whole-plant biomass.
3. Test root age as transport maturation separately from mortality. Defaulting living-length capacity to a linear function is a controlled approximation; mature conducting xylem and uptake-active tissue are not physiologically interchangeable (Melchior & Steudle; Boursiac above).
4. Separate fungal C delivery for maintenance/front growth from P exported to the host. Zero P trade does not imply zero C transport (Bago above). If only P transport is implemented, state this limitation explicitly.
5. Fit an effective capacity scale first, then explore a separate energetic scale. Whole-organism growth alone will confound these with `jmax`, `kappa_c`, growth efficiency, geometry tortuosity and attachment fraction. Keep zero-cost/high-capacity reference cases and sensitivity ranges rather than claim one identified mechanistic parameter set.

Standing `kappa_c` is already an operating respiration cost in [organism traits](../../src/mycormarl/fungus/traits.py). Audit its provenance before adding transport respiration: either estimate a separable increment or repartition maintenance. A carbon-per-P penalty automatically tied to newly accepted uptake also penalises fungal retained P as if already delivered; a charge on host export needs provenance and misses internal fungal use. Name which process the experiment prices.

## Proposed verification before adoption

These are proposed checks; no model tests or transport benchmarks were run for this research note.

- Analytical one-edge and two-edge series/parallel examples: path subdivision leaves conductance and linear distance cost invariant; two equal distal branches sharing one trunk cannot each claim the full trunk allowance.
- Resource accounting through the public numerical environment: depleted C, zero uptake, absent partner, biological death, inventory-capped competition and unequal prices. Verify soil + central + local + structural P + all P exports closes; C inputs minus pool/structure/export/respiration changes closes within numerical tolerance.
- A direct comparison where one organism is carbon/transport limited while both request the same depleted soil cell: verify the declared redistribution rule and no P disappearance. Zero transport capacity with zero internal storage leaves rejected P in soil.
- Oldest-first interior thinning, still-active young distal uptake, floor exhaustion, front retreat and regrowth without repair. Retained floor tissue remains both maintenance-paying and uptake-active under #89; capacity is a separate prediction.
- Dynamic model only: tracer impulse delay, closed-graph conservation, nonnegative stocks, death/remapping export accounting, no delivery across a missing attachment, no spurious P when empty node volume returns during regrowth.
- Fixed physical action schedule, identical initial C/P stocks and matched duration across refined environment/soil/transport clocks. Qualify both uptake and sink delivery; a stable solver need not be accurate. Scalar/JIT/vmap agreement and checkpoint continuation exercise fixed-shape state.
- Benchmark production grid, declared coarse graph and representative training batch: compile time, warmed steps/day, peak memory, and extra output size separately. Accept or reject a coarse mesh using biological outputs under refinement, not throughput alone.

## Decision ready for discussion

The scientifically useful first comparison is **A versus B versus B+A**, each with the same uptake closure, oldest-first mortality, attachment assumption and fixed physical experiment. This isolates an energetic distance effect from a surviving-capacity effect. Add C if the observed/research question concerns arrival lags, fungal storage, or maintained uptake with reduced host delivery. Implementing C without data for storage feedback and internal mortality fate would buy extra unidentifiable mechanisms; A alone cannot answer the strongest consequence of old interior thinning.

## Cross-review of independent assessment B

Reviewed [assessment B](transport-model-independent-b.md) after completing this draft. Both assessments prefer coarse shared routes with separate energetic parameters, distinguish fungal collection from plant trade, and reserve dynamic stocks for storage/delay questions. B makes a useful additional comparison between service capacity and a quasi-steady conductance-network solve; this note initially grouped those too loosely.

**Accepted correction:** parent and B identified that the original draft assigned every serial edge the same complete driving-concentration difference while defining its cap as `G*Delta_cmax`. That fails path-subdivision invariance. The recommendation above now uses length-independent streaming service capacity, with distance entering explicit costs, and treats true resistance as a separate node-potential solve. This is a material model correction, not a numerical detail.

**Remaining modelling choices:** B's fungal global-collection interpretation needs a declared destination just as A's does. A root-interface mixing pool is workable but forces P used for distal fungal growth to return inward first; a purely global fungible pool cannot identify nearest-root distance or the root route of later host trade. Both reports explicitly leave this as a partial P-path model until endpoint/provenance and tipward C supply are specified. Neither report's microscopy evidence calibrates the assumed net P service rate.

**Implementation differences:** this note lays out a coarse conservative dynamic graph directly; B separately lists a quasi-steady conductance solve. Both are valuable comparators because the latter introduces resistance without travel-time stocks. B's elementary arithmetic check is limited evidence, not a runnable model comparison or benchmark. A's listed verification remains proposed and unexecuted.

No additional blocking defect found in B's stated conservative tree allocation: child admissible amounts enter parent demand, and top-down scaling propagates shared ancestor rationing. Both first-pass arbitration schemes leave blocked P for later substeps, so their competitive predictions need substep refinement or an explicit joint allocation solve before scientific interpretation.
