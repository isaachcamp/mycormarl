# Transport models: independent research and implementation comparison

> **Subsequent accepted decision, 27 October 2026:**
> [ADR-0021](../adr/0021-treat-internal-transport-capacity-as-non-limiting.md)
> treats internal transport capacity as non-limiting. The capacity-based
> recommendations below are retained as research alternatives considered;
> they are not the selected implementation direction. Explicit energetic
> transport costs remain open for separate design.

27 October 2026. Two independently tasked research agents investigated the same transport-cost question, examined the production code and [mortality specification #89](https://github.com/isaachcamp/mycormarl/issues/89), then cross-reviewed their proposals. The user requested design comparison with pseudocode. No transport model was implemented in production and no comparative runtime benchmark was run.

Detailed reports: [investigation A](transport-model-independent-a.md), [investigation B](transport-model-independent-b.md). The [earlier synthesis](transport-costs-and-absorber-age.md) provides additional literature and geometry proposals. Equations below are candidate closures, not fitted biological laws.

## What the investigations agree on

Separate membrane uptake, internal delivery, storage, host exchange and transport respiration. Density already determines absorbing surface in the current uptake model; its role in axial conducting capacity is an additional hypothesis. Root absorbing length is not a measurement of vascular area. Root hydraulic experiments show why axial and radial conductance must be distinguished, but do not calibrate phosphate-specific respiratory costs. [Boursiac et al. 2022](https://doi.org/10.1093/plphys/kiac281).

AM fungi have bidirectional flows and wider/faster trunk routes. These observations motivate shared routes and heterogeneous capacity, not a universal linear distance cost or inverse-distance throughput. The reported maintained root transport efficiency in the 2025 study is a normalised graph-theoretical shortest-path metric; it is not constant C/P efficiency, transit time or nutrient throughput. [Oyarte Galvez et al. 2025](https://doi.org/10.1038/s41586-025-08614-x).

The agreed mortality extension is oldest-first **maintenance-deficit** mortality, not an intrinsic age hazard. It retains living uptake-active interior tissue at a density floor, followed provisionally by retreat. Under a linear density-capacity proxy, thinning a particular shared segment to fraction `f` reduces its service capacity to `f` times baseline. The root and fungal floors imply 20% and 5% of that segment's baseline capacity. They do not establish whole-network capacity: another segment may already dominate, parallel routes may bypass it, and the initial equal-age cohort does not imply an old centre/young edge. [Specification #89](https://github.com/isaachcamp/mycormarl/issues/89).

## Four concrete designs

| Design | Relationship implemented | Added live state | Main prediction and limitation |
|---|---|---|---|
| Distance cost only | `C_resp = sum_i x_i*(e0 + eL*d_i)` | Scalar expenditure counters; route geometry/maps | Tests profitability of transport distance; cannot represent a shared interior bottleneck or arrival lag. |
| Shared throughput, optional distance cost | `Qmax_e = v_e*Aeff_e*cstar_e`; sum of distal deliveries must fit each edge | Coarse fixed routing forest, capacities and temporary tree buffers; no local P stocks | Tests whether thinning restricts distal benefit. Longer routes do not automatically reduce steady throughput. Delivery is instantaneous within each substep. |
| Quasi-steady resistance network | `F_e = G_e*(c_u-c_v)`, `G_e = D_eff*Aeff_e/length_e`, with node conservation | Coarse edge operators and node solve buffers; no dynamic local P stocks | Tests serial/parallel resistance and limited driving force. Needs consistent source/sink conditions and a network solve. |
| Dynamic conservative transport | Local P amounts updated by uptake, conservative edge fluxes and sink delivery | Internal P compartments, potentially C compartments and attachment history | Predicts lag, accumulation and uptake–delivery separation. Requires storage feedback, death/remapping fate and additional numerical qualification. |

Units: `x_i` is mg P delivered per substep; `e0` is g C/mg P; `eL` is g C/(mg P cm); `d` is cm; `v` is cm/day; `Aeff` is cm² conducting area; `cstar` and `c` are mg P/cm³ conducting material; `Qmax` and `F` are mg P/day; `G` is cm³/day. For each organism, an initial conducting-area proxy is `Aeff = chi*pi*a²*lambda_live*A_corridor`; its connected/oriented/conducting fraction `chi` needs calibration. The root absorber radius must not be mistaken for its xylem lumen radius.

A and B independently favoured shared coarse-route constraints plus a separately varied distance cost. A initially explored conductance-derived independent edge limits; review identified a subdivision problem and distinguished them from a consistent resistance solve. B preferred a streaming-inspired service limit because its steady capacity is invariant when a uniform edge is subdivided. Neither recommendation identifies an empirical physiological coefficient.

## Important difference between throughput and resistance

A service limit and a resistance law are alternative constitutive assumptions. They are not interchangeable ways to implement the same biology. For a uniform route, a fixed-speed service capacity `v*Aeff*cstar` stays constant as length increases; arrival time increases, and a separately assumed distance price may increase. For a fixed total concentration drop, the resistance model's steady flux decreases with total length.

Do not give every edge its own full source-to-sink concentration drop. Splitting an edge in half doubles each half's conductance; independently capping each at `G_half*Delta_c_total` would falsely double permitted throughput. A true network solve divides the total drop consistently, preserving the original flow. This is a decisive numerical test.

Likewise, multiplying one segment's resistance by `1/f` does not multiply the whole route's resistance by that factor. If that segment initially contributed 10% of a series route's resistance, total resistance changes by `0.9 + 0.1/f`: 1.4 at the root floor and 2.9 at the fungal floor. These are illustrative algebra, not calibration.

## Geometry and pool location are decisions, not implementation details

Use root routes to the stem collar, allowing a radial layer route and shared axial backbone. The current fine-root discs do not resolve that backbone: parameterise its capacity separately or explicitly assume unlimited backbone transport.

For hyphae, use routes to assigned connected exchange-capable roots. Nearest-root distance alone neither establishes colonisation nor identifies a path through living fungus. A small fixed set of attachment basins is a tractable closure. Central hemispherical routing should be a separately named alternative, not described as nearest-root transport. Flux-weighted mean distances are useful diagnostics; retain route-level information for bottlenecks.

The [current state](../../src/mycormarl/state.py) contains globally fungible free P. Fungal uptake enters fungal P; the fungus later chooses plant transfer through its trade action. Constraining fungal collection is therefore not automatically constraining host delivery. A model claiming the full hypha–root–stem route needs an exchange-local pool or an explicitly documented approximation for the location/provenance of traded P. Carbon supply to fungal maintenance and the growth front also persists when P export is zero. P-only transport pricing is a partial metabolism model. [Bago et al. 2003](https://doi.org/10.1104/pp.102.007765).

## Pseudocode for the smallest shared-capacity experiment

This is a proposed no-local-storage transaction. It preserves the current abstraction that delivered uptake enters the organism's available pool. A fungal collection sink and its relationship to host exchange must be declared first.

```text
After maintenance mortality and growth:
    derive surviving uptake density and coarse conducting capacity
    use declared fixed attachment/routing forest

For each soil substep of duration h:
    diffuse soil P
    compute both organisms' membrane uptake requests from same concentration
    preview existing proportional allocation of finite soil P

    for each organism:
        aggregate preview requests onto coarse source nodes
        bottom-up: cap each subtree's demand by shared edge rate * h
        top-down: propagate proportional reductions to source cells
        calculate C price of these candidate deliveries
        scale candidate deliveries down to fit carried available C

    atomically debit soil and credit organism P from final accepted amounts
    debit only the C expenditure for those realised deliveries
    carry reduced C into next substep
```

This deliberately does not redistribute a blocked organism's share immediately. Rejected P stays in soil for later substeps. A joint allocation with redistribution is a different design needing an explicit fairness rule and convergence/refinement checks. Global C scaling is feasible but may reduce cheap and expensive routes together; cost-minimising allocation would add another biological preference.

Modify the [soil transaction](../../src/mycormarl/soil/soil.py) before final soil removal/pool credit, rather than clipping delivery after P has disappeared from soil. Trade-related costs belong inside the [coupled allocation integrator](../../src/mycormarl/allocation.py), where incoming trade, biology and transport expenditure can compete for evolving pools. Current photosynthate is added before soil processing, so funding uptake transport from it is an explicit timing choice. Existing fungal maintenance is a broader operating-respiration proxy: repartition it or identify the new charge as additional sensitivity to avoid claiming independent calibration. [Parameter register](../model-parameter-register.md).

## JAX and memory comparison

The default [configuration](../../src/mycormarl/params.py) has 2,000,000 soil cells. One float32 field occupies 7.63 MiB. Two full-grid internal P fields add 15.26 MiB per environment, or 3.81 GiB across 256 environments, before intermediates and other state. These values were calculated from shape and dtype; they are not peak-memory measurements. Dynamic transport can instead use a coarse mesh, reducing state while adding another resolution assumption.

With `N` soil cells, `K` coarse transport nodes, `E` edges and `S` soil substeps, distance-only work is approximately `O(SN)`; the routing forest adds `O(S(K+E))` tree passes plus source mapping; a linear tree resistance solve can also use elimination/back-substitution, while loops or nonlinearities add solver work. Dynamic transport introduces internal substeps or implicit solves whose stability/convergence cannot be inherited from the soil diffusion scheduler. Fixed topology and bounded arrays suit JAX `scan`/`fori_loop` and batching. None of these asymptotic claims establishes acceptable training throughput.

Transport nodes are not chronological cohorts. Avoid an `N*cohort_capacity` mapping and avoid returning large transport/cohort vectors at every timestep. Store necessary histories in resumable live state, with compact evaluation outputs as required by #89.

## Recommended comparison before a production decision

Compare distance-cost-only, throughput-capacity-only, and their combination with identical uptake, mortality, attachment and maintenance assumptions. Add a genuine quasi-steady resistance version, matched at baseline throughput, to reveal whether conclusions depend on fixed-speed versus fixed-driving-force physiology. Reserve dynamic compartments for questions about lag, storage or fungal retention; those require uptake and host delivery to be distinct endpoints. The distinction is physiologically relevant but not quantitatively resolved by current parameters. [Schnepf & Roose 2006](https://doi.org/10.1111/j.1469-8137.2006.01771.x), [Hammer et al. 2011](https://doi.org/10.1111/j.1574-6941.2011.01043.x).

Require P/C accounting, shared-trunk and zero-connection cases, route-subdivision invariance, oldest-first thinning/recovery, matched-time soil/transport refinement, scalar/JIT/batch agreement, and measured warmed throughput/peak memory. Results should remain conditional on unresolved attachment, conducting fractions, service/driving-force parameters, cost coefficients and the interpretation of global pools.
