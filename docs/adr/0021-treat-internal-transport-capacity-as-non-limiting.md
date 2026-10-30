# ADR-0021: Treat internal transport capacity as non-limiting

**Status:** Accepted, 27 October 2026. This records a model-scope decision;
it does not implement an explicit transport-carbon cost.

Treat internal root and external-hyphal transport capacity as non-limiting
within the model's assumed living, connected infrastructure. Reduced interior
length density will not introduce a transport-throughput cap, delivery delay,
congestion penalty or compensatory carbon charge. This keeps the proposed
transport-cost extension focused on energetic expenditure without adding an
uncalibrated capacity mechanism.

## Reasoning and alternatives

The alternatives considered included distance-dependent expenditure, shared-route throughput limits,
quasi-steady resistance networks and dynamic internal transport. Lower
conducting density can reduce capacity, but that alone does not imply greater
carbon expenditure per unit delivered: delivery might decrease, capacity might
remain above demand, or the organism might expend energy to compensate.
Choosing the compensatory response would require an additional physiological
law, rather than follow from density loss itself.

The available evidence does not identify a transferable relationship between
living length density, net resource throughput and respiratory carbon cost for
this model's plant–fungus system. Fungal imaging resolves bidirectional flows
and wider, faster trunk routes, but its maintained root-route efficiency is a
graph-theoretical metric, not a calibration of throughput or energetic cost.
Root hydraulic experiments distinguish absorbing and conducting functions;
fine-root length density does not establish vascular capacity. See
[Oyarte Galvez et al. (2025)](https://doi.org/10.1038/s41586-025-08614-x)
and [Boursiac et al. (2022)](https://doi.org/10.1093/plphys/kiac281).

Explicit capacity would also require choices about attachment, shared routes,
the location of globally available fungal P, and competition for constrained
delivery. Dynamic transport would add internal inventories and their fate
during mortality. Those choices would change the interpretation of uptake,
trade and spatial death, with further state, solver and qualification costs.
We defer them to avoid making the transport-cost study depend on several
unidentified mechanisms. This is a tractability and identifiability decision,
not evidence that biological transport bottlenecks are negligible.

## Consequences and scope

**Root-specific revision:** [ADR-0022](0022-separate-fine-root-mortality-from-persistent-backbone-connectivity.md)
removes the proposed fine-root floor and assumes an implicit persistent
conducting-root backbone. The fungal floor/retreat geometry is also under
revision. References below to retained floor tissue describe the earlier fungal
proposal conditionally; they do not select a final fungal mortality geometry.

- Density continues to affect absorbing geometry, soil-to-membrane uptake,
  represented tissue investment and standing maintenance through their
  existing or separately agreed relationships. This decision removes only
  feedback from reduced internal transport capacity.
- For the planned [oldest-first maintenance mortality extension (#89)](https://github.com/isaachcamp/mycormarl/issues/89),
  any retained living fungal tissue remains uptake-active and maintenance-paying.
  If a fungal density floor is retained after the pending geometry decision,
  it is a continuity approximation, not a calibrated flux threshold. Interior
  thinning will not independently throttle distal delivery.
- Do not infer an inverse-density carbon surcharge or an age-dependent
  conductivity penalty from mortality. Density may still affect a separately
  justified route-distance approximation; that is distinct from compensation
  for reduced capacity.
- Distance-dependent carbon expenditure remains open for separate design and
  calibration. No coefficient, attachment rule, pool location, cost payer or
  update timing is selected here. Carbon affordability and finite soil P remain
  genuine resource constraints if an energetic cost is later implemented.
- The approximation does not establish root–fungus attachment or authorise
  transport across biologically disconnected tissue. Existing absence/death
  gates and the agreed continuity approximation remain applicable.
- Results cannot establish whether a sparse interior can sustain distal demand
  in a real organism, or predict transport-induced accumulation, lag or failure.
  Revisit this decision if those become scientific endpoints or matched
  measurements constrain the relevant conducting infrastructure and fluxes.

This explicitly declines the capacity-based extension suggested in the
**Potential extension** section of [ADR-0004](0004-derive-uptake-geometry-from-current-biomass.md).
It does not supersede ADR-0004's current memoryless geometry or implement
#89's planned replacement. Soil diffusion and uptake remain distinct from internal organism transport. Historical research
recommendations remain alternatives considered, rather than an approved
capacity implementation plan.
