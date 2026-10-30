# ADR-0022: Separate fine-root mortality from persistent backbone connectivity

**Status:** Accepted modelling assumptions, 27 October 2026; not implemented.
The mortality specification and dependent tickets must follow this revision
before implementation. Exchange availability throughout the footprint is accepted;
the energetic transport law remains a separate design.

Represent mortality in the absorbing fine-root population without a positive
fine-root density floor. Assume an unresolved conducting-root backbone remains
connected to surviving distal fine roots wherever roots have previously grown,
while the plant is operational. For fungal transport-cost geometry, use the
closest root-containing cell, including cells containing this implicit
conducting backbone after their absorbing fine roots have died. Every cell
in this footprint is assumed available for fungal exchange without separately
tracked colonisation or an establishment delay. The nearest cell therefore
provides an eligible exchange endpoint, not merely a distance proxy.

## Reasoning

Absorbing fine-root density and stem connectivity describe different functions.
An empty patch of the represented fine-root population need not imply a break
in an unresolved conducting-root system. Retaining living absorbers solely to
represent that system would preserve uptake and maintenance for the wrong
reason. Separating these concepts supports the
[non-limiting transport approximation](0021-treat-internal-transport-capacity-as-non-limiting.md)
without requiring an explicitly resolved coarse-root network.

This is a functional modelling approximation, not a claim that all higher-order
roots always survive or support mycorrhizal exchange. Anatomical differentiation
between absorbing distal orders and conducting proximal orders supports the
separation, but species and root order affect the association. For example,
[Guo et al. (2008)](https://doi.org/10.1111/j.1469-8137.2008.02573.x)
found transport-oriented higher orders without mycorrhizal colonisation in
their sampled trees. Exchange eligibility of our implicit backbone is therefore
an explicitly accepted additional approximation, not an inference from anatomy.

## Consequences

- Keep whole-plant maintenance-induced biomass loss and the fixed fine-root
  fraction. Root cohorts account for the associated fine-root loss; this is not
  selective fine-root sacrifice with protected coarse-root mass.
- Root cohorts may thin to zero and disappear oldest first. Plant
  `death_fraction` continues to govern organism death; it no longer defines a
  fine-root density floor. The root floor-exhaustion retreat rule loses its
  rationale and must be removed from the revised root specification.
- Distinguish living absorbing-root density from a persistent root-containing
  footprint. A cell can have zero fine-root uptake while still containing an
  implicit conducting route. Track this footprint separately from current
  absorbing density; growth can extend it and fine-root loss does not erase it.
- Do not add coarse-root absorbing surface or an additional protected biomass,
  construction or maintenance compartment. Backbone connectivity is implicit
  within the aggregate plant approximation. Biological termination and absence
  still disable plant activity and bilateral trade.
- For fungal cost distance, use distance to the closest root-containing cell
  plus an average distance to a root within that cell derived from root length
  density. Use the existing fine-root density parameter
  (`PlantTraits.root_length_density`) as the provisional within-occupied-region
  spacing proxy; do not introduce a separate conducting-root density parameter.
  This reuses a reference density, not the post-mortality uptake-density grid:
  spacing remains defined in backbone-only cells. It does not assert equal
  biological densities of conducting and absorbing roots. Same-cell
  co-occurrence has zero intercell distance but need not have zero total
  distance. The spacing coefficient, distinct-cell metric and root-to-stem cost
  remain design work. This is not a reduced-capacity compensation penalty.
- Every cell in the persistent footprint is exchange-available while the plant
  is operational, including cells with no surviving absorbing fine roots. No
  colonisation state, attachment history or establishment delay is represented.
  Availability does not force trade or create resources: realised exchange
  remains subject to organism presence/liveness, the existing bilateral trade
  commands and available donor pools. This does not permit transport across
  disconnected fungal tissue or resolve fungal network routing.
- Fungal contact loss is allowed: do not preserve contact artificially when
  mortality leaves the living network disconnected from every root-containing
  cell. In that state bilateral C/P trade is unavailable, while existing stocks
  and ordinary maintenance/survival dynamics continue. Loss of contact is not
  itself immediate organism death. Global actions and pools remain appropriate
  for a single internally connected, rapidly mixed fungal network; contact
  detection and its transition timing require specification before coding.
- The fungal floor/retreat rule is being revisited to allow a central empty
  region; the precise mortality geometry is not settled by the contact-loss
  decision. There is still no spontaneous age-dependent death or physiological
  age penalty.

This revises the root-floor assumptions of
[specification #89](https://github.com/isaachcamp/mycormarl/issues/89)
and [ticket #92](https://github.com/isaachcamp/mycormarl/issues/92), and qualifies
ADR-0021's general floor language. It does not yet replace the current
memoryless production geometry in ADR-0004/ADR-0017. The published specification and dependent tickets track agreed changes and
remaining implementation gates. In particular, fungal mortality geometry and
contact detection/transition timing must be settled explicitly before coding;
this ADR does not choose those unresolved rules.
