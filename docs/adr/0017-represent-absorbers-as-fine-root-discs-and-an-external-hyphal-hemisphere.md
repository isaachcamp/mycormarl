# ADR-0017: Represent absorbers as fine-root discs and an external-hyphal hemisphere

**Status:** Accepted; retrospective record of the implemented geometry,
recorded 21 October 2026.

## Context and decision

Growth must change absorbing length and surface in a physically interpretable
way without introducing individual root or fungal network simulation. Retain
two biomass-derived spatial closures on the common axisymmetric soil grid:

- Convert whole-plant dry biomass to absorbing fine-root length using
  `kfroot * SRL`, applying the fine-root fraction once. All of this represented
  length is uptake-active; there is no additional inactive/coarse-root
  fraction hidden in that conversion.
- Partition root length with the prescribed depth profile normalised over
  the intended maximum rooting depth, not the simulated soil depth. Give
  each layer a disc radius implied by its assigned length and the common
  within-disc length density. The profile controls depth allocation while
  growth changes radial extent.
- Convert fungal dry biomass through structural C and tissue C density to
  cylindrical external-hyphal length. Represent that length as an isotropic
  hemisphere at fixed local saturation density. Spores, intraradical hyphae,
  vesicles and arbuscules are not separate geometry compartments.
- Volume-average partially occupied cells using the disc/hemisphere
  intersection. Clip structure outside the represented soil rather than
  redistributing its length into the domain or increasing local density.
  Fungal boundary contact does not instantly saturate every soil cell.

## Trade-offs and consequences

These closures preserve analytically checkable represented length and expose
how structural biomass changes uptake. They avoid persistent branching,
connectivity, directional growth and ownership state. The entire organism
still pays construction and maintenance costs: outside-domain structure has
no in-domain absorbing surface but is not removed from biomass accounting.
The soil boundary limits represented uptake geometry, not fungal biomass.

Reconstruction from current living biomass, including contraction and possible
regrowth, is governed by
[ADR-0004](0004-derive-uptake-geometry-from-current-biomass.md). Local senescence,
damage, turnover and absorber age remain unrepresented. Constant allometry
and uniformly active fine roots are provisional modelling assumptions; the
species-level trait aggregates are not matched measurements of one plant.

Volume averaging conserves length, but applying nonlinear uptake closures to
an averaged front-cell density is still a spatial approximation. Grid
refinement must assess uptake as well as geometric conservation. Adding a
network, sub-cell quadrature, an active-length fraction or age dependence
would require an explicit model extension and new qualification.

## Evidence

The [growth model](../growth-model.md#plant-biomass-to-root-geometry) gives
conversions, intersection invariants and provenance. The
[root](../../src/mycormarl/plant/roots.py) and
[mycelium](../../src/mycormarl/fungus/mycelium.py) modules own the pure
geometry functions. [Geometry tests](../../tests/test_growth_geometry.py)
cover contained length, partial fronts, clipping, zero biomass and
post-maintenance/growth reconstruction. Numerical qualification follows
[ADR-0019](0019-separate-policy-numerical-and-soil-clocks-and-qualify-resolution.md).
