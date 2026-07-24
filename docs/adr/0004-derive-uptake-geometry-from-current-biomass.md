# ADR-0004: Derive uptake geometry from current biomass

**Status:** Accepted

## Context

Maintenance deficits can destroy plant or fungal biomass. Root and
external-hyphal length-density fields determine the organisms' soil phosphate
uptake geometry.

One possible model would preserve the historical growth envelope and thin its
density after biomass loss. That would make damage path-dependent, but it would
also permit extremely sparse legacy networks and would require new rules for
connected roots, fungal attachment, tissue abandonment, and whether subsequent
growth repairs or extends the network.

The current model instead reconstructs both density fields from post-allocation
biomass before soil uptake.

## Decision

Root and external-hyphal uptake geometry will remain a memoryless function of
current living biomass and static traits.

- Maintenance-induced biomass loss has no separate persistent damage state.
- Fungal loss retracts the saturated hemispherical colony boundary.
- Plant loss reduces the radial extent of the root disc in each represented
  depth layer; the configured rooting-depth distribution remains unchanged.
- Later regrowth may recreate previously occupied geometry.
- The model will not currently track a historical front, local density damage,
  a minimum-density backbone, or explicit root–fungus connectivity.

Density fields continue to be reconstructed after maintenance and allocation,
so maintenance loss cannot be offset by growth within the same transition:
any maintenance deficit exhausts at least one essential resource required for
growth.

## Consequences

- Uptake geometry remains connected by construction under its axisymmetric
  geometric assumptions and cannot become a near-zero-density legacy network.
- Equal current biomass and traits imply equal uptake geometry, regardless of
  maintenance history.
- Policies may rebuild retracted uptake territory in later transitions by
  allocating resources to growth.
- The biological model does not represent persistent tissue damage or enforce
  fungal attachment to living roots.
- Introducing path-dependent density loss later would reopen this decision and
  require explicit spatial-connectivity and regrowth semantics.

## Potential extension

Maintenance-deficit mortality could later reduce length density within the
established root or mycelial envelope instead of retracting the uptake
geometry. This extension should not be introduced as proportional density
scaling alone: repeated deficits could otherwise leave an arbitrarily sparse
network that remains spatially extensive but is no longer biologically capable
of supporting resource transport.

A complementary extension would therefore model network transport capacity and
use it to define the minimum viable connected density. Density loss, survival,
front extension, and any removal of disconnected tissue would then be governed
by whether the remaining root or mycelial network can transport sufficient
resources between its uptake surfaces and the plant–fungus exchange region.
