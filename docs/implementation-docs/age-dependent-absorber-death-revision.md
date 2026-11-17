# Age-dependent absorber mortality: root-backbone revision

27 October 2026. Discussion state, not an executable replacement implementation
plan. [ADR-0022](../adr/0022-separate-fine-root-mortality-from-persistent-backbone-connectivity.md)
records the accepted root assumptions. GitHub #89 and #90–93 remain the published
specification/tickets, but their root-floor requirements are stale relative to
this decision and require reconciliation before implementation.

## Latest transport/mortality clarification, 11 November 2026

Assign zero explicit carbon cost to fungal polyphosphate translocation for now.
Keep soil uptake, P exchange commands, structural requirements, conservation
and contact-loss gates unchanged. This does not assert that every P-processing
operation is free or remove existing maintenance expenditure.

Preserve the outer front under oldest-first mortality while the organism is
operational. The earlier provisional outer-retreat fallback is no longer the
selected behaviour. Its removal is an explicit rule, not a consequence of the
name age-dependent mortality alone. Reconcile any retained fungal floor with
the required loss budget and non-retracting front before implementation; never
silently forgive biomass loss. Whole-organism death/absence still disables
active uptake geometry and bilateral exchange.

The transport shell is the current step's new radial growth, with no retreat
correction. Zero growth gives zero construction-related carbon-transport charge;
ongoing P transport has zero explicit cost, eliminating the pending front-surface
P-distance fallback. Standing maintenance continues. Carbon-delivery cost
coefficients and its resource-funded transaction remain to be designed. The selected transport model below incorporates this clarification.

## Agreed changes

1. Remove the positive fine-root density floor. Maintenance-induced fine-root
   loss is allocated oldest first, permitting partial thinning and complete
   removal of a cohort. Plant `death_fraction` retains its organism-death role.
2. Assume the implicit conducting-root backbone remains connected to surviving
   distal roots while the plant is operational. Root-containing occupancy
   persists where roots have grown, even if absorbing fine-root density becomes
   zero. It is separate from the uptake-active density field.
3. Preserve whole-plant biomass loss, the fixed fine-root fraction, depth
   allocation, structural-P accounting, death gates and front-only growth.
   There is no new protected coarse-root biomass or direct coarse-root uptake.
4. Use distance to the closest root-containing cell, including backbone-only
   cells, plus the average distance to a root within that cell derived from
   length density. This latest clarification replaces the earlier cell-only
   distance decision. Reuse the existing fine-root density parameter
   (`PlantTraits.root_length_density`) as the provisional spacing proxy, not the
   post-mortality density field. No new non-fine-root density parameter is added;
   the proxy remains defined in backbone-only cells.
   Root-to-stem transport remains a distinct leg whose energetic treatment has
   not been selected.
5. Keep exact numerical-step cohorts, unchanged policy observations/actions,
   compact evaluation outputs and the no-background-senescence agreement.
   Fungal floor/retreat is now under revision; do not treat the previous
   positive-floor rule as a final decision for the hollow-network proposal.
6. Every cell in the persistent root footprint is available for fungal exchange,
   including cells whose fine roots have died. Do not track colonisation,
   attachment history or establishment delay separately. The closest
   root-containing cell is an eligible exchange endpoint; availability does not
   force trade, credit P automatically to the plant, bypass donor-pool limits
   or preserve activity after organism death.
7. Permit loss of every fungal contact with the persistent root footprint.
   Bilateral C/P exchange then becomes unavailable; existing stocks and ordinary
   maintenance/survival dynamics continue. Do not guarantee contact geometrically
   or classify contact loss itself as immediate biological death. The three
   whole-organism Rate actions remain unchanged.

## Consequences for the old plan

### Total fungal biomass versus external tissue

[Issue #96](https://github.com/isaachcamp/mycormarl/issues/96) introduces static
`irm_fraction` allocation: IRM is `f * B_f` and ERM is `(1-f) * B_f`.
Whole-fungus maintenance, construction and structural-P accounts use total
`B_f`. External cohorts must represent only ERM, including external tissue
outside the soil grid. A total growth increment or mortality loss `delta_B_f`
changes external cohort mass by `(1-f) * delta_B_f` and implicit IRM by
`f * delta_B_f`. IRM has no independent cohort state or protected survival.
Do not allocate the entire total-biomass loss to external cohorts or apply the
fraction again to total structural-P mortality exports. IRM/ERM quantities are
derived when needed, without additional stock diagnostics or actor inputs.

The published #89/#91 fungal mass-to-cohort wording requires reconciliation
with this split before implementation, alongside the existing root and contact
revisions. The allocation itself adds no colonisation state, establishment
delay, root-capacity constraint or exchange gate. ADR-0022's exchange-available
footprint remains a separate agreed future approximation.

### Spatial mortality revision

The plant and fungus can no longer share one positive-floor/retreat policy.
Share cohort accounting where useful, but specify their mortality rules
separately. Root oldest-first removal has all remaining fine-root tissue
available to meet its prescribed loss, so it needs no special floor-exhaustion
retreat mechanism. Do not reinterpret complete root-cohort deletion as forced
youngest-first outer retreat.

With front-grown, chronologically ordered root cohorts, older inner tissue can
disappear while younger outer absorbers persist. The backbone footprint records
the root-containing region independently. Ordinary oldest-first loss does not
remove the youngest outer cohort before all older cohorts have been consumed;
avoid inventing a live-plant retreat scenario that contradicts this order and
the fixed fine-root fraction. Define reset occupancy, zero-growth steps,
whole-organism termination and front growth from the surviving state explicitly
when refining the plan.

The historical footprint is not an anatomical reconstruction of every coarse
root. Occupied shell/disc descriptors may provide a compact representation;
assessing a grid mask versus analytic occupied geometry belongs in planning.
Domain clipping must still not erase biological mass or maintenance accounts.

## Resolved exchange question

The user confirmed that every persistent root-containing cell is an available
exchange location without separately tracked colonisation. This is stronger
than potential colonisability and is now an accepted model approximation.
It does not claim that all real transport roots are colonisable. No separately
resolved exchange surface or cellwise exchange kinetics is introduced by this
availability decision; the existing organism-level trade abstraction remains.

## Remaining design work

The root-floor, exchange-availability, provisional density proxy and permission
for contact loss are resolved. A central empty fungal zone is under discussion;
the exact floor/removal geometry remains to be finalised subject to the latest
non-retracting-front decision. Assess fungal
self-connectivity separately from root contact: a hollow connected shell can
lose every root connection. The user accepts that outcome, with trade disabled
rather than artificial preservation of contact.

The existing specification already uses strict oldest-first eligible loss,
not a smooth age-weighted mortality preference. Clarify whether the user's
simplification means complete removal of earlier cohorts with a partially
depleted final affected cohort (exact loss budget), or all-or-nothing cohort
deletion (which would require a different budget/geometry treatment). The
existing floor reused organism `death_fraction`, so removing a floor does not
remove that parameter's whole-organism death role.

Whole-organism Rate actions can remain valid for a connected hollow fungal
network under assumed rapid mixing and non-limiting transport. They do not
choose exchange sites, allocate growth locally or repair the interior. Loss
of every root contact will require an explicit exchange gate; fragmented
living components would require guaranteed connectivity or separate resource
accounting. The gate is agreed behaviour but is not implemented. Its timing
relative to maintenance, allocation, growth and contact restoration needs
specification, without retrospectively cancelling already executed transfers.

Planning must define the compact footprint representation,
reset/growth/termination updates, contact-loss and restoration timing,
distance metric between distinct cells, root-spacing coefficient and resolution
checks. The root-density proxy reuses the existing parameter for now; no matched
non-fine-root density estimate has been selected. Explicit
transport-carbon coefficients, cost ownership and update timing remain a
separate extension. Do not introduce those costs as part of mortality work.

## Selected transport-shell model

Use the current numerical step's realised radial expansion as the transport
shell width. Under the non-retracting operational-front rule, this is the band
between the old and new outer fronts, with no retreat correction or free width
parameter. Weight delivery distance by new absorbing length: nearest persistent
root footprint plus reference-density spacing for fungi, and straight-line
distance to the stem base for fine roots. Price delivered construction carbon
with a linear distance charge, using separate, unfitted organism coefficients.

Zero construction has zero construction-related transport charge. Explicit
fungal polyphosphate-translocation cost is zero, so no ongoing P-distance
fallback is required. Uptake and ordinary maintenance continue unchanged.
The [selected model and figures](../technical-summaries/transport-shell-geometries/README.md)
record the equations, scope and remaining funded-budget integration decisions.
This numerical growth increment requires cumulative-cost convergence under
timestep refinement; it is not a fixed biological active-zone width.

## Ticket reconciliation after discussion

- **#89:** separate root and fungal mortality, replace root-floor connectivity
  with the backbone assumption, identify nearest-cell geometry and record
  exchange availability throughout the operational plant's footprint.
- **#92:** remove root floor and floor-exhaustion retreat; add complete old-root
  loss, distal fine-root connectivity, separate historical occupancy and
  unchanged whole-plant mass accounting to the acceptance criteria.
- **#93:** qualify root interior gaps and fungal floor/retreat separately; check
  that backbone-only cells do not gain direct P uptake and termination disables
  activity. Preserve conservation, refinement and performance gates.
- **#90/#91:** review interface/diagnostic assumptions for a root-positive-floor
  dependency; the fungal biological rule and compact-output requirement stand.

No production changes, GitHub edits or numerical tests were performed in this
discussion revision. The next step is to reconcile the published specification
and tickets, then use the existing-plan workflow to refine the implementation
sequence. Do not execute the old root-floor criteria as if the new decision
had not occurred.
