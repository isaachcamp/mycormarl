# Age-dependent absorber mortality: root-backbone revision

27 October 2026. Discussion state, not an executable replacement implementation
plan. [ADR-0022](../adr/0022-separate-fine-root-mortality-from-persistent-backbone-connectivity.md)
records the accepted root assumptions. GitHub #89 and #90–93 remain the published
specification/tickets, but their root-floor requirements are stale relative to
this decision and require reconciliation before implementation.

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
the exact floor/removal geometry remains to be finalised. Assess fungal
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

## Accepted transport-shell scope, 28 October 2026

Use a configurable physical shell width in cm for transport-distance averaging;
its numerical baseline remains unset. The shell is anchored at the outer front
and intersected with surviving tissue, so its inner radius is
`max(a,b-w,0)`. This changes only the transport-demand proxy: existing uptake,
outer-boundary growth, global actions and cohort-budget rules remain in scope.
It does not spread actual tissue deposition over the band. Width must be
independent of timestep and grid spacing. FWHM estimation and Edelstein tip-field
dynamics are deferred; neither blocks refinement of the existing mortality plan.

### Earlier FWHM proposal (deferred)

Uptake is explicitly unchanged. The user proposes deriving a transport-shell
width from the FWHM of the uptake peak behind the front as a proxy for a
spatially dispersed growth front. The [derivation notes](../technical-summaries/transport-shell-geometries/README.md)
record the mathematical proposal and the distinction between the measured peak
band and borrowing its width for a shell anchored at the growth front. Profile
normalisation, multiple/absent peaks, truncated peaks and a no-uptake fallback
remain to be specified. This does not restrict active uptake or change where
new tissue is constructed; explicit transport expenditure remains separate
from the mortality implementation.

## Edelstein-style alternative (deferred, 28 October 2026)

The user asked whether evolving growing-tip and hyphal-length densities would
produce a dispersed growth front directly. The [primary-source assessment](../technical-summaries/edelstein-tip-density-front.md)
finds conditional support for a finite-width tip pulse, with gross deposition
`v*n` supplying a direct growth-transport weight. The user chose a parameterised
transport width instead of this morphology change for now.
This would replace the sharp outer-shell spatial-growth rule, require exact
coupling to policy-funded C/P construction and alter age-history storage because
new birth cohorts overlap in space. Neither tip/length fields alone nor their
anastomosis terms implement oldest-first structural mortality, root attachment
or internal resource transport. The uptake law remains unchanged; geometry-driven
uptake outcomes can still differ. Those changes are deferred; the selected
parameterised transport shell preserves disjoint outer-boundary birth shells
as the current growth approximation.

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
