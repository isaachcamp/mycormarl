# Compare ordered and fully simultaneous state updates

Status: intended investigation, recorded 22 October 2026. This document proposes
a comparison; it does not select a new production update scheme or authorise
implementation of one. Tracked in [issue #87](https://github.com/isaachcamp/mycormarl/issues/87).
The current baseline is the three-rate allocation model
in [ADR 0020](../adr/0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md)
and [specification #79](https://github.com/isaachcamp/mycormarl/issues/79).

## Purpose

Determine whether the current ordering of physiological processes materially
affects resource allocation, survival or experimental conclusions compared with
a fully coupled treatment. Measure how differences change with numerical
timestep, and whether the current scheme is sufficiently accurate and efficient
at the intervals used in experiments. A fully simultaneous production model is
an option to evaluate, not a predetermined outcome.

## Current baseline and proposed comparison

The environment currently applies maintenance and death, followed by coupled
trade/growth/reproduction, followed by photosynthesis, biomass-derived uptake
geometry and soil uptake. Within allocation, both organisms' free pools and
biological expenditures evolve simultaneously; received trade is immediately
eligible. Photosynthate and newly acquired soil P become eligible for allocation
in the next numerical timestep.

| Interaction | Current ordered treatment | Proposed fully coupled reference |
|---|---|---|
| Trade, growth and reproduction | Simultaneous through evolving C/P pools | Preserve the same coupling |
| Photosynthesis and allocation | Carbon acquired after allocation | Newly produced carbon participates as it is acquired |
| Soil uptake and allocation | P acquired after allocation | Newly acquired P participates during the interval |
| Biomass and acquisition capacity | Post-allocation biomass determines acquisition | Capacity follows evolving biomass |
| Maintenance and allocation | Maintenance paid first | Interact throughout the interval under an explicit scarcity rule |
| Starvation and death | Assessed before allocation/acquisition | Events assessed along the evolving trajectory |

The ordering has potentially opposing effects. Delaying new resources can reduce
growth or survival, while applying post-growth acquisition capacity for the
acquisition update can increase acquisition relative to continuously evolving
capacity. Their net effect must be measured, not assumed.

## Meaning of fully simultaneous

Integrate organism pools, biomass and soil P as a coupled dynamical system. For
each organism, the conceptual elemental balances are

\[
\frac{dC}{dt}=A_C+T_C^{in}-T_C^{out}-M_C-\gamma_C\dot G-s\gamma_C\dot R,
\]
\[
\frac{dP}{dt}=U_P+T_P^{in}-T_P^{out}-M_P-\gamma_P\dot G-s\gamma_P\dot R.
\]

Here acquisition terms are zero where unavailable: plant photosynthesis supplies
`A_C`; soil uptake supplies `U_P`; trade directions retain the current plant-C
and fungus-P meanings. `M` denotes actual maintenance expenditure. Growth costs
are `gamma`, reproductive cost scale is `s`, and the existing allocation laws
remain `Q=min(C/gamma_C,P/gamma_P)`, `dot G=kg*Q`, `dot R=kr*Q/s`.
Maintenance-associated biomass loss, death-related stock handling and their
elemental ledgers must also be specified explicitly. Soil uptake must remove
exactly the amount credited to the organisms.

Rates must respond to evolving state during integration. Evaluating every
process once from the same initial state would not provide this coupling.
The numerical implementation may still use suitable implicit or split methods
for the soil system; its accuracy against the intended coupled equations must
be demonstrated. Neither a particular solver nor a mandatory single monolithic
solver is selected here.

## Decisions required before implementing the reference

- **Maintenance under scarcity:** specify priority relative to discretionary
  expenditure, use of free pools, biomass loss and behaviour at zero stocks.
  Simultaneous integration alone does not define these rules.
- **Death events:** define threshold evaluation, historical maximum biomass,
  irreversible termination, cessation of trade/acquisition and stock accounting.
- **Dynamic acquisition:** preserve current photosynthesis, geometry, transport
  and uptake constitutive laws while identifying which quantities update during
  integration and how shared soil uptake remains conservative.

Where a different biological rule is necessary, report it as a separate model
choice. Do not attribute its effects solely to numerical update ordering.

## Staged comparison

1. **Resource supply and allocation:** disable maintenance and death; compare
   photosynthesis plus allocation, then controlled P supply plus allocation.
   Include isolated analytical or independently solved cases before spatial soil.
2. **Biomass feedback and soil coupling:** enable biomass-dependent acquisition,
   then spatial soil transport and shared uptake. Separate changing uptake
   capacity from soil depletion and transport effects.
3. **Maintenance and survival:** start well supplied, then approach scarcity and
   death thresholds under the explicitly agreed maintenance/event rules.
4. **Experimental consequences:** compare fixed strategies across representative
   resource conditions. Assess changes in strategy ranking and qualitative
   outcomes before considering trained-policy comparisons.

Include plant-only, fungus-only and coupled configurations where each process
is meaningful, plus zero-resource, zero-process and imbalanced-resource cases.

## Controls for a fair numerical comparison

Keep physical action rates, initial biomasses and explicit initial free pools,
initial soil inventories, trait values, forcing and physical duration identical.
Unspecified reset pools currently depend on `dt`; using those defaults would
confound refinement with a changed initial problem.

Hold the policy interval fixed while refining the outer environment timestep.
An illustrative sequence is a policy interval of `0.05` days with numerical
steps of `0.05`, `0.025`, `0.0125` days and finer as required. Begin with fixed
commands, so policy feedback and observation-window changes do not confound the
comparison. Later feedback comparisons must use the same observation schedule
and explicitly account for current final-substep trade observations.

Do **not** recompute `K=-log(epsilon)/delta_t` during refinement. Select physical
rates once and hold them fixed. Recomputing K would change the allocation
strategy. Use the unrestricted fixed-rate reference seam rather than treating
this as a declared static screen with equal policy/environment intervals; see
the [static allocation guide](../static-allocation.md).

Refine internal allocation integration separately from the outer timestep, and
hold the spatial grid fixed initially. Establish adequate soil temporal/spatial
accuracy before interpreting physiology-ordering differences. Resolve or bound
reference error below the differences being reported. Keep comparison cases
away from the numerical biomass guard; its stage-wise diagnostic limitation is
deferred and must not confound this investigation.

## Evidence and decision criteria

Record trajectories and integrated amounts at matching physical times:

- Biomass, reproductive output, free C/P stocks and soil P inventory.
- Photosynthesis, uptake, trade, maintenance expenditure and biomass loss.
- Per-organism and whole-system elemental residuals, including structural stock,
  reproductive exports, maintenance and mortality according to the model ledger.
- Survival and time of death, limitation changes and strategy rankings.
- Solver qualification, compile cost, warm runtime and numerical work.

Conservation and nonnegative-stock checks must pass independently of agreement
between methods. Define absolute/relative tolerances and scientifically material
effect sizes before accepting results; none are selected in this document.
Establish the coupled reference's own convergence, then compare the ordered
scheme across outer timesteps. If the same continuous rate laws are represented,
smooth-case differences should shrink under refinement. Persistent differences
require checking equation/event mismatches as well as numerical accuracy.

The investigation should conclude whether the existing timestep is adequate,
whether reducing it is practical, and whether full coupling or a better split
scheme is warranted. Report threshold-sensitive outcomes separately from smooth
trajectory errors. Retain reproducible configurations, source revision, commands,
solver settings and runtime versions. A production migration, new biological
rules or PPO retraining requires a subsequent decision.

## Related documentation

- [Current model and update sequence](../model-overview.md)
- [Growth and reproduction equations](../growth-model.md)
- [Earlier rates/grid/update-order review](rates-grid-and-update-order-review.md)
- [Known action-application concerns](action-application-bug-notes.md)
- [Three-rate qualification baseline](../qualification/three-rate-allocation.md)
