# Biological environment simplification: discussion record

## Agreed scope — 15 October 2026

- Review the repository for readability and unnecessary complexity, with the
  current biological environment as the target of this pass. Keep PPO separate.
- Preserve existing biological behaviour. Record suspected existing bugs for
  separate investigation rather than silently correcting or approving them as
  part of a refactor.
- Review the existing contracts with the user before selecting reference runs
  or implementing simplifications.
- Use core tests and a small set of sample runs with reference outputs to
  establish preservation evidence, as detailed below.

## Contract review in progress

Relevant architectural decisions are ADR-0001–0004 and ADR-0009–0012. ADR-0014
also establishes ownership of generic rollout facilities. ADR-0012 replaces
the earlier fractional/simplex action scheme with Rate actions; older wording
in ADR-0003 and the model overview must be read in that historical context.

The user accepted the biological contracts and preservation of the environment's
observable interface, subject to a separate reconsideration of actions. PPO
implementation and training remain outside this pass; the physical action space
and PPO/environment boundary may be redesigned in another chat.

## Action-design dependency

**Action meanings may change before this plan is implemented.** The existing
trade priority, rate meanings, action-construction behaviour, and action-related
interface details are not permanently frozen by this discussion.

Before implementing this plan, review the outcome of the separate action-design
discussion and identify the agreed action contract and baseline revision. Capture
reference outputs against that agreed behaviour before making simplification
changes. An intentional action redesign must not be mistaken for a regression
caused by simplification, and reference expectations must not be silently updated
to conceal an unintended behaviour change.

## Agreed reference-run design — 15 October 2026

Use explicit configurations and fixed actions for four scenario families:

1. Mixed, resource-sufficient: both organisms survive, trade, grow, reproduce,
   and take up soil P.
2. Mixed, resource-stressed: maintenance deficits, biomass loss, death, and
   cancellation of trade.
3. Plant-only: plant dynamics and inactive absent fungus.
4. Fungus-only: fungal dynamics and inactive absent plant.

Choose and verify conditions that actually exercise these behaviours. Retain
final summaries, the initial state, and five evenly spaced intermediate snapshots
per run; do not save every step. Focused core tests protect exact death and
trade-cancellation boundaries that the snapshot schedule may miss.

Compare biomass, free C/P pools, soil-P fields, rewards, lifecycle status,
observations, and selected Transition facts at the chosen checkpoints, alongside
explicit P-balance checks. Precise field selection remains to be settled.

Require exact agreement for discrete facts such as shapes, flags, and step
counts. Compare floating-point values with tight, explicitly recorded tolerances
established from baseline measurements before refactoring. Investigate failures
rather than widening tolerances merely to pass a change.

Target approximately one minute for the routine reference suite after JAX
compilation on the user's current machine. Measure and report compilation time
separately. Verify the budget before fixing scenario sizes and durations.

## Reference artifacts and updates

Commit small, structured numerical fixtures to Git with the configuration,
actions, snapshot schedule, source revision, and runtime information needed to
reproduce them. Keep reference regeneration as an explicit command separate from
checking a candidate change. Failed comparisons must produce readable difference
reports. Reference updates require an explained, reviewed change.

## Implementation sequence and completion

1. Review any action redesign from the separate chat and establish the agreed
   contract and baseline revision.
2. Select concrete scenarios and verify that they exercise the intended
   behaviours. Measure runtime and numerical variation to establish scenario
   sizes, durations, and tight comparison tolerances.
3. Establish passing core tests and capture the reference suite before
   simplification. Record existing suspected defects separately.
4. Review the biological environment and supporting code for duplication,
   unnecessary indirection, obsolete code, and misleading naming. Select bounded
   changes with a concrete readability benefit.
5. Implement each bounded change and run the relevant core tests and reference
   comparisons. Investigate differences; do not silently revise expectations.
6. Finish when the agreed review findings are resolved or explicitly deferred,
   and preservation checks pass. There is no line-count reduction target.

## Implementation-time measurements and prerequisites

- Final action contract and baseline revision, after the separate discussion.
- Concrete configurations, actions, durations, comparison fields, and measured
  numerical tolerances, selected and verified under the agreed rules above.
- A compact structured fixture encoding suitable for those fields.

## Published specification and tickets — 15 October 2026

The user approved the specification testing seam and ticket breakdown for
publication. The authoritative specification is
[GitHub issue #72](https://github.com/isaachcamp/mycormarl/issues/72).
Execution tickets are #73–#78: baseline and sufficient mixed coverage,
single-consumer coverage, stressed mixed coverage, simplification review, the
first bounded simplification, and final verification. The review may generate
additional bounded execution tickets that must gate final verification.

Publication does not resolve the action-design prerequisite or start
implementation. Concrete configurations and tolerances are established during
baseline work under the agreed requirements above.
