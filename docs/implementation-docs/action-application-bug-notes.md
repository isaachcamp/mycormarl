# Action-application bugs for later follow-up

Originally recorded 15 October 2026 at the user's request. The historical
findings below are preserved; their original paths refer to the earlier package
layout. Current code is under `src/mycormarl/`. This investigation record
supplements the GitHub tracker.

## Resolution status — 22 October 2026

The [three-rate implementation](../adr/0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md)
resolves items 1–5. The [qualification report](../qualification/three-rate-allocation.md)
and retained 487-test result provide public behavioral evidence.

| Original item | Current status |
|---|---|
| 1. Timestep-dependent storage hazard | Removed in #80; retention is implicit free stock. |
| 2. Execution before later interval death discarded | Fixed in #83; any-step allocation execution reaches PPO validity. |
| 3. Disabled trade reported as realised | Fixed in #83; component availability and requested commands remain distinct. |
| 4. Held horizon overrun | Fixed in #83; shortened final intervals and terminal padding freeze state, flows and reward. |
| 5. Mixed final-step/cumulative diagnostics | Fixed in #83; flows sum, events use any-step aggregation, qualification requires all executed steps. |
| 6. Reset inventory depends on dt | Deliberately deferred by #79. Numerical comparisons explicitly pin physical initial pools. |

Additional confirmed defects found and fixed during the migration:

- PPO's scalar trade entropy was summed over the batch dimension; #81 retains
  per-sample scalar entropy and sums only the biological latent vector.
- Growth C/P offers aliased realised use, hiding essential-resource limitation.
  #83 integrates non-consuming `kg*C`/`kg*P` offers separately.
- The wrapper's old fitness recomputation was wrong at nonunit reproductive
  costs. #83 sums realised output directly.
- Static resource ledgers omitted reproductive exports; #82 adds them.
- Raw/batched static and evaluation/learning/qualification paths could accept
  exhausted allocation budgets and non-finite results. #84 rejects these paths.
- Final static P inventory excluded retained dead structure, creating a false
  `0.5 mg` residual in a public death fixture. #84 includes it alongside actual
  mortality exports.
- Static single-consumer summaries treated absent counterparts as deaths. #84
  restricts biological-failure classification to configured consumers.
- Static limitation traces read acquired C/P from the wrong columns and
  reported first-row maintenance/prior-acquisition ratios near `5e9` instead
  of the documented zero. Both serializer calculations are corrected in #84.
- Guard contact could be missed after later maintenance reduced biomass. #84
  retains an any-step contact event and rejects unqualified workflows. Batched
  stops also mask death counters from discarded later dynamics.
- Existing `1000/day` analytical cases exposed float32 cumulative roundoff
  (`9.9999857` growth versus `10`). Compensated updates fix this without wider
  tolerances. Eager/JIT reassociation also selected different substep counts
  at an integer boundary; factoring the time scale fixes the discrepancy.
- Held evaluation used numerical dt for duration and growth-rate denominators;
  #84 uses elapsed physical time, including shortened final intervals.
- Summary evaluation reused a full-control compiled actor for reduced-control
  parameters on the same environment. #84 includes actor family in its cache key.
- Final #85 audit found large finite reproductive costs could round output to
  zero and then erase spending reconstructed as `scale * output`. Integrating
  expenditure directly fixes both stock and export ledgers, with both-species
  regressions at `1e37`/`1e38`. Output overflowing float32 now explicitly fails
  allocation qualification rather than reporting successful infinite fitness.

The final-substep received-trade observation window remains the explicit
existing interface, as does reset-stock dependence on dt. Wider continuous
maintenance/source/soil reordering remains outside this migration. The
investigation findings and proposals below should be read in their original
15 October context rather than as unresolved claims about the current engine.

## 1. Storage has a numerical-timestep-dependent effect

- Evidence: `mycormarl/mycormarl/environments/base_mycor.py`, `_apply_allocation` (around lines 689–719).
- Storage enters the competing hazard sum but never transfers resources out of the free pool. Its suppression of growth/reproduction vanishes as numerical `dt` tends to zero.
- Reproduction: balanced initial resources supporting one biomass unit, growth rate 1/day, reproduction 0, storage 10/day, no other processes. Cumulative growth over one day is 0.090908 with one numerical step, 0.465087 with ten, and 0.630282 with 1,000. Without storage it is 0.632121 for all these subdivisions.
- Consequence: a learned retention strategy changes with numerical resolution. This is a defect in the model formulation, even though it implements ADR-0012's documented retention semantics.
- Follow-up: the action-design proposal removes this hazard and makes retention implicit; explicitly supersede the corresponding ADR text if adopted.

## 2. Partial-interval biological action execution is discarded on death

- Evidence: `mycormarl/mycormarl/environments/policy_interval.py`, final `Transition` construction sets `allocation_executed=end`; `mycormarl/mycormarl/algos/ppo.py`, `transition_to_ppo_fields` maps this to biological actor validity.
- Trigger: an organism executes growth/reproduction in an early numerical substep and dies in a later substep of the same policy interval.
- Consequence: its earlier action affected state/reward, but the complete interval is marked invalid for the biological actor update.
- Follow-up: preserve whether allocation executed anywhere during the interval, with explicit lifecycle facts. Add a partial-interval death regression.

## 3. Wrapper reports trade commands as realised when trade was disabled

- Evidence: `policy_interval.py` constructs `realised_action` using only the owner's final operational status. `base_mycor.py`, `completed_transition`, separately gates realised trade on both partners being operational.
- Trigger: a surviving single-consumer organism requests nonzero trade, or its partner is unavailable during part/all of an interval.
- Consequence: wrapper action provenance disagrees with the numerical environment and can claim nonexecuted trade. The wrapper's separately aggregated `trade_executed` flag does not have this same error.
- Follow-up: distinguish requested rates, enabled execution, and cumulative physical transfers. Test absent/dead partners and partial-interval execution.

## 4. Held intervals can integrate past the administrative horizon

- Evidence: `policy_interval.py` scans a fixed number of numerical steps without freezing after administrative termination; `base_mycor.py` continues biological updates when passed an already terminal state.
- Trigger: a numerical episode horizon falls inside a held policy interval; aligned horizons avoid this case.
- Consequence: state, resource fluxes, and reward can include biological time after the declared horizon.
- Follow-up: either validate horizon alignment or support a shortened/frozen final interval. Test a horizon inside a policy interval.

## 5. Interval diagnostics mix cumulative flows and final-substep flows

- Evidence: `policy_interval.py` explicitly sums selected diagnostics but leaves other entries from the final numerical substep.
- Affected examples: growth C/P offered/used and maintenance consumption/deficits remain final-substep values while `growth`, reproductive exports, uptake, and trade are summed. Cancellation events can also be missed when retaining only the last flag.
- Consequence: interpreting all returned flow diagnostics as policy-interval totals produces incorrect resource-budget reconstructions; this is a reporting defect, not demonstrated resource loss in the state update.
- Follow-up: declare each diagnostic as a state, rate, integrated amount, or event and aggregate accordingly. Test interval totals against the constituent numerical steps.

## 6. Default reset inventories change with numerical timestep

- Evidence: `base_mycor.py`, `_initial_state.reset_pool` (around lines 247–254) sets an unspecified free pool to `kappa * initial_biomass * config.dt`.
- Trigger: numerical timestep is changed while initial C/P pool fields remain unspecified.
- Consequence: a timestep comparison also changes the initial physical resource stocks. It is not a pure convergence experiment and can change early survival and growth.
- Follow-up: specify physical initial stocks or a reserve horizon independent of numerical `dt`. Until changed, explicitly set all four initial free pools in numerical-convergence experiments. The current formula is deliberate code behavior; its incompatibility with fixed-physical-problem refinement is the defect recorded here.

## Related concerns that are not yet classified as bugs

- Sequential maintenance → outgoing trade → allocation → incoming trade → photosynthesis/uptake creates finite-step priorities and delays. Determine intended biological semantics and measure convergence before selecting fixes.
- `rate_action` sanitizes malformed commands to zero, as existing tests require. This is an explicit permissive interface policy; reconcile it with strict-boundary wording before changing behavior.
- Static 90/10 biology describes a hazard split, not guaranteed realised C/P use under stoichiometric limitation. Clarify wording and expose realised budgets.
- Existing numerical qualification does not adequately cover nonzero storage and general competing process modes.
- Received-trade observations use the final numerical substep's received amount and a maintenance reference proportional to numerical `dt` (`base_mycor.py`, `get_obs`). This can consistently represent a recent-step rate proxy when numerator and denominator scale together, but is not an interval-total receipt feature. Audit zero-maintenance guards, finite-step effects, and intended observation timescale before choosing a replacement.

See [the action-space proposal](action-space-design-review.md) for the mathematical derivation, both requested reproductive-model options, and the distinction between policy grouping invariance and numerical convergence. The subsequent [rates, grid, and update-order review](rates-grid-and-update-order-review.md) found the reset-inventory coupling in item 6 and the trade-history observation concern above. It found correct volume scaling and conservative shared soil-uptake accounting; nonlinear front-cell averaging is recorded there as a spatial approximation requiring qualification, rather than a verified implementation defect.
