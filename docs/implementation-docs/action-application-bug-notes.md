# Action-application bugs for later follow-up

Recorded 15 October 2026 at the user's request. No fixes are implemented by this note. These findings come from code inspection; the storage defect additionally has a reproducible analytical calculation. Dedicated environment regression tests remain to be added when fixes are undertaken. This is an investigation record, not a replacement for the repository's GitHub issue tracker.

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
