# ADR-0020: Use three-rate actions and stoichiometric reproductive output

**Status:** Accepted and implemented, 22 October 2026. This is the new action
baseline for subsequent biological simplification. It implements
[specification #79](https://github.com/isaachcamp/mycormarl/issues/79).

The four-rate model allowed unmatched reproductive exports, gave trade first
claim on resources, and made its storage hazard depend on numerical resolution.
Use three nonnegative intensities `[trade, growth, reproduction]` in inverse
days, integrating both organisms' free-pool balances and biological extents
together. This preserves physical resource accounts and allows received trade
to participate during allocation without an artificial destination for surplus.

For each organism let `Q=min(C/gamma_c,P/gamma_p)`. Instantaneous structural
growth is `kg*Q`; reproductive output is `kr*Q/s`, with positive finite
`reproduction_cost_scale=s`, independently configurable and defaulting to one.
Growth consumes its paired costs; reproduction exports `s*gamma*output` and
reward sums realised output. Changing `s` at fixed commands changes output by
its reciprocal while leaving expenditure unchanged. Reproduction creates no
parental uptake infrastructure or offspring state. Blocked growth at the
numerical biomass guard spends no resource and is not redirected to reproduction.

Plant C trade and fungal P trade are donor-pool hazards. Each accepted physical
amount is debited and credited once while the limiting resource is reevaluated.
Stoichiometric surplus and finite-rate residue remain free stocks; there is no
storage action, liquidation sweep, excretion sink, or trade-first priority.
The outer sequence remains maintenance/death, coupled allocation, photosynthesis,
biomass-derived geometry and soil uptake. New photosynthate and soil P are first
usable in the next numerical step. Wider continuous maintenance/source/soil
reordering is a separate modelling decision.

The learned full-control actor samples three Gaussian latents: trade intensity,
biological intensity and reproductive expenditure share. Softplus maps the
intensities; sigmoid maps the share `a`, giving `kg=(1-a)*kb`, `kr=a*kb`.
PPO scores sampled Gaussian latents, and evaluation uses the same transform.
The biological environment remains independent of PPO. Typed Transition facts
retain requested commands and any-step execution, including execution before
later death; flows/output sum across held intervals and terminal padding freezes.

For declared static allocation screens alone, require policy interval equal
to outer environment interval `delta_t`, default residual `epsilon=.001`, and
`K=-log(epsilon)/delta_t`. Trade share `p` gives `kt=p*K`, `kb=(1-p)*K`, then
the same biological split. At `.05` days, `K=138.1551055796/day`. Validate and
record declarations, cost scales, commands and solver settings. This is nominal
isolated-pool turnover, not guaranteed depletion of both essential resources.
Changing `delta_t` and recomputing `K` changes the physical strategy; internal
refinement holds commands fixed. Learned policies have no screen K constraint.
Existing reduced-control biology retains its explicit `6.907755/day` and 90/10
split; no new trade-only scientific strategy is selected.

The pure-JAX allocation solver uses conservative SSPRK3 substeps and compensated
float32 updates. It integrates reproductive expenditure directly and derives
output afterward, so a tiny rounded output cannot erase material expenditure.
A dimensionless outgoing-hazard step bound controls refinement;
the default is `.02`, with a finite budget of `65536` substeps. Exhaustion marks
a failed transition rather than clipping rates or repairing stocks; non-finite
integrated stocks or output also fail qualification. Public
static, evaluation, qualification and PPO workflows reject failed allocation
and numerical-guard contact. The [qualification report](../qualification/three-rate-allocation.md)
states tested accuracy, runtime and limits; these are numerical evidence, not
ecological calibration or proof of timestep-exact full physiology.

This explicitly supersedes the four-component/storage decision in
[ADR-0012](0012-adopt-rate-actions-for-held-policy-intervals.md), selects the
reproductive objective left open by
[ADR-0018](0018-account-for-structural-phosphorus-and-explicit-biological-exports.md),
and supersedes the separate outgoing/incoming allocation ordering in
[ADR-0019](0019-separate-policy-numerical-and-soil-clocks-and-qualify-resolution.md).
Their remaining accounting, held-rate, finite-inventory and geometry principles
remain in force. Historical qualification outputs keep their original model
identity; they are not relabelled as three-rate evidence.

This is a compatibility break: old physical vectors and policy interfaces fail
explicitly, `alpha` is removed, the actor interface is
`three-latent-stoichiometric-rate-v2`, and numerical qualification manifests use
schema `3`. State fields remain `state-v2`. Old learned outcomes cannot be
reinterpreted as this model. The [static experiment guide](../static-allocation.md),
[core tests](../../tests/test_stoichiometric_actions.py),
[interval tests](../../tests/test_three_rate_intervals.py), and
[workflow failure tests](../../tests/test_numerical_workflow_failures.py) provide
executable evidence. Biological simplification and its parent issue retain
their own scope.
