# Biologically Plausible IPPO Actor Interface Implementation Plan

**Status:** Reviewed proposal; implementation has not started.

## Summary

Replace the current four-way maintenance/trade/growth/reproduction allocation
and unconstrained Gaussian PPO policy with a compact, biologically plausible
independent-PPO walking skeleton for the plant and fungal actors.

Each actor will observe five bounded local signals: its own biomass, carbon
reserve, phosphorus reserve, most recently received partner resource, and
current partner association. Each actor will produce one independent trade
fraction plus a growth/reproduction/reserve simplex. Maintenance will become
an unavoidable automatic C/P debit paid before learned allocation. PPO will
sample three latent Gaussian coordinates through two actor output heads and
transform them into the four physical actions without clipping or
renormalising the sampled Gaussian action.

The implementation will preserve intentionally partial observability,
independent local critics, the existing `0.025 day` environment/action
timestep, deterministic maintenance-deficit mortality, and Cobb-Douglas
reproductive fitness. It will distinguish biological termination from an
administrative episode truncation so that death does not bootstrap while the
arbitrary maximum-step boundary does.

This plan includes explanations for the key decisions because the interface is
part scientific model and part learning system: several alternatives are
technically possible but imply different biological objectives.

`CONTEXT.md` is not present. This plan therefore takes the repository code,
tests, implementation documentation, and the MycorMARL working diary as its
authoritative context.

## Execution Route

`undecided`

The constrained route is optional. No execution route has been selected yet.

## Goals

- Provide a small continuous action space whose executed resource transaction
  matches the action probability used by PPO.
- Keep `BaseMycorMarl` algorithm-independent and translate its typed
  per-agent `Transition` instances through one small pure PPO helper.
- Remove learned maintenance allocation while preserving unavoidable
  biomass-dependent C/P costs and deterministic mortality.
- Prevent the trade fraction from reserving the non-traded currency.
- Give plant and fungus bounded, biologically plausible local observations
  without exposing partner internals or global soil state.
- Keep the plant and fungal actor-critics independent and feed-forward as the
  baseline.
- Preserve resource accounting, delayed availability of incoming resources,
  sticky death, and the ability of a surviving facultative partner to
  continue after the other agent dies.
- Make the arbitrary episode limit a true training truncation while keeping
  biological death terminal.
- Add executable correctness tests for every new contract and retain a finite
  JIT-compiled PPO smoke run.
- Keep physical-time discount sensitivity configurable while using
  undiscounted cumulative reproductive fitness as the default objective.

## Non-goals

- Centralised critics, MAPPO, parameter sharing between species, or privileged
  critic observations.
- Recurrent actors, frame stacking, belief-state estimation, or removal of
  intentional partial observability.
- Separate actor-decision and numerical integration intervals, held daily
  actions, or timestep-invariant action-rate conversion.
- Stochastic/background mortality, accumulated stress, ageing, senescence, or
  maintenance-sacrifice/terminal-investment strategies.
- Partitioning maintenance P among explicit repair, recycling, turnover, and
  soil-return pools.
- Treating Cobb-Douglas reward as a conserved reproductive-biomass pool or
  introducing explicit propagule size and composition.
- Broad convergence tuning, reward/return scaling studies, cross-play,
  heuristic-partner evaluation, exploitability analysis, or claims of global
  policy optimality.
- Transformed-action entropy regularisation, separate actor/critic optimisers,
  policy-sample wrapper types, or a general adapter framework. The walking
  skeleton uses initial Gaussian exploration, one optimiser per species, and
  one small pure PPO conversion helper.
- Running the deferred adversarial-policy, discount-timescale, spatial,
  timestep, and large scientific validation matrices. This plan only exposes
  the necessary configuration and covers walking-skeleton correctness.
- Changing the qualified `0.025 day` environment/action timestep.
- Turning the arbitrary maximum-step boundary into a genuine annual lifecycle
  endpoint. A future annual-species model will require seasonal/phenological
  state and explicit terminal life-history semantics.

## Users

- The model developer connecting `BaseMycorMarl` to the repository's PPO
  implementation.
- Researchers using the plant/fungus policies as a transparent independent-PPO
  baseline before expanding action, observation, memory, or critic structure.
- Future model-development work comparing richer MARL designs against a small,
  explicitly documented walking skeleton.

## Requirements

### 1. Observation contract

Each live agent receives a float32 vector with shape `(5,)` in this order:

1. own biomass;
2. own carbon reserve;
3. own phosphorus reserve;
4. most recently received partner resource;
5. current partner association (`0.0` or `1.0`).

Use the fixed numerical guard:

```text
OBS_EPS = 1e-8
denom_safe = max(denom, OBS_EPS)
```

Use `max`, not unconditional epsilon addition, so ordinary ratios are not
systematically biased. Clamp physical inputs non-negative, define `0/0` as
zero, and clip each transformed observation to `[0, 1]`.

#### Biomass

Use the same saturating transform for both species:

```text
o_B = B / (B + s_B)
```

For the plant:

```text
s_B,plant = 0.5 * plant.biomass_cap
```

For the fungus, first calculate the biomass whose existing saturated
hemisphere reaches the configured outer soil radius `R`:

```text
L_radial_fill = (2 / 3) * pi * fungus.saturation_density * R^3
B_radial_fill = (
    L_radial_fill
    * fungus.hyphal_tissue_carbon_density
    * pi
    * fungus.hyphal_radius^2
    / fungus.gamma_c
)
s_B,fungus = 0.5 * B_radial_fill
```

This is a radial-extent reference, not a fungal biomass cap and not a claim
that the entire cylindrical domain is filled.

#### Carbon and phosphorus reserves

Express free pools relative to the resource required to construct an amount of
biomass equal to current biomass:

```text
o_C = C / max(C + gamma_C * B, OBS_EPS)
o_P = P / max(P + gamma_P * B, OBS_EPS)
```

#### Most recently received trade

Scale the received resource against the recipient's next-step maintenance
demand:

```text
plant_need_P = plant.kappa_p * plant_biomass * dt
plant_trade_obs = plant_last_p_received / max(
    plant_last_p_received + plant_need_P,
    OBS_EPS,
)

fungus_need_C = fungus.kappa_c * fungus_biomass * dt
fungus_trade_obs = fungus_last_c_received / max(
    fungus_last_c_received + fungus_need_C,
    OBS_EPS,
)
```

#### Association and dead observations

Association is one only while both configured consumers are present, alive,
and functionally able to exchange resources. It is zero in either independent
consumer mode and after either partner dies.

Dead and absent agents emit an all-zero observation. The per-agent termination
and PPO validity masks, rather than an additional operational observation,
distinguish death from an extremely resource-poor live state.

#### Why this observation design

- Saturating ratios give fixed bounded inputs without a running normaliser
  whose meaning drifts during training.
- Species-specific biomass scales avoid incorrectly normalising indeterminate
  fungal growth by the plant cap while retaining the same transform shape.
- Reserve ratios represent actionable internal physiology rather than raw
  pools whose scale changes with biomass.
- Received trade is biologically observable and gives one step of interaction
  feedback without exposing the partner's pools, actions, or intentions.
- Association separates a partner choosing zero trade from there being no
  functioning partner.
- Feed-forward partial observability is intentional: the baseline should not
  solve partner-state inference with recurrence before the resource interface
  itself is verified.

### 2. State additions and accounting

Add float32 arrays with the existing single-agent shape for:

- `plant_last_p_received`;
- `fungus_last_c_received`;
- `cumulative_plant_p_maintenance_loss_mg`;
- `cumulative_fungus_p_maintenance_loss_mg`.

Initialise all four fields to zero. Last-received fields are observation
memory, not resource pools. Update them from realised incoming trade and zero
them for absent or dead recipients. `_get_obs(state)` must reconstruct the
complete observation without an out-of-band `last_trade` parameter.

Accumulate all actually paid maintenance P in the corresponding maintenance
loss counter. Do not count unmet demand as a physical P loss.

#### Why store received trade

The actor observes a property of the immediately preceding transition. If it
is not carried in environment state, the same saved `State` cannot reconstruct
the observation that PPO received, making checkpoint/resume, deterministic
replay, wrappers, and debugging unreliable.

#### Why use a maintenance-P loss counter

Real maintenance P can be recycled, repaired into tissue, or lost through
turnover. Modelling those pools is outside the walking skeleton. Treating paid
maintenance P as an explicit external maintenance/turnover loss preserves the
current cost while closing accounting instead of silently deleting P.

### 3. Physical action contract

Each environment action is a float32 vector with shape `(4,)`:

```text
[trade, growth, reproduction, reserve]
```

The constraints are:

```text
0 <= trade <= 1
growth >= 0
reproduction >= 0
reserve >= 0
growth + reproduction + reserve = 1
```

Trade is not on the biological allocation simplex. Plant trade spends only C;
fungal trade spends only P. The shared growth/reproduction/reserve fractions
are applied separately to each resource pool remaining after maintenance and
outgoing trade.

Growth remains essential-resource limited:

```text
growth = min(C_growth / gamma_C, P_growth / gamma_P)
```

Charge only the C and P actually used by realised growth; unused
non-limiting-resource growth allocation remains in the pool. Reproduction
consumes both allocated resource amounts and retains the existing
Cobb-Douglas fitness reward. Reserve is the portion not offered to growth or
reproduction during the current step.

Every caller must generate a valid physical action before invoking the
environment. PPO uses its sigmoid and centred-simplex transforms; heuristics,
tests, examples, and qualification scripts use one shared public
action-construction helper. `BaseMycorMarl` executes the supplied action
unchanged and must not clip, sanitise, project, renormalise, or substitute a
fallback allocation. An explicit validation helper may fail loudly in tests
or debug workflows outside the compiled training path.

#### Why this action design

- A single four-way simplex incorrectly reserves the trade fraction of the
  currency that species cannot trade.
- Two resource-specific simplexes would require five latent degrees of freedom
  and permit many mismatched growth requests before the basic interface is
  established.
- One trade coordinate plus a three-part biological simplex uses three latent
  degrees of freedom while retaining reserve behavior and the intended
  Cobb-Douglas response to C/P imbalance.
- Reserve must be explicit after maintenance becomes automatic; otherwise the
  actor would be forced to spend all disposable resources every step.

### 4. Automatic maintenance and transition order

Remove maintenance from `Actions` and from the learned simplex. For each
operational organism, execute one step in this order:

1. Compute start-of-step C/P maintenance demand from biomass and `dt`.
2. Pay as much demand as the free pools permit.
3. Record paid maintenance P in the new loss counter.
4. Convert C/P shortfall through the existing deterministic
   maintenance-deficit biomass-loss rule.
5. Update sticky death and current association.
6. Cancel all sending and receiving for an organism that died during
   maintenance; the partner retains any proposed outgoing transfer.
7. Calculate surviving partners' simultaneous outgoing trade from their
   post-maintenance traded-resource pools.
8. Deduct outgoing trade and apply the shared growth/reproduction/reserve
   simplex separately to the remaining C and P pools.
9. Apply realised growth, reproduction export, and plant biomass cap.
10. Update active root and hyphal geometry from surviving biomass.
11. Run photosynthesis and the soil P evolution/uptake transaction.
12. Credit incoming trade and newly acquired resources after allocation.
13. Store realised received trade, increment the step, and construct reward,
    observation, termination, truncation, and information outputs.

Incoming trade, photosynthate, and soil uptake cannot fund allocation in the
same step. An organism that dies during maintenance executes no learned
action in that transition.

#### Why maintenance is automatic

Maintenance is an unavoidable physiological cost rather than a discretionary
investment in the baseline. Making the random initial policy explicitly fund
maintenance creates a sharp early-death exploration problem. Automatic debit
retains the life-history trade-off indirectly: reproduction, growth, trade,
and reserve choices determine whether future pools can meet unavoidable
maintenance.

#### Why deterministic mortality remains

Maintenance deficit already provides a dense causal path from resource
allocation to biomass loss and eventual death. Adding uncalibrated stochastic
mortality would increase return variance and change the scientific model.
Background and accumulated-stress hazards remain future extensions.

### 5. Dead-agent behavior

Death is sticky. After death:

- freeze remaining free C/P and retained biomass as inert corpse bookkeeping;
- keep retained resources in whole-system accounting but make them unavailable
  to every active process;
- perform no maintenance, trade, growth, reproduction, photosynthesis,
  geometry formation, or soil uptake;
- emit zero reward, zero action effect, zero observation, zero received-trade
  memory, and association zero;
- keep the surviving partner active until its own death or a global reset.

Do not add corpse decomposition, recycling, or soil return.

#### Why retain inert corpse resources

This prevents dead agents from affecting the environment without inventing a
decomposition model. Frozen pools make the simplification visible in balance
diagnostics instead of silently transferring resources to another compartment.

### 6. PPO actor and critic contract

Retain separate plant and fungal actor-critic parameter trees. Each critic sees
only its own five-element actor observation.

Within each actor, use one shared feed-forward encoder and two minimal output
heads:

- trade head: one Gaussian latent `z_trade`;
- allocation head: two diagonal-Gaussian latents `z_allocation_1`,
  `z_allocation_2`.

Transform them as:

```text
trade = sigmoid(z_trade)
allocation_logits = [
    z_allocation_1 / sqrt(2) + z_allocation_2 / sqrt(6),
   -z_allocation_1 / sqrt(2) + z_allocation_2 / sqrt(6),
   -2 * z_allocation_2 / sqrt(6),
]
[growth, reproduction, reserve] = softmax(allocation_logits)
```

Initialise:

- trade location bias to `logit(0.1)`, giving a 10% initial median;
- both allocation location biases to zero and use equal initial allocation
  scales, so the location-transformed action is exactly uniform and the
  centred logistic-normal distribution is exchangeable across growth,
  reproduction, and reserve, giving equal expected allocation;
- exploration scale away from zero and away from a value that makes most
  transformed samples saturate at the boundaries.

Represent trade and allocation as factorised distributions even though they
share a policy encoder. Keep separate latent log-probability and KL terms.
Store both latent actions and executed physical actions in trajectories. The
critic retains its own tower; the shared encoder is shared by the two policy
heads, not between policy and value function.

When association is absent at the decision state, force physical trade to zero
and exclude the trade factor from PPO likelihood and KL. Continue to train
allocation. If automatic maintenance kills an agent before its sampled action
can execute, mask both actor factors for that transition but retain the
pre-death critic target when the minibatch otherwise contains valid actor
samples; the preceding action receives the future consequence through GAE.
Bilateral trade cancellation also masks the surviving partner's trade factor
for that transition, while its executed allocation factor remains trainable.

#### Why two output heads

The heads do not create separate actors or a sequential second decision. They
make two semantically different action factors explicit, permit different
initial biases, allow trade-only masking when association is absent, and keep
trade versus allocation likelihood/KL inspectable. A single three-output diagonal
Gaussian could be split manually, but would provide no advantage and would
make these contracts easier to violate accidentally.

#### Why centred allocation coordinates

The simpler reference-logit map `softmax([z_1, z_2, 0])` has two degrees of
freedom, but its sampled distribution is not symmetric: the reserve logit has
no exploration noise, so zero means do not give equal expected allocation at
non-zero variance. The orthonormal zero-sum contrast above keeps the same two
degrees of freedom while treating growth, reproduction, and reserve
exchangeably at initialisation. It is still a one-to-one map between two
latent coordinates and the interior of the simplex.

#### Why latent PPO likelihoods

The current Gaussian sample is clipped and renormalised before execution, so
many sampled actions map to the same physical allocation while PPO scores the
untransformed Gaussian. Retaining the three latent samples makes the
probability ratio correspond exactly to the stochastic decision that generated
the deterministic physical action.

The likelihood ratio and KL may be computed in latent space because the
sigmoid and centred-simplex transforms are fixed bijections: their Jacobian
terms cancel between the old and new policies for the same sampled action.

Set the entropy-bonus coefficient to zero in the walking skeleton. Initial
Gaussian scales provide exploration without introducing transformed-density
and additional sampling machinery into the first implementation. Do not apply
latent Gaussian entropy as a substitute because it can reward large latent
variance while physical sigmoid/softmax actions concentrate near boundaries.
A correctly Jacobian-adjusted physical-action entropy bonus remains a future
extension if exploration collapse becomes an observed problem.

### 7. Rollout validity, death, and sample masking

For each species, trajectories must carry enough information to distinguish:

- critic-valid pre-death state;
- allocation action actually executed;
- trade action actually active;
- per-agent biological termination;
- administrative truncation;
- final pre-reset observation or its bootstrap value;
- reset boundary.

The environment expresses lifecycle through each per-agent `Transition` using
`operational_at_start` and `operational_at_end`. The PPO transition adapter
derives:

```text
critic_valid = operational_at_start
terminated = operational_at_start AND NOT operational_at_end
allocation_actor_valid = allocation_executed
trade_actor_valid = trade_executed
```

`truncated` and `final_observation` are consumed directly from the same
per-agent `Transition`; they remain environment facts, not PPO masks.

Retain the transition in which maintenance causes death for critic/value
learning, but do not apply an actor loss to an action the environment did not
execute. Exclude every later dead-agent sample from actor loss, critic loss,
advantage normalisation, and learning aggregates.

Normalise advantages using actor-valid samples for that species only. Use safe
masked reductions for actor and value losses. Keep the existing single
actor-critic optimiser per species. If a minibatch has no actor-valid samples,
skip the entire species update so Adam momentum cannot move inactive policy
parameters; a rare isolated critic-only death target may therefore be omitted
in the walking skeleton. Record actor-valid, trade-active, and critic-valid
sample counts or fractions per species so obligate fungal mortality cannot
silently starve its policy of data.

#### Why per-species masks matter

The fungus is obligately dependent on association while the plant is
facultative. Fungal death may systematically occur earlier, leaving many fixed
API steps in which only the plant acts. Padding fungal learning with inert
post-death transitions would bias its critic and entropy objective rather than
representing useful experience.

### 8. Termination, truncation, and discounting

Expose distinct concepts:

```text
terminated[agent] = biological death for that agent
truncated = arbitrary maximum step reached
reset = all configured agents dead OR truncated
```

The inherited JaxMARL `MultiAgentEnv.step` auto-resets and returns reset state
and observation when `dones["__all__"]` is true. Preserve the final pre-reset
observation in `info` (or an equivalently tested JAX-compatible return field)
before auto-reset so PPO never bootstraps from the reset observation.

For agent `i`, compute:

```text
delta_i = reward_i + gamma * (1 - terminated_i) * final_or_next_value_i - value_i
```

Stop the recursive GAE trace at either biological termination or
administrative truncation so the reset episode's rewards are not mixed into
the preceding trace. At truncation, the delta still bootstraps from the final
pre-reset observation. At death, bootstrap value is zero.

Use undiscounted cumulative reproductive fitness by default:

```text
gamma = 1.0
discount_half_life_days = infinity / None
```

Provide one authoritative helper/configuration path for optional physical-time
discount sensitivity:

```text
gamma = exp(-log(2) * env.config.dt / half_life_days)
```

Support the previously selected future sensitivity values of 30, 90, 365 days,
and infinity, but do not run the empirical sensitivity study in this walking
skeleton plan.

The undiscounted, time-limit-bootstrapped objective is valid only as a proper
episodic problem with finite expected lifetime return. Under the walking
skeleton, strictly positive P maintenance and death thresholds for each active
organism, positive initial biomass, and a finite closed P supply make eventual
maintenance-deficit death unavoidable. Validate that condition when
`gamma == 1`. If a future configuration permits zero P maintenance, a zero
death threshold, external P replenishment, or another indefinitely viable
trajectory, it must use a genuine biological terminal horizon, a finite
discount, or a separately designed average-reward formulation.

#### Why distinguish termination and truncation

Death removes all future reproductive opportunity and must not bootstrap. The
current maximum-step limit is arbitrary: treating it as death would make
healthy biomass, reserves, and association suddenly worthless and could induce
end-of-window liquidation strategies. Truncation inserts a training boundary,
not a biological event.

#### Why default to undiscounted fitness

The reward is a reproductive-fitness index accumulated over the organism's
life. Mortality already removes future fitness. Without a calibrated external
mortality or demographic-compounding model, an additional finite discount
would impose an otherwise unsupported preference for earlier reproduction.

## Assumptions

- There remains exactly one plant policy and one fungal policy under the fixed
  two-agent JaxMARL API.
- `EnvConfig.dt = 0.025 day` is both the numerical and actor decision interval
  in the walking skeleton.
- Current species traits and units remain valid inputs for the observation and
  maintenance calculations.
- Plant biomass cap is a meaningful plant normalization reference.
- Existing saturated hemispherical fungal geometry is the authoritative
  biomass-to-radial-extent mapping.
- For the undiscounted baseline, every active organism has positive initial
  biomass, P maintenance, and death fraction, and the environment has finite
  total P with no external P replenishment, making lifetime reproductive
  return finite.
- Cobb-Douglas output is a dimensionless or indexed proxy for reproductive
  fitness, not conserved biomass.
- Actor observations always use the five bounded biological transforms.
  Qualification and diagnostic code reads raw quantities from `State` or
  explicit `info` fields rather than changing the actor observation contract.
- Existing policy parameter files and serialized `State` values are
  development artifacts rather than stable external formats.
- The current dirty worktree contains user-owned untracked
  `technical-summaries/` content that this plan and its execution must not
  modify unless separately requested.

## Constraints

- Preserve JAX/JIT/vmap compatibility; avoid Python data-dependent branches in
  environment and loss kernels.
- Keep PPO masks, GAE controls, advantages, value targets, and optimiser
  concepts out of `BaseMycorMarl`. The environment reports only
  an algorithm-independent typed `Transition`; loose diagnostics remain
  separate.
- Keep the public two-agent names `plant` and `fungus` and stable dictionary
  structure required by JaxMARL.
- Return the same typed per-agent `Transition` schema under fixed `plant` and
  `fungus` keys in every consumer mode and lifecycle state. Duplicated
  truncation flags must agree.
- Preserve plant-only and fungus-only modes under the fixed two-agent API.
- Keep all physical pools non-negative and all biological allocation within
  the available post-maintenance budgets.
- Require physical actions to satisfy their bounds and simplex invariant before
  entering `BaseMycorMarl`; do not repair actions inside the environment.
- Preserve the existing growth-before-uptake geometry convention and
  next-step availability of newly acquired resources.
- Do not silently load old policy parameters into the new observation/head
  architecture.
- Migrate all four-element action callers atomically because the old and new
  vectors have the same shape but different meanings.
- Preserve unrelated user changes and use focused, reviewable phases.
- Use executable tests rather than comments as the authoritative correctness
  record.

## Proposed Approach

Implement the contract from environment state outward, then update PPO only
after the physical action and observation interfaces are independently tested.
This ordering keeps biological accounting failures separate from policy-loss
failures.

1. Establish red tests and inventory all action/state/observation call sites.
2. Add observation-memory and maintenance-loss state fields, then implement
   the complete five-element observation contract.
3. Refactor the environment transaction to automatic maintenance and the new
   trade plus biological-simplex action.
4. Add the pure PPO transition-conversion helper, then introduce the two-head latent policy,
   factorised likelihoods, validity masks, and correct
   termination/truncation bootstrapping.
5. Migrate scripts/docs/callers, run the integrated PPO smoke test, and perform
   a final review/fix pass.

This is an atomic development migration rather than a compatibility rollout.
The action vector shape remains `(4,)`, so accepting old ordering would be more
dangerous than failing explicitly. Update all repository callers in the same
phase and document that saved state/policy artifacts must be regenerated.

## Implementation Phases

### P0 — Baseline contracts and migration inventory

- Run the existing full suite and record the baseline result and warning set.
- Inventory every `State` constructor/replacement, observation assumption,
  `EnvConfig.norm_obs` use, action vector, action-index enum use, PPO
  trajectory field, and saved-policy entry point.
- Add or adapt focused tests that fail on the old behavior for:
  - five bounded observations and exact feature order;
  - state-only reconstruction of last received trade;
  - the new independent trade plus biological-simplex contract;
  - automatic maintenance and maintenance-P accounting;
  - maintenance-death trade cancellation;
  - inert corpse behavior;
  - latent-to-physical policy transformations;
  - death versus truncation bootstrap masks.
- Record explicit checkpoint incompatibility and the old/new four-action
  ordering in the phase handoff before changing source behavior.

**Exit gate:** The existing baseline is known, every affected caller is listed,
and red tests express the new contract without unrelated failures.

### P1 — State-backed bounded observations

- Add last-received trade and cumulative maintenance-P loss fields to `State`.
- Update every state constructor, reset fixture, replacement, and balance
  helper for the expanded dataclass.
- Implement reusable non-negative saturating-ratio helpers with `OBS_EPS`.
- Implement plant and fungal biomass scales, including the fungal radial-fill
  inverse of the current geometry traits.
- Replace `_get_obs(state, last_trade=...)` with state-only observation
  construction.
- Add the association feature, dead/absent zero mask, float32 cast, and `(5,)`
  observation spaces with fixed bounds `[0,1]`.
- Remove `EnvConfig.norm_obs` and migrate qualification/tests that set it;
  obtain raw diagnostic quantities from `State` or explicit `info` fields.

**Exit gate:** Reset and transition observations are reconstructable from
`State`, finite, correctly ordered, bounded, species-scaled, JIT-compatible,
and covered at zero and extreme inputs.

### P2 — Automatic maintenance and compact physical resource transaction

- Replace the action enum/order with
  `[trade, growth, reproduction, reserve]`, add the shared public constructor,
  and keep optional validation outside the compiled environment path.
- Move maintenance demand/payment/deficit ahead of learned allocation for both
  species and remove maintenance allocation from policy-controlled code.
- Accumulate paid maintenance P in the new explicit loss counters.
- Resolve sticky death before trade; cancel bilateral transfer if either
  partner dies during maintenance.
- Apply outgoing trade only to plant C or fungal P, then apply the same
  growth/reproduction/reserve simplex separately to remaining C and P.
- Preserve essential-resource realised growth, return non-limiting leftovers,
  preserve Cobb-Douglas fitness reward, and keep incoming resources delayed.
- Freeze all dead-agent state and active effects according to the corpse
  contract.
- Update `info` diagnostics to report automatic maintenance, requested and
  realised action components, cancelled trade, association, and new P loss.
- Migrate every test, example, qualification helper, and script action vector
  atomically to the new ordering.

**Exit gate:** Focused environment tests prove no overdraft, correct ordering,
maintenance-P closure, trade cancellation, reserve behavior, delayed incoming
resources, inert death, independent consumer modes, JIT, and vmap.

### P3 — Two-head IPPO policy, valid-sample losses, and boundary semantics

- Refactor `ActorCritic` to a shared encoder with separate trade and allocation
  latent distributions plus the independent local critic.
- Implement exact initial location biases, equal allocation scales, the
  orthonormal zero-sum allocation contrast, and non-saturating exploration
  scales.
- Add one deterministic latent-to-physical transformation helper with tests.
- Keep sampling, transformation, and trajectory storage explicit in the
  rollout loop; do not introduce a policy-sample wrapper type.
- Extend trajectories to store latent actions, physical actions, factorised old
  log-probabilities, action-valid/trade-active/critic-valid masks,
  termination, truncation, and bootstrap observation/value data.
- Add one small pure PPO helper that converts the typed `Transition` into
  trajectory fields and derives every PPO validity/bootstrap/trace mask. Do
  not introduce an adapter class or derive/name PPO masks inside
  `BaseMycorMarl`.
- Populate every per-agent `Transition` with start/end operational status,
  allocation/trade execution, truncation, and the final pre-reset observation.
- Preserve final pre-reset observations through JaxMARL auto-reset.
- Replace the current GAE `done` mask with separate bootstrap and trace masks.
- Implement masked advantage normalisation, actor/value reductions, and a
  whole-species no-actor-sample update guard separately for plant and fungus.
- Keep one actor-critic optimiser per species and set the walking-skeleton
  entropy coefficient to zero.
- Mask trade likelihood when association is absent or bilateral trade is
  cancelled by maintenance death; mask both actor factors only for the
  organism whose learned allocation cannot execute.
- Make undiscounted fitness the default and add tested physical-half-life to
  per-step-gamma conversion without running the deferred empirical study;
  reject or clearly fail an undiscounted configuration that violates the
  proper-episode condition.
- Update batching/minibatching to preserve environment dimensions and all new
  masks under `NUM_ENVS > 1`.

**Exit gate:** Unit and integration tests prove transformed physical actions,
factorised likelihoods, initial trade median/allocation exchangeability,
masked losses and whole-update guards, per-agent death handling,
final-observation truncation bootstrap, and finite JIT PPO updates for one and
multiple environments.

### P4 — Integration, compatibility documentation, and verified handoff

- Update `scripts/train_ppo.py`, README examples, docstrings, and any public
  action/observation documentation.
- Make saved parameter output identify the new actor-interface version or fail
  clearly when an incompatible old parameter tree is supplied.
- Run the required walking-skeleton tests, focused regression groups, and the
  complete suite.
- Run a small reproducible JIT PPO training smoke job and verify finite
  parameters, losses, observations, rewards, actions, and values.
- Check default mixed mode plus plant-only and fungus-only API stability.
- Review the final diff for stale action ordering, out-of-band trade memory,
  unmasked dead samples, reset-observation bootstrapping, and P-accounting
  omissions; fix findings and repeat verification.
- Document deferred extensions and avoid presenting smoke success as evidence
  of convergence, robustness, cooperation, or biological validity.

**Exit gate:** All required tests and smoke commands pass, documentation and
examples match the implemented interface, old artifacts cannot load silently,
and no actionable review finding remains.

## Validation Plan

### Required executable correctness tests

- **Observation bounds:** reset, ordinary, zero-pool, high-pool, unassociated,
  absent, and dead observations are finite float32 vectors of shape `(5,)`;
  actor observations lie in `[0,1]`.
- **Observation meaning:** plant/fungal biomass reference points, reserve
  ratios, maintenance-need trade ratios, association, and feature order match
  their equations.
- **State reconstruction:** `_get_obs(state)` reproduces received-trade
  observations after reset, ordinary trade, no trade, partner death, and saved
  state round-trip.
- **Action constraints:** the PPO transform and shared public constructor
  produce finite actions with trade in `[0,1]` and non-negative biological
  fractions summing to one; `BaseMycorMarl` receives and executes those values
  unchanged. The explicit debug validator rejects invalid inputs.
- **Budget safety:** automatic maintenance, trade, growth, reproduction, and
  reserve never overdraw C or P; non-limiting growth allocation returns to its
  pool.
- **Maintenance P:** only paid maintenance P enters the plant/fungal counters;
  unmet demand and reproduction/mortality remain separately accounted.
- **Death:** maintenance-caused death cancels both trade directions, surviving
  partners retain proposed transfers, corpse pools freeze, active geometry and
  uptake disappear, rewards/actions become inert, and the survivor continues.
- **Timing:** received trade, photosynthate, and uptake cannot fund allocation
  until the next environment step.
- **Policy transform:** trade initial median is 0.1; the allocation location
  transforms to the uniform simplex; the zero-sum contrast is orthonormal and
  permutation-symmetric with equal initial scales; latent and physical shapes
  are correct; physical actions satisfy constraints without clipping.
- **PPO likelihood:** recomputed factor log-probabilities match stored latent
  samples; inactive trade is excluded; valid allocation remains trainable.
- **Sample validity:** death-causing critic samples remain, unexecuted/dead
  actor samples are masked and post-death value samples are excluded. A
  minibatch with no actor-valid samples skips the complete species update and
  leaves parameters/optimiser state unchanged. If one partner dies during
  maintenance, the survivor's cancelled trade is masked but its executed
  allocation remains actor-valid.
- **Lifecycle derivation:** operational-to-non-operational means biological
  termination; non-operational at both boundaries means absent/dead padding;
  operational at both boundaries remains critic-valid.
- **Termination/truncation:** death uses zero bootstrap; administrative
  truncation uses the final pre-reset value; GAE stops at both and never uses
  the reset observation or reset episode reward.
- **Discount conversion:** infinity/None gives `gamma=1`; selected finite
  half-lives produce the expected per-step values from physical `dt`; an
  undiscounted configuration without guaranteed finite lifetime is rejected.
- **PPO smoke:** one-update JIT training succeeds with one and at least two
  environments; parameter trees, losses, returns, and actions are finite.

### Verification commands

Use the repository's managed environment and record exact output during
execution. At minimum:

```bash
uv run pytest -q tests/test_actions.py
uv run pytest -q tests/test_base_mycor_refactor.py tests/test_review_repairs.py
uv run pytest -q
uv run python scripts/train_ppo.py --total-timesteps 256 --num-steps 128 --num-envs 1
```

Add focused commands for new observation, termination/truncation, and PPO
policy tests once their final file names are chosen.

### Deferred validation

Do not treat the following as exit gates for this plan: multi-seed convergence
curves, cross-play, heuristic or adversarial partners, optimal-strategy claims,
discount-half-life strategy comparisons, separate control intervals, spatial
saturation experiments, or broad scientific calibration. Preserve them in the
working diary for a later evaluation plan.

## Compatibility and Rollout

- This is a breaking development migration for environment state, observation
  shape, observation configuration, action meaning, trajectory schema, and
  PPO parameter trees.
- Update all repository callers atomically. Do not provide an automatic adapter
  from the old four-action vector because the unchanged shape makes silent
  semantic conversion unsafe.
- Regenerate serialized policy parameters and any saved environment state.
- Keep the change in phase-sized commits or equivalent reviewable checkpoints
  so source and tests can be reverted together if a phase fails verification.
- Preserve the fixed agent dictionary/API and consumer modes so external
  JaxMARL orchestration changes only where the new state/action/observation
  contracts require it.

## Risks

- **Silent action-order compatibility:** old and new actions both have shape
  `(4,)`. Missing one caller could silently reinterpret maintenance as
  reproduction or reserve. Mitigate with atomic inventory/migration and named
  helpers/enums in P0–P2.
- **Incorrect PPO density:** scoring transformed physical fractions as
  Gaussian latents would recreate the current likelihood mismatch. Mitigate
  with stored latent actions and latent PPO ratios/KL.
- **Invalid direct callers:** the environment no longer repairs malformed
  arrays. Mitigate by migrating all repository callers to one public physical
  action constructor and testing exact unchanged passage through the
  environment boundary.
- **Auto-reset bootstrap error:** JaxMARL returns reset observations at global
  boundaries. Mitigate by preserving and testing final observations before
  auto-reset.
- **Outcome-dependent action masking:** maintenance may kill an agent before
  its sampled action executes. Mitigate with explicit environment-reported
  execution facts, a tested pure PPO conversion helper, and separate critic
  versus actor validity.
- **Boundary semantic drift:** environment facts and PPO masks could evolve
  independently and silently disagree. Mitigate with adapter contract tests
  covering ordinary action, absent association, maintenance death, survivor
  continuation, biological termination, and truncation.
- **Duplicated joint boundary:** per-agent `Transition` instances both carry
  administrative truncation. Enforce equality as an environment invariant and
  fail focused tests if the two instances disagree.
- **Fungal sample starvation:** obligate fungal dependence may create many
  post-death API steps. Mitigate correctness-wise with valid-sample counts,
  masked reductions, and empty-batch guards; broader training responses remain
  deferred.
- **No entropy bonus:** initial Gaussian exploration may narrow prematurely.
  Monitor action scales and defer Jacobian-adjusted physical-action entropy
  unless exploration collapse is actually observed.
- **Undiscounted critic variance:** `gamma=1` can be harder to estimate over
  long horizons. Preserve it as the scientific objective and defer tuning of
  GAE/rollout/value architecture unless the smoke test becomes numerically
  invalid.
- **Improper undiscounted return:** later enabling zero P maintenance, a zero
  death threshold, or external P replenishment could permit endless positive
  reward, for which time-limit bootstrapping with `gamma=1` has no finite value
  target. Guard the current proper-episode assumptions and require an explicit
  alternative objective when they no longer hold.
- **Single-timestep action semantics:** stock fractions are chosen every
  `0.025 day`, so reserve and spending behavior remain tied to that interval.
  This is an accepted walking-skeleton limitation; separate control intervals
  remain deferred.
- **Maintenance-P simplification:** treating all paid P as external loss omits
  recycling and repair. The explicit counter makes the approximation auditable.
- **Cobb-Douglas interpretation:** the reward discounts imbalance but does not
  impose hard reproductive stoichiometry. It must remain documented as a
  fitness index, not physical offspring biomass.
- **Corpse sequestration:** inert retained resources are unavailable forever
  within an episode. This avoids invented decomposition but can affect long
  surviving-partner trajectories.
- **Fixed epsilon:** `1e-8` has different physical meaning across units. It is
  accepted only as a numerical zero guard under current scales.
- **Raw observation compatibility:** removing `EnvConfig.norm_obs` requires
  migration of qualification fixtures and scripts that currently set it.
  Mitigate by reading raw diagnostics from `State`/`info`, keeping one
  invariant bounded actor interface instead of two observation meanings.
- **No broad policy validation:** passing this plan demonstrates interface and
  numerical correctness, not convergence, robust mutualism, or biological
  optimality.

## Success Criteria

- Default plant and fungal observations are finite bounded `(5,)` vectors with
  the agreed biological meaning and state-backed trade memory.
- Environment physical actions are `[trade, growth, reproduction, reserve]`,
  with independent trade and an exact three-part simplex generated from three
  PPO latents.
- Automatic maintenance is paid before allocation; maintenance P is explicitly
  accounted; maintenance-caused death cancels bilateral trade.
- No transition overdraws a resource pool, and incoming resources remain
  unavailable until the next step.
- Dead agents remain completely inert while surviving partners continue.
- PPO uses separate trade/allocation output heads, exact factorised latent
  likelihoods, centred exchangeable allocation exploration, valid-sample
  masks, and the agreed initial policy. The walking skeleton uses no entropy
  bonus.
- Biological death and administrative truncation produce the correct distinct
  bootstrap and trace behavior even through JaxMARL auto-reset.
- Default training uses `gamma=1`; finite physical half-life conversion is
  configurable and unit-tested.
- All required focused tests, the complete repository suite, JIT/vmap checks,
  and minimal one-/multi-environment PPO smoke tests pass without non-finite
  values.
- Scripts and documentation match the implemented interface, and old policy or
  state artifacts cannot be mistaken for compatible new artifacts.
- Deferred evaluation and scientific extensions remain explicitly documented
  rather than being implied by walking-skeleton completion.

## Open Questions

No behavioral question blocks implementation planning.

The plan-wide execution route also remains `undecided` until the human selects
`standard`, `constrained`, or `undecided` for execution.
