# Three-rate allocation: changes, implementation and qualification

Implemented for [#79](https://github.com/isaachcamp/mycormarl/issues/79) and
#80–#85 in [PR #86](https://github.com/isaachcamp/mycormarl/pull/86).
Qualification below applies to model source `4515864`, verified 22 October 2026.

## Model changes

The physical action is now three nonnegative per-day rates:
**trade, growth and reproduction**. The storage action is removed; unused
resources remain in the free C/P pools, including legitimate stoichiometric
surplus and finite-rate residue.

Both growth and reproduction require essential C and P. For each organism,

\[
Q=\min(C/\gamma_C,P/\gamma_P),\qquad
\dot G=k_gQ,\qquad \dot R=k_rQ/s.
\]

Growth consumes its structural costs `gamma`. Reproduction uses those same
costs multiplied by `reproduction_cost_scale=s`, independently configurable
for plant and fungus, defaulting to one. Increasing `s` reduces output without
changing expenditure at fixed rates. Reward sums realised reproductive output;
reproduction does not add parental biomass. The former Cobb–Douglas objective
and its `alpha` configuration are removed.

Trade, growth and reproduction now integrate simultaneously for both organisms.
Plant C and fungal P transfers are debited and credited once, and received
resources can participate during allocation. The wider update remains ordered:
**maintenance/death → coupled allocation → photosynthesis → updated geometry
and soil uptake**. Newly acquired photosynthate/soil P is usable next step.

## Implementation

- **Numerics:** a pure-JAX, JIT/vmap-compatible SSPRK3 solver subdivides allocation
  using an outgoing-hazard bound (default `0.02`, maximum `65,536` substeps).
  Compensated float32 accumulation controls high-rate ledger drift. Reproductive
  expenditure is integrated directly, then converted to output, preserving
  spending even when tiny output rounds to zero. Budget exhaustion or non-finite
  results fail explicitly; workflows reject failed integrations.
- **PPO:** three Gaussian latents specify trade intensity, biological intensity
  and reproductive expenditure share. Softplus maps intensities and sigmoid maps
  share `a`, giving `kg=(1-a)kb`, `kr=a*kb`. PPO scores the sampled latents;
  evaluation shares the transform. Legacy actor artifacts are rejected through
  the new `three-latent-stoichiometric-rate-v2` interface; state remains `state-v2`.
- **Static experiments:** the declared screen requires equal policy/outer
  intervals and `K=-log(epsilon)/delta_t`, with default `epsilon=0.001`.
  At `0.05` days, `K=138.1551/day`. Trade share `p` and reproductive share `a`
  produce `[pK,(1-a)(1-p)K,a(1-p)K]`; declarations and rates are recorded.
  This specifies nominal turnover, not guaranteed depletion of both elements.
  General fixed-rate references remain available. Existing reduced-control
  scientific rates were retained.
- **Execution/accounting:** interval flows and reproductive output sum across
  numerical steps; execution before later death is retained; disabled trade is
  reported correctly; terminal padding freezes state and flows. Resource ledgers
  include reproductive exports. Active callers and artifact workflows migrated.

## Qualification and limits

The full suite passed **487 tests in 159.77 seconds**, with one existing JAXopt
deprecation warning, no skips and no widened tolerances. Coverage includes
analytical allocation, both-species cost scales, elemental conservation,
single-consumer/death/horizon behaviour, JIT/vmap, PPO and artifact compatibility.

Six public-environment cases matched independent DOP853 references: maximum
default reference error **2.5443e-6**, maximum elemental ledger residual
**4.4704e-8**. Fixed-command refinement reduced the maximum recorded-observable
error from **6.427e-5** to **7.831e-8**. Fast static allocation, four static
examples and a PPO checkpoint save/load/evaluation smoke also passed. Exact
configurations, runtime measurements and versions are retained in the
[qualification report](qualification/three-rate-allocation.md) and its JSON
records. These establish numerical/workflow behaviour, not ecological calibration
or learned-policy performance.

A subsequent independent review found a deferred diagnostic limitation:
intermediate-stage biomass limiting can alter allocation without the final
biomass reaching the cap, so guard contact can go unreported. The cap is enforced;
guard-affected trajectories are not fully detected. Comparisons should stay away
from it. Arbitrary new parameter regimes still require qualification.

The full environment is not timestep-exact: source/maintenance/soil ordering,
dt-dependent unspecified reset pools and final-substep trade observations remain.
Numerical comparisons must pin physical initial stocks and action rates.
[Issue #87](https://github.com/isaachcamp/mycormarl/issues/87) records the intended
ordered-versus-simultaneous comparison. [ADR 0020](adr/0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md)
and the [static guide](static-allocation.md) provide the detailed contracts.
