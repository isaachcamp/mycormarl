# Biological simplification (#72)

Baseline: `9a68fa43fb33e9fbb8c60176d2a01772f2d4cfe1`, after implemented ADR-0020
and completed #79–#85. Three rates, coupled trade/allocation and stoichiometric
reproductive output are the agreed model. PPO, scientific equations and the
user's pre-existing domain/qualification edits are outside this cleanup.

## Reference evidence

The existing `evaluate_static_policy` seam captures initial/final states and
five intermediate checkpoints for sufficient mixed, stressed mixed, plant-only
and fungus-only scenarios. It uses production biology, not a second integrator.
[Reference-check guidance](../biological-reference-checks.md) documents commands,
provenance, update policy and fields. Fixtures, configurations, traits, actions,
seeds, source hashes and runtime versions are retained in Git.

Core tests protect precise lifecycle/resource boundaries between checkpoints.
A controlled stress case independently checks fungal biomass .512 at step three,
.4096 and cancelled trade at death on step four, then inactive dead padding.
Expected death remains a rejected living qualification trajectory. Absent-partner
inactivity, P conservation and sensitivity to numerical/discrete mutations are
checked independently; checking never regenerates fixtures.

Pre-cleanup repeat and frozen comparisons were exact on the recorded CPU runtime.
The fixed float32 comparison allowance is relative 2e-6 / absolute 1e-8 (about
17 float32 epsilons relative). The independent whole-system P ledger allows
1e-7 * max(1 mg, initial total). These are numerical checks, not scientific
uncertainty bounds or cross-device guarantees. No tolerance widening is allowed
merely to pass a cleanup. Initial four-run measurements total ~3.26 s including
setup, ~1.80 s compilation and ~.004 s compiled execution; repeat runs including
recompilation total ~2.34 s. See [measurement data](../qualification/biological-reference-baseline.json).

## Completed work and checks

- Original #73–#78 and follow-up #94: sparse reference workflow, shared P ledgers,
  no unrequested limitation-history allocation, unused private-helper removal,
  and corrected biological ordering/accounting explanations. Baseline core checks:
  163 passed; original completed suite: 506 passed in 194.23 s, one existing JAXopt
  warning. Red-before-green covered new checkpoint/scenario capabilities.
- Further user-approved cleanup, one commit per recommendation: shared final
  summaries/rejection rules (18 focused tests); inlined photosynthesis (30);
  folded uptake adapter into its sole substep caller (52); retired legacy
  growth/density-scaling modules (109); removed isolated-organism step methods
  (90). Each code change passed all four unchanged reference checks. These are
  refactorings verified by existing independent assertions, not new behaviour.
- Consolidated the seven redundant phase notes into this report. The command
  guide and numerical evidence remain separate. No dependency changes or extra
  frameworks were needed. Use `uv run python -m pytest -q`: direct `pytest`
  misses the repository-root import needed by an existing script benchmark test.

## Compatibility and retained boundaries

The user explicitly authorized retiring `grow`, `remove_density_from_biomass`,
and `BaseMycorMarl.step_plant` / `step_fungus`. No callers were found in core,
REDbyRBD or the animations repository. External users of these legacy APIs
must use the production environment's `step_env`/`step` entry points instead.
`MycorMarlState` remains an alias; blended uptake and its diagnostics remain.

Keep the single/batched public interfaces, batching, limitation traces,
checkpoint summaries and actor-metadata helpers: downstream studies use them.
Issue #95 replaces the separate execution loops as described below.
Numerical/geometry modules retain their unit and conservation boundaries.
The previously deferred SSPRK3 stage-wise biomass-guard diagnostic concern
remains: intermediate limiting can be missed by initial/final guard checks.
Reset-stock dt dependence, last-substep trade observations, #64, #87 and
#89–#93 scientific extensions are also separate work. No new biological defect
was established or silently corrected by this simplification.

Final follow-up verification: **506 passed**, one existing JAXopt warning,
199.82 s. All four references also match exactly at zero tolerance, with no
fixture regeneration or changed tests. The five code recommendations remove
136 net production lines; the documentation consolidation removes the seven
phase files while retaining their unique evidence and deferrals here. Final
review found no blocking change-specific defects.


## Shared static trajectory (#95)

Single execution and homogeneous batches now use one pure-JAX trajectory;
`vmap` supplies per-member stopping/freezing for batches. Host validation and
result serialization stay outside it. The original key conventions remain:
single steps use `seed + step + 1`, batches use `seed + 1`. Optional single-run
recording is selected statically; batches return no recording arrays or histories.
The duplicated batch stepping/accumulation loop is removed (26 net production
lines removed), with no changes to biology, PPO or the Transition contract.

One intentional correction was explicitly approved: a run rejected at the
initial plant biomass guard executes zero steps and reports zero accumulated
totals. The old batch loop could contaminate growth with NaNs from discarded
allocation calculations (`0 * NaN`). Executed allocation failures still retain
their failure step, NaNs and rejection. The public regression test fails on the
old implementation and passes on the shared trajectory.

Performance comparison on CPU (`TFRT_CPU_0`, JAX 0.9.0), against `600d808`:
compile each executable after clearing JAX caches; warm both, then alternate
30 synchronized executions of each, reporting medians. Compilation includes
lowering. These bounded measurements are not production-scale guarantees.

| Grid | Steps | Batch | Warm ms before → after | Compile s before → after | Temporary bytes before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4×4 | 12 | 1 | 0.118 → 0.122 | 0.621 → 0.745 | 1872 → 1528 |
| 4×4 | 12 | 4 | 0.225 → 0.232 | 0.685 → 0.807 | 3472 → 3128 |
| 4×4 | 12 | 16 | 0.478 → 0.488 | 0.712 → 0.777 | 9168 → 8824 |
| 40×40 | 24 | 1 | 1.796 → 1.814 | 0.644 → 0.768 | 39952 → 39608 |
| 40×40 | 24 | 4 | 3.230 → 3.406 | 0.685 → 0.816 | 156688 → 156344 |
| 40×40 | 24 | 16 | 9.979 → 9.978 | 0.706 → 0.823 | 624016 → 623672 |
| 100×100 | 24 | 1 | 4.332 → 4.356 | 0.692 → 0.762 | 281424 → 281080 |
| 100×100 | 24 | 4 | 13.474 → 13.546 | 0.663 → 0.804 | 1122768 → 1122424 |
| 4×4 uneven | 24 | 4 | 0.269 → 0.281 | 0.662 → 0.784 | 3472 → 3128 |

Cases use dt=.25 days and actions plant [.3,.2,.1], fungus [.4,.15,.1].
The 4×4 grid spans 2 cm; larger grids span 10 cm in each dimension. Normal
cases use the reference mixed traits and initial P levels `5 + member_index`.
The uneven case uses the public stopping-parity test's traits/P levels, extended
to 24 steps, and stops members at [4,5,24,24]. Input/output bytes are unchanged
in every case (largest: 480300/480764); temporary memory falls by 344 bytes in
every case. Warm medians differ by 0–5.5%; compilation adds .065–.140 s.
The user explicitly accepted this measured tradeoff for the simpler implementation.

Final #95 verification: **511 passed** in 328.29 s, one existing JAXopt
deprecation warning. All four frozen references match at zero tolerance;
fixtures and comparison tolerances are unchanged. Public-result comparisons
cover different member stopping times, horizon/death, initial and later guard
contact, and executed allocation failure. Final diff review found no blocking
change-specific defects.
