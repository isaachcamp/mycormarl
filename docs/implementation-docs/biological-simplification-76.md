# #76 — measured suite and simplification review

Depends on #74/#75. Goal: choose bounded improvements after establishing evidence.

- [v] Measure first/repeat execution and compilation separately for all scenarios.
- [v] Verify repeat and frozen evidence agree exactly before simplification.
- [v] Review biological ordering, allocation, soil, geometry, traits, observations,
  lifecycle, generic rollouts and their tests for unnecessary complexity.
- [v] Select bounded changes with explicit compatibility and verification limits.

Measurements: see `../qualification/biological-reference-baseline.json`.
The four first runs total about 3.26 s, compilation about 1.80 s, compiled
execution about .004 s. Repeat runs including recompilation total about 2.34 s.
This is comfortably below the one-minute post-compilation target. No scenario
resizing is necessary. Commands record setup/serialization in total time.

Every repeat and fixture comparison is exact on this CPU runtime. Comparator
allowance is 2e-6 relative / 1e-8 absolute, roughly 17 float32 epsilons relative,
with absolute allowance far below the ~.01–.1 mg organism P stocks. This is a
small arithmetic-reassociation allowance, not scientific uncertainty or evidence
of cross-device equivalence. Snapshot whole-system P uses an independent
1e-7 * max(1 mg, initial total) ledger bound. No tolerance widening after cleanup.

## Findings and disposition

1. **#77: duplicate static P bookkeeping.** Single and batched rollouts repeat
   inventory and export definitions. Consolidate those definitions while retaining
   the public summaries and arithmetic order. Also avoid allocating a full
   limitation history when it was not requested. Verify single and batched
   summaries against frozen references and existing qualification tests.
2. **Additional bounded ticket: unused private soil wrapper and stale step
   explanations.** `_apply_blended_uptake` has no callers; the diagnostics entry
   point is the actual shared transaction. Remove the unused wrapper. Remove
   unused private maintenance helpers and correct stale comments about zeroing
   dead pools, separate incoming trade and the obsolete geometry TODO. Preserve
   public names and behaviour. Verify full reference/core checks.
3. **Defer removal of public legacy helpers.** Growth and maintenance modules have
   no internal callers, but public functions may have external callers. Removing
   those APIs would contradict the agreed interface preservation. Private dead
   helpers can be removed without deleting public entry points.
4. **Keep explicit numerical and geometry modules.** They encode distinct units,
   solver/qualification boundaries and conservation rules. Merging them would
   obscure responsibilities without demonstrated readability benefit.
5. **Keep single/batched execution loops distinct.** Their stopping/masking rules
   differ; share bookkeeping rather than introducing a generic execution framework.
6. **Separate scientific follow-ups.** #64, #87, and #89–#93 concern model or
   execution design beyond this cleanup. Reset-stock dt dependence and the
   last-substep trade observation remain documented existing contracts.
7. **Retain the explicitly deferred biomass-guard diagnostic concern.** The
   earlier independent review found that an internal SSPRK3 stage can limit
   growth even when initial/final biomass stay below the cap; the current guard
   diagnostic can miss that stage contact. The user explicitly deferred its fix
   on 22 October. This simplification neither repairs it nor claims to protect
   that unrepresented event. Reference scenarios stay well below the guard.

No new biological defect was established by this review. The old trace serializer
and interval bugs were already fixed in #84; do not reapply historical findings.
