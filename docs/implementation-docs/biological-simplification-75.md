# #75 — stressed mixed preservation

Goal: protect maintenance deficits, shrinkage, death and cancelled trade while
the plant survives. Depends on #73. No fixes to scientific semantics.

- [v] Demonstrate the new stressed command fails before scenario implementation.
- [v] Capture initial/final plus five intermediate checkpoints at fixed intervals.
- [v] Independently check P balance, expected fungal death and absent later uptake.
- [v] Check steps three/four/five against literal biomass .512/.4096 in a
  controlled public-rollout case; cancellation occurs on death, padding inactive.
- [v] Retain existing resource-eligibility and lifecycle core tests.

Evidence: checkpoint and command tests passed (14 tests at this stage); all four
scenarios also passed independent repeat checks with zero exact differences.
Expected death remains a rejected living qualification trajectory; the reference
asserts that exact rejection reason rather than weakening qualification.

Exit review: no changes to biology, fixture overwrites during check, or widened
tolerances. Reset-stock dependence on dt and #64 remain separate known concerns.
