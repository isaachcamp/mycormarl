# #94 — remove obsolete private helpers and stale explanations

Depends on #76. This is the additional bounded finding from the review, separate
from #77. Final #78 depends on both. Preserve public helpers and model behaviour.

- [v] Verify private wrapper/helper names have no callers, then remove them.
- [v] Correct step explanations to match ADR-0020 and retained dead-state accounts.
- [v] Run focused soil/lifecycle tests and all four reference comparisons.
- [v] Review removal boundaries and confirm no public entry point was removed.

Exit: dead private code removed, accurate explanations, unchanged evidence.

Verification: full suite 506 passed (194.23 s); four frozen references also
match exactly after simplification. No fixtures were regenerated after cleanup.
Public names retained; final review found no blocking regressions in this diff.
