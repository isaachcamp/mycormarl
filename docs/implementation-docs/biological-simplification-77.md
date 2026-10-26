# #77 — simplify static-rollout P bookkeeping

Depends on #76. Scope: share inventory/export definitions and omit unrequested
limitation-history storage. No changes to static public outputs or dynamics.

- [v] Consolidate the repeated P inventory/export definitions, preserving order.
- [v] Avoid history allocation when limitation tracing is disabled.
- [v] Recheck frozen single and batched summaries in all four scenarios.
- [v] Recheck focused rollout/qualification tests, then review the diff.

Pre-change evidence: all four batched summaries match the frozen single-run
references (4 passed). This is a refactoring review stage, not a new behavioural
feature; existing and frozen assertions remain unchanged throughout.

Verification: full suite 506 passed (194.23 s); four frozen references also
match exactly after simplification. No fixtures were regenerated after cleanup.
Public names retained; final review found no blocking regressions in this diff.
