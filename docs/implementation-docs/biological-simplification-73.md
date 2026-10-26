# #73 — baseline and mixed reference check

Goal: one complete reproducible reference check via the existing static rollout.
Dependencies: ADR-0020 is implemented; #79–#85 closed. No further design gate.
Scope: optional selected checkpoints, mixed fixture, check/regenerate command,
comparison failures, baseline provenance. Non-scope: model changes and PPO.

- [v] Establish passing biological core tests and record baseline/provenance.
- [v] Add sparse checkpoint capture through public rollout, red before green.
- [v] Add explicit mixed scenario and fixture with independent coverage/P checks.
- [v] Add check/regenerate workflow and demonstrate numerical/discrete failures.
- [v] Verify all above checks a second time and review the diff.

Exit: a frozen baseline passes, mutations fail readably, no per-step state archive,
and default rollout callers retain existing outputs.

Evidence: original biological baseline 163 passed; sparse/default rollout and
comparator checks passed (24 at first integration). A second full-scenario repeat
produced zero exact fixture differences. Invalid schedules and early terminal
capture are checked at the public seam. No scientific changes were needed.
