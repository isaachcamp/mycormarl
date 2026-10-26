# #74 — single-consumer preservation

Goal: check plant-only and fungus-only through the existing reference workflow.
Depends on #73's complete mixed check. No biological or action changes.

- [v] Add public command tests; observe rejection of unsupported scenarios (red).
- [v] Add explicit scenarios and capture five intermediate snapshots plus endpoints.
- [v] Independently assert absent biomass, pools, geometry, observations, rewards,
  uptake and execution flags remain inactive, with fixed two-agent structure.
- [v] Run both command checks again and review for unintended contract changes.

Exit: both scenarios pass frozen comparisons and independent P/inactivity checks.

Evidence: both CLI scenarios passed; exact repeat differences zero. Batched
summaries also match their frozen references before simplifying bookkeeping.
