# Biological simplification (#72)

Execution Route: standard. Baseline: `9a68fa43fb33e9fbb8c60176d2a01772f2d4cfe1`.
ADR-0020 and completed #79–#85 resolve the separate action-design prerequisite:
three rates, simultaneous allocation/trade, stoichiometric reproductive output.
Preserve the current model; #87 and #89 are separate proposed model changes.
Existing uncommitted domain/qualification edits belong to the user and are not
part of this change. No dependency changes are required (Python/JAX/pytest).

The walking skeleton is the existing `evaluate_static_policy` seam extended
with optional sparse checkpoints, checked against a committed mixed fixture.
All four scenarios use this seam; no second biological implementation.

## Phases

- [v] #73: baseline, sparse checkpoint capture and mixed reference check.
- [v] #74: plant-only and fungus-only references.
- [v] #75: resource-stress/death reference and exact boundary checks.
- [v] #76: measure full suite and review bounded simplification opportunities.
- [v] #77: implement selected bounded simplification with unchanged references.
- [v] #94: remove obsolete private helpers and stale step explanations.
- [v] #78: full checks, final review, findings and deferrals.

Each phase has a companion phase note with scope, checks and evidence. Tests
use the already approved public fixed-action rollout and environment boundaries.
Reference artifacts contain five intermediate snapshots plus initial/final facts;
checking never regenerates expectations. Numerical tolerances are established
before refactoring. No changes to PPO or scientific equations are intended.

## Final evidence — 26 October 2026

`MPLCONFIGDIR=/tmp/mycormarl-mpl uv run python -m pytest -q`: **506 passed**,
one existing JAXopt deprecation warning, 194.23 s. No skipped checks, changed
expectations, or widened tolerances. All four post-cleanup reference results
also match the pre-cleanup fixtures exactly with zero tolerance; normal check
mode retains the documented small float32 allowance. Baseline repeat/performance
measurements are recorded separately from the comparison data.

Resolved findings: repeated P inventory/export bookkeeping, unrequested history
allocation, unused private soil/maintenance helpers and inaccurate step comments.
Public legacy helpers remain for compatibility. The numerical/geometry module
boundaries and differing single/batch execution loops remain explicit. Known
stage-wise biomass-guard reporting, reset-stock dt dependence, observation-window
semantics and proposed scientific extensions are deferred as detailed in phase 76.

Use `python -m pytest`, matching the qualification workflow: direct `pytest`
does not put the repository root on the import path for one existing script
benchmark test. README now gives the working invocation. No packaging machinery
was added to work around this command difference.

Final inline review checked public entry points, snapshot scheduling/early stops,
comparison sensitivity, provenance, unchanged fixtures, P arithmetic order,
removed-helper callers and scope. No blocking findings remain in this change.
