# Documentation, provenance, and handoff

## Objective

Make the completed phosphate model usable without conversation history by
providing one authoritative guide, auditable provisional-parameter provenance,
small executable examples, and an accurate repository entry point.

## Checklist

- [v] Document the model equations, units, canonical and derived state,
  timestep order, configuration, diagnostics, and limitations.
- [v] Record a source, derivation, or explicit "modelling choice" label for
  every provisional phosphate and geometry default.
- [v] Highlight assumptions still requiring validation: fungal P fraction,
  root radius, plant root allocation/SRL compatibility, and the MDPI-derived
  carrot P value.
- [v] Add tested plant-only, fungus-only, and mixed uptake examples using the
  production soil transaction.
- [v] Replace the obsolete training entry point and stale pre-P5 status text.
- [v] Run focused tests, the complete suite, and a final diff review.

## Gate

Another developer can configure, run, validate, and interpret the phosphate
model without relying on conversation history.

## Verification record

- Phase status: **verified** on 2026-07-21.
- [`../../../phosphate-model.md`](../../../phosphate-model.md) is the authoritative model
  and provenance guide; `README.md` links the guide, qualification evidence,
  examples, and test command.
- The three examples use the shared production transaction and expose
  consumer-specific uptake, non-negative inventory, regime weight, cap
  frequency, numerical substeps, and balance error.
- Focused examples passed `6 passed, 1 warning`; the complete suite passed
  `200 passed, 1 warning` in 21.25 seconds. `git diff --check` passed. The
  warning is the unchanged upstream JAXopt deprecation.
- The documented `uv run python scripts/phosphate_examples.py --mode
  plant-only` command completed successfully.
- The initial P7 review found no actionable defect, but a subsequent delegated
  whole-plan audit identified integration gaps. These are tracked and verified
  in `post-implementation-audit-repairs.md`; this historical record no longer claims that
  the untouched PPO/evaluation modules were compatible.
- Remaining scientific risks are maintained explicitly in the guide and main
  plan rather than treated as completed calibration.
