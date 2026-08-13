# ADR-0006: Use global time only in the depletion-gradient diagnostic

**Status:** Accepted

## Context

The sparse phosphate-uptake closure uses a fixed `T_ref` when estimating the
outer radius of an unresolved radial depletion gradient. A trajectory figure
needs elapsed time in that calculation to show how sparse uptake departs from
the continuous closure.

Global simulation time is suitable for an experiment in which all absorbing
cylinders exist from `t=0` and never grow. It is not a defensible age for roots
or hyphae introduced later by production growth dynamics.

## Decision

Create a diagnostic-only, fixed-reservoir experiment in which elapsed global
experiment time controls the diffusion travel distance in `R_eff(t)` and hence
`k(t)`. Keep absorber radius, length density, represented length, reservoir
concentration, and all uptake traits fixed.

Plot cumulative blended uptake by the represented cell in one panel for each
representative absorber radius. The diagnostic applies the existing blend with
elapsed experiment time in place of `T_ref`, so
`w_cont(t_sim) = 1 / (1 + (t_diff / t_sim)^p)` and the request moves from its
zero-time sparse/continuous identity toward the continuous closure. Mark each
trajectory at its density- and radius-specific `t_diff` when that point lies
within the plotted time range.

Do not change the production closure or add a production switch as part of the
diagnostic. A future production implementation must model absorber cohort age
rather than substitute global simulation time.

## Consequences

- The diagnostic cleanly exposes depletion-gradient development without
  confounding growth or finite-inventory depletion.
- The compact two-panel figure shows the accumulated cell-scale consequence of
  the time-dependent blended closure.
- Its global clock is valid because all diagnostic absorbers have the same
  known age.
- The diagnostic cannot be reused unchanged as a production-model time switch.
- Cohort creation, mixed ages, turnover, and state storage remain a separate
  modelling and implementation problem.
