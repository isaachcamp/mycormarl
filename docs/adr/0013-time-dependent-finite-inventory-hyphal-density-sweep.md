# ADR-0013: Sweep hyphal density with a time-dependent finite-P closure

**Status:** Accepted

## Context

We need to test how hyphal length density affects both realised phosphate
uptake and carbon-for-phosphorus efficiency, motivated by the density--speed
trade-off in arbuscular mycorrhizal fungal growth. The existing diagnostics
separate fixed-reference-time construction economics from the diagnostic that
uses elapsed time to develop the unresolved depletion gradient. Neither alone
provides the requested 30-day, finite-inventory experiment.

Normalising uptake by construction carbon is an efficiency endpoint, not a
control that removes the density effect. Absolute uptake and uptake per unit
structural hyphal carbon must therefore be reported together.

## Decision

Run an isolated, single-cell closure diagnostic with one fixed absorber. Do
not construct or step `BaseMycorMarl`, organism growth, policy code, or a
spatial soil grid.

- Hold absorber radius at the default fungal hyphal radius and cell volume at
  the one-cell diagnostic default.
- Sweep hyphal length density from `25` through `5000 cm cm^-3` in increments
  of `25`.
- Hold geometry fixed over a 30-day global experiment clock.
- At each numerical step, use elapsed time to calculate the dynamic effective
  depletion radius, sparse resistance, and continuous-closure weight using
  the existing `t_diff` and transition exponent semantics.
- Evaluate uptake at the current finite-reservoir bulk P state, deduct accepted
  uptake, and continue until 30 days or reservoir exhaustion.
- Price hyphae using structural fungal tissue carbon only:

      C_hypha = (lambda V) pi r_h^2 rho_C,hypha

  Exclude CUE, maintenance, transport, host carbon transfer, and whole-fungus
  budgets from the primary metric.
- Produce both a full time series and one summary row per density. The summary
  includes total 30-day P uptake, construction carbon, and P uptake per
  construction carbon.

The dynamic global clock is valid only because every diagnostic absorber is
present at time zero and remains fixed. This decision does not alter
production-model closure semantics.

## Consequences

- Absolute uptake measures the cell-scale consequence of density, including
  finite-reservoir depletion.
- P per construction carbon measures realised construction efficiency but can
  still vary with density through diffusion, closure blending, and depletion.
- Results intentionally ignore diffusion from neighbouring cells and should
  not be interpreted as a spatially resolved soil prediction.
- The dense sweep is computationally modest but requires explicit early-time
  handling and conservation checks because the blend changes fastest near
  `t=0`.

