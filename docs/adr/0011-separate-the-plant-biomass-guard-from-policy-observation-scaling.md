# ADR-0011: Separate the plant biomass guard from policy-observation scaling

**Status:** Accepted

## Context

The plant model requires a finite numerical guard against unrepresented
biomass growth, but the previous `100 g DM` default had no support as a
120-day carrot maximum. The strongest directly comparable field observations
place favourable whole-plant dry mass near `22--25 g` at harvest. The `Forto`
shoot-plus-storage-root trajectory reached `23.26 g` at 120 days and has a
summed fitted asymptote of `35.05 g`; that asymptote was not observed, excludes
fine roots, and is not a carrying-capacity estimate.

The previous implementation also derived the actor's plant-biomass reference
as `0.5 * biomass_cap`. Changing the guard would therefore change both growth
truncation and the policy input, confounding biological and learning effects.

## Decision

Use `biomass_cap = 50 g DM` as the provisional plant numerical growth guard.
This is approximately `1.43` times the fitted `Forto` asymptote and about twice
the largest independent cultivar-mean endpoint recovered in the source
review. It provides headroom above the `25--35 g DM` biological trajectory
reference without retaining the unsupported `100 g` scale.

Do not interpret `50 g` as a measured maximum or mechanistic carrying
capacity. A qualification trajectory that contacts the guard fails
qualification; cap contact is not evidence of physiological growth
saturation.

Represent actor scaling independently as
`biomass_observation_reference = 50 g DM`. This preserves the previous actor
input scale, since the former `0.5 * 100 g` calculation also yielded `50 g`.
Future changes to either value must be justified and tested separately.

## Consequences

- The cap cannot manufacture an accepted high-P growth plateau: contacting it
  invalidates the run.
- The 50 g guard is an evidence-bounded modelling choice, not a carrot trait.
- Changing the cap no longer silently changes policy observations.
- The biological comparison remains the complete biomass trajectory,
  including the `25--35 g` reference range and declining windowed relative
  growth rate, rather than agreement with one endpoint.

## Evidence

The primary-source comparison and derivation are recorded in
[`carrot-growth-biomass-cap-and-carbon-fixation.md`](../research/carrot-growth-biomass-cap-and-carbon-fixation.md).
