# ADR-0009: Represent AM engagement using complete consumer modes

**Status:** Accepted

## Context

In `mixed`, the model begins with a living established fungus, fungal reserves,
hyphal uptake geometry, and an active bilateral association. The plant and
fungus independently choose continuous transfer fractions. Finite trade
latents pass through a sigmoid, so a living actor can request arbitrarily small
but not exactly zero trade.

Disabling transfers while retaining the fungus would not represent absence of
association. The fungus would still consume soil phosphorus through its
initial biomass and uptake geometry, turning the putative off state into a
fungal-competition treatment.

## Decision

Interpret phase-one `mixed` outcomes as the value of access to an established
AM association. Near-zero requested trade must be reported quantitatively and
must not be described as exact cessation or zero colonisation.

Represent the later abstract AM-engagement choice using complete consumer
modes:

- gate-on selects the complete `mixed` initial condition;
- gate-off selects the complete `plant-only` initial condition.

The choice occurs at reset and is inferred by comparing separately trained
replicate outcomes. Do not add an in-episode Bernoulli gate or infer engagement
by thresholding the continuous trade fraction. Do not retain fungal biomass,
reserves, geometry, or uptake in the off state.

Define plant association advantage at initial P level or environmental
distribution `x` as:

```text
Delta_AM(x) = E[Y_mixed(x)] - E[Y_plant_only(x)]
```

Positive association advantage means the complete established-association
mode is favoured under the modeled conditions. The initial abstraction adds no
separate initiation or persistence cost. Conditional on `mixed`, the existing
continuous plant and fungal trade policies operate unchanged.

## Consequences

- The extensive association choice remains distinct from the intensive choice
  of how much resource to transfer.
- The off treatment contains no biologically implausible non-partnered fungal
  competitor.
- The model can locate conditions under which association is favoured without
  claiming to simulate colonisation dynamics.
- Omitted initiation, persistence, establishment delay, and colonisation-
  dependent uptake mechanisms may shift or qualitatively alter an inferred
  threshold.
- A sign change brackets a threshold for refinement; it does not establish a
  precise universal P threshold.

## Related studies

- [`uniform-p-association-response-pilot-spec.md`](../implementation-docs/uniform-p-association-response-pilot-spec.md)
- [`spatial-p-heterogeneity-pilot-spec.md`](../implementation-docs/spatial-p-heterogeneity-pilot-spec.md)
