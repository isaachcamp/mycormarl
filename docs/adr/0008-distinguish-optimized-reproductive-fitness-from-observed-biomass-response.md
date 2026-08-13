# ADR-0008: Distinguish optimized reproductive fitness from observed biomass response

**Status:** Accepted

## Context

MycorMARL rewards each organism with a per-transition Cobb–Douglas
reproductive-fitness index. Independent PPO therefore optimizes cumulative
reproductive fitness, while greenhouse studies commonly report plant biomass
responses. These quantities can diverge because reproduction exports carbon
and phosphorus that could otherwise support biomass growth.

Independent PPO also produces two interacting learned policies in `mixed`; it
does not establish a globally optimal joint strategy or a game-theoretic
equilibrium.

## Decision

Use cumulative plant reproductive fitness as the primary plant-performance
endpoint in policy studies. It is the undiscounted sum of the plant's
per-transition Cobb–Douglas reproduction rewards over the stated evaluation
episode and matches the plant PPO objective.

Report final living plant biomass and cumulative gross plant growth as
secondary endpoints. For comparison with greenhouse observations, calculate
mycorrhizal growth response from mode-level mean biomass:

```text
MGR = 100 * (mean_biomass_mixed / mean_biomass_plant_only - 1)
```

MGR is missing rather than epsilon-adjusted when mean `plant-only` biomass is
zero. Do not average seedwise biomass ratios. Reproductive fitness is reported
as an absolute difference of mode-level means; no reproductive-fitness ratio
is required.

Evaluate saved policies primarily with a deterministic **latent-location
policy**: transform each actor's learned latent Gaussian location through the
production action mapping without Gaussian sampling. This is not the mean
physical action because the mapping is nonlinear. Repeated sampled-policy
rollouts are a separate sensitivity diagnostic.

Refer to evaluated trained behaviour as a **learned IPPO outcome**, not an
"optimal strategy." Qualify every result by its environment, training
protocol, evaluation protocol, and seed replication.

## Consequences

- The primary scientific inference follows the implemented learning objective.
- Biomass enhancement or suppression remains observable without being
  conflated with reproductive fitness.
- Fitness and biomass responses may disagree and must both be reported.
- Deterministic primary evaluation is reproducible while sampled evaluation
  can test sensitivity to residual policy stochasticity.
- PPO sampling noise is not by itself evidence of biological bet hedging.
- Claims are limited to learned outcomes and do not imply global optimality or
  equilibrium convergence.

## Related study

The first application of this decision is specified in
[`uniform-p-association-response-pilot-spec.md`](../implementation-docs/uniform-p-association-response-pilot-spec.md).
