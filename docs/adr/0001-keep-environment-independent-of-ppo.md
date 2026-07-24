# ADR-0001: Keep `BaseMycorMarl` independent of PPO

**Status:** Accepted

## Context

`BaseMycorMarl` is a biological multi-agent environment. Independent PPO needs
additional trajectory bookkeeping, including actor-valid, trade-active, and
critic-valid masks plus termination/truncation bootstrap controls.

Returning those PPO concepts directly from the environment would couple the
scientific model to one learning algorithm and make the transition semantics
harder to reuse or test independently.

## Decision

`BaseMycorMarl` will emit only an algorithm-independent `Transition` through
its standard JaxMARL boundary. Its facts include:

- requested and realised biological action components;
- whether allocation and trade actually executed;
- biological termination and administrative truncation;
- the final pre-reset observation needed at an auto-reset boundary.

A small pure PPO transition-adapter function in the algorithm layer will
convert the `Transition` into the trajectory schema used by independent PPO.
It will derive actor, trade, critic, bootstrap, and GAE trace masks without
introducing an adapter class or framework.

The environment must not import PPO modules or expose fields named in terms of
PPO losses, advantages, value targets, GAE, or policy optimisation.

## Consequences

- Environment dynamics and accounting can be tested without constructing PPO.
- PPO boundary semantics have one explicit, independently testable conversion
  point.
- Other algorithms can consume the same `Transition`.
- The adapter becomes a required integration component and must have contract
  tests against representative environment transitions.
