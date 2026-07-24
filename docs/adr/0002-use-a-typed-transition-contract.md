# ADR-0002: Use a typed `Transition` contract

**Status:** Accepted

## Context

The small PPO transition-adapter function needs execution, termination, truncation, and
pre-reset observation facts from `BaseMycorMarl`. If these essential facts are
mixed into an informal diagnostic dictionary, spelling or shape drift can
silently change learning semantics.

The repository currently has a PPO `Trajectory` type but no type named
`Transition`.

## Decision

`BaseMycorMarl` will expose essential step facts through a stable, typed JAX
PyTree named `Transition`. `Transition` is a per-agent type. Every environment
step returns a fixed mapping:

```text
{
    "plant": Transition(...),
    "fungus": Transition(...),
}
```

The fixed mapping is preserved in mixed, plant-only, fungus-only, living, and
dead-agent states.

`Transition` is algorithm-independent. It describes what happened during one
environment step and does not contain PPO masks, advantages, value targets,
GAE controls, or optimiser state.

Loose diagnostics are not fields in the stable `Transition` contract. They
remain a separate output intended for inspection rather than learning
correctness.

Administrative truncation is a joint boundary fact but is copied into both
per-agent instances so each independent-policy adapter input is self-contained.
The two values must be identical and this is an environment invariant.

Each per-agent `Transition` includes at least:

- `operational_at_start`;
- `operational_at_end`;
- `allocation_executed`;
- `trade_executed`;
- `truncated`;
- `final_observation`.

Lifecycle uses start and end operational status rather than a single sticky
termination flag. The PPO transition-adapter function derives biological termination
from an operational-to-non-operational change and excludes already dead or
absent transitions using `operational_at_start`.

## Consequences

- JIT, scan, and vmap see a fixed transition structure and fixed field shapes.
- The PPO transition-adapter function depends on typed fields rather than stringly typed
  diagnostic keys.
- Plant and fungal PPO processing can consume the same per-agent schema
  independently.
- Joint truncation is duplicated deliberately and requires an equality test.
- Death transitions and already-dead padding remain distinguishable without
  adding PPO validity masks to the environment.
- Schema changes become explicit compatibility changes with focused tests.
- `Transition` and PPO `Trajectory` have distinct meanings documented in the
  project glossary.
