# MycorMARL Domain Context

## Glossary

### Operational agent

An agent that is configured as present, biologically alive, and able to
participate in the environment process at the specified transition boundary.
`operational_at_start` and `operational_at_end` record this status before and
after one environment step.

An absent single-consumer counterpart and an already dead organism are both
non-operational. A change from operational to non-operational during a
transition is a biological termination event.

### Physical action

The bounded resource-allocation command passed to `BaseMycorMarl`:
`[trade, growth, reproduction, reserve]`, where trade is independently bounded
and the remaining three components form a simplex.

A physical action is valid by construction before it reaches the environment.
`BaseMycorMarl` executes it unchanged and does not clip, sanitise, project, or
renormalise it. PPO produces physical actions through its latent transforms;
non-policy callers use the shared public action-construction helper.

### Transition

A stable, typed JAX PyTree produced by `BaseMycorMarl` for one agent during a
completed environment step. A `Transition` contains algorithm-independent
facts, including realised biological actions, whether an action component
executed, biological termination, administrative truncation, and the agent's
final pre-reset observation.

A `Transition` describes what happened in the model. It is not a PPO
trajectory sample and does not contain PPO loss masks, GAE controls,
advantages, value targets, or other learning-algorithm bookkeeping. Loose
human-facing diagnostics remain separate from this stable contract.

Every step returns a fixed mapping with one `Transition` for `plant` and one
for `fungus`, including in single-consumer modes and after death. The
administrative truncation flag is identical in both instances.

Lifecycle is represented by `operational_at_start` and
`operational_at_end`, not only by a sticky termination flag. This preserves
the distinction between an organism that dies during the current transition
and one that was already dead or absent.

### PPO transition adapter

A small pure function in the PPO layer that converts a `Transition` into the
trajectory fields needed by independent PPO. It derives PPO validity masks and
bootstrap/trace controls without making `BaseMycorMarl` depend on PPO. It is
not a class, framework, or extensibility layer.

### Biomass-derived uptake geometry

The root and external-hyphal length-density fields reconstructed from an
organism's current biomass before soil uptake. These fields represent current
living uptake infrastructure, not a persistent record of previously occupied
soil.

Maintenance-induced biomass loss therefore changes uptake geometry through the
same biomass-to-geometry conversion as any other biomass change. The model does
not retain thinning, damage, abandoned territory, or a historical growth front
once biomass is restored.
