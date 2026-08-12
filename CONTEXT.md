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

### Construction-carbon efficiency diagnostic

A figure-level diagnostic that divides plant phosphate uptake by the
structural carbon required to construct the absorbing length represented in a
cell. For target absorber length `L = lambda * V_i`, plant-side construction
carbon is computed analytically from the existing biomass-to-root-length
relation and plant `gamma_c`; above-ground growth is not included.

The diagnostic distinguishes fixed-bulk-concentration uptake (an intrinsic
transport comparison) from finite-inventory uptake (realised depletion of a
cell's phosphate pool). Absolute uptake is reported separately from uptake per
construction carbon.

Under plant economics, construction carbon is a constant multiple of absorber
length. Uptake per construction carbon and uptake per unit absorber length
therefore have the same structure across a geometry sweep and differ only by a
constant scale factor.

### Reference depletion timescale

The finite-inventory time `t_1%` at which surface concentration first falls
below one percent of its initial value. It is measured for a reference
absorber, rather than used as the horizon for the construction-carbon
efficiency surfaces. The configured uptake reference time `T_ref` remains the
common horizon for those surfaces.

### Time-dependent depletion-gradient diagnostic

A fixed-reservoir, fixed-absorber experiment in which global experiment time
controls the travel distance of the unresolved radial phosphate depletion
gradient around an absorbing cylinder. The absorber radius, represented
length, and length density do not grow or otherwise change.

The diagnostic's global time coordinate is not colony propagation, organism
age, or a production-model approximation for the ages of roots and hyphae.
Using cohort age in the full model is a separate modelling problem.
