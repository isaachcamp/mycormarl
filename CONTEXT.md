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
cell. Root-tissue structural-carbon density is inferred from the supplied
plant `gamma_c`, specific root length, and reference root radius. For target
absorber length `L = lambda * V_i`, construction carbon is
`L * pi * r_a^2 * rho_C,root`. Root allocation fraction, above-ground growth,
maintenance, reproduction, and whole-organism carbon budgets are excluded.

The diagnostic distinguishes fixed-bulk-concentration uptake (an intrinsic
transport comparison) from finite-inventory uptake (realised depletion of a
cell's phosphate pool). Absolute uptake is reported separately from uptake per
construction carbon.

Cells with `r_a >= R_terr`, where
`R_terr = 1 / sqrt(pi * lambda)`, are touching or overlapping absorber
geometries. They are retained and flagged in tabular output but masked from
scientific surfaces.

### Fungus-equivalent plant geometry

A panel-specific reference on a construction-carbon efficiency surface. At
native fungal length density, its plant absorber radius is solved continuously
so that plant-economics P uptake per construction C equals the native fungal
P-per-C value for the displayed metric. It is not the nearest sampled grid
cell and can differ between integrated-uptake and instantaneous-rate panels.

Efficiency panels also show actual fungal geometry evaluated with plant uptake
and root-tissue construction economics. This counterfactual remains at the
fungal coordinates, whereas the fungus-equivalent plant geometry is an
on-surface, panel-specific plant geometry.

Absolute-uptake and depletion-timescale panels instead use actual
fungus-native geometry because they do not display construction economics.

### Reference depletion timescale

The finite-inventory time `t_1%` at which surface concentration reaches one
percent of its initial value. It is obtained from a semi-analytical event-time
calculation with uptake resistance and blending frozen at configured `T_ref`,
rather than by stopping a fixed-duration timestep simulation. The configured
`T_ref` remains the common horizon for the construction-efficiency and
absolute-uptake surfaces.

### Time-dependent depletion-gradient diagnostic

A fixed-reservoir, fixed-absorber experiment in which global experiment time
controls the travel distance of the unresolved radial phosphate depletion
gradient around an absorbing cylinder. The absorber radius, represented
length, and length density do not grow or otherwise change.

The diagnostic's global time coordinate is not colony propagation, organism
age, or a production-model approximation for the ages of roots and hyphae.
Using cohort age in the full model is a separate modelling problem.
