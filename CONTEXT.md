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

### Rate action

The action command passed to `BaseMycorMarl`. It commands non-negative
first-order **per-day** rates for trade, growth, reproduction, and storage.
Growth, reproduction, and storage compete as pool outflow hazards rather than
as a simplex; storage initially means retention in the existing free pool, not
a new compartment. Newly acquired soil P, photosynthate, and received trade
are eligible for the held rate during subsequent numerical substeps.

The policy wrapper holds the Rate action unchanged across numerical substeps.
PPO records sampled latents, the executed Rate action, and likelihoods in
latent space. Non-policy callers use `mycormarl.actions.rate_action`.

### Finite-horizon PPO return

The undiscounted sum of rewards up to a declared administrative episode
boundary. Its terminal transition has zero continuation value: an
administrative truncation is a return boundary in this experiment, rather than
a bootstrap into the reset episode. This is distinct from the existing
time-limit-bootstrap behaviour used for continuing tasks.

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

For the plant, the represented infrastructure is exactly its fine-root biomass:
`kfroot` is the whole-plant mass fraction converted to absorbing length, and
all of that represented fine-root length is uptake-active. The model has no
separate inactive, coarse-root, or active-absorber fraction within `kfroot`.

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

### Cumulative reproductive fitness

The time-sum of an organism's per-transition Cobb–Douglas reproduction reward
over an evaluation episode. Plant cumulative reproductive fitness is the
primary plant-performance endpoint for phosphorus-response policy studies
because it is the quantity optimized by the plant PPO objective.

Final living plant biomass and cumulative gross plant growth are secondary
plant-performance endpoints. They make the simulated response comparable with
greenhouse growth measurements, but they are not interchangeable with
cumulative reproductive fitness and are not themselves the PPO objective.

### Plant biomass numerical guard

The `50 g DM` hard limit on represented plant biomass and realised structural
growth. It provides numerical headroom above the `25--35 g DM` favourable
carrot trajectory reference; it is not an observed maximum or mechanistic
carrying capacity. Contact with the guard invalidates a qualification
trajectory and must not be interpreted as physiological growth saturation.

The independent `50 g DM` plant biomass observation reference controls only
the bounded actor input `B / (B + reference)`. It preserves the earlier policy-
input scale and prevents changes to the numerical guard from silently changing
policy observations.

### Learned IPPO outcome

The evaluated behaviour of independently trained plant and fungal PPO policies
under a stated environment, training protocol, and seed. A learned IPPO
outcome is an empirical candidate produced by optimization; it is not evidence
of a globally optimal strategy or a game-theoretic equilibrium.

### Uniform-P initial condition

A reset condition in which every soil cell in the configured domain begins at
the same solution-P concentration. A supplied depth profile is the separate
opt-in initial vertical-heterogeneity treatment. After reset, the production
finite-inventory soil model evolves normally: diffusion and root and fungal
uptake may generate spatial and temporal concentration differences.

"Uniform" therefore describes the externally imposed initial condition, not a
continuously replenished reservoir and not a constraint that keeps the field
spatially uniform. Variation generated by the organisms and soil dynamics is
endogenous P variation; deliberately added spatial or temporal variation is an
externally heterogeneous P treatment.

### Latent-location policy evaluation

A deterministic evaluation in which each PPO actor uses the location of its
learned latent Gaussian and transforms that location through the production
latent-to-physical-action mapping. It is the primary evaluation protocol for
P-response studies.

This protocol does not compute the expected physical action, because the
nonlinear sigmoid and centred-simplex transforms generally make the transform
of a latent mean differ from the mean transformed action. Repeated sampled
policy rollouts are a separate diagnostic of residual policy-distribution
variability, not the primary estimate of learned strategy and not by itself
evidence of biological bet hedging.

### AM engagement gate

A reset-level association choice for later experiments. Gate-on selects the
complete `mixed` initial condition, including the established fungal partner,
and gate-off selects the complete `plant-only` initial condition. It is not an
in-episode threshold on continuous trade and does not leave fungal biomass or
fungal uptake geometry in the soil when off.

The gate is an abstract formation-or-acceptance choice, not a mechanistic
measure of root colonisation. Its initial version adds no initiation or
persistence cost. Conditional on `mixed`, the existing plant and fungal
continuous trade policies operate unchanged.

### Ex-ante association advantage

The difference in expected plant performance between separately trained
`mixed` and `plant-only` modes over the same distribution of environmental
conditions. The association choice is conceptually made before the particular
heterogeneous initial P field is drawn. Paired evaluation uses the same initial
field realization in both modes.

Positive ex-ante association advantage means that access to an established AM
partner is favoured under the modeled environmental distribution. Under the
current expected-reward objective this is an insurance or risk-buffering
result, not by itself evidence that the policy optimizes evolutionary bet
hedging or geometric-mean fitness.
