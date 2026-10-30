# MycorMARL Domain Context

## Glossary

### Static intraradical mycelium allocation

`irm_fraction` is the fixed fraction of total fungal dry biomass allocated to
intraradical mycelium (IRM). Extraradical mycelium (ERM) is the remaining
`1 - irm_fraction`; only ERM produces external soil-uptake geometry. Both
fractions share the existing fungal construction stoichiometry and maintenance
coefficients. Growth and mortality change both fractions proportionally; IRM
is not protected from biomass loss. Total biomass remains the single state
and accounting quantity, including retained residual mass after death.

The trait defaults to zero, must be finite in `[0, 1)`, and must be zero in
`fungus-only` mode. Varying it holds configured total initial biomass fixed.
It is not a colonised-root fraction, root-capacity limit or exchange rule.
IRM/ERM mass and volume can be derived when needed; no separate state, stock
diagnostics or policy inputs are added. See [ADR-0017](docs/adr/0017-represent-absorbers-as-fine-root-discs-and-an-external-hyphal-hemisphere.md)
and [issue #96](https://github.com/isaachcamp/mycormarl/issues/96).

### Non-limiting internal transport capacity

An approximation under which retained living, connected root and hyphal tissue
does not constrain resource delivery through its conducting capacity. Reduced
interior density therefore causes no independent throughput limit, delivery
delay or compensatory transport-carbon surcharge. Density can still affect
uptake, tissue investment, maintenance and a separately specified route distance.
See [ADR-0021](docs/adr/0021-treat-internal-transport-capacity-as-non-limiting.md).

### Oldest-first maintenance mortality

Tissue loss triggered by insufficient maintenance resources, allocated to the
oldest eligible living tissue first. The planned fungal rule retains a minimum
living density; the revised fine-root rule permits complete cohort removal,
with stem connectivity supplied by an implicit conducting-root backbone.
This is not current production behaviour. [Issue #89](https://github.com/isaachcamp/mycormarl/issues/89)
and dependent tickets #90–93 require reconciliation with
[ADR-0022](docs/adr/0022-separate-fine-root-mortality-from-persistent-backbone-connectivity.md)
before implementation of the revised root rule.

### Implicit conducting-root backbone

An unresolved root transport structure assumed to maintain stem connectivity
to surviving distal fine roots while the plant is operational. Its
root-containing footprint persists where roots have grown even when local
absorbing fine-root density becomes zero. It adds no separately represented
biomass or uptake surface. All cells in this footprint are assumed available
for fungal exchange without separately tracked colonisation or establishment
delay, including backbone-only cells. They count as root-containing cells for
fungal transport-cost distance. Availability does not compel trade or bypass
organism-death gates. See [ADR-0022](docs/adr/0022-separate-fine-root-mortality-from-persistent-backbone-connectivity.md).

### Fungal root contact

A surviving connection between the living fungal network and an
exchange-available cell in the persistent root footprint. Contact can be lost
without immediate organism death; loss of all contact disables bilateral C/P
trade while ordinary maintenance and survival dynamics continue. This is agreed
future behaviour, not current production behaviour. The distance approximation
uses the existing reference fine-root density as its within-cell spacing proxy,
independently of post-mortality absorbing density. See
[ADR-0022](docs/adr/0022-separate-fine-root-mortality-from-persistent-backbone-connectivity.md).

### Transport-front shell

A finite-width band of living hyphae immediately behind the outer growth front,
used as a proxy for transport source/destination distance. Its width is a
physical parameter in cm, independent of timestep or grid spacing. It does not
restrict phosphate uptake or spread actual new tissue construction over that
band. Uptake-FWHM and tip-density dynamics are deferred alternatives; no width
value or energetic coefficient is selected. See the
[transport-shell derivation](docs/technical-summaries/transport-shell-geometries/README.md).

### Front-only absorber growth

Addition of new absorbing tissue exclusively at the current outer growth
boundary, without replacement or densification of established inner tissue.
This is a core approximation of the planned absorber-mortality extension;
tissue retained at its minimum density remains living and uptake-active.

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
first-order **per-day** rates for trade, growth and reproduction. Trade and
stoichiometric biology compete simultaneously for evolving free C/P pools;
received trade is eligible during allocation. Retention is the free resource
stock left after realised spending, including essential-resource surplus and
finite-rate residue. It has no separate controlled hazard or compartment.
New soil P and photosynthate become eligible in the next numerical step.

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

### Time-dependent finite-inventory hyphal-density sweep

An isolated, single-cell closure diagnostic that holds one fungal absorber
fixed, sweeps hyphal length density, advances the global diagnostic clock for
30 days, and deducts accepted phosphate uptake from a finite reservoir. The
elapsed clock controls the existing depletion-gradient resistance and
sparse/continuous blend. It reports both absolute P uptake and P uptake per
structural hyphal construction carbon. It is not a production-model spatial
simulation and does not include diffusion from neighbouring cells.

### Structural hyphal construction carbon

The carbon physically represented by the fixed hyphal cylinder in the
diagnostic: represented hyphal length times cylindrical cross-sectional area
times fungal hyphal tissue carbon density. It excludes carbon-use-efficiency
losses, maintenance, host transfer, and whole-fungus budgets.

### Cumulative reproductive fitness

The time-sum of an organism's realised stoichiometric reproductive output
over an evaluation episode. Both essential resources constrain output; their
costs equal growth costs times the organism's reproductive cost scale. The
output is exported and adds no parental biomass. Plant cumulative reproductive fitness is the
primary plant-performance endpoint for phosphorus-response policy studies
because it is the quantity optimized by the plant PPO objective.

Final living plant biomass and cumulative gross plant growth are secondary
plant-performance endpoints. They make the simulated response comparable with
greenhouse growth measurements, but they are not interchangeable with
cumulative reproductive fitness and are not themselves the PPO objective.

### Reproductive cost scale

A finite, positive multiplier of both growth C/P costs per unit reproductive
output, specified independently for plant and fungus and defaulting to one.
At fixed commands, changing it scales output efficiency while preserving
reproductive resource expenditure and composition.

### Biological intensity and reproductive expenditure share

Biological intensity is the sum of growth and reproduction rates. The
reproductive expenditure share divides that intensity between the two paired
resource sinks. It is an expenditure preference; output also depends on
reproductive cost scale and essential-resource availability.

### Declared static allocation experiment

A fixed-share experiment that declares equal policy and outer environment
intervals, a nominal residual target, and the corresponding total processing
intensity. This contract is distinct from unrestricted fixed-rate references
and learned policies. Numerical refinement holds the physical commands fixed.

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
nonlinear softplus intensity and sigmoid share transforms generally make the transform
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
