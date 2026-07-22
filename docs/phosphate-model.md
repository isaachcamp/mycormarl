# Phosphate model guide

## Scope and implementation status

The implemented model represents labile inorganic phosphorus (Pi) in an
axisymmetric `r-z` soil domain. It couples closed-boundary diffusion, linear
instantaneous buffering, root and fungal uptake, sparse/continuous uptake
blending, competition for finite cell inventory, and organism growth geometry.
The complete transport, uptake, competition, and geometry pipeline is
implemented and numerically qualified. Historical planning and verification
records are retained under [`archive/`](archive/).

This is a mechanistic provisional parameterisation, not an empirically
calibrated soil. In particular, an initial `1 µM` solution concentration does
not mean that extractable P or total soil P is `1 µM`.

## State and units

| Quantity | Symbol / field | Unit | Status |
|---|---|---|---|
| Labile P per cell | `State.soil_labile_p` | µmol P cell⁻¹ | canonical conserved soil state |
| Solution Pi | `C` | µmol P cm⁻³ | derived from canonical state |
| Plant/fungal P pools | `plant_p_pool`, `fungus_p_pool` | mg P | canonical organism state |
| Root/hyphal length density | `root_length_density`, `hyphae_length_density` | cm structure cm⁻³ bulk soil | derived from structural biomass |
| Cell volume | `V` | cm³ bulk soil | static grid geometry |
| Biological time | `dt` | day | configuration/scheduler |
| Physical time | `dt_s` | s | derived for flux kernels |

The soil-to-organism conversion occurs once, when accepted uptake is credited:
`1 µmol elemental P = 0.0309738 mg P`. Concentration is never stored as a
second authoritative field.

## Equations

### Initial inventory and buffering

Configured concentration in micromolar is converted by
`C [µmol cm⁻³] = 10⁻³ C [µM]`. With volumetric water content `theta` and
linear volumetric buffer power `B`, each cell stores

```text
M_labile = C V (theta + B)
C = M_labile / [V (theta + B)].
```

`B = dC_sorbed,bulk/dC_solution` represents the instantaneously reversible
sorbed inventory. The dissolved fraction is `theta/(theta+B)` and the
retardation factor is `(theta+B)/theta`. The default `theta=0.3`, `B=239`
therefore stores about 0.125% of labile P in solution at equilibrium.

### Diffusion

Amount transfer across internal finite-volume faces uses

```text
q = -D_l theta f_l grad(C),
```

with equal and opposite transfers between neighbours. All external soil faces
are no-flux; the `r=0` face also has zero area. Buffering is included through
the concentration–amount relation, not inserted a second time into the flux.
The implied propagation diagnostic is

```text
D_app = D_l theta f_l / (theta + B).
```

An explicit cellwise CFL ceiling is calculated from storage capacity and face
conductances. When a biological step exceeds `0.8 dt_CFL`, the complete
fixed-geometry diffusion-then-uptake transaction is repeated in equal
substeps. The `1 day` reference time used below is a biological regime
parameter; it is not a diffusion substep or numerical stability interval.

### Growth to absorbing geometry

Plant dry biomass `G_p` produces total fine-root length

```text
L_root = G_p k_root SRL.
```

Differences of `F(d)=1-beta^d` partition this length among depth layers. Every
occupied disc uses the same provisional density `lambda_root`, so each layer
has its own radius

```text
R_k = sqrt[L_root w_k / (pi lambda_root dz_k)].
```

Fungal dry biomass `G_f` produces external hyphal length through

```text
L_hypha = G_f gamma_C,f / (M_C pi r_h²).
```

That length fills a saturated hemisphere of radius

```text
R_colony = [3 L_hypha / (2 pi lambda_sat)]^(1/3).
```

Both geometries volume-average cells crossed by their fronts and clip at the
soil boundary. Density is recomputed from surviving post-growth biomass.

### Uptake, regime transition, and competition

Integrated length in a cell is `L=lambda V`; lateral absorbing area is
`A=2 pi r L` (end caps excluded). Continuous uptake requests use

```text
J(C) = J_max C / (K_m + C)
U_cont = J(C_b) A dt_s.
```

The sparse closure assigns each absorber a cylindrical territory
`R_soil=1/sqrt(pi lambda)`. Its effective depletion radius is bounded by the
territory and finite propagation over `T_ref`. A logarithmic diffusion
resistance gives an analytical, non-negative Michaelis–Menten surface
concentration `C_s`, from which `U_sparse=J(C_s)A dt_s`.

Hyphal density alone defines the first-model overlap time `t_diff`. Both roots
and fungi use the shared weight

```text
Omega = T_ref / t_diff
w_cont = Omega^p / (1 + Omega^p)
U_request = (1-w_cont) U_sparse + w_cont U_cont.
```

With no hyphae, `w_cont=0`; a plant-only run therefore remains sparse. Roots
become part of the continuous regime wherever sufficiently dense hyphae make
the shared depletion field continuous.

Plant and fungal requests are calculated from the same post-diffusion bulk
concentration. Competition is then applied once per cell:

```text
s = min[1, M_available / (U_root,request + U_fungus,request)]
U_root = s U_root,request
U_fungus = s U_fungus,request.
```

This prevents overdraw and preserves request shares. It does not fully resolve
sub-cell interference between separate sparse depletion zones when demand is
below the inventory cap.

## Environment step order

1. Constrain both allocation actions and snapshot start-of-step pools.
2. Calculate trade, realised stoichiometric growth, maintenance, reproduction,
   and biomass loss using only those starting pools.
3. Record structural-P mortality loss and reproduction export; apply trade.
4. Fix plant carbon, then recompute root and hyphal geometry from surviving
   post-growth biomass.
5. Calculate geometry-only sparse resistances and overlap weights once.
6. For every CFL substep: diffuse canonical amount, derive fresh `C_b`, form
   sparse and continuous requests, blend, cap competition, subtract accepted
   uptake, and credit organism P pools in mg.
7. Increment time. Newly acquired P can fund growth only on the next step.

## Configuration

`EnvConfig` owns soil and numerical controls; `PlantTraits` and `FungusTraits`
own organism parameters. The selected production defaults are a `50 cm`
maximum radius, `100 cm` depth, `0.1 cm` radial/depth intervals, `25 cm`
P-bearing topsoil, `1 µM` initial solution Pi, `dt=0.025 day`, and `14,600` steps.
The resulting grid is `500 x 1000` cells. Smaller explicit configurations
should be used for development and tests.

Important controls are:

| Configuration field | Meaning |
|---|---|
| `consumer_mode` | `mixed`, `plant-only`, or `fungus-only`; independent modes retain a dormant partner in the two-agent API |
| `initial_solution_p_um` | solution Pi used only to initialise the canonical inventory |
| `theta_water`, `buffer_power` | water-filled storage and reversibly sorbed storage |
| `phosphate_diffusion_coefficient_cm2_s`, `phosphate_impedance_factor` | solution diffusion and tortuosity/impedance multiplier |
| `diffusion_cfl_safety` | explicit stability safety factor |
| `uptake_reference_time_days`, `uptake_transition_exponent` | static sparse/continuous transition (`T_ref`, `p`) |
| `radial_interval_cm`, `depth_interval_cm` | requested cell intervals; a final shortened cell is allowed |

Construction validates all state-forming organism pools, growth/maintenance
rates, death fractions, photosynthesis traits, uptake/geometry traits, and
finite physical soil ranges. It then precomputes grid geometry, face
conductances, the CFL schedule, and static array shapes.

## Parameter provenance

| Runtime default | Value | Provenance and status |
|---|---:|---|
| Initial solution Pi | `1 µM` | Order-of-magnitude soil-solution reference associated with Vance et al. (2003), [doi:10.1046/j.1469-8137.2003.00695.x](https://doi.org/10.1046/j.1469-8137.2003.00695.x); not a universal mean and not extractable P. |
| `theta`, `B` | `0.3`, `239` | Schnepf & Roose (2006), attributed there to Barber (1995), [doi:10.1111/j.1469-8137.2006.01771.x](https://doi.org/10.1111/j.1469-8137.2006.01771.x). Soil-specific calibration required. |
| `D_l`, `f_l` | `1e-5 cm² s⁻¹`, `0.308` | Provisional Schnepf–Roose parameterisation; yields `D_app≈3.86e-9 cm² s⁻¹`. |
| Root/fungal `J_max`, `K_m` | `3.26e-6 µmol cm⁻² s⁻¹`, `5.8e-3 µmol cm⁻³` | Tinker & Nye (2000) values as reported by Schnepf & Roose (2006). Separate fields intentionally retained. |
| Hyphal radius | `5e-4 cm` | Schnepf & Roose (2006). |
| Root radius | `0.01 cm` | Inherited, unsourced provisional value; replace with cultivar/absorbing-order data. |
| Plant `gamma_C` | `0.402 g C g⁻¹ dry mass` | Untreated dry carrot-root elemental analysis, Kaur et al. (2022), *Scientific Reports*, [doi:10.1038/s41598-022-20971-5](https://doi.org/10.1038/s41598-022-20971-5). Root-dominated proxy, not a whole-plant measurement. |
| Plant `gamma_P` | `1.92 mg P g⁻¹ dry mass` | Dry-mass-weighted value derived from unfertilised carrot roots/leaves in Kováčik et al. (2022), *Agronomy* (MDPI), [doi:10.3390/agronomy12112770](https://doi.org/10.3390/agronomy12112770). Independent non-MDPI corroboration required. |
| `k_root`, `SRL` | `0.62`, `25,434.3 cm g⁻¹` | *Daucus carota* medians from GRooT: Guerrero-Ramírez et al. (2021), [doi:10.1111/geb.13179](https://doi.org/10.1111/geb.13179). Standing root-mass fraction is used as marginal allocation; the two medians are not matched observations. |
| Fungal `gamma_C`, `M_C` | `0.5 g C g⁻¹`, `0.1155 g C cm⁻³ tissue` | Bisot et al. (2026), [doi:10.1073/pnas.2512182123](https://doi.org/10.1073/pnas.2512182123). Precise AM-fungal tissue density remains uncertain. |
| Fungal `gamma_P` | `40 mg P g⁻¹ dry mass` | Maximum 4% mass fraction reported by the Bisot et al. literature search. This is an upper bound, not a validated representative default; underlying studies must be reviewed. |
| `lambda_sat` | `168.75 cm⁻²` | Converted from `1500 µm mm⁻²` in Oyarte-Galvez et al. (2025), [doi:10.1038/s41586-025-08614-x](https://doi.org/10.1038/s41586-025-08614-x), using the stipulated `rho_3D=3 rho_2D²/4`; not a direct 3-D soil measurement. |
| `lambda_root`, `beta` | `1 cm cm⁻³`, `0.96` | Inherited/provisional geometry choices; no empirical calibration is claimed. |
| `T_ref`, `p` | `1 day`, `2` | Explicit modelling choices. `T_ref` approximates exposure time while all structure remains indefinitely active. |
| Grid, timestep | `0.1 cm`, `0.025 day` | Numerical qualification selected the grid under the 5% rule, but no coarser timestep passed after coupled P uptake and free-pool outputs were added. The timestep is therefore the finest tested fallback, not demonstrated convergence. |

## Running and interpreting examples

After installing the project environment, run:

```bash
uv run python scripts/phosphate_examples.py --mode all
```

Use `--mode plant-only`, `fungus-only`, or `mixed` for one scenario. Each
four-cell example reports initial/final soil amount, uptake credited to each
consumer, the soil-to-pool balance error, minimum cell inventory, mean
continuous weight, capped-cell fraction, and numerical substep count. These
are fixed-geometry uptake examples, not calibrated ecosystem predictions.

For complete environment trajectories, set `EnvConfig.consumer_mode` to the
same three values. The environment always exposes `plant` and `fungus` to keep
one stable JaxMARL interface. In an independent mode, the absent partner starts
with zero biomass and pools, remains dead/dormant, cannot trade or allocate,
has zero geometry and uptake, and receives zero reward.

Death is also absorbing for an organism that was initially active. Its stored
biomass is retained as historical state, but subsequent actions, fixation,
trade, growth, maintenance, reproduction, active geometry, P uptake, and
reward are zero. There is currently no recovery or remobilisation pathway.

The PPO stack uses these same identifiers. A small runnable training job is:

```bash
uv run python scripts/train_ppo.py --total-timesteps 256 --num-envs 1
```

The launcher uses a typed `PPOConfig`, an explicit small soil domain, and saves
both parameter trees under `outputs/` by default. Production experiments
should construct and persist their own explicit environment, species, and PPO
configuration rather than relying on launcher defaults.

For the full numerical study, run `scripts/phosphate_qualification.py`; its
canonical reports are
`docs/qualification/phosphate-numerical-qualification.md` and `.json`. The
selected spatial comparison remained below the provisional 5%
rule and the reported closed-system errors were below `1e-5`. No timestep
above `0.025 day` passed once coupled uptake and endpoint free-P pools were
included; the selected timestep is consequently a conservative fallback
pending a finer study or a justified change to the endpoint metric.

## Known limitations

- Maintenance P is removed from free pools but has no explicit destination.
  Do not claim whole-system P conservation for maintenance-active runs until
  reservation, recycling, or loss is chosen and implemented.
- Organism P lost with mortality is recorded as system loss; no litter,
  mineralisation, or recycling pool exists.
- All represented roots/hyphae remain uptake-active indefinitely. Local colony
  age and growth-front exposure are deferred.
- Linear buffering is instantaneous, homogeneous, and zero-intercept. Soil
  calibration may require nonlinear or kinetic sorption.
- The axisymmetric root and saturated-hemisphere fungal geometries are closures,
  not explicit branching networks.
- Initial organism pools and every empirical proxy above require calibration
  against the eventual experiment; extractable-P observations need an
  observation model rather than direct comparison with solution `C`.

The maintained list of decisions still to be made is in
[open questions](open-questions.md). For code ownership and pipeline
boundaries, see the [module map](module-map.md).
