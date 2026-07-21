# Phosphate foundations: function and test reference

This page documents the implemented P0–P6 phosphate functions and tests: what
each addition does, why it exists, and where it sits in the transport–uptake
pipeline. The source docstrings are the local API reference; this page explains
their composition. For equations, configuration, provenance, and limitations,
start with `implementation-docs/phosphate-model-guide.md`.

## Pipeline position

```text
EnvConfig (cm extents/intervals, µM, theta, B)
  -> scalar validation
  -> axisymmetric edges, volumes, and face areas
  -> topsoil-averaged solution concentration
  -> µM to µmol cm^-3
  -> M_labile = C * V_cell * (theta + B)
  -> State.soil_labile_p  [canonical conserved state]

Plant/fungal biomass (g dry mass)
  -> organism-specific biomass-to-length conversion
  -> axisymmetric root / saturated-hemisphere hyphal distribution
  -> State root/hyphal length density  [cm cm^-3]
  -> recomputed after growth and biomass loss, before soil uptake

Soil uptake step:
State.soil_labile_p
  -> C = M_labile / (V_cell * (theta + B))
  -> conservative internal-face diffusion of canonical amount
  -> repeat at exact CFL-safe substep duration when required
  -> sparse and continuous Michaelis-Menten requests
  -> fungal-overlap-derived smooth request blending
  -> one shared cell-inventory competition cap
  -> accepted uptake in µmol P
  -> µmol P to mg P
  -> plant/fungal P pools

```

Reset, biological growth-to-geometry, continuous uptake, conservative
diffusion, CFL-based subcycling, analytical sparse uptake, smooth regime
blending, and qualification diagnostics are active through P6. Blending does
not change the conservative transport or allocation boundaries.

## Canonical state and units

- Soil geometry: centimetres and cubic centimetres.
- Configured solution P: micromolar (`µM`).
- Derived kernel concentration: `µmol P cm^-3`.
- Stored soil state: `State.soil_labile_p`, `µmol P cell^-1`, shape
  `(n_r, n_z)`.
- Root/hyphal density: `cm structure cm^-3 bulk soil`, shape `(n_r, n_z)`.
- Physical rates: seconds.
- Organism P pools and loss/export diagnostics: `mg P`.

Storing amount makes cell inventory finite and conservation testable. Solution
concentration remains the quantity that drives gradients and Michaelis–Menten
kinetics, but it is derived rather than independently mutable.

## `phosphate_units.py`

### Constants

| Constant | Purpose | Pipeline role |
|---|---|---|
| `MICROMOLAR_TO_MICROMOL_PER_CM3` | `1 µM = 10^-3 µmol cm^-3` | Converts user-facing initial conditions to kernel units. |
| `MICROMOL_P_TO_MG_P` | `1 µmol P = 0.0309738 mg P` | Converts accepted soil uptake when crediting organism pools. |
| `SECONDS_PER_DAY` | `86,400 s day^-1` | Converts the biological timestep before physical flux calculations. |

### Functions

| Function | What it does | Why / pipeline fit |
|---|---|---|
| `micromolar_to_micromol_per_cm3` | Converts µM to `µmol cm^-3`; scalar or array, JAX-compatible. | Prevents a factor-of-1000 error when configuration enters buffering and uptake. |
| `micromol_p_to_mg_p` | Converts soil amount from µmol P to mg P. | Defines the single soil-to-organism unit boundary used in P3. |
| `days_to_seconds` | Converts model days to seconds. | Makes day-based scheduling compatible with per-second diffusion and uptake parameters. |
| `labile_capacity_factor` | Returns `theta + B`. | This is the local storage multiplier in `M_labile = C V (theta+B)`. |
| `retardation_factor` | Returns `(theta+B)/theta`. | Diagnostic for the slowing of apparent P propagation by reversible buffering. |
| `dissolved_labile_fraction` | Returns `theta/(theta+B)`. | Quantifies how much of the total labile inventory is instantaneously dissolved. |
| `michaelis_menten_surface_flux` | Evaluates `J_max C/(K_m+C)` in `µmol cm^-2 s^-1`. | Shared kinetic kernel for later continuous and sparse root/fungal uptake closures. |
| `cylindrical_lateral_area` | Evaluates `2πrL`, excluding end caps. | Converts cell-integrated root/hyphal length to absorbing surface before uptake. |
| `validate_linear_buffer_parameters` | Requires finite `theta > 0` and `B >= 0`. | Fails before allocation/JIT and protects conversion denominators and storage sign. |
| `validate_michaelis_menten_parameters` | Requires finite `J_max >= 0` and `K_m > 0`. | Keeps transformed kinetic code branch-free while preventing invalid configuration. |

The conversion and numerical functions accept arrays and are suitable for JAX
transformations. The validators accept scalar configuration and intentionally
run before JAX kernels are built.

## `phosphate_grid.py`

| Function | What it does | Why / pipeline fit |
|---|---|---|
| `validate_axisymmetric_grid_parameters` | Validates finite positive radius, depth, and radial/depth intervals. | Prevents malformed or empty grid allocations. |
| `_edges_from_interval` | Generates one zero-based edge vector, with tolerance for exactly divisible extents and an exact final maximum. | Centralises the shortened-boundary-cell rule used for both axes. |
| `axisymmetric_edges_from_intervals` | Builds interval-spaced radial/depth boundaries and ends exactly at each maximum. | Supplies one shared geometry to amount, density, uptake, and diffusion calculations while allowing a shortened final cell. |
| `axisymmetric_cylindrical_cell_volumes` | Computes `π(r_o²-r_i²) dz` for every `(r,z)` cell. | Converts concentration to amount and later integrates length densities. |
| `axisymmetric_radial_face_areas` | Computes `2πr dz` on `(n_r+1,n_z)` faces. | P4 uses these for conservative radial flux; the `r=0` face is exactly zero. |
| `axisymmetric_vertical_face_areas` | Repeats annular area on `(n_r,n_z+1)` horizontal faces. | P4 uses these for conservative vertical flux and top/bottom boundaries. |
| `axisymmetric_topsoil_fractions` | Computes the fraction of each depth cell above the topsoil limit. | Preserves inventory when 25 cm cuts through a cell instead of rounding to a layer. |
| `axisymmetric_uniform_p_conc` | Builds the temporary reset concentration field, volume averaging a crossed layer. | Represents the configured solution field immediately before amount conversion; it is not state. |
| `solution_concentration_to_labile_amount` | Applies `M = C V (theta+B)`. | Creates the canonical conserved cell inventory. |
| `labile_amount_to_solution_concentration` | Applies `C = M/[V(theta+B)]`. | Derives the field later used by diffusion gradients and uptake kinetics. |
| `initial_labile_p_from_micromolar` | Composes µM conversion, topsoil occupancy, volumes, and buffering. | Produces `State.soil_labile_p` during reset in one auditable path. |

## `phosphate_uptake.py`

| Function | What it does | Why / pipeline fit |
|---|---|---|
| `continuous_uptake_request` | Integrates length density over cell volume, converts length to lateral area, and multiplies Michaelis–Menten flux at bulk concentration by seconds per timestep. | Provides the coupled-depletion alternative for each consumer before P5 blending. |
| `territory_radius_cm` | Calculates `1/sqrt(pi lambda)` and returns infinity at zero density. | Converts local absorber density into the cylindrical soil territory available to the sparse closure and overlap diagnostic. |
| `effective_uptake_radius_cm` | Adds to the absorber radius the smaller of territory gap and `sqrt(D_app T_ref)`. | Bounds the analytical depletion annulus by both neighbouring absorbers and finite propagation time. |
| `sparse_uptake_resistance` | Calculates `k = r J_max ln(R_eff/r)/D_flux`, with infinite resistance when diffusive supply is disabled. | Caches the geometry/transport part of the nonlinear surface boundary once per biological step without double-counting buffering. |
| `hyphal_overlap_time_seconds` | Calculates the time for buffered diffusion to cross the hyphal territory gap; absent hyphae or diffusion returns infinity. | Supplies the static local depletion-coupling timescale and deliberately ignores root density in the first implementation. |
| `continuous_regime_weight` | Evaluates the stable bounded form `1/[1+(t_diff/T_ref)^p]`. | Produces one fungal-derived interpolation weight shared by root and fungal requests, including exact sparse/continuous limits. |
| `sparse_surface_concentration` | Evaluates the positive uptake/supply quadratic root using sign-dependent stable forms. | Recovers temporary `C_s` without an iterative solve, avoiding cancellation and enforcing `0 <= C_s <= C_b`. |
| `sparse_uptake_request` | Applies Michaelis–Menten kinetics at `C_s` and integrates cylindrical area and timestep. | Provides the isolated-absorber alternative in `µmol P cell^-1` using current post-diffusion bulk concentration. |
| `blend_uptake_requests` | Interpolates the two alternative request arrays with a clipped continuous weight. | Prevents double-counting by blending rather than summing sparse and continuous sinks before competition. |
| `PhosphateUptakeDiagnostics` | Names cellwise surface ratios, weights, alternative/final requests, and cap status. | Gives qualification code explicit units and fields while remaining a JAX-compatible tuple. |
| `blended_uptake_transaction` | Calculates both closures, blends with the shared weight, caps demand once, and returns accepted amounts plus diagnostics. | Is the single scientific transaction used by both production evolution and P6 reporting, preventing diagnostic-model drift. |
| `allocate_competing_uptake` | Caps combined root and fungal requests once against available labile amount and divides accepted uptake proportionally. | Guarantees non-negative inventory, preserves request shares under over-demand, and supplies separate accepted cell amounts for pool credit. |

## `phosphate_qualification.py`

| Function | What it does | Why / pipeline fit |
|---|---|---|
| `reference_relative_change` | Compares a candidate with its finer/smaller-step reference and suppresses meaningless relative errors when both values are below an explicit floor. | Implements the P6 5% selection arithmetic consistently across uptake, shares, biomass, and extent. |
| `annual_runtime_projection` | Converts a selected day-based timestep and warmed seconds per step into annual step count and runtime. | Separates a measured per-step cost from the one-off compilation cost and documents what the annual estimate means. |

## `phosphate_diffusion.py`

| Function | What it does | Why / pipeline fit |
|---|---|---|
| `apparent_diffusivity_cm2_s` | Evaluates `D_l theta f_l/(theta+B)`. | Supplies the buffered propagation diagnostic without incorrectly using `D_app` in the amount flux. |
| `axisymmetric_diffusion_conductances` | Computes internal-face `D_l theta f_l A/d` using actual cell-centre distances. | Caches static radial/vertical coupling, handles shortened final cells, and omits closed external faces. |
| `cell_outgoing_diffusion_conductance` | Sums incident internal conductances per cell. | Provides the denominator of the exact cellwise explicit stability limit. |
| `explicit_diffusion_cfl_seconds` | Returns `min_i[V_i(theta+B)/sum_j G_ij]`. | Defines the positivity ceiling used by the environment scheduler. |
| `required_diffusion_substeps` | Applies `max(1, ceil(dt_bio/(safety dt_CFL)))`. | Chooses a static equal-duration schedule; disabled diffusion yields one substep. |
| `diffuse_labile_amount` | Applies equal-and-opposite radial and vertical amount transfers from solution-concentration differences. | Conserves canonical µmol P under closed boundaries while redistributing it before uptake. |
| `validate_diffusion_parameters` | Validates `D_l >= 0`, `theta > 0`, `0 <= f_l <= 1`, and `0 < safety <= 1`. | Stops invalid transport schedules before arrays or JIT kernels are constructed. |

## Plant growth geometry

| Function | What it does | Why / pipeline fit |
|---|---|---|
| `validate_plant_growth_geometry_traits` | Requires valid growth, root-geometry, and uptake traits; bounds `kroot` and the depth parameter. | Prevents invalid traits from reaching JAX geometry and P3 uptake kernels. |
| `root_length_from_plant_biomass` | Applies `L_root = biomass * kroot * SRL`, defensively clipping negative biomass to zero. | Converts current plant dry biomass to absorbing-root length while applying `kroot` exactly once. |
| `_root_depth_cdf` | Evaluates `1 - beta^depth`. | Defines the provisional cumulative depth distribution. |
| `_depth_weights_from_edges` | Differences and normalises the CDF at actual depth edges. | Conserves total root length over the configured, potentially shortened grid. |
| `axisymmetric_disc_overlap_fractions` | Computes annular area fractions for either one radius or one radius per depth layer. | Volume-averages radial cells crossed by each depth-specific disc edge. |
| `root_disc_radii_from_biomass` | Applies `R_k = sqrt(L_root w_k / (pi lambda_root dz_k))`. | Converts beta-weighted layer lengths into slower radial expansion at greater depth while keeping within-disc density uniform. |
| `axisymmetric_stacked_disc_root_density` | Multiplies each layer's overlap fractions by the common `lambda_root`. | Conserves biomass-implied root length while discs fit and represents exact disc–domain intersections after radial clipping. |
| `density_field_from_biomass` | Provides the environment-facing root-density entry point. | Ensures reset and post-growth steps use the same pure construction path. |

## Fungal growth geometry

| Function | What it does | Why / pipeline fit |
|---|---|---|
| `validate_fungus_growth_geometry_traits` | Requires valid growth, hyphal-geometry, and uptake traits. | Prevents division by zero and invalid fungal geometry/kinetics before JIT. |
| `hyphal_length_from_fungal_biomass` | Applies `L_h = biomass * gamma_c / (M_C * pi * r_h²)`. | Implements the dry-biomass → structural-carbon → tissue-volume → cylindrical-length chain. |
| `colony_radius_from_length_axisymmetric` | Inverts `L = lambda_sat * (2/3) pi R³`. | Determines saturated hemispherical extent from total external-hyphal length. |
| `_cylinder_sphere_intersection_primitive` | Evaluates the axial integral used by sphere–cylinder intersections. | Supports analytical partial-cell volume rather than numerical quadrature. |
| `_volume_under_sphere_within_radius` | Calculates hemisphere volume within a cylindrical radius and depth interval. | Supplies exact occupied volumes for annular cells. |
| `axisymmetric_hemisphere_cell_fractions` | Converts intersection volumes to clipped cell fractions. | Volume-averages the moving colony front and clips growth at soil boundaries. |
| `axisymmetric_hemisphere_density` | Multiplies occupied fractions by `lambda_sat`. | Produces the local saturated/partially occupied hyphal density. |
| `axisymmetric_density_from_biomass` | Composes fungal length, colony radius, occupancy, and density. | Conserves length while the colony fits; after boundary contact it represents the exact hemisphere–domain intersection and approaches saturated capacity as the domain fills. |
| `density_field_from_biomass` | Provides the environment-facing fungal-density entry point. | Ensures reset and post-growth steps use the same pure construction path. |

## Environment and state integration

| Function / field | Responsibility |
|---|---|
| `BaseMycorMarl._validate_soil_config` | Runs consumer-mode, episode, timestep, grid, topsoil, concentration, buffer, diffusion, impedance, CFL-safety, `T_ref`, blend-exponent, and observation validation before constructing JAX arrays. |
| plant/fungal trait validators | Reject invalid initial pools, maintenance and photosynthesis rates, death fractions, biomass capacity, stoichiometry, geometry, and uptake traits before state construction. |
| `BaseMycorMarl.agents` | Exposes canonical `plant`/`fungus` identifiers used by the environment and PPO stack. |
| `EnvConfig.consumer_mode` | Keeps the two-agent API stable while masking the absent partner completely in `plant-only` or `fungus-only` trajectories. |
| `BaseMycorMarl.max_episode_steps` | Uses `EnvConfig.max_steps` unless the constructor receives an explicit override, allowing the selected annual configuration to request all 14,600 steps instead of silently retaining the historical 256-step episode limit. |
| `BaseMycorMarl._initial_state` | Uses precomputed geometry for labile P and converts initial plant/fungal biomass into 2D length-density fields; P diagnostics start at zero. |
| `BaseMycorMarl.step_env` | Caps plant growth before charging resources, records reproduction export and structural-P mortality loss, recomputes density from post-growth biomass, then invokes uptake so new P cannot fund same-step growth. |
| `BaseMycorMarl` static soil geometry | Caches conductances, exact CFL seconds, substep count, and substep duration once at construction. |
| `BaseMycorMarl.step_phosphorus_field` | Supplies cached transport geometry, schedule, and P5 controls to the integrated soil transaction. |
| `uptake_geometry_coefficients` | Derives root/fungal sparse resistances and one hyphal-density overlap weight from post-growth geometry. It runs once outside the substep loop because geometry stays fixed while concentration changes. |
| `_apply_blended_uptake` | Recalculates both closures from current concentration, blends requests, caps shared demand once, and credits accepted mg P. |
| `soil_diffusion_uptake_substep` | Applies conservative diffusion, then derives refreshed concentration and invokes blended uptake with cached coefficients. |
| `soil_diffusion_uptake_substep_with_diagnostics` | Runs the same substep but returns exact post-diffusion uptake diagnostics, letting offline qualification observe the concentration at which production uptake actually occurred. |
| `evolve_soil_p` | Calculates P5 geometry coefficients once, repeats equal-duration soil substeps, and accumulates accepted consumer uptake. |
| `evolve_soil_p_with_diagnostics` | Uses the same coefficient builder and substep transaction while retaining one diagnostic tuple per substep, providing an offline deterministic P6 path without changing environment state or the JIT production API. |
| `State.soil_labile_p` | The only mutable soil-P inventory, in `µmol P cell^-1`. |
| cumulative loss/export fields | Accumulate explicit `mg P` removed through structural mortality and reproduction. |

Geometry arrays (`r_edges`, `z_edges`, `cell_volumes`, radial face areas, and
vertical face areas) belong to the environment because they are static. The
dynamic state stores only quantities that evolve during an episode.

## Tests as executable documentation

### `test_phosphate_unit_contracts.py`

| Test | Contract protected |
|---|---|
| `test_phosphate_conversion_constants_have_expected_values` | Exact unit constants. |
| `test_configuration_units_convert_to_physical_kernel_units` | µM/day/µmol conversions at the model boundaries. |
| `test_schnepf_roose_linear_buffer_identities` | `239.3` capacity, `797.67` retardation, and `0.001254` dissolved fraction. |
| `test_michaelis_menten_reference_at_one_micromolar` | Nominal flux `4.794e-7 µmol cm^-2 s^-1`. |
| `test_one_centimetre_cylinder_lateral_area_excludes_end_caps` | Root and hyphal reference areas. |
| `test_zero_concentration_and_zero_length_have_zero_flux_and_area` | Physical zero limits. |
| `test_array_helpers_are_jittable` | JAX compatibility and finite non-negative array results. |
| `test_invalid_linear_buffer_parameters_fail_fast` | Zero, negative, NaN, and infinite buffer inputs. |
| `test_invalid_michaelis_menten_parameters_fail_fast` | Negative/non-finite `J_max` and non-positive/non-finite `K_m`. |

### `test_axisymmetric_geometry.py`

| Test | Contract protected |
|---|---|
| `test_axisymmetric_cylindrical_cell_volumes` | Analytical annular volumes and `(n_r,n_z)` shape. |
| `test_axisymmetric_edges_follow_intervals_and_end_at_maxima` | Requested spacing, exact outer boundaries, and shortened final-cell behaviour. |
| `test_axisymmetric_face_areas_have_expected_geometry_and_axis_face_is_zero` | Radial/vertical face formulas and symmetry at `r=0`. |
| `test_topsoil_fractions_volume_average_a_crossed_layer` | Fractional rather than rounded topsoil occupancy. |
| `test_invalid_axisymmetric_grid_parameters_fail_fast` | Allocation-time geometry validation. |
| `test_axisymmetric_uniform_p_conc_with_partial_topsoil_layer` | Temporary reset concentration averaging. |
| `test_hemisphere_cell_fractions_use_annular_volume_fraction` | Partial fungal occupancy. |
| `test_hemisphere_density_scales_fraction_by_saturation_density` | Fungal density scaling. |
| `test_colony_radius_from_length_axisymmetric_inverts_hemisphere_volume` | Analytical radius inversion. |
| `test_axisymmetric_stacked_disc_root_density_conserves_total_length` | Root-length conservation. |

### `test_labile_phosphate_state.py`

| Fixture/test | Contract protected |
|---|---|
| `species` | Distinct initial pools expose unintended biological reset changes. |
| `small_config` | Cheap grid with a deliberately crossed topsoil layer. |
| `test_buffered_amount_concentration_round_trip` | Reversibility and non-negative amount. |
| `test_default_domain_initial_inventory_matches_configured_extents` | Analytical inventory implied by the default cylinder and zero subsoil. |
| `test_partial_topsoil_amount_uses_fractional_cell_volume` | Correct amount in a partially occupied cell. |
| `test_buffered_transformations_are_jittable` | JAX compatibility of both directions. |
| `test_default_config_describes_reference_domain_and_initial_condition` | Physical defaults and provisional buffering values. |
| `test_reset_stores_only_axisymmetric_labile_amount_and_zero_diagnostics` | Canonical state name, 2D shapes, static geometry, and diagnostic initialisation. |
| `test_reset_recovers_configured_solution_field_and_preserves_biological_pools` | Correct reset composition with unchanged organism resources. |
| `test_environment_rejects_invalid_soil_configuration` | Fail-fast environment construction, including invalid P5 reference time and transition exponent. |
| `test_soil_step_updates_canonical_amount_without_storing_concentration` | Active diffusion/uptake retains canonical non-negative amount without a duplicate concentration state. |

### `test_phosphate_diffusion.py`

| Test group | Contract protected |
|---|---|
| defaults, `D_app`, and validation | Schnepf–Roose parameters, buffered reference value, and fail-fast physical ranges. |
| conductance and CFL references | Actual shortened-cell centre distances and independent two-annulus stability arithmetic. |
| radial/vertical hand calculations | Signed equal-and-opposite finite-volume transfers in both axes. |
| uniform, single-cell, and safety-limit cases | Uniform-field invariance, closed boundaries/axis, conservation, and non-negativity. |
| cached schedule and one/multi-step integration | Exact substep count, equal duration, explicit repetition, diffusion-before-uptake ordering, accumulated uptake, and JIT. |

### `test_continuous_phosphate_uptake.py`

| Test group | Contract protected |
|---|---|
| kinetic defaults and reference request | Agreed `J_max`/`K_m` and independent flux × area × time arithmetic. |
| zero, monotonicity, and JIT tests | Physical limits and transformed-array compatibility. |
| uncapped and oversubscribed competition | Full acceptance below supply and proportional scaling above it. |
| zero-demand/inventory and symmetry tests | NaN-free empty limits, non-negativity, equal treatment, and cellwise conservation. |

### `test_p3_environment_uptake.py`

| Test group | Contract protected |
|---|---|
| plant-only and fungus-only uptake | Correct consumer-specific sparse/blended request and mg-P pool credit. |
| mixed reconstruction, oversubscription, and full-step balance | Same fungal-derived weight for both consumers, one post-blend cap, non-negative soil, and soil-to-free-pool conservation. |
| growth sequencing and JIT | Uptake cannot fund same-step growth and the full soil kernel compiles. |
| mortality accounting | Structural P removed with biomass is accumulated as loss. |

### `test_sparse_phosphate_uptake.py`

| Test group | Contract protected |
|---|---|
| territory and effective-radius references | Cylindrical area partition, territory bound, and finite diffusion-propagation bound. |
| nominal overlap and blend limits | Approximately 5.5-day saturated-hypha result; absent/dense/zero-diffusion behaviour; exact bounded weights. |
| surface-concentration parameterisation | Stable analytical residual, physical bounds, no-resistance/high-resistance limits, and cancellation resistance at large `C_b`. |
| sparse request arithmetic | Independent `surface flux × cylindrical area × time`, zero absorber/concentration limits, and no-transport result. |
| blending and JIT | Alternative requests are interpolated rather than summed, retain both limits, and compile under JAX. |

### `test_phosphate_qualification.py`

| Test group | Contract protected |
|---|---|
| transaction and diagnostic evolution equivalence | Qualification requests, cap, accepted amounts, post-diffusion state, and pool credits exactly match production evolution. |
| diagnostic limits and conservation | Surface ratios remain bounded, capped demand is identified, and accepted consumer amounts equal soil loss. |
| comparison and annual projection | The near-zero reporting floor, reference-relative error, annual step ceiling, and runtime conversion are explicit. |
| executable scenario runner | Repeated fixed-geometry studies are deterministic, close P balance to `1e-5`, and preserve monotonic uptake over the concentration range. |

### `test_phosphate_examples.py`

| Test group | Contract protected |
|---|---|
| all three public modes | Plant-only, fungus-only, and mixed examples use production evolution, retain non-negative soil, and close the soil-to-pool transaction balance. |
| absent-consumer checks | A disabled consumer receives exactly zero uptake. |
| mixed and invalid-mode checks | Mixed uptake credits both pools reproducibly, exposes a bounded regime weight, and rejects unsupported scenario names clearly. |

### `test_review_repairs.py`

| Test group | Contract protected |
|---|---|
| public subprocess entry points | The documented example wrapper and `main.py` execute rather than only exposing an importable helper. |
| independent-consumer trajectories | An absent partner remains dormant through multiple complete steps and cannot trade, grow, form geometry, take up P, or receive reward. |
| absorbing death and observation shape | Once dead, an organism cannot resume allocation, photosynthesis, geometry, uptake, trade, or reward; observations match the declared flat `(4,)` space. |
| PPO smoke | One complete two-policy PPO update uses canonical `plant`/`fungus` identifiers and the flat observation contract. |
| state-forming trait validation | Negative and non-finite pools/rates, invalid death fractions, and invalid plant capacity/photosynthesis traits fail before reset. |

## PPO integration

`PPOConfig` is the typed training configuration. `make_train` validates actor,
minibatch, and update counts and uses `env.agents` throughout. The compact
`scripts/train_ppo.py` launcher constructs an explicit development domain and
saves both policy parameter trees. The obsolete evaluation plotting package,
which depended on removed `AgentState` and `State.agents` representations, has
been removed rather than maintained.

## P6 qualification runner and artifacts

`scripts/phosphate_qualification.py` owns the reproducible offline scenario
matrix, coupled trajectory, selection arithmetic, JIT timing, core-array memory
estimate, and JSON/Markdown rendering. It calls production environment and soil
functions rather than maintaining separate model equations. The canonical
outputs are `implementation-docs/qualification/p6-results.json` and
`p6-results.md`; measured timings are platform-specific and include device
metadata.

### `test_growth_geometry.py`

| Fixture/test | Contract protected |
|---|---|
| `test_provisional_growth_geometry_trait_defaults` | Agreed plant/fungal trait values and units. |
| `test_one_gram_biomass_has_expected_root_and_hyphal_length` | Independent per-gram reference calculations. |
| `test_length_conversions_have_physical_zero_and_nonnegative_limits` | Zero and defensive negative-biomass limits. |
| `test_length_conversions_are_jittable` | JAX compatibility of both conversion chains. |
| `test_root_density_conserves_length_on_shortened_boundary_cells` | Root integral conservation on the interval/maxima grid. |
| `test_root_disc_radii_decrease_with_beta_weighted_depth` | Analytical layer radii and slower expansion with depth. |
| `test_root_density_is_uniform_inside_depth_specific_discs` | Common `lambda_root` in fully occupied cells and volume-averaged fronts. |
| `test_root_geometry_clips_each_layer_at_radial_domain_boundary` | Analytical disc–domain intersection after radial clipping. |
| `test_sphere_cylinder_intersection_helper_returns_physical_volume` | Physical cm³, including the cylindrical factor `pi`, in the intersection helper. |
| `test_fungal_density_conserves_length_with_partial_front_cells` | Hyphal integral conservation before boundary contact. |
| `test_fungal_extent_is_monotonic_with_biomass` | Monotonic hemispherical occupancy. |
| `test_fungal_density_clips_at_saturated_domain_capacity` | Explicit saturated capacity after boundary contact. |
| `test_zero_biomass_produces_zero_spatial_density` | Spatial zero limits. |
| `_geometry_species`, `_geometry_config` | Small, physically controlled environment fixtures. |
| `test_reset_initialises_geometry_from_initial_structural_biomass` | Initial structure is present before the first soil step. |
| `test_realised_growth_updates_geometry_before_soil_stage` | Immediate growth credit and step ordering. |
| `test_maintenance_biomass_loss_contracts_geometry` | Density follows surviving rather than historical biomass. |
| `test_environment_rejects_invalid_growth_geometry_traits` | Fail-fast trait validation before JAX kernels. |

### `test_geometry_growth_video.py`

| Test | Contract protected |
|---|---|
| `test_growth_radii_advance_by_radial_grid_interval` | One frame target per radial edge, including a shortened final cell. |
| `test_growth_radii_reject_more_than_one_thousand_frames` | Hard 1,000-frame maximum. |
| `test_fungal_biomass_inverts_the_p2_colony_radius_pipeline` | Fungal biomass targets the displayed hemisphere radius through production functions. |
| `test_root_biomass_inverts_maximum_depth_specific_disc_radius` | Plant biomass targets the largest root-disc radius while retaining beta-dependent radii. |

## Extension rules for later phases

- Mutate `soil_labile_p`; never store a second authoritative concentration.
- Derive concentration after every diffusion update and again from the updated
  amount at the next substep; never store it independently.
- Use cell volumes for amount and length-density integration.
- Recompute absorbing geometry from current structural biomass after growth and
  biomass loss; do not construct length directly from gross resource allocation.
- Use face areas only for conservative neighbour fluxes.
- Derive neighbour distances from cell centres because the final radial or
  depth cell may be shorter than the configured interval.
- Convert accepted uptake to mg exactly once when crediting organism pools.
- Keep `evolve_soil_p` amount-conservative; do not reintroduce the removed
  concentration-as-inventory helpers.
- Add a test docstring and update this reference whenever a foundation helper
  or invariant is added or materially changed.
