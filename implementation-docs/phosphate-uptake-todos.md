# Phosphate uptake modelling TODOs

> Implementation status (2026-07-21): this file is the retained scientific
> requirements record. P0–P6 implement the transport, uptake, competition,
> geometry, and qualification items below. Remaining research decisions are
> explicitly labelled unresolved or deferred. See
> `implementation-docs/phosphate-model-guide.md` for the authoritative runtime
> description.

The original uptake calculation prevented negative cell values by
proportionally scaling plant and fungal requests, but was not dimensionally
consistent. The requirements below record how it was replaced.

Required changes:

- Store total labile phosphate amount per cell as the canonical conserved
  state. Expose soil-solution concentration as the quantity used to configure
  the initial field and to evaluate diffusion and Michaelis-Menten uptake.
- Use `1 µM` soil-solution inorganic phosphate as the nominal initial
  condition. Treat this as a representative order-of-magnitude reference, not
  a universal empirical mean.
- Initialise this concentration only in the upper `25 cm` of soil and
  initialise cells below that depth with zero labile phosphate. If the
  `25 cm` boundary cuts through a cell, volume-average the intersected cell.
- For the provisional axisymmetric domain with maximum radius `50 cm` and
  maximum depth `100 cm`, the upper `25 cm` contains approximately
  `196,349.54 cm^3` of soil. With `theta = 0.3`, `B = 239`, and
  `C_0 = 1 µM = 0.001 µmol cm^-3`, the initial labile inventory is
  approximately `46,986.45 µmol = 1,455.35 mg P`.
- Include logarithmically spaced initial-condition scenarios of
  `0.1, 0.3, 1, 3, and 10 µM`. Regard approximately `0.1–2 µM` as the core
  range for low-P, unfertilised soil and `3–10 µM` as enriched or upper-end
  conditions until a particular experimental soil is selected.
- Use instantaneous-equilibrium linear buffering for the first
  solution-to-total-labile conversion:
  `M_labile = V_cell * (theta + B) * C`, where `C` is soil-solution Pi
  concentration, `theta` is volumetric water content, and `B` is the
  volumetric P buffer capacity. Use `theta` and `B` as the active model
  parameters. Define `B` explicitly as
  `dC_sorbed,bulk / dC_solution`, with effective units of solution-water volume
  per bulk-soil volume. Document `B = rho_b * K_d` as a calibration
  relationship rather than requiring `rho_b` and `K_d` at runtime. Later
  calibrate the buffering relationship and initial inventory against the soil
  and P measurement used in the target experiment.
- Use the Schnepf and Roose (2006) parameter set as the provisional default:
  `theta = 0.3 cm^3 cm^-3` and `B = 239 cm^3 cm^-3` (both attributed there to
  Barber, 1995). This gives `theta + B = 239.3` and a retardation factor
  `R = (theta + B) / theta = 797.67`; only about `0.125%` of the equilibrium
  labile inventory is in solution. Include this identity as a parameter/unit
  test because the buffering is intentionally very strong.
- Convert root and hyphal length density to length in each cell using the cell
  volume. Define both fields as centimetres of absorbing structure per cubic
  centimetre of bulk soil (`cm cm^-3`). Initially treat all represented length
  as uptake-active (`active_fraction = 1`).
- Convert that length to absorbing surface area:
  `A = 2 * pi * radius * length`.
- Use the Schnepf and Roose (2006) hyphal radius
  `r_h = 5e-4 cm = 5 µm` as the provisional fungal absorbing radius. Do not
  infer a root radius from that paper. Retain the code's existing
  `r_root = 0.01 cm = 100 µm` as the provisional root absorbing radius,
  explicitly marking it as an unsourced value for later calibration.
- Use the same Michaelis-Menten kinetic parameters for plant roots and fungal
  hyphae in the first model, taking the Tinker and Nye (2000) values reported
  by Schnepf and Roose (2006):
  `J_max = F_max = 3.26e-6 µmol cm^-2 s^-1` and
  `K_m = 5.8e-3 µmol cm^-3 = 5.8 µM`. Keep distinct plant and fungal parameter
  fields in the interface even though their provisional values are identical,
  so they can later be calibrated independently.
- Calculate the unconstrained amount requested during a step as
  `U_request = J(C) * A * dt`, with
  `J(C) = J_max * C / (K_m + C)` and consistent concentration units.
- Add reference-value tests for the kinetic conversion. At the nominal
  `C = 1 µM`, the provisional parameters give
  `J / J_max = 1 / (5.8 + 1) = 0.1471` and
  `J approximately 4.79e-7 µmol cm^-2 s^-1`.
- Convert the phosphate field to an available amount in each cell before
  applying the competition constraint. Include volumetric water content and
  the chosen sorption/buffer model when `soil_p` denotes solution
  concentration.
- If several consumers draw from one cell, constrain the sum of their requests
  to the available amount using proportional demand scaling. Calculate both
  final blended requests simultaneously from the same pre-uptake `C_b`, then
  define
  `s = min(1, M_available / (U_root_request + U_fungus_request))` and
  `U_root = s * U_root_request`,
  `U_fungus = s * U_fungus_request`. Handle zero total demand without
  division. This is a neutral equal-fraction rule with no consumer priority.
- Define `M_available` as the canonical labile amount remaining in the cell
  after the diffusion update, not as solution concentration or solution-only
  P. This is consistent with the initial assumption of instantaneous
  equilibrium buffering. If kinetic desorption is introduced later, the
  supply cap will need revision.
- Apply the competition cap once, after sparse/continuous blending. Do not cap
  the sparse and continuous components separately. Update the cell by
  `M_labile_new = M_available - U_root - U_fungus` and require
  `M_available - M_labile_new = U_root + U_fungus` within numerical tolerance.
- Aggregate accepted spatial amounts into the correct organism pools:
  `delta_P_plant = sum_i(U_root,i)` and
  `delta_P_fungus = sum_i(U_fungus,i)` for the current single plant and fungus.
  Do not add grid-shaped sink arrays directly to agent-pool arrays. A later
  multi-agent version will require agent-specific spatial ownership fields.
- Keep soil transport, requests, and the canonical soil inventory in
  `µmol P`, matching the uptake-flux units. At the soil-organism boundary,
  convert each accepted uptake amount to organism-pool units using
  `1 µmol P = 0.0309738 mg P`, and store plant and fungal P pools in `mg P`.
  Use fungal `gamma_P,fungus = 40 mg P g^-1 dry biomass`, corresponding to
  the `4%` maximum P mass fraction identified by Bisot et al. in their
  literature search. This is an upper-bound parameterisation, not yet a
  validated representative AM-fungal tissue concentration; independently
  review the underlying studies before calibration. Keep
  `gamma_P,plant` separate, with the provisional *Daucus carota* value defined
  below.
  Conservation tests must convert all terms to one common unit.
- Initially do not return organism P to the soil after mortality. Treat it as
  loss from the simulated system and accumulate `P_lost_to_mortality` so an
  extended whole-system mass balance can still be checked. Defer litter,
  mineralisation, and recycling pools.
- **Unresolved maintenance-P accounting:** the current biological maintenance
  logic subtracts `maint_p_used` from the organism's free P pool, but does not
  transfer that P to structural biomass, soil, or a recorded export/loss pool.
  It therefore leaves the tracked system without an explicit fate. Retain the
  existing C-and-P maintenance behaviour for now; do not silently replace it
  with carbon-only maintenance. Before whole-system P balance is claimed for
  maintenance-active steps, decide whether maintenance P represents temporary
  reservation/recycling, turnover loss, or another explicit destination, then
  implement and test that choice.
- Record as an accepted first-model approximation that proportional inventory
  scaling prevents mass overdraw but does not fully resolve sub-grid
  root-hypha diffusive interference when two nominally sparse depletion zones
  overlap and total demand remains below the cell inventory. Shared `C_b`
  updates and timestep-convergence tests will limit and diagnose this error.
- Convert the amount remaining after uptake back to the stored field variable.
- Implement conservative diffusion between cells using the axisymmetric cell
  geometry, and test conservation, non-negativity, grid-size dependence, and
  limiting cases.
- Compute diffusive transfer from the soil-solution concentration gradient:
  `q = -D_l * theta * f_l * grad(C)`. Transfer the resulting phosphate amount
  between neighbouring canonical `M_labile` cell states, then re-establish
  local linear equilibrium with `C = M_labile / (V_cell * (theta + B))`.
  Do not also put the buffered apparent diffusivity into this amount-flux
  calculation, because that would count retardation twice.
- Keep the first transport model diffusion-only. Hold `theta` fixed and omit
  water-driven advection/Darcy flux, hydrodynamic dispersion, and water-content
  dynamics. These processes may be added later without changing the canonical
  labile-amount state.
- Apply closed external boundaries to the axisymmetric soil domain. Use
  zero normal phosphate flux at the soil surface, bottom, and outer radial
  boundary. At `r = 0`, impose the cylindrical symmetry condition
  `dC/dr = 0` (the central face also has zero area). These conditions apply to
  the simulated soil boundary, not to root or hyphal surfaces: roots and
  hyphae remain internal uptake sinks. If a non-axisymmetric angular dimension
  is retained anywhere, its azimuthal boundary must be periodic rather than
  treated as a physical no-flux wall.
- For constant properties, verify equivalence with the concentration equation
  `(theta + B) * dC/dt = div(D_l * theta * f_l * grad(C)) - S` and apparent
  diffusivity `D_app = D_l * theta * f_l / (theta + B) = D_l * f_l / R`.
  With the provisional Schnepf and Roose values `D_l = 1e-5 cm^2 s^-1`,
  `f_l = 0.308`, `theta = 0.3`, and `B = 239`, the expected
  `D_app` is approximately `3.86e-9 cm^2 s^-1`.
- Add unit annotations and tests covering concentration, volume, surface area,
  flux, timestep, and phosphate amount.
- Test the surface-area conversion using `r_h = 5e-4 cm`, including that one
  centimetre of hypha has lateral absorbing area
  `2 * pi * r_h * 1 cm approximately 3.142e-3 cm^2`. Exclude end caps.
- Also test that one centimetre of root with `r_root = 0.01 cm` has lateral
  absorbing area `2 * pi * r_root * 1 cm approximately 6.283e-2 cm^2`.
- Replace the current dimensionally incomplete fungal biomass-to-length
  conversion with the tissue-volume method used by Bisot et al. (2026).
  Rename their `M_C` concept descriptively as fungal or hyphal tissue carbon
  density: it is carbon mass per unit volume of living fungal tissue, not per
  soil grid-cell volume. For new structural carbon,
  `delta_V_hypha = delta_M_C,structural / M_C` and, under the provisional
  constant-radius cylindrical approximation,
  `delta_L_hypha = delta_V_hypha / (pi * r_h^2)`.
- Define `M_C = d_cell * f_dry * f_carbon`, following Bisot et al., and keep
  all carbon-mass and tissue-volume units explicit. Use the provisional value
  `M_C = 0.1155 g C cm^-3` from Bisot et al. (2026); retain it as a
  configurable parameter because they note that the precise AM-fungal value
  is not known.
- Do not identify fungal `M_C` with fungal `gamma_C`. With realised biomass
  growth measured in grams of dry biomass, define
  `gamma_C,fungus` in `g C g^-1 dry biomass`; it converts dry-biomass growth
  into incorporated carbon. `M_C`, in `g C cm^-3 fungal tissue`, then converts
  that carbon into fungal tissue volume. The full dimensional chain is
  `delta_G_fungus -> gamma_C,fungus * delta_G_fungus
  -> delta_M_C,structural / M_C -> delta_V_hypha
  -> delta_V_hypha / (pi * r_h^2) -> delta_L_hypha`.
- Maintain four distinct stoichiometric growth parameters:
  `gamma_C,plant`, `gamma_P,plant`, `gamma_C,fungus`, and
  `gamma_P,fungus`. Their units are respectively
  `g C g^-1 dry biomass` and `mg P g^-1 dry biomass`.
- Use `gamma_C,fungus = 0.5 g C g^-1 fungal dry biomass`, matching the
  `f_carbon = 0.5` assumption in Bisot et al. (2026). Together with
  `M_C = 0.1155 g C cm^-3`, this preserves the distinction between fungal
  carbon fraction and carbon density per tissue volume.
- Use *Daucus carota* as the provisional plant parameterisation:
  `gamma_C,plant = 0.402 g C g^-1 plant dry biomass`, based on elemental
  analysis of untreated dry carrot-root material reported by Kaur et al.
  (2022) in *Scientific Reports* (Nature Portfolio), and
  `gamma_P,plant = 1.92 mg P g^-1 plant dry biomass`. The latter is the
  dry-mass-weighted whole-plant value calculated from the unfertilised-soil
  treatment of Kováčik et al. (2022): root and leaf P concentrations were
  `1.975` and `1.688 mg g^-1`, respectively. Kováčik et al. was published in
  *Agronomy*, an MDPI journal; retain that publisher information with the
  provenance and seek corroboration outside MDPI before calibration. Treat
  the carbon value as root-dominated rather than a direct whole-plant
  measurement, and keep both values configurable for later calibration.
- Use the realised, stoichiometrically constrained fungal biomass increment
  returned by `_grow_biomass_essential_resources` as the authoritative driver
  of new hyphal structure:
  `delta_G = min(C_alloc / gamma_C, P_alloc / gamma_P)`, where `gamma_C` and
  `gamma_P` are the carbon and phosphorus requirements per unit biomass.
  Derive incorporated structural carbon and resource consumption from that
  realised growth:
  `delta_M_C,structural = gamma_C * delta_G` and
  `delta_M_P,used = gamma_P * delta_G`. Then use
  `delta_V_hypha = gamma_C * delta_G / M_C` and
  `delta_L_hypha = gamma_C * delta_G / (M_C * pi * r_h^2)`.
  Do not use gross carbon allocation directly to construct length; any carbon
  allocated but not required because P limits growth remains outside the new
  structural volume.
- Increase root length from realised plant dry-biomass growth using empirical
  specific root length:
  `delta_M_root = k_root * delta_G_plant` and
  `delta_L_root = SRL * delta_M_root`, where `k_root` is the dry-biomass
  fraction allocated to roots and `SRL` is in `cm root g^-1 root dry mass`.
  Do not also derive root length from `pi * r_root^2 * L`; that would impose a
  second, potentially inconsistent root mass-volume relationship.
- Distribute this root length through stacked depth discs with one uniform
  provisional within-disc density `lambda_root = 1.0 cm root cm^-3 soil`.
  For layer `k`, calculate `L_k = L_root * w_k` from differences of
  `F(d) = 1 - beta^d`, then
  `R_k = sqrt(L_k / (pi * lambda_root * dz_k))`. Deeper layers therefore
  expand radially more slowly rather than sharing one broadcast radius. Use
  annular overlap fractions for front cells. If `R_k` exceeds the soil radius,
  clip the represented disc-domain intersection without redistributing its
  excess length to another depth.
- Use the *Daucus carota* species medians in the GRooT database as provisional
  absorbing-root traits: `k_root = 0.6198` (rounded runtime default `0.62`)
  from median root mass fraction and
  `SRL = 254.343 m g^-1 = 25,434.3 cm g^-1`. The aggregate values are based
  on 8 root-mass-fraction and 10 SRL study-site entries. Root mass fraction is
  a standing-biomass ratio rather than a directly measured marginal
  allocation fraction, so using it as `k_root` assumes approximately steady
  biomass partitioning. GRooT's SRL describes fine roots and is preferable to
  calculating SRL from the carrot storage taproot.
- Record source provenance:
  Kaur et al. (2022), *Scientific Reports*,
  `https://doi.org/10.1038/s41598-022-20971-5`;
  Kováčik et al. (2022), *Agronomy* (MDPI),
  `https://doi.org/10.3390/agronomy12112770`; and
  Guerrero-Ramírez et al. (2021), GRooT database,
  `https://doi.org/10.1111/geb.13179`, with the public species aggregation at
  `https://github.com/GRooT-Database/GRooT-Data`.
- For the first external-mycelium model, assign new fungal structural volume
  to cylindrical hyphae and omit spores and intraradical fungal structures.
  Document this simplification; Bisot et al.'s whole-network carbon accounting
  also includes spore volume.
- When distributing new length spatially, enforce
  `sum_i(delta_lambda_h,i * V_cell,i) = delta_L_hypha` so the length-density
  field conserves the length implied by fungal structural carbon.
- Retain the provisional spatial-growth closure of a hemispherical external
  mycelium at fixed saturation length density. Determine colony radius from
  cumulative external hyphal length using
  `R_colony = (3 * L_total / (2 * pi * lambda_sat))^(1/3)`, and use
  volume-weighted partial occupancy in cells crossed by the colony front.
- Convert the approximately `1,500 µm mm^-2` planar saturation density reported
  by Oyarte-Galvez et al. (2025) using the stipulated closure
  `rho_3D = 3 * rho_2D^2 / 4`. Since `rho_2D = 1.5 mm^-1`, use
  `lambda_sat = rho_3D = 1.6875 mm^-2 = 168.75 cm^-2`
  (`168.75 cm hypha cm^-3 soil`). Treat this as a provisional conversion from
  a two-dimensional experimental network, not a direct three-dimensional soil
  measurement.
- With `lambda_sat = 168.75 cm^-2`, `r_h = 5e-4 cm`, and
  `D_app approximately 3.86e-9 cm^2 s^-1`, the static overlap calculation gives
  `R_soil approximately 0.04343 cm` and `t_diff approximately 5.5 days`.
  Thus, with the current provisional `T_ref = 1 day` and `p = 2`,
  `Omega approximately 0.18` and `w_cont approximately 0.032`; include this
  consequence explicitly in transition-parameter sensitivity tests.
- Do not track local hyphal colonisation age in the first implementation.
  Accept that a newly occupied growth-front cell may be classified as though
  its local density were already equilibrated. This approximation should be
  limited because the established hemispherical interior remains at nearly
  constant saturation density.
- Document an age-aware transition as a later extension. It would track time
  since first hyphal arrival in each cell and compare that age with the local
  depletion-zone overlap time; it must be revisited if turnover or substantial
  density decline is added.
- Under the closed soil-boundary configuration, test that diffusion alone
  conserves domain-total labile P exactly within numerical tolerance; only
  biological uptake may reduce the soil total.
- Test uptake across the full initial-concentration scenario range because the
  Michaelis-Menten response makes concentration sensitivity nonlinear and
  dependent on `K_m`.
- Run timestep sensitivity for representative root-only, fungus-only, and
  mixed cells, including a sparse mixed cell. Confirm that reducing `dt`
  converges total uptake, root/fungus shares, and the evolving `C_b`; this
  checks the within-step lag created by simultaneous requests from the same
  pre-uptake concentration.

The existing aggregate hyphal length-density field represents competition
among hyphae only implicitly. Whether that homogenisation is appropriate should
be checked by comparing the simulation timestep and biological time horizon
with the density-dependent depletion-zone overlap time.

## Sparse-to-continuous uptake transition

- Treat the uptake regime as a property of the local shared depletion
  environment, not permanently of each consumer. In root-only cells, use the
  sparse cylindrical sub-grid root closure because root density alone is
  assumed insufficient to homogenise depletion. When sufficiently dense
  hyphae are present, use the same fungal-derived continuous-regime weight for
  both root and fungal uptake: roots in that cell also draw from the coupled
  depletion field.
- Use the sparse closure for both consumers when local hyphal density gives a
  long statically estimated depletion-zone overlap time, and the continuous
  closure for both when that time is short relative to the model-relevant
  reference timescale. Root density does not drive the initial regime
  calculation.
- Estimate the cylindrical soil-territory radius from local hyphal length
  density as `R_soil = 1 / sqrt(pi * lambda_h)`. Define the diffusion distance
  from hyphal surface to the territory boundary as
  `ell_gap = max(R_soil - r_h, 0)` and the overlap timescale as
  `t_diff = ell_gap^2 / D_app`. Treat `lambda_h = 0` as no fungal uptake and
  infinite `t_diff`.
- In the first implementation, do not use local age in the regime calculation.
  Compare `t_diff` with one fixed biological/reference timescale `T_ref`.
  Use `T_ref = 24 h` as a provisional configurable default, not as a universal
  biological constant, and include it in sensitivity analysis. It is a static
  proxy for absorber exposure/lifetime while the model assumes indefinitely
  active roots and hyphae with no maximum age. Form the static overlap ratio
  `Omega_static = T_ref / t_diff`. A smooth
  blend may use
  `w_cont = Omega_static^p / (1 + Omega_static^p)`, so that short `t_diff`
  favours the continuous closure and long `t_diff` favours the sparse closure.
  Use configurable `p = 2` provisionally and test sensitivity at `p = 1` and
  `p = 4`. A bare `t_diff` cannot determine the regime without a comparison
  timescale.
- In the later age-aware extension, replace `T_ref` with local time since
  colonisation, giving `Omega_age = local_hyphal_age / t_diff`.
- Blend each consumer's uptake requests, not phosphate states, using the same
  hyphal-derived local weight:
  `U_root_request = (1 - w_cont) * U_root_sparse + w_cont * U_root_continuous`
  and
  `U_fungus_request = (1 - w_cont) * U_fungus_sparse + w_cont * U_fungus_continuous`.
  When no hyphae are present, define `w_cont = 0`, so root-only uptake is
  sparse.
  Apply the shared plant-fungus inventory constraint only after forming the
  final root and fungal requests. Never add sparse and continuous closures as
  independent sinks.

## Analytical sparse uptake closure

- Avoid a per-cell radial PDE and iterative nonlinear solve. Use a
  quasi-steady cylindrical concentration profile with current cell-average
  solution concentration `C_b`, absorbing radius `r_a`, and effective outer
  radius
  `R_eff = r_a + min(sqrt(D_app * T_ref), max(R_soil - r_a, 0))`.
  For the root sparse component, derive `R_soil` from root length density; for
  the fungal sparse component, derive it from hyphal length density.
- Define the amount-flux transport coefficient
  `D_flux = D_l * theta * f_l`. Use `D_app` only to set the propagation length
  in `R_eff`; using it again in the steady amount-flux resistance would
  double-count buffering.
- Balance cylindrical diffusive supply and Michaelis-Menten surface uptake:
  `2*pi*D_flux*(C_b-C_s)/ln(R_eff/r_a) =
  2*pi*r_a*J_max*C_s/(K_m+C_s)`. With
  `k = r_a*J_max*ln(R_eff/r_a)/D_flux`, recover the physical surface
  concentration analytically:
  `C_s = (C_b - K_m - k + sqrt((C_b - K_m - k)^2 + 4*C_b*K_m)) / 2`.
  Handle `C_b = 0` and `R_eff -> r_a` explicitly and use a numerically stable
  equivalent of the positive quadratic root where needed.
- Calculate
  `U_sparse = 2*pi*r_a*L*J_max*C_s/(K_m+C_s)*dt`.
  Precompute or cache geometry-dependent quantities (`R_soil`, `R_eff`,
  logarithmic resistance, `k`, and the blend weight) whenever density or
  geometry changes. Recalculate `C_s` and uptake each phosphorus step because
  `C_b` evolves.
- Preserve `M_labile` as the stored state. After diffusion and after actual
  uptake, update `C_b = M_labile / (V_cell * (theta + B))`; `C_s` is a
  temporary diagnostic and is not stored as a depletion-profile state.
- Distinguish two scale-separation checks. `T_ref / t_diff` tests whether
  neighbouring depletion zones have had time to couple. The surface-depletion
  ratio `delta_s = 1 - C_s / C_b` tests whether the additional approximation
  `C_s approximately C_b` is accurate. For small resistance,
  `delta_s approximately k / (K_m + C_b)`. Record `C_s / C_b` in validation
  runs across the concentration and density ranges. Timestep convergence
  cannot by itself detect a biased surface-concentration closure.

## Timestep and credit-assignment order

- Prioritise a short reinforcement-learning credit path from growth allocation
  to increased absorbing surface and then increased P acquisition. At the
  start of a biological step, calculate realised C- and P-constrained growth
  using only the resource pools available before that step's new uptake.
  Immediately update biomass, root length, hyphal length, and their spatial
  density fields.
- Use the post-growth geometry for the full soil diffusion-and-uptake update:
  recover `C_b` from `M_labile`, apply conservative diffusion, recover the
  post-diffusion `C_b`, calculate simultaneous blended root and fungal
  requests, apply proportional competition, and update soil and organism
  pools.
- Credit newly acquired P only after the allocation/growth calculation, so it
  is visible to the policy at the next step and cannot recursively fund the
  same growth that created its absorbing surface.
- Interpret this as growth occurring at the beginning of the discrete
  interval. It overestimates the exposure time of newly created surface if
  physical growth is gradual within a coarse timestep. Require timestep
  convergence of growth, acquired P, colony extent, and learned-policy
  outcomes. If this bias is material, use midpoint geometry
  `lambda_effective = lambda_old + 0.5 * delta_lambda` for uptake while still
  committing the complete new geometry at the end of the step.

## Spatial and temporal resolution

- Retain the axisymmetric `r-z` representation and configure it directly with
  maximum radius, maximum depth, and radial/depth interval sizes. Use
  provisional defaults of `R_domain = 50 cm`, depth `100 cm`, and
  `dr = dz = 0.1 cm`, giving `500 x 1000 = 500,000` cells. Generate edges at
  the requested interval and end exactly at each configured maximum; if an
  extent is not divisible by its interval, shorten only the final cell. No
  exact correspondence to a specified three-dimensional soil volume is
  required.
- Use centimetres for length, seconds internally for physical transport and
  uptake rates, and days as the user-facing model time unit. Convert every
  configured biological timestep to seconds before applying `D_l`, `D_app`,
  or `J_max`. The existing `dt = 0.05 day` therefore denotes `1.2 h`.
- For explicit finite-volume diffusion, calculate the exact axisymmetric
  stability limit from cell geometry:
  `dt_CFL,i = V_i * (theta + B) /
  sum_j(D_l * theta * f_l * A_ij / d_ij)`, equivalently
  `V_i / sum_j(D_app * A_ij / d_ij)`. Require the soil diffusion step to be no
  larger than `0.8 * min_i(dt_CFL,i)`.
- Calculate neighbour distances from the generated cell centres rather than
  assuming globally uniform spacing, because a boundary cell may be shorter
  than the configured interval.
- Treat this CFL value as a hard stability ceiling, not as the target
  biological timestep. Choose the largest fixed biological `dt` that passes
  temporal convergence for uptake, growth, competition shares, colony extent,
  and policy outcomes. The uptake cap guarantees non-negativity but does not
  make a coarse timestep accurate.
- Begin accuracy studies from the current approximately
  `dr = dz = 0.1 cm` grid. Compare `0.1`, `0.05`, and `0.025 cm` resolutions,
  and retain the coarsest grid whose key outputs differ by no more than a
  provisional `5%` from the next finer grid.
- For each candidate grid, test upward and downward timestep changes starting
  from `0.05 day`, for example `0.025`, `0.05`, `0.1`, `0.2`, and `0.4 day`,
  subject to the CFL ceiling. Select the largest converged value rather than
  increasing directly to the diffusion limit. At `0.1 cm` spacing and the
  provisional `D_app`, the approximate two-dimensional diffusion-only limit
  is several days, which is too coarse to adopt without biological convergence
  evidence.
- If the chosen biological timestep exceeds the diffusion ceiling on a finer
  grid, use
  `N_sub = ceil(dt_bio / (0.8 * dt_CFL_min))` and
  `dt_soil = dt_bio / N_sub`. Hold post-growth geometry fixed during soil
  substeps, repeatedly update diffusion, `C_b`, uptake, and competition, and
  accumulate accepted uptake for credit at the end of the biological step.

## Agreed interpretation of buffer power

- Labile Pi comprises soil-solution Pi plus Pi reversibly associated with the
  solid phase. It excludes irreversibly fixed, occluded, and otherwise
  nonlabile mineral P.
- `B` is a local slope, `dC_sorbed,bulk / dC_solution`, rather than itself a
  fraction of P. Under the initial linear, zero-intercept assumption,
  `C_sorbed,bulk = B * C_solution`; the equilibrium solution fraction is
  `theta / (theta + B)`.
- Pi uptake is from the soil solution. The first model assumes instantaneous
  local equilibrium, so reversibly sorbed Pi immediately replenishes solution
  Pi after uptake or diffusive redistribution. It does not yet represent
  finite desorption kinetics, hysteresis, precipitation/dissolution, or
  concentration- and pH-dependent buffering.
- Buffering does not reduce the molecular diffusivity of an ion while it is
  dissolved. It slows the macroscopic propagation of a solution-concentration
  disturbance because diffusing Pi repeatedly partitions between liquid and
  solid phases and must change the full labile inventory.
- Fe- and Al-bearing surfaces are important phosphate sorbents. Associations
  with Ca-bearing phases may also control Pi, especially in alkaline soils,
  but precipitation or poorly reversible mineral formation must not
  automatically be counted in the reversible linear buffer pool.
