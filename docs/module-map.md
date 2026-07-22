# Phosphate model module map

This is a compact guide to where the implemented model lives. For the
scientific assumptions, equations, units, and parameter provenance, read the
[phosphate model](phosphate-model.md). Source docstrings and tests remain the
authoritative API-level detail.

## End-to-end flow

```text
configuration and organism traits
  -> axisymmetric grid and initial buffered labile-P amount
  -> biological allocation and realised biomass change
  -> biomass-derived root and hyphal length-density fields
  -> conservative diffusion of labile P
  -> sparse and continuous uptake requests
  -> root–fungus competition for each cell's finite inventory
  -> soil amount reduction and organism P-pool credit
  -> observations, rewards, diagnostics, and qualification outputs
```

## Runtime modules

| Module | Responsibility | Main consumers |
|---|---|---|
| `params.py` | Environment and species configuration containers. | Environment construction and experiment setup. |
| `state.py` | Canonical dynamic state, including cellwise labile P and organism pools. | Environment, soil evolution, and policies. |
| `environments/base_mycor.py` | Owns step ordering, trade, growth, mortality, geometry refresh, soil evolution, observations, and termination. | JaxMARL training and complete simulations. |
| `soil/phosphate_units.py` | Unit conversion, buffering identities, Michaelis–Menten flux, and scalar validation. | Grid initialisation and uptake kernels. |
| `soil/phosphate_grid.py` | Axisymmetric edges, cell volumes, face areas, topsoil fractions, and concentration–amount transforms. | Environment construction and soil state conversion. |
| `soil/phosphate_diffusion.py` | Conservative face conductances, explicit diffusion, CFL ceiling, and substep scheduling. | Integrated soil evolution. |
| `soil/phosphate_uptake.py` | Continuous and analytical sparse requests, depletion-overlap blending, and proportional competition. | Integrated soil evolution and qualification diagnostics. |
| `soil/soil.py` | Composes diffusion and uptake into one substep and repeats it on the cached numerical schedule. | `BaseMycorMarl.step_phosphorus_field`. |
| `plant/roots.py` | Converts plant biomass to depth-weighted stacked-disc root density. | Reset and post-growth geometry refresh. |
| `fungus/mycelium.py` | Converts fungal biomass to a volume-averaged saturated hemispherical hyphal field. | Reset and post-growth geometry refresh. |
| `plant/traits.py`, `fungus/traits.py` | Organism stoichiometry, kinetics, geometry, maintenance, and initial pools. | Biological and soil calculations. |
| `growth.py`, `maintenance.py` | Shared realised-growth and maintenance calculations. | Plant and fungal environment steps. |
| `observations.py` | Optional numerical normalisation for policy observations. | Environment observation construction. |
| `algos/ppo.py` | Two-policy PPO integration using the `plant` and `fungus` interfaces. | Training launcher and experiments. |

## Executable and evidence modules

| Path | Purpose |
|---|---|
| `scripts/phosphate_examples.py` | Runs small deterministic plant-only, fungus-only, or mixed uptake examples. |
| `scripts/phosphate_qualification.py` | Regenerates the convergence, conservation, sensitivity, and performance evidence. |
| `scripts/geometry_growth_video.py` | Visualises the production root and fungal geometry mappings. |
| `scripts/train_ppo.py` | Runs a small explicit PPO development job. |
| `tests/test_phosphate_*.py` | Protects units, state, diffusion, uptake, competition, and qualification contracts. |
| `tests/test_environment_phosphate_uptake.py` | Protects soil-to-organism pool credit, competition, loss accounting, and environment integration. |
| `tests/test_growth_geometry.py` | Protects biomass-to-length and spatial conservation rules. |
| `tests/test_base_mycor_refactor.py` | Protects environment ordering, death, trade, growth, and observations. |
| `tests/test_review_repairs.py` | Protects public entry points, independent modes, PPO integration, and repaired regressions. |

## Key ownership boundaries

- `State.soil_labile_p` is the only mutable soil-P inventory. Solution
  concentration is always derived from it.
- Geometry modules decide where absorbing length exists; uptake modules decide
  the request per unit geometry; the competition transaction decides what the
  finite soil inventory can supply.
- The environment owns biological ordering. The soil package does not decide
  allocation, growth, reward, death, or trade.
- Production evolution and numerical qualification share the same uptake
  transaction, preventing the diagnostic calculations from becoming a second
  model.
