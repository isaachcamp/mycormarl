# MycorMARL

MycorMARL is a JAX/JaxMARL model of plant–mycorrhizal resource exchange. Its
soil phosphate component stores finite labile P amounts on an axisymmetric
grid and couples buffered diffusion with root/fungal uptake and competition.

## Start here

- [Model overview](docs/model-overview.md):
  shared state, axisymmetric geometry, and growth–phosphate coupling.
- [Growth model](docs/growth-model.md):
  resource allocation, biomass change, and root/fungal geometry.
- [Phosphate model](docs/phosphate-model.md):
  buffering, diffusion, uptake regimes, competition, and verification.
- [Module map](docs/module-map.md):
  code ownership, pipeline boundaries, executables, and test locations.
- [Numerical qualification](docs/qualification/phosphate-numerical-qualification.md):
  convergence, balance, sensitivity, and performance evidence.
- [Open questions](docs/open-questions.md):
  unresolved scientific, calibration, and modelling decisions.

Install and run the deterministic examples:

```bash
uv sync
uv run python scripts/phosphate_examples.py --mode all
```

Run the tests:

```bash
uv run pytest -q
```

Run a small PPO development job and save both policy parameter trees:

```bash
uv run python scripts/train_ppo.py --total-timesteps 256 --num-envs 1
```

`main.py` is a convenience alias for the example runner. Scientific runs
should construct explicit `EnvConfig`, `PlantTraits`, `FungusTraits`, and
`SpeciesParams` values so the parameterisation is recorded with the output.
`EnvConfig.consumer_mode` accepts `mixed`, `plant-only`, or `fungus-only`.
Independent modes retain the two-policy JaxMARL API while keeping the absent
partner permanently dormant.
