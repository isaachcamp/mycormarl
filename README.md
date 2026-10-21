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
- [Independent PPO walking skeleton](docs/implementation-docs/ippo-walking-skeleton.md):
  actor observations, Physical actions, lifecycle semantics, training, and
  saved-policy compatibility.
- [Model Parameter Register](docs/model-parameter-register.md): register
  for all model parameters, estimated values and related literature.
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

Run a small reproducible PPO development job and save a versioned bundle with
both policy parameter trees:

```bash
uv run python scripts/train_ppo.py --finite-horizon-returns \
  --mode mixed --total-timesteps 256 --num-steps 128 --num-envs 1 --seed 42
```

The trainer also accepts `--mode plant-only` and `--mode fungus-only` while
retaining the fixed two-agent API. Saved bundles must be loaded with
`mycormarl.policy_artifacts.load_policy_artifact`; unversioned policies and
artifacts from the previous actor or environment-state schema are rejected.

`main.py` is a convenience alias for the example runner. Scientific runs
should construct explicit `EnvConfig`, `PlantTraits`, `FungusTraits`, and
`SpeciesParams` values so the parameterisation is recorded with the output.
`EnvConfig.consumer_mode` accepts `mixed`, `plant-only`, or `fungus-only`.
Independent modes retain the two-policy JaxMARL API while keeping the absent
partner permanently dormant.

The PPO command is an interface and numerical smoke check. It does not by
itself establish learning convergence, cooperation, robustness, or biological
optimality.

## Research and animations

[REDbyRBD](https://github.com/isaachcamp/REDbyRBD) owns studies, uncertainty screens,
foraging diagnostics, figures and the reproducible report.
[mycormarl-animations](https://github.com/isaachcamp/mycormarl-animations) owns
all animation scenes and builders. Both install an immutable core Git revision.
See [source version conventions](docs/reproducibility/version-control.md).

Historical local outputs and plots were relocated to the sibling data archive
`../mycormarl-data/exploration/mycormarl-checkout-2026-10-21/`.
Its `relocation.json` records original paths and file-preservation checks.
Historical analysis reports are archived in REDbyRBD under
`docs/archive/core-analysis-2026-08/`.
