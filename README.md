# MycorMARL

MycorMARL is a JAX/JaxMARL model of plant–mycorrhizal resource exchange. Its
soil phosphate component stores finite labile P amounts on an axisymmetric
grid and couples buffered diffusion with root/fungal uptake and competition.

## Start here

- [Phosphate model guide](implementation-docs/phosphate-model-guide.md):
  equations, units, configuration, provenance, diagnostics, and limitations.
- [Function and test reference](implementation-docs/phosphate-foundations-reference.md):
  module-level pipeline and protected contracts.
- [P6 qualification](implementation-docs/qualification/p6-results.md):
  convergence, balance, sensitivity, and performance evidence.

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
