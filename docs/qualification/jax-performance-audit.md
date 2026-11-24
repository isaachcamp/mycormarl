# JAX performance audit

Audit date: 24 November 2026. Core source: `8cfb50a2e8db09ba6e817729d678ce1ad8124281`.
Scope: current independent PPO and its environment path. This is a small code
inspection and CPU smoke benchmark, not an A100 profile or a learning study.

The main training path already uses JAX effectively. The first improvement to
pursue is reducing retained training history, followed by measuring environment
batch sizes on Orchid. There is no evidence here that a broad vectorisation
rewrite is needed.

## What is already compiled and batched

- [Training launcher](../../scripts/train_ppo.py): one outer `jax.jit` call.
- [PPO](../../src/mycormarl/algos/ppo.py): `vmap` over environment resets and
  steps; batched neural-network inputs; `lax.scan` over rollout steps, GAE,
  minibatches, epochs and training updates. Plant and fungus have separate
  network calls and updates, both inside the compiled computation.
- [Allocation](../../src/mycormarl/allocation.py): a JAX `fori_loop` with a
  rate-dependent numerical substep count.
- [Soil evolution](../../src/mycormarl/soil/soil.py): array operations across
  the soil grid and a `fori_loop` over sequential diffusion/uptake substeps.
  Geometry-dependent uptake coefficients are already calculated outside that loop.
- [Policy intervals](../../src/mycormarl/environments/policy_interval.py): a
  `scan` over numerical steps. The benchmark uses the bare environment,
  matching the smoke launcher, rather than this wrapper.

Python loops over the fixed agent names and configuration validation are not
Python loops over rollout timesteps. Successive state transitions and optimiser
updates have sequential dependencies; replacing their scans with `vmap` would
change the computation.

## Findings, in priority order

1. **Retained history grows with run length.** The outer PPO scan returns all
   trajectories, diagnostics, advantages and targets for every update. The CLI
   receives that full result but saves only the final policy parameters. A
   compact compiled wrapper returning runner state and metrics reduces the
   returned payload in this experiment. For production, also consider bounded
   training chunks through the existing resumed-runner-state seam. Preserve
   full debug output for its existing consumers. Keep numerical qualification
   checks; compact output does not remove their required reductions.
2. **Batch-size tuning is the simplest vectorisation opportunity.** The CLI
   defaults to one environment; `PPOConfig` defaults to sixteen. CPU results
   confirm that batching is functional, but the A100 optimum needs measurement
   on representative soil grids. More environments also increase memory and
   samples per PPO update, so a throughput sweep is not a learning comparison.
3. **Profile numerical work before restructuring policies.** Allocation has
   action-dependent substeps, soil evolution has grid/timestep-dependent
   substeps, and auto-reset constructs reset state on every step before
   selecting it. These are inspection candidates, not measured bottlenecks.
   Combining the two policy updates is a later experiment: it must preserve
   independent parameters, masks, normalisers and random-stream semantics.

The only explicit host callback in the trainer is the numerical qualification
check after the outer update scan. It is not called once per environment step.
The inspected hot path uses JAX arrays, with no explicit per-step NumPy
conversion or `device_get` found. Compiled kernels still need GPU profiling
before claiming good accelerator utilisation.

## Benchmark method and limits

[Runner](../../scripts/benchmark_ppo.py), [raw measurements](jax-performance-audit.json).
Each case uses seed 42, a 1 cm by 1 cm domain at 0.1 cm resolution, `dt=0.025`
days, 32 rollout steps, two updates, one epoch and one minibatch per policy,
with finite-horizon returns. Species and environment settings are in the JSON.

`full` returns the existing trainer result. `compact` selects only runner state
and metrics **inside** the outer JIT; it does not modify the trainer. Both run
the same updates and numerical checks. Each executable is compiled explicitly,
warmed once, then timed three times with `block_until_ready` and an effects
barrier. This follows [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html)
on compilation and asynchronous execution. Imports and file I/O are excluded.
Full is always measured before compact; timings are short and order-sensitive.

The runner checks every returned leaf for finiteness and compares full/compact
runner state at `rtol=1e-6, atol=1e-7`. Metrics use `rtol=1e-5, atol=1e-6`:
a stricter preliminary comparison differed by about `1.01e-7` in one near-zero
metric at sixteen environments. These are float32 tolerance checks, not claims
of bitwise identity or long-run equivalence.

Output bytes sum array leaves, including repeated references such as targets
and returns. They measure logical payload, **not peak memory, unique allocated
bytes, or host-transfer volume**. The tiny grid and short runs cannot establish
production speed, memory requirements, convergence or numerical qualification
for a scientific study.

## Local measurements

Device: `TFRT_CPU_0`; JAX `0.9.0`; `macOS-26.6.2-arm64-arm-64bit`.

| Environments | Output | Compile (s) | Warm median (ms) | Env transitions/s | Logical output (bytes) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | full | 3.67 | 3.54 | 18,098 | 251,107 |
| 1 | compact | 2.72 | 3.05 | 20,982 | 226,083 |
| 4 | full | 3.82 | 6.35 | 40,297 | 330,124 |
| 4 | compact | 2.57 | 6.41 | 39,963 | 230,028 |
| 16 | full | 4.27 | 14.44 | 70,920 | 646,192 |
| 16 | compact | 3.18 | 12.05 | 84,992 | 245,808 |

At sixteen environments compact output reduces the logical payload by about
62%. Runtime differences vary across these short cases; treat them as smoke
measurements rather than a robust speedup estimate. Throughput counts one
environment transition with both agents, not two agent transitions.

To repeat from the repository root in an installed environment:

```bash
python scripts/benchmark_ppo.py \
  --source-revision "$(git rev-parse HEAD)" \
  --output /tmp/mycormarl-jax-audit.json
```

On Orchid, verify that the JSON reports a GPU device before interpreting it as
a GPU benchmark. Increase `--num-envs` for a batching sweep; use a representative
study configuration in a later benchmark before choosing production settings.
