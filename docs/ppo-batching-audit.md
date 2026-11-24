# PPO batching audit

The training CLI now starts from the agreed baseline: 16 environments, 128
rollout steps, four update epochs, four minibatches, learning rate `2.5e-4`,
clip `0.2`, GAE lambda `0.95` and value-loss coefficient `0.5`.
This is a configuration to investigate, not evidence of convergence or an
optimal setting for the biological model.

`--episode-steps` controls the episode horizon independently of `--num-steps`,
which controls the rollout buffer. The default horizon is 14,600 environment
steps; at the default `dt=0.025` days that is 365 days. This does not introduce a
separate policy cadence: there is one policy decision per public environment
step. Removal of the older held-interval wrapper remains deferred in #101.

The CLI's default total budget is 16,384 environment transitions, eight PPO
updates at the baseline. This is a development budget. Set the intended total
budget before starting a learning experiment; changing it during resume would
change the learning-rate schedule and is rejected.

Inspect configuration without training:

```bash
uv run python scripts/train_ppo.py --finite-horizon-returns --dry-run
```

## Two comparisons

```bash
uv run python scripts/sweep_ppo.py \
  --output-dir /tmp/mycormarl-batching-preview --dry-run
```

The default manifest contains six distinct configurations and three independent
training seeds (`42 43 44`), for eighteen runs. The shared baseline is executed
once per seed and belongs to both suites.

| Suite | Environments | Rollout steps | Batch per policy | Minibatches | Minibatch size | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed batch | 8 | 256 | 2,048 | 4 | 512 | 4 |
| Both: baseline | 16 | 128 | 2,048 | 4 | 512 | 4 |
| Fixed batch | 32 | 64 | 2,048 | 4 | 512 | 4 |
| Fixed rollout | 4 | 128 | 512 | 4 | 128 | 4 |
| Fixed rollout | 8 | 128 | 1,024 | 4 | 256 | 4 |
| Fixed rollout | 32 | 128 | 4,096 | 4 | 1,024 | 4 |

Fixed-batch comparisons retain update frequency and optimiser work per batch,
while changing trajectory length. Fixed-rollout comparisons deliberately
change batch size and therefore update frequency at a fixed sample budget.
Neither comparison guarantees equal effective samples after lifecycle masks.
Inspect both policies' validity metrics in `metrics.jsonl`.

Evaluation defaults to every 4,096 environment transitions plus the final
boundary. The runner translates this to the appropriate number of PPO updates
for each configuration. Chunking must align with these evaluation boundaries;
the default sweep chunk size is one update. All runs use the same episode
horizon, environment overrides, species defaults and evaluation protocol.
The total budget must contain whole batches for every selected case.

## Execute and resume

A small execution check can use a short, explicitly declared episode horizon:

```bash
uv run python scripts/sweep_ppo.py \
  --suite fixed-rollout --case n4-t128 --seeds 42 \
  --total-timesteps 1024 --episode-steps 4 \
  --evaluation-interval-timesteps 512 \
  --output-dir /tmp/mycormarl-batching-smoke
```

For an accelerator audit, use a representative, qualified environment:

```bash
uv run python scripts/sweep_ppo.py \
  --env-config /path/to/qualified-environment.json \
  --episode-steps 14600 --seeds 42 43 44 \
  --output-dir ../mycormarl-data/training/batching-audit
```

`--env-config` is a JSON object of `EnvConfig` field overrides, shared by every
case. Unspecified fields use the training CLI's development configuration,
including a small 1 cm by 1 cm soil domain at 0.1 cm resolution. Explicit
`--episode-steps` and `--mode` take precedence over JSON values. A tiny-grid
execution check does not establish production-grid throughput or biological
qualification. Device details in results must show a GPU before interpreting a
run as an A100 measurement. No A100 runs were performed to implement this tool.

Use `--suite fixed-batch` or `--suite fixed-rollout` for one comparison, and
`--case n16-t128` to select a configuration. `--seeds` selects independent
training replicates; environments within each run share the evolving policies.

Repeat the same command with `--resume` to continue matching subruns. Existing
manifests must match the requested experiment. A dry-run preview can be
followed by execution in the same output directory. Use one writer per sweep
directory; separate directories are needed for independently scheduled jobs.

## Retained evidence

- `manifest.json`: configurations, seed identities, suite membership, fixed
  horizon and evaluation spacing. Environment JSON overrides are embedded and
  copied to `environment.json` for child runs.
- `<case>-seed<seed>/run/`: normal compact metrics, full resumable checkpoints,
  run provenance and selected evaluation weights/summaries.
- `<case>-seed<seed>/policy.msgpack`: final policy export.
- `results.json`: incrementally saved summaries of successful subruns, including
  compilation time, warmed execution throughput, device details, allocator peak
  bytes when supported, and selected evaluation learning curves.

Learning curves include environment transitions, cumulative compute and elapsed
wall seconds, plant/fungal fitness and final living biomass. Compute seconds include PPO
compilation and synchronous execution; they exclude initialisation, persistence,
Python startup and evaluation. This clock is persisted across resume. Values
are recorded at chunk ends rather than estimating unmeasured per-update times.

Each learning point's `wall_elapsed_seconds` measures time inside
`run_training` through its compact evaluation, including initialisation,
compilation, execution and persistence. The saved clock continues across resume;
time between invocations and work lost before a saved boundary are excluded.
Optional missing/partial evaluation output does not prevent checkpoint resume.

`wall_seconds` measures the current child-process invocation end to end,
including Python startup. After resume it measures that resume invocation,
not the original run's lifetime. Use fresh runs for end-to-end wall-time
comparisons. Keep both wall clocks distinct from compute-only time.

Compilation is measured explicitly with JAX lowering/compilation. Execution is
synchronised with `block_until_ready` and an effects barrier. The first execution
of each chunk-size executable is excluded from warmed throughput; a one-chunk
run therefore reports no warmed estimate. GPU allocator peak statistics, when
available, are sampled at training-chunk boundaries. They can include prior
evaluation allocations and omit a final evaluation peak; they are not peak
process RSS or total device memory. CPU runs may report no allocator statistics. All existing numerical checks remain active.

## Later hyperparameter comparisons

Both CLIs accept `--learning-rate`, `--clip-eps`, `--gae-lambda`, `--vf-coef`,
`--update-epochs` and `--num-minibatches`. After the batching comparison, use
separate output directories to compare learning rates `1e-4` and `5e-4`, clip
`0.1`, epochs `8`, or minibatch counts `1` and `2`, changing one factor at a time.
No automatic Cartesian hyperparameter search is launched.

Discounting remains unchanged: the sweep explicitly uses undiscounted
finite-horizon returns (`DISCOUNT_HALF_LIFE_DAYS=None`). Investigating alternative
discount factors, their physical interpretation and their interaction with GAE
and rollout duration is recorded as deferred [issue #102](https://github.com/isaachcamp/mycormarl/issues/102).


## Known resume limitations

Independent review found two recovery gaps that remain deferred. An interruption
after checkpointing can leave a selected evaluation summary missing or partial;
the sweep may then omit a learning point or fail while parsing that summary.
Also, the sweep currently rewrites an existing manifest non-atomically, so an
interruption during that rewrite can damage it. Ordinary completed-run resume
is tested, but these interruption cases need repair before relying on automatic
recovery of a full audit. The initial Slurm wrapper uses a fresh job directory
and does not automatically resume a previous sweep.
