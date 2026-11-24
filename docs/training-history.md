# Training history and restart

The training CLI uses compact, bounded PPO chunks. PPO baseline defaults and
controlled sweeps are documented in the [batching audit](ppo-batching-audit.md). By default each chunk
contains at most ten updates. Rollout trajectories are retained only while
needed for the current PPO update; they are not stacked across the whole run.
The PPO objective, numerical dynamics and global learning-rate schedule are
unchanged by chunking. Chunk records now separate compilation from synchronous
execution time and retain a cumulative compute clock across restart.

## Run a training job

```bash
uv run python scripts/train_ppo.py --finite-horizon-returns \
  --total-timesteps 4096 --num-steps 128 --num-envs 4 \
  --chunk-updates 2 \
  --output /tmp/mycormarl-training/policies.msgpack \
  --run-dir /tmp/mycormarl-training/run
```

A total budget must contain whole PPO updates (`num_steps * num_envs`). Keep
the same total budget when restarting, because it defines LR annealing. Fresh
runs require an empty run directory, preventing accidental history overwrite.
Without explicit paths, files go to the sibling
`../mycormarl-data/training/` directory, outside the Git checkout.

Run the same command with `--resume` to continue from `latest.msgpack`.
If interrupted after `run.json` is saved but before the first checkpoint,
`--resume` validates that manifest and recreates the initial state from the
recorded seed. Any first-chunk metrics written without a committed checkpoint
are discarded before that chunk is repeated. Explicit `--resume-from` paths
still require the selected checkpoint to exist.

Interruption loses only the unfinished chunk. `--chunk-updates` may change on
resume; environment, species, PPO settings, seed, core source hash and recorded
dependency versions must match. Completed runs can be reopened to regenerate
the final policy export without taking another training step.

If the latest checkpoint is damaged, use the same command with
`--resume-from /tmp/mycormarl-training/run/previous.msgpack` instead of
`--resume`. Logs are rolled back to that checkpoint before training continues.
Use one writer per run directory.

## Files retained

| File | Contents |
| --- | --- |
| `run.json` | Environment, species, PPO configuration, seed, named RNG identities, source revision and core source hash, dependency versions, platform, devices and initial retention settings at run creation |
| `metrics.jsonl` | One record per update: both policies' loss/KL/entropy/LR/critic-scale/validity metrics, completed-episode aggregates and numerical qualification flags; chunk-boundary records include timing and retention settings |
| `latest.msgpack` | Latest successful full runner state, including both policies, optimiser states, critic normalisers, batched environment state, observations, RNG states and partial-episode counters |
| `previous.msgpack` | Previous successful checkpoint as a recovery fallback |
| Policy output path | Final versioned policy weights for evaluation |
| `evaluations/update-*.msgpack` and `.json` | Selected policy weights and fixed-protocol evaluation summaries, when evaluation is enabled |
| `debug/update-*.msgpack` | Full chunk trajectories, advantages, targets and diagnostics, only with `--debug-history` |
| `evaluations/update-*.trace.json` | Detailed evaluation traces, only when evaluation and debug history are both enabled |

Metrics remain on disk across the entire run. Checkpoint storage is bounded to
the latest and previous states; selected evaluation policies and debug outputs
accumulate deliberately. No automatic pruning deletes exploration records.

Episode summaries are compact aggregates per policy and update: completed
count, fitness sum, length sum, biological-termination count and administrative-
truncation count. Divide sums by completed count for means; zero-count records
have zero sums, not evidence of zero-fitness episodes. Partial totals cross
rollout and checkpoint boundaries. Biological death ends that organism's
summary immediately; an already dead or absent organism is not counted again
at the environment horizon. Fitness is cumulative reproductive output. Length
is counted in policy decisions, not days. These are training-policy summaries,
separate from fixed-policy evaluation results.

## Optional evaluation and debugging

Add `--evaluate-every 10 --evaluation-episodes 3` to evaluate at chunk boundaries
that cross multiples of ten updates, plus the final boundary. With chunks
larger than the interval, only the boundary policy is evaluated; no intermediate
weights are reconstructed. Evaluation uses the same seed and latent-location
protocol at every checkpoint, without consuming training RNG streams.
Evaluation is disabled by default because its cost depends on the workload.
Selected summaries retain fitness, biomass and other existing evaluator
endpoints, together with evaluation seed, episode count and actor settings. Training checkpoints are also readable by the existing public
checkpoint evaluator.

`--debug-history` enables detailed training output for each chunk and detailed
traces at selected evaluations. Use it for short diagnostic runs: its disk
usage intentionally grows with run length.

## Qualification and persistence

Allocation convergence and biomass-guard contact are reduced to flags at every
update, checked across the chunk, and saved with metrics. All returned values
must be finite. JAX execution and callback effects finish before successful
metrics or checkpoints are written. A rejected chunk leaves the last committed
checkpoint in place.

Metrics are flushed before an atomic replacement of the checkpoint; each
checkpoint contains the number of committed updates. Restart removes incomplete
or uncommitted trailing log rows and rejects missing committed history. Debug
and evaluation files are auxiliary: an interruption after checkpointing may
leave the last boundary without its optional debug/evaluation outputs. Restart
still continues from the committed training state.

The low-level `make_train` API keeps detailed output by default for existing
consumers. Set `retain_history=False` for compact output. When resuming directly,
pass both the dynamic runner state and returned `episode_state`:

```python
updater = jax.jit(make_train(env, chunk_config,
                           initial_runner_state=runner,
                           retain_history=False))
result = updater(key, runner, episode_state)
```

The runner-state tuple remains compatible with existing consumers. The new
`run_training` helper owns log/checkpoint persistence and supplies the partial-
episode state automatically. Older study checkpoints remain evaluable through
the existing evaluator, but are not accepted for this new runner's resumable
logging protocol.
