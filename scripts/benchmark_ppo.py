"""Small CPU/GPU PPO audit: compilation, warmed runtime and returned array size."""

import argparse
from dataclasses import asdict
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import numpy as np
from mycormarl.algos.ppo import PPOConfig, make_train
from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.fungus.traits import FungusTraits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--source-revision', default='unrecorded')
    args = parser.parse_args()
    if any(n < 1 for n in args.num_envs):
        parser.error('--num-envs must be positive')
    rows = []
    env_config = EnvConfig(
        max_steps=32, dt=0.025, soil_radius_cm=1., soil_depth_cm=1.,
        radial_interval_cm=0.1, depth_interval_cm=0.1,
    )
    for nenv in args.num_envs:
        env = BaseMycorMarl(env_config, SpeciesParams(plant=PlantTraits(), fungus=FungusTraits()))
        config = PPOConfig(
            TOTAL_TIMESTEPS=64 * nenv, NUM_STEPS=32, NUM_ENVS=nenv,
            NUM_MINIBATCHES=1, UPDATE_EPOCHS=1, FINITE_HORIZON_RETURNS=True,
        )
        train = make_train(env, config)
        key = jax.device_put(jax.random.PRNGKey(42))
        previous_result = None
        for variant in ('full', 'compact'):
            def run(key):
                result = train(key)
                return result if variant == 'full' else {k: result[k] for k in ('runner_state', 'metrics')}
            start = time.perf_counter()
            compiled = jax.jit(run).lower(key).compile()
            compile_seconds = time.perf_counter() - start
            result = jax.block_until_ready(compiled(key))
            jax.effects_barrier()
            durations = []
            for _ in range(3):
                start = time.perf_counter()
                result = jax.block_until_ready(compiled(key))
                jax.effects_barrier()
                durations.append(time.perf_counter() - start)
            leaves = jax.tree.leaves(result)
            assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
            if previous_result is not None:
                for a, b in zip(
                    jax.tree.leaves(previous_result['runner_state']),
                    jax.tree.leaves(result['runner_state']), strict=True,
                ):
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
                # Output pruning can change float32 fusion/reduction rounding,
                # particularly for near-zero approximate KL measurements.
                for a, b in zip(
                    jax.tree.leaves(previous_result['metrics']),
                    jax.tree.leaves(result['metrics']), strict=True,
                ):
                    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
            previous_result = result
            row = dict(
                num_envs=nenv, variant=variant, compile_seconds=compile_seconds,
                warmed_seconds=durations, median_seconds=statistics.median(durations),
                environment_transitions_per_second=64 * nenv / statistics.median(durations),
                output_logical_bytes=sum(x.nbytes for x in leaves),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    evidence = dict(
        source_revision=args.source_revision, environment=asdict(env_config),
        species=asdict(env.species), update_epochs=1, num_minibatches=1,
        finite_horizon_returns=True, platform=platform.platform(),
        jax_version=jax.__version__, devices=[str(d) for d in jax.devices()],
        seed=42, num_steps=32, updates=2, repeats=3, rows=rows,
    )
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == "__main__":
    main()
