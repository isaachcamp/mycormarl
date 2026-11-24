"""Compare PPO environment counts at fixed batch size or fixed rollout length."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def summarize(case, run_dir, wall_seconds):
    """Read compact records; retain learning points at selected evaluations."""
    metadata = json.loads((run_dir/'run.json').read_text())
    compile_seconds = warmed_seconds = 0.0
    warmed_transitions = 0
    peak_bytes = []
    learning_curve = []
    with (run_dir/'metrics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            chunk = row.get('chunk')
            if not chunk:
                continue
            compile_seconds += chunk['compile_seconds']
            if not chunk['first_execution']:
                warmed_seconds += chunk['execution_seconds']
                warmed_transitions += (chunk['end_update'] - chunk['start_update'] + 1) * case['batch_size']
            for device in chunk['device_memory_stats']:
                peak = (device['stats'] or {}).get('peak_bytes_in_use')
                if peak is not None:
                    peak_bytes.append(peak)
            path = run_dir/'evaluations'/f"update-{row['update']:08d}.json"
            if path.exists():
                evaluation = json.loads(path.read_text())
                summary = evaluation['summary']
                learning_curve.append(dict(
                    timesteps=row['timesteps'], compute_elapsed_seconds=row['compute_elapsed_seconds'],
                    wall_elapsed_seconds=summary['evaluation_configuration']['wall_elapsed_seconds'],
                    fitness=summary['fitness'], final_living_biomass=summary['final_living_biomass'],
                    evaluation_configuration=summary['evaluation_configuration'],
                ))
    return dict(
        id=case['id'], run_dir=str(run_dir), wall_seconds=wall_seconds,
        compile_seconds=compile_seconds,
        warmed_environment_transitions_per_second=(warmed_transitions / warmed_seconds if warmed_seconds else None),
        peak_device_bytes_in_use=max(peak_bytes) if peak_bytes else None,
        devices=metadata['devices'], batch_size=case['batch_size'], seed=case['seed'],
        learning_curve=learning_curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--env-config', type=Path, help='JSON EnvConfig overrides, shared by all runs.')
    parser.add_argument('--suite', choices=['all', 'fixed-batch', 'fixed-rollout'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--total-timesteps', type=int, default=16384)
    parser.add_argument('--episode-steps', type=int, default=14600)
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--num-minibatches', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--clip-eps', type=float, default=.2)
    parser.add_argument('--gae-lambda', type=float, default=.95)
    parser.add_argument('--vf-coef', type=float, default=.5)
    parser.add_argument('--chunk-updates', type=int, default=1)
    parser.add_argument('--evaluation-interval-timesteps', type=int, default=4096,
                        help='Matched evaluation spacing in environment transitions; 0 disables.')
    parser.add_argument('--evaluation-episodes', type=int, default=1)
    parser.add_argument('--mode', choices=['mixed', 'plant-only', 'fungus-only'], default='mixed')
    parser.add_argument('--case', nargs='+', help='Select cases, e.g. n4-t128, for smoke checks.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0
            or min(args.total_timesteps, args.episode_steps, args.update_epochs,
                   args.num_minibatches, args.chunk_updates, args.evaluation_episodes) < 1
            or args.evaluation_interval_timesteps < 0):
        parser.error('counts must be positive; seeds must be unique and nonnegative')
    cases = {}
    suites = ['fixed-batch', 'fixed-rollout'] if args.suite == 'all' else [args.suite]
    for suite in suites:
        pairs = [(8, 256), (16, 128), (32, 64)] if suite == 'fixed-batch' else [
            (4, 128), (8, 128), (16, 128), (32, 128)]
        for nenv, steps in pairs:
            name = f'n{nenv}-t{steps}'
            cases.setdefault(name, dict(num_envs=nenv, num_steps=steps, suites=[]))['suites'].append(suite)
    if args.case and not set(args.case) <= cases.keys():
        parser.error('unknown case for this suite')
    rows = []
    for name, case in cases.items():
        if args.case and name not in args.case:
            continue
        batch = case['num_envs'] * case['num_steps']
        if (args.total_timesteps % batch or case['num_steps'] % args.num_minibatches
                or args.evaluation_interval_timesteps % batch):
            parser.error(f'{name}: budget/evaluation spacing must divide into whole batches; '
                         'rollout steps must divide into whole minibatches')
        if (args.evaluation_interval_timesteps
                and (args.evaluation_interval_timesteps // batch) % args.chunk_updates):
            parser.error(f'{name}: chunk boundaries must align with evaluation times')
        for seed in args.seeds:
            ppo = dict(
                NUM_ENVS=case['num_envs'], NUM_STEPS=case['num_steps'],
                TOTAL_TIMESTEPS=args.total_timesteps, NUM_MINIBATCHES=args.num_minibatches,
                UPDATE_EPOCHS=args.update_epochs, LR=args.learning_rate,
                CLIP_EPS=args.clip_eps, GAE_LAMBDA=args.gae_lambda, VF_COEF=args.vf_coef,
                DISCOUNT_HALF_LIFE_DAYS=None, FINITE_HORIZON_RETURNS=True,
            )
            rows.append(dict(
                id=f'{name}-seed{seed}', seed=seed, ppo=ppo, suites=case['suites'],
                episode_steps=args.episode_steps, batch_size=batch,
                minibatch_size=batch // args.num_minibatches,
                evaluation_interval_timesteps=args.evaluation_interval_timesteps,
                evaluate_every_updates=args.evaluation_interval_timesteps // batch,
            ))
    environment = json.loads(args.env_config.read_text()) if args.env_config else {}
    manifest = dict(environment_overrides=environment, format='mycormarl-ppo-sweep', version=1, mode=args.mode,
                    evaluation_episodes=args.evaluation_episodes, runs=rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        parser.error('existing sweep manifest differs; use another output directory')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.dry_run:
        print(json.dumps(dict(manifest=str(path), runs=len(rows))))
        return
    env_path = args.output_dir/'environment.json'
    env_path.write_text(json.dumps(environment, indent=2) + '\n')
    results = []
    for row in rows:
        directory = args.output_dir/row['id']
        run_dir = directory/'run'
        command = [sys.executable, str(Path(__file__).with_name('train_ppo.py')),
                   '--env-config', str(env_path),
                   '--finite-horizon-returns', '--total-timesteps', str(args.total_timesteps),
                   '--num-envs', str(row['ppo']['NUM_ENVS']), '--num-steps', str(row['ppo']['NUM_STEPS']),
                   '--episode-steps', str(args.episode_steps), '--seed', str(row['seed']),
                   '--update-epochs', str(args.update_epochs), '--num-minibatches', str(args.num_minibatches),
                   '--learning-rate', str(args.learning_rate), '--clip-eps', str(args.clip_eps),
                   '--gae-lambda', str(args.gae_lambda), '--vf-coef', str(args.vf_coef),
                   '--chunk-updates', str(args.chunk_updates), '--mode', args.mode,
                   '--evaluate-every', str(row['evaluate_every_updates']),
                   '--evaluation-episodes', str(args.evaluation_episodes),
                   '--output', str(directory/'policy.msgpack'), '--run-dir', str(run_dir)]
        if args.resume and (run_dir/'run.json').exists():
            command.append('--resume')
        start = time.perf_counter()
        child = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
        print(child.stdout, end='', flush=True)
        results.append(summarize(row, run_dir, time.perf_counter()-start))
        (args.output_dir/'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(dict(manifest=str(path), results=str(args.output_dir/'results.json'), runs=len(rows))))


if __name__ == '__main__':
    main()
