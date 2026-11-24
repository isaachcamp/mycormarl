"""Bounded PPO chunks, append-only metrics and atomic resumable checkpoints."""
from __future__ import annotations

from dataclasses import asdict, replace
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from flax import serialization
import jax
import numpy as np

from mycormarl.algos.ppo import (
    ActorCritic, initial_episode_state, initialize_runner_state, make_train,
)
from mycormarl.policy_artifacts import (
    ACTOR_INTERFACE_VERSION, ENVIRONMENT_STATE_SCHEMA_VERSION, save_policy_artifact,
)
from mycormarl.random_streams import derive_random_streams


def _plain(tree):
    return jax.tree.map(lambda x: x.tolist() if hasattr(x, 'tolist') else x, tree)


def _atomic_write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _reconcile_metrics(path, completed):
    """Discard uncommitted rows after a crash between logging and checkpointing."""
    seen, offset = 0, 0
    if path.exists():
        with path.open('rb') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if row['update'] > completed:
                    break
                if row['update'] != seen + 1:
                    raise ValueError('metrics history has missing or duplicate updates')
                seen = row['update']
                offset = stream.tell()
        if seen != completed:
            raise ValueError('metrics history is missing committed updates')
        with path.open('r+b') as stream:
            stream.truncate(offset)
    elif completed:
        raise ValueError('metrics history is missing committed updates')


def _metadata(env, config, seed):
    ppo = asdict(config)
    ppo.pop('RUN_TIMESTEPS')
    configuration = json.loads(json.dumps(dict(
        environment=asdict(env.config), species=asdict(env.species), ppo=ppo, seed=seed,
    )))
    revision = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.rglob('*.py')):
        source_hash.update(str(source.relative_to(Path(__file__).parent)).encode())
        source_hash.update(source.read_bytes())
    return dict(
        source_tree_sha256=source_hash.hexdigest(),
        configuration=configuration,
        actor_interface_version=ACTOR_INTERFACE_VERSION,
        environment_state_schema_version=ENVIRONMENT_STATE_SCHEMA_VERSION,
        actor_configuration=dict(activation=config.ACTIVATION, trade_only=config.TRADE_ONLY),
        random_streams=derive_random_streams(seed).to_dict(),
        source_revision=revision.stdout.strip() if revision.returncode == 0 else 'unknown',
        dependencies={name: version(name) for name in ('jax', 'jaxlib', 'flax', 'optax', 'jaxmarl')},
        python_version=platform.python_version(), platform=platform.platform(),
        devices=[dict(id=d.id, platform=d.platform, kind=d.device_kind) for d in jax.devices()],
    )


def run_training(env, config, *, seed, run_dir, chunk_updates=10, resume=False,
                 debug=False, max_chunks=None, evaluate_every=0, evaluation_episodes=1):
    """Train in bounded chunks; each successful chunk commits logs and state.

    ``max_chunks`` permits a deliberate stop at a resumable boundary. The total
    budget and LR schedule remain fixed. Evaluation is opt-in and uses a fixed
    latent-location protocol/seed independent of training random streams.
    """
    if chunk_updates < 1 or (max_chunks is not None and max_chunks < 1):
        raise ValueError('chunk_updates and max_chunks must be positive')
    if evaluate_every < 0 or evaluation_episodes < 1:
        raise ValueError('invalid evaluation schedule')
    rollout = config.NUM_STEPS * config.NUM_ENVS
    if rollout < 1 or config.TOTAL_TIMESTEPS < rollout or config.TOTAL_TIMESTEPS % rollout:
        raise ValueError('total timesteps must contain whole PPO updates')
    if config.RUN_TIMESTEPS is not None:
        raise ValueError('run_training controls RUN_TIMESTEPS through chunk_updates')
    total_updates = config.TOTAL_TIMESTEPS // rollout
    directory = Path(run_dir)
    latest = directory / 'latest.msgpack'
    metrics_path = directory / 'metrics.jsonl'
    metadata = _metadata(env, config, seed)
    metadata['retention_configuration'] = dict(
        chunk_updates=chunk_updates, debug_history=debug,
        evaluate_every=evaluate_every, evaluation_episodes=evaluation_episodes,
    )
    if not resume and directory.exists() and any(directory.iterdir()):
        raise ValueError('run directory is not empty; use resume or a new directory')
    streams = derive_random_streams(seed)
    key = jax.random.PRNGKey(seed)
    runner = initialize_runner_state(env, config, streams, key)
    episodes = initial_episode_state(config.NUM_ENVS)
    completed = 0
    resume_path = Path(resume) if isinstance(resume, (str, Path)) else latest
    backup_path = resume_path if resume else latest
    if resume:
        payload = serialization.msgpack_restore(resume_path.read_bytes())
        if (payload.get('format') != 'mycormarl-ppo-checkpoint'
                or payload.get('format_version') != 1
                or payload.get('retention_version') != 1):
            raise ValueError('incompatible resumable training checkpoint')
        saved = payload['metadata']
        for field in ('configuration', 'actor_interface_version',
                      'environment_state_schema_version', 'dependencies', 'source_tree_sha256'):
            if saved.get(field) != metadata[field]:
                raise ValueError(f'checkpoint {field} differs from requested configuration')
        runner = jax.device_put(serialization.from_state_dict(runner, payload['runner_state']))
        episodes = jax.device_put(serialization.from_state_dict(episodes, payload['episode_state']))
        completed = payload['completed_updates']
        if not 0 <= completed <= total_updates:
            raise ValueError('invalid completed update count')
    # Validate the PPO contract before creating logs or a fresh run directory.
    first_count = min(chunk_updates, max(total_updates - completed, 1))
    first_updater = make_train(
        env, replace(config, RUN_TIMESTEPS=first_count * rollout), streams,
        initial_runner_state=runner, retain_history=debug,
    )
    directory.mkdir(parents=True, exist_ok=True)
    _reconcile_metrics(metrics_path, completed)
    if not resume:
        _atomic_write(directory / 'run.json', (json.dumps(metadata, indent=2) + '\n').encode())
    executables = {first_count: jax.jit(first_updater)}
    chunks = 0
    while completed < total_updates and (max_chunks is None or chunks < max_chunks):
        count = min(chunk_updates, total_updates - completed)
        if count not in executables:
            chunk_config = replace(config, RUN_TIMESTEPS=count * rollout)
            updater = make_train(env, chunk_config, streams,
                                 initial_runner_state=runner, retain_history=debug)
            executables[count] = jax.jit(updater)
        start = time.perf_counter()
        result = jax.block_until_ready(executables[count](key, runner, episodes))
        # debug.callback effects must succeed before any successful record is written.
        jax.effects_barrier()
        elapsed = time.perf_counter() - start
        for leaf in jax.tree.leaves(result):
            if not np.isfinite(np.asarray(leaf)).all():
                raise ValueError('non-finite training output; checkpoint not saved')
        runner, episodes = result['runner_state'], result['episode_state']
        with metrics_path.open('a') as stream:
            for i in range(count):
                row = dict(
                    update=completed + i + 1, timesteps=(completed + i + 1) * rollout,
                    metrics={a: _plain(jax.tree.map(lambda x: x[i], m))._asdict()
                             for a, m in result['metrics'].items()},
                    episodes=_plain(jax.tree.map(lambda x: x[i], result['episode_summaries'])),
                    qualification=_plain(jax.tree.map(lambda x: x[i], result['qualification'])),
                    chunk=(dict(
                        start_update=completed + 1, end_update=completed + count,
                        seconds_including_first_compile=elapsed,
                        **metadata['retention_configuration'],
                    ) if i == count - 1 else None),
                )
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        completed += count
        payload = dict(
            format='mycormarl-ppo-checkpoint', format_version=1, retention_version=1,
            metadata=metadata, completed_updates=completed,
            runner_state=serialization.to_state_dict(runner),
            episode_state=serialization.to_state_dict(episodes),
            chunk_seconds_including_first_compile=elapsed,
        )
        if backup_path.exists():
            previous_temp = directory / 'previous.msgpack.tmp'
            shutil.copyfile(backup_path, previous_temp)
            previous_temp.replace(directory / 'previous.msgpack')
        _atomic_write(latest, serialization.msgpack_serialize(payload))
        backup_path = latest
        if debug:
            target = directory / 'debug' / f'update-{completed:08d}.msgpack'
            target.parent.mkdir(exist_ok=True)
            _atomic_write(target, serialization.to_bytes(result))
        if evaluate_every and (completed // evaluate_every != (completed-count) // evaluate_every
                               or completed == total_updates):
            from mycormarl.checkpoint_evaluation import (
                evaluate_policy_summary, save_evaluation_summary_artifact,
            )
            evaluation_dir = directory / 'evaluations'
            evaluation_dir.mkdir(exist_ok=True)
            policy_path = evaluation_dir / f'update-{completed:08d}.msgpack'
            parameters = {a: s.params for a, s in runner[0].items()}
            save_policy_artifact(policy_path, parameters,
                                 consumer_mode=env.config.consumer_mode, random_streams=streams)
            summary = evaluate_policy_summary(
                env, parameters, episodes=evaluation_episodes, seed=seed,
                actor=ActorCritic(activation=config.ACTIVATION, trade_only=config.TRADE_ONLY),
            )
            summary['evaluation_configuration'] = dict(
                seed=seed, episodes=evaluation_episodes, update=completed,
                actor_configuration=metadata['actor_configuration'],
            )
            if debug:
                from mycormarl.checkpoint_evaluation import (
                    evaluate_policy_parameters, save_evaluation_artifact,
                )
                trace = evaluate_policy_parameters(
                    env, parameters, episodes=evaluation_episodes, seed=seed,
                    actor=ActorCritic(activation=config.ACTIVATION, trade_only=config.TRADE_ONLY),
                )
                save_evaluation_artifact(policy_path.with_suffix('.trace.json'), trace,
                                         checkpoint=policy_path)
            save_evaluation_summary_artifact(policy_path.with_suffix('.json'), summary,
                                             checkpoint=policy_path)
        chunks += 1
        # Keep no reference to the previous chunk's trajectories or summaries.
        del result
    return dict(runner_state=runner, episode_state=episodes, completed_updates=completed)
