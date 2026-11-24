"""Bounded PPO output and durable continuation preserve the training process."""
from dataclasses import replace
import json

from flax import serialization
import jax
import numpy as np
import pytest

from mycormarl.algos.ppo import PPOConfig, make_train
from mycormarl.training import run_training
from test_ppo_actor import _small_environment


def config():
    return PPOConfig(TOTAL_TIMESTEPS=6, NUM_STEPS=2, NUM_ENVS=1,
                     NUM_MINIBATCHES=1, UPDATE_EPOCHS=1,
                     FINITE_HORIZON_RETURNS=True)


def assert_tree_close(a, b):
    a, b = serialization.to_state_dict(a), serialization.to_state_dict(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if np.issubdtype(np.asarray(x).dtype, np.inexact):
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(x, y)


def test_compact_output_matches_debug_without_retaining_rollouts():
    env = _small_environment()
    key = jax.random.PRNGKey(9)
    detailed = jax.jit(make_train(env, config()))(key)
    compact = jax.jit(make_train(env, config(), retain_history=False))(key)
    jax.block_until_ready(compact)
    jax.effects_barrier()
    assert not {'trajectories', 'advantages', 'targets', 'returns'} & compact.keys()
    for field in ('runner_state', 'metrics', 'episode_state', 'episode_summaries', 'qualification'):
        assert_tree_close(detailed[field], compact[field])
    assert compact['episode_summaries']['plant']['count'].sum() == 3


def test_disk_resume_matches_uninterrupted_training_and_preserves_history(tmp_path):
    from mycormarl.environments.base_mycor import BaseMycorMarl
    original = _small_environment()
    env = BaseMycorMarl(replace(original.config, max_steps=3), original.species)
    uninterrupted = run_training(env, config(), seed=7, run_dir=tmp_path/'full', chunk_updates=3)
    first = run_training(env, config(), seed=7, run_dir=tmp_path/'split', chunk_updates=2,
                         max_chunks=1, debug=True)
    assert first['completed_updates'] == 2
    resumed = run_training(env, config(), seed=7, run_dir=tmp_path/'split', chunk_updates=2,
                           resume=True)
    assert_tree_close(uninterrupted['runner_state'], resumed['runner_state'])
    assert_tree_close(uninterrupted['episode_state'], resumed['episode_state'])
    records = [json.loads(row) for row in (tmp_path/'split'/'metrics.jsonl').read_text().splitlines()]
    assert [r['update'] for r in records] == [1,2,3]
    assert all(r['qualification']['allocation_converged'] for r in records)
    assert all('plant' in r['episodes'] and 'fungus' in r['metrics'] for r in records)
    assert sum(r['episodes']['plant']['count'] for r in records) == 2
    assert sum(r['episodes']['plant']['length_sum'] for r in records) == 6
    latest = serialization.msgpack_restore((tmp_path/'split'/'latest.msgpack').read_bytes())
    previous = serialization.msgpack_restore((tmp_path/'split'/'previous.msgpack').read_bytes())
    assert latest['completed_updates'] == 3
    assert previous['completed_updates'] == 2
    assert list((tmp_path/'split'/'debug').glob('*.msgpack'))
    assert not (tmp_path/'full'/'debug').exists()
    with pytest.raises(ValueError, match='configuration'):
        run_training(env, replace(config(), LR=.1), seed=7, run_dir=tmp_path/'split', resume=True)


def test_failed_chunk_does_not_write_a_successful_checkpoint(tmp_path):
    from test_stoichiometric_actions import pool_environment
    env = pool_environment(dt=1., max_steps=1, allocation_max_substeps=1)
    with pytest.raises(RuntimeError, match='allocation'):
        run_training(env, replace(config(), NUM_STEPS=1), seed=1,
                     run_dir=tmp_path/'failed', chunk_updates=1)
    assert not (tmp_path/'failed'/'latest.msgpack').exists()
    assert not (tmp_path/'failed'/'metrics.jsonl').exists()


def test_recovery_uses_previous_checkpoint_and_discards_uncommitted_metrics(tmp_path):
    directory = tmp_path/'run'
    expected = run_training(_small_environment(), config(), seed=5,
                            run_dir=directory, chunk_updates=1)
    (directory/'latest.msgpack').write_bytes(b'corrupted latest checkpoint')
    with (directory/'metrics.jsonl').open('a') as stream:
        stream.write('{"partial":')
    recovered = run_training(_small_environment(), config(), seed=5,
                             run_dir=directory, chunk_updates=1,
                             resume=directory/'previous.msgpack')
    assert_tree_close(expected['runner_state'], recovered['runner_state'])
    rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == [1, 2, 3]


def test_selected_evaluation_retains_policy_summary_and_optional_trace(tmp_path):
    from mycormarl.checkpoint_evaluation import evaluate_checkpoint
    result = run_training(_small_environment(), replace(config(), TOTAL_TIMESTEPS=2),
                          seed=5, run_dir=tmp_path/'run', evaluate_every=1, debug=True)
    files = tmp_path/'run'/'evaluations'
    assert (files/'update-00000001.msgpack').exists()
    summary = json.loads((files/'update-00000001.json').read_text())
    assert summary['protocol'] == 'latent-location'
    assert summary['summary']['evaluation_configuration']['seed'] == 5
    assert summary['summary']['evaluation_configuration']['episodes'] == 1
    assert (files/'update-00000001.trace.json').exists()
    # Resumable files also retain the existing public checkpoint-evaluation ABI.
    evaluation = evaluate_checkpoint(tmp_path/'run'/'latest.msgpack', _small_environment(), episodes=1)
    assert len(evaluation.episodes) == 1
    assert result['completed_updates'] == 1


def test_episode_summaries_end_at_death_and_do_not_count_absent_agents():
    from types import SimpleNamespace
    import jax.numpy as jnp
    from mycormarl.algos.ppo import initial_episode_state, update_episode_state
    state = initial_episode_state(2)
    fields = dict(
        plant=SimpleNamespace(critic_valid=jnp.array([True, True]),
                              terminated=jnp.array([False, True]),
                              truncated=jnp.array([False, False])),
        fungus=SimpleNamespace(critic_valid=jnp.array([False, False]),
                               terminated=jnp.array([False, False]),
                               truncated=jnp.array([False, False])),
    )
    state, first = update_episode_state(state, {'plant':jnp.array([2.,3.]), 'fungus':jnp.zeros(2)}, fields)
    assert first['plant']['fitness_sum'].tolist() == [0.,3.]
    fields['plant'] = SimpleNamespace(critic_valid=jnp.array([True, False]),
                                     terminated=jnp.array([False, False]),
                                     truncated=jnp.array([True, True]))
    fields['fungus'].truncated = jnp.array([True, True])
    _, second = update_episode_state(state, {'plant':jnp.array([5.,0.]), 'fungus':jnp.zeros(2)}, fields)
    assert second['plant']['count'].tolist() == [1,0]
    assert second['plant']['fitness_sum'].tolist() == [7.,0.]
    assert second['plant']['length_sum'].tolist() == [2,0]
    assert second['fungus']['count'].tolist() == [0,0]


def test_invalid_ppo_configuration_does_not_create_run_history(tmp_path):
    with pytest.raises(ValueError, match='divisible'):
        run_training(_small_environment(), replace(config(), NUM_MINIBATCHES=3),
                     seed=7, run_dir=tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
