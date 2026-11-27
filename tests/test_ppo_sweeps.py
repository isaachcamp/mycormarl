"""Public CLIs preserve the experiment while varying PPO collection settings."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def cli(script, *args):
    result = subprocess.run([sys.executable, str(ROOT/'scripts'/script), *map(str, args)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_training_cli_baseline_and_rollout_length_keep_a_fixed_episode_horizon():
    baseline = cli('train_ppo.py', '--dry-run', '--finite-horizon-returns')
    assert baseline['ppo']['NUM_ENVS'] == 16
    assert baseline['ppo']['NUM_STEPS'] == 128
    assert baseline['ppo']['UPDATE_EPOCHS'] == 4
    assert baseline['ppo']['NUM_MINIBATCHES'] == 4
    assert baseline['ppo']['LR'] == 2.5e-4
    assert baseline['ppo']['GAE_LAMBDA'] == .95
    assert baseline['ppo']['CLIP_EPS'] == .2
    assert baseline['ppo']['VF_COEF'] == .5
    assert baseline['ppo']['DISCOUNT_HALF_LIFE_DAYS'] is None
    assert baseline['environment']['max_steps'] == 14600
    shorter = cli('train_ppo.py', '--dry-run', '--finite-horizon-returns', '--num-steps', 64, '--episode-steps', 14600)
    assert shorter['environment'] == baseline['environment']
    assert shorter['ppo']['NUM_STEPS'] == 64


def test_sweep_cli_declares_both_comparisons_without_changing_the_experiment(tmp_path):
    output = tmp_path/'sweep'
    cli('sweep_ppo.py', '--output-dir', output, '--dry-run', '--seeds', 3, 7)
    manifest = json.loads((output/'manifest.json').read_text())
    rows = manifest['runs']
    assert len(rows) == 12  # Six configurations, two independent training seeds.
    fixed_batch = {(r['ppo']['NUM_ENVS'], r['ppo']['NUM_STEPS'])
                   for r in rows if 'fixed-batch' in r['suites']}
    assert fixed_batch == {(8,256), (16,128), (32,64)}
    fixed_rollout = {(r['ppo']['NUM_ENVS'], r['ppo']['NUM_STEPS'])
                     for r in rows if 'fixed-rollout' in r['suites']}
    assert fixed_rollout == {(4,128), (8,128), (16,128), (32,128)}
    assert {r['seed'] for r in rows} == {3,7}
    assert all(r['episode_steps'] == 14600 for r in rows)
    assert all(r['ppo']['TOTAL_TIMESTEPS'] == 16384 for r in rows)
    assert all(r['ppo']['NUM_MINIBATCHES'] == 4 and r['ppo']['UPDATE_EPOCHS'] == 4 for r in rows)
    assert all(r['ppo']['DISCOUNT_HALF_LIFE_DAYS'] is None for r in rows)
    assert all(r['evaluation_interval_timesteps'] == 4096 for r in rows)
    assert all(r['batch_size'] * r['evaluate_every_updates'] == 4096 for r in rows)
    assert not list(output.rglob('latest.msgpack'))


def test_sweep_cli_runs_a_case_and_retains_timing_learning_curve_and_configuration(tmp_path):
    environment = tmp_path/'environment.json'
    environment.write_text(json.dumps(dict(soil_radius_cm=.2, soil_depth_cm=.2,
                                           radial_interval_cm=.1, depth_interval_cm=.1)))
    output = tmp_path/'actual'
    cli('sweep_ppo.py', '--output-dir', output, '--suite', 'fixed-rollout',
        '--case', 'n4-t128', '--seeds', 5, '--total-timesteps', 1024,
        '--episode-steps', 4, '--env-config', environment,
        '--learning-rate', .0001, '--clip-eps', .1,
        '--evaluation-interval-timesteps', 512)
    result, = json.loads((output/'results.json').read_text())
    assert result['wall_seconds'] > 0
    assert result['compile_seconds'] > 0
    assert result['warmed_environment_transitions_per_second'] > 0
    assert result['devices']
    assert [point['timesteps'] for point in result['learning_curve']] == [512,1024]
    assert all(set(point['fitness']) == {'plant','fungus'} for point in result['learning_curve'])
    times = [point['compute_elapsed_seconds'] for point in result['learning_curve']]
    assert times[1] > times[0] > 0
    wall_times = [point['wall_elapsed_seconds'] for point in result['learning_curve']]
    assert wall_times[1] > wall_times[0] > 0
    assert all(wall >= compute for wall, compute in zip(wall_times, times, strict=True))
    run_dir = Path(result['run_dir'])
    saved = json.loads((run_dir/'run.json').read_text())['configuration']
    assert saved['environment']['max_steps'] == 4
    assert saved['environment']['soil_radius_cm'] == .2
    assert saved['ppo']['NUM_ENVS'] == 4 and saved['ppo']['NUM_STEPS'] == 128
    assert saved['ppo']['LR'] == .0001 and saved['ppo']['CLIP_EPS'] == .1
    assert saved['ppo']['UPDATE_EPOCHS'] == 4 and saved['ppo']['NUM_MINIBATCHES'] == 4
    assert saved['ppo']['DISCOUNT_HALF_LIFE_DAYS'] is None
    before = (run_dir/'metrics.jsonl').read_bytes()
    cli('sweep_ppo.py', '--output-dir', output, '--suite', 'fixed-rollout',
        '--case', 'n4-t128', '--seeds', 5, '--total-timesteps', 1024,
        '--episode-steps', 4, '--env-config', environment,
        '--learning-rate', .0001, '--clip-eps', .1,
        '--evaluation-interval-timesteps', 512, '--resume')
    assert (run_dir/'metrics.jsonl').read_bytes() == before


def test_sweep_cli_rejects_chunking_that_would_misalign_evaluation_times(tmp_path):
    output = tmp_path/'misaligned'
    result = subprocess.run(
        [sys.executable, str(ROOT/'scripts/sweep_ppo.py'), '--output-dir', str(output),
         '--dry-run', '--chunk-updates', '3'], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'align' in result.stderr
    assert not output.exists()


def test_throughput_only_sweep_saves_measurements_without_policy_evaluation(tmp_path):
    output = tmp_path/'throughput'
    cli('sweep_ppo.py', '--output-dir', output, '--suite', 'fixed-rollout',
        '--case', 'n4-t128', '--seeds', 5, '--total-timesteps', 1024,
        '--episode-steps', 4, '--evaluation-interval-timesteps', 0)
    result, = json.loads((output/'results.json').read_text())
    assert result['learning_curve'] == []
    assert result['compile_seconds'] > 0
    assert result['warmed_environment_transitions_per_second'] > 0
    run_dir = Path(result['run_dir'])
    assert (run_dir/'latest.msgpack').exists()
    assert not (run_dir/'evaluations').exists()
    records = [json.loads(line) for line in (run_dir/'metrics.jsonl').read_text().splitlines()]
    assert [row['timesteps'] for row in records] == [512,1024]
    assert all(row['qualification']['allocation_converged'] for row in records)
    assert not any(row['qualification']['biomass_guard_contact'] for row in records)
