"""Public protocol contracts for the 10-µM two-arm plant-learning study."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import jax

from mycormarl.algos.ppo import make_train
from mycormarl.environments.base_mycor import PLANT
from mycormarl.random_streams import derive_random_streams

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ippo_learning_10_time_aware.py"
_SPEC = spec_from_file_location("ippo_learning_10_time_aware", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
study = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = study
_SPEC.loader.exec_module(study)


def test_two_arm_study_plan_has_matched_500_episode_daily_protocol():
    """ADR-0015's declared training units must be recoverable before launch."""
    plan = study.study_plan()

    assert plan.initial_solution_p_um == 10.0
    assert plan.horizon_days == 120.0
    assert plan.numerical_timestep_days == 0.025
    assert plan.decision_interval_days == 1.0
    assert plan.policy_steps_per_episode == 120
    assert plan.episodes_per_seed == 500
    assert plan.workers == 2
    assert plan.seeds == (0, 1, 2, 3, 4)
    assert plan.environment["consumer_mode"] == "plant-only"
    assert plan.environment["action_interface"] == "rate-action-daily-held"
    assert plan.environment["initial_solution_p_depth_profile"][-1] == (100.0, 0.069)
    assert [(arm.identifier, arm.include_episode_clock) for arm in plan.arms] == [
        ("state-only", False),
        ("time-aware", True),
    ]
    assert plan.ppo["discount_half_life_days"] is None
    assert plan.ppo["finite_horizon_returns"] is True
    assert plan.ppo["num_envs"] == 1
    assert plan.control_identifiers == (
        "static-growth-10-reproduction-90",
        "static-growth-90-reproduction-10",
        "grow-to-reproduce-day-90",
    )


def test_arm_environment_and_ppo_config_change_only_the_clock_observation():
    """Both arms retain identical biology and finite-horizon PPO settings."""
    plan = study.study_plan()
    state_only, time_aware = plan.arms

    state_environment = study.study_environment(plan, state_only)
    time_environment = study.study_environment(plan, time_aware)
    state_ppo = study.ppo_config(plan)
    time_ppo = study.ppo_config(plan)

    assert state_environment.config.include_episode_clock is False
    assert time_environment.config.include_episode_clock is True
    assert state_environment.observation_spaces["plant"].shape == (5,)
    assert time_environment.observation_spaces["plant"].shape == (6,)
    assert state_environment.config == time_environment.config.replace(
        include_episode_clock=False
    )
    assert state_ppo == time_ppo
    assert state_ppo.FINITE_HORIZON_RETURNS is True
    assert state_ppo.DISCOUNT_HALF_LIFE_DAYS is None


def test_each_task_uses_one_atomic_resume_checkpoint(tmp_path):
    """Five hundred episodes must not create a checkpoint-file sprawl."""
    plan = study.study_plan()
    task_dir = study._task_dir(tmp_path, plan.arms[0], seed=0)

    assert study._checkpoint_path(task_dir) == task_dir / "checkpoint.msgpack"
    assert study._latest_checkpoint(task_dir) is None


def test_task_resume_advances_from_checkpoint_without_replaying_episode(tmp_path):
    """A restored PPO runner must perform only the one remaining episode."""
    protocol = study.study_plan()
    tiny_plan = replace(
        protocol,
        horizon_days=0.05,
        decision_interval_days=0.05,
        policy_steps_per_episode=1,
        episodes_per_seed=2,
        environment={
            **protocol.environment,
            "soil_radius_cm": 0.2,
            "soil_depth_cm": 0.2,
            "radial_interval_cm": 0.1,
            "depth_interval_cm": 0.1,
            "initial_solution_p_depth_profile": ((0.0, 1.0), (0.2, 1.0)),
        },
        ppo={**protocol.ppo, "num_steps": 1, "update_epochs": 1},
    )
    arm = tiny_plan.arms[0]
    environment = study.study_environment(tiny_plan, arm)
    config = study.ppo_config(tiny_plan)
    first_update = jax.jit(make_train(environment, config, derive_random_streams(0)))(
        jax.random.PRNGKey(0)
    )
    task_dir = study._task_dir(tmp_path, arm, seed=0)
    study._write_checkpoint(
        study._checkpoint_path(task_dir),
        plan=tiny_plan,
        arm=arm,
        seed=0,
        episode=1,
        runner_state=first_update["runner_state"],
    )

    study.train_arm_seed(tiny_plan, arm, seed=0, output_dir=tmp_path)
    completed, resumed_state = study._restore_checkpoint(
        study._checkpoint_path(task_dir),
        plan=tiny_plan,
        arm=arm,
        seed=0,
        environment=environment,
        config=config,
    )

    assert completed == 2
    assert int(resumed_state[0][PLANT].step) == 2
    assert (task_dir / "evaluations" / "episode-00002.json").exists()
