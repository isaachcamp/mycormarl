"""Public contracts for the reduced-control trade-only study."""

import jax.numpy as jnp
import pytest
import jax
import json
from dataclasses import replace

from mycormarl.algos.ppo import (
    ActorCritic,
    PPOConfig,
    initialize_runner_state,
    make_train,
)
from mycormarl.checkpoint_evaluation import evaluate_policy_parameters
from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.trade_only import (
    FUNGUS_MIXED_TRADE_FRACTION_PER_DAY,
    PLANT_MIXED_TRADE_FRACTION_PER_DAY,
    TOTAL_BIOLOGICAL_RATE_PER_DAY,
    fixed_allocation_rate_action,
    plant_only_actions,
    pool_fraction_to_rate,
    run_trade_only_baseline,
)
from mycormarl.trade_only_study import study_plan, study_tasks
import mycormarl.trade_only_study as trade_study
from mycormarl.random_streams import derive_random_streams


def test_fixed_allocation_action_preserves_learned_trade_and_declared_90_10_rates():
    """The constrained seam emits the ordinary four-component Rate action."""
    action = fixed_allocation_rate_action(jnp.asarray(1.25))

    assert action.tolist() == pytest.approx(
        [1.25, 0.9 * TOTAL_BIOLOGICAL_RATE_PER_DAY,
         0.1 * TOTAL_BIOLOGICAL_RATE_PER_DAY, 0.0]
    )


def test_pool_fraction_to_rate_exchanges_the_declared_daily_current_pool_fraction():
    """Fixed controls retain the requested current-pool percentages as hazards."""
    assert 1.0 - jnp.exp(-pool_fraction_to_rate(0.05)) == pytest.approx(0.05)
    assert 1.0 - jnp.exp(-pool_fraction_to_rate(0.75)) == pytest.approx(0.75)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        pool_fraction_to_rate(1.0)


def test_plant_only_actions_hold_static_plant_allocation_and_zero_trade():
    """The no-partner control contains no unidentifiable learned trade action."""
    actions = plant_only_actions()

    assert actions["plant"].tolist() == pytest.approx(
        [0.0, 0.9 * TOTAL_BIOLOGICAL_RATE_PER_DAY,
         0.1 * TOTAL_BIOLOGICAL_RATE_PER_DAY, 0.0]
    )
    assert actions["fungus"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_fixed_trade_baseline_preserves_provenance_and_realised_exchange():
    """Baseline artifacts declare fixed allocation separately from realised flow."""
    result = run_trade_only_baseline(
        initial_p_micromolar=(1.0,), seeds=(0,), days=0.1, timestep_days=0.05,
    )

    assert result["protocol"]["fixed_allocation"]["growth_fraction"] == 0.9
    assert result["protocol"]["fixed_allocation"]["reproduction_fraction"] == 0.1
    plant_only = next(row for row in result["entries"] if row["mode"] == "plant-only")
    mixed = next(row for row in result["entries"] if row["mode"] == "mixed")
    assert plant_only["commanded_rate_actions"]["plant"][0] == 0.0
    assert plant_only["commanded_rate_actions"]["fungus"] == [0.0, 0.0, 0.0, 0.0]
    assert mixed["commanded_rate_actions"]["plant"][0] == pytest.approx(
        pool_fraction_to_rate(PLANT_MIXED_TRADE_FRACTION_PER_DAY)
    )
    assert mixed["commanded_rate_actions"]["fungus"][0] == pytest.approx(
        pool_fraction_to_rate(FUNGUS_MIXED_TRADE_FRACTION_PER_DAY)
    )
    assert result["protocol"]["mixed_trade_fraction_of_current_post_maintenance_pool_per_day"] == {
        "plant": 0.05, "fungus": 0.75,
    }
    assert set(mixed["transfers"]) == {"plant_c_out", "fungus_p_out"}
    assert set(mixed["cumulative_reproductive_fitness"]) == {"plant", "fungus"}
    assert set(mixed["final_living_biomass"]) == {"plant", "fungus"}


def test_trade_only_study_plan_declares_p_modes_and_fixed_allocation_provenance():
    """The runner's public plan keeps independent P conditions explicit."""
    plan = study_plan()

    assert plan.initial_p_micromolar == (0.75, 3.0, 10.0)
    assert plan.modes == ("plant-only", "mixed")
    assert plan.episodes_per_seed == 500
    assert plan.evaluation_interval_episodes == 500
    assert plan.environment["soil_radius_cm"] == 40.0
    assert plan.environment["soil_depth_cm"] == 60.0
    assert plan.ppo["trade_only"] is True
    assert plan.fixed_allocation["total_biological_rate_per_day"] == pytest.approx(
        TOTAL_BIOLOGICAL_RATE_PER_DAY
    )
    tasks = study_tasks(plan)
    assert len(tasks) == 18
    assert {task["mode"] for task in tasks} == {"plant-only", "mixed"}
    assert {(task["mode"], task["initial_p_micromolar"], task["seed"])
            for task in tasks} >= {("plant-only", 0.75, 0), ("mixed", 10.0, 4)}
    assert {(task["initial_p_micromolar"], task["seed"])
            for task in tasks if task["mode"] == "plant-only"} == {
                (0.75, 0), (3.0, 0), (10.0, 0)
            }


def test_trade_only_plan_accepts_an_explicit_positive_unique_p_cohort():
    """A selectable cohort keeps every requested initial P explicit in provenance."""
    plan = trade_study.study_plan((0.3, 1.5, 5.0), workers=2)

    assert plan.initial_p_micromolar == (0.3, 1.5, 5.0)
    assert plan.workers == 2
    with pytest.raises(ValueError, match="unique"):
        trade_study.study_plan((0.3, 0.3))
    with pytest.raises(ValueError, match="positive"):
        trade_study.study_plan((0.0,))


def test_run_study_refuses_to_mix_a_different_p_cohort_into_existing_output(tmp_path):
    """A new P cohort cannot overwrite the active study's checkpoint provenance."""
    (tmp_path / "study-plan.json").write_text(
        json.dumps(trade_study.plan_payload(study_plan()))
    )

    with pytest.raises(ValueError, match="different study plan"):
        trade_study.run_study(tmp_path, initial_p_micromolar=(0.3, 1.5, 5.0))


def test_run_study_reuses_a_matching_completed_baseline(tmp_path, monkeypatch):
    """Restarting learning must not recompute already compatible controls."""
    plan = study_plan()
    (tmp_path / "study-plan.json").write_text(json.dumps(trade_study.plan_payload(plan)))
    entries = [
        {"mode": mode, "initial_p_micromolar": p_level, "seed": 0}
        for mode in plan.modes for p_level in plan.initial_p_micromolar
    ]
    (tmp_path / "fixed-allocation-baseline.json").write_text(json.dumps({
        "format": "mycormarl-trade-only-baseline",
        "protocol": {
            "fixed_allocation": plan.fixed_allocation,
            "mixed_trade_fraction_of_current_post_maintenance_pool_per_day": {
                "plant": 0.05, "fungus": 0.75,
            },
            "mixed_trade_rate_per_day": {
                "plant": pool_fraction_to_rate(0.05),
                "fungus": pool_fraction_to_rate(0.75),
            },
        },
        "entries": entries,
    }))
    started = []
    compiled = []
    monkeypatch.setattr(trade_study, "study_plan", lambda *_, **__: plan)
    monkeypatch.setattr(
        trade_study, "run_trade_only_baseline",
        lambda **_: pytest.fail("matching controls must be reused"),
    )
    monkeypatch.setattr(
        trade_study, "train_mixed_seed",
        lambda plan, p_level, seed, output_dir, **_: started.append((p_level, seed)),
    )
    monkeypatch.setattr(
        trade_study, "_compile_resumed_updater",
        lambda environment, config, seed: compiled.append(
            environment.config.initial_solution_p_um
        ) or object(),
    )

    trade_study.run_study(tmp_path)

    assert set(started) == {
        (p_level, seed) for p_level in plan.initial_p_micromolar for seed in plan.seeds
    }
    assert sorted(compiled) == list(plan.initial_p_micromolar)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert set(progress["tasks"]) == {
        f"p-{p_level:g}/seed-{seed}"
        for p_level in plan.initial_p_micromolar for seed in plan.seeds
    }


def test_legacy_zero_trade_baseline_is_not_reused(tmp_path):
    """The new mixed current-pool control replaces rejected zero-trade artifacts."""
    plan = study_plan()
    (tmp_path / "study-plan.json").write_text(json.dumps(trade_study.plan_payload(plan)))
    (tmp_path / "fixed-allocation-baseline.json").write_text(json.dumps({
        "format": "mycormarl-trade-only-baseline",
        "protocol": {"fixed_allocation": plan.fixed_allocation},
        "entries": [],
    }))

    assert not trade_study._has_matching_baseline(tmp_path, plan)


def test_fixed_baseline_reuses_completed_plant_only_entries_when_replacing_mixed(tmp_path, monkeypatch):
    """Replacing mixed zero-trade controls avoids re-running unchanged controls."""
    plan = study_plan((1.0,))
    plant_only_entry = {
        "mode": "plant-only", "initial_p_micromolar": 1.0, "seed": 0,
        "status": "completed", "final_living_biomass": {"plant": 2.0},
        "cumulative_reproductive_fitness": {"plant": 3.0},
    }
    (tmp_path / "fixed-allocation-baseline.json").write_text(json.dumps({
        "entries": [plant_only_entry],
    }))
    calls = []
    monkeypatch.setattr(trade_study, "run_trade_only_baseline", lambda **kwargs: (
        calls.append(kwargs) or {
            "entries": [{"mode": "mixed", "initial_p_micromolar": 1.0,
                         "seed": 0, "status": "completed"}],
            "protocol": {}, "completion": {"completed": 1, "requested": 1},
            "status": "complete",
        }
    ))

    trade_study.run_fixed_baseline(tmp_path, plan)

    assert calls[0]["include_plant_only"] is False
    entries = json.loads((tmp_path / "fixed-allocation-baseline.json").read_text())["entries"]
    assert entries == [plant_only_entry, {"mode": "mixed", "initial_p_micromolar": 1.0,
                                           "seed": 0, "status": "completed"}]


def test_trade_only_task_resumes_after_the_last_completed_episode(tmp_path):
    """The bounded checkpoint advances training instead of replaying it."""
    protocol = study_plan()
    tiny_plan = replace(
        protocol, horizon_days=0.05, numerical_timestep_days=0.05,
        decision_interval_days=0.05, policy_steps_per_episode=1,
        episodes_per_seed=2, evaluation_interval_episodes=2,
        environment={"soil_radius_cm": 0.2, "soil_depth_cm": 0.2,
                     "radial_interval_cm": 0.1, "depth_interval_cm": 0.1},
        ppo={**protocol.ppo, "num_steps": 1, "update_epochs": 1},
    )
    environment = trade_study.study_environment(tiny_plan, 1.0)
    config = trade_study.ppo_config(tiny_plan)
    first = jax.jit(make_train(environment, config, derive_random_streams(0)))(
        jax.random.PRNGKey(0)
    )
    task_dir = trade_study._task_dir(tmp_path, 1.0, 0)
    trade_study._write_checkpoint(
        trade_study._checkpoint_path(task_dir), plan=tiny_plan, p_micromolar=1.0,
        seed=0, completed_episodes=1, runner_state=first["runner_state"],
    )

    trade_study.train_mixed_seed(tiny_plan, 1.0, 0, tmp_path)
    completed, resumed = trade_study._restore_checkpoint(
        trade_study._checkpoint_path(task_dir), plan=tiny_plan, p_micromolar=1.0,
        seed=0, environment=environment, config=config,
    )

    assert completed == 2
    assert int(resumed[0]["plant"].step) == 2
    assert (task_dir / "evaluation.json").exists()


def test_trade_only_runner_initialization_is_seeded_and_has_not_trained():
    """A reusable update executable must accept independent fresh seed states."""
    protocol = study_plan()
    tiny_plan = replace(
        protocol, horizon_days=0.05, numerical_timestep_days=0.05,
        decision_interval_days=0.05, policy_steps_per_episode=1,
        episodes_per_seed=2, evaluation_interval_episodes=2,
        environment={"soil_radius_cm": 0.2, "soil_depth_cm": 0.2,
                     "radial_interval_cm": 0.1, "depth_interval_cm": 0.1},
        ppo={**protocol.ppo, "num_steps": 1, "update_epochs": 1},
    )
    environment = trade_study.study_environment(tiny_plan, 1.0)
    config = trade_study.ppo_config(tiny_plan)

    first = initialize_runner_state(
        environment, config, derive_random_streams(0), jax.random.PRNGKey(0)
    )
    second = initialize_runner_state(
        environment, config, derive_random_streams(1), jax.random.PRNGKey(1)
    )

    assert int(first[0]["plant"].step) == 0
    assert int(second[0]["fungus"].step) == 0
    assert not jnp.array_equal(
        first[0]["plant"].params["params"]["policy_encoder_0"]["kernel"],
        second[0]["plant"].params["params"]["policy_encoder_0"]["kernel"],
    )


def test_trade_only_ppo_updates_scalar_trade_policy_and_executes_fixed_allocation():
    """Mixed reduced-control PPO has one sampled latent per operational actor."""
    environment = BaseMycorMarl(
        EnvConfig(max_steps=2, dt=0.05, soil_radius_cm=0.2, soil_depth_cm=0.2,
                  radial_interval_cm=0.1, depth_interval_cm=0.1),
        SpeciesParams(PlantTraits(), FungusTraits()),
    )
    config = PPOConfig(
        TOTAL_TIMESTEPS=2, RUN_TIMESTEPS=2, NUM_STEPS=2, NUM_ENVS=1,
        NUM_MINIBATCHES=1, UPDATE_EPOCHS=1, DISCOUNT_HALF_LIFE_DAYS=30.0,
        TRADE_ONLY=True,
    )

    output = jax.jit(make_train(environment, config))(jax.random.PRNGKey(5))

    for agent, trajectory in zip(
        ("plant", "fungus"), output["trajectories"], strict=True
    ):
        assert trajectory.rate_action.shape[-1] == 4
        assert jnp.allclose(
            trajectory.rate_action[..., 1:],
            jnp.asarray([0.9 * TOTAL_BIOLOGICAL_RATE_PER_DAY,
                         0.1 * TOTAL_BIOLOGICAL_RATE_PER_DAY, 0.0]),
        )
        assert not jnp.any(trajectory.biological_rate_actor_valid)
        assert jnp.any(trajectory.trade_actor_valid)
        assert "biological_rate_head" not in output["runner_state"][0][agent].params["params"]
        assert int(output["runner_state"][0][agent].step) == 1


def test_trade_only_latent_location_evaluation_records_commanded_and_realised_exchange():
    """Evaluation keeps the ordinary action and the realised bilateral flows distinct."""
    environment = BaseMycorMarl(
        EnvConfig(max_steps=2, dt=0.05, soil_radius_cm=0.2, soil_depth_cm=0.2,
                  radial_interval_cm=0.1, depth_interval_cm=0.1),
        SpeciesParams(PlantTraits(), FungusTraits()),
    )
    actor = ActorCritic(trade_only=True)
    observations, _ = environment.reset(jax.random.PRNGKey(0))
    parameters = {
        agent: actor.init(jax.random.PRNGKey(index + 1), observations[agent])
        for index, agent in enumerate(("plant", "fungus"))
    }

    result = evaluate_policy_parameters(
        environment, parameters, episodes=1, seed=3, actor=actor,
    )
    row = result.episodes[0].trace[0]

    assert row["actions"]["plant"][1:] == pytest.approx(
        [0.9 * TOTAL_BIOLOGICAL_RATE_PER_DAY,
         0.1 * TOTAL_BIOLOGICAL_RATE_PER_DAY, 0.0]
    )
    assert set(row["transfers"]["plant"]) == {"proposed_out", "out", "in"}
    assert row["transfers"]["plant"]["out"] <= row["transfers"]["plant"]["proposed_out"]
