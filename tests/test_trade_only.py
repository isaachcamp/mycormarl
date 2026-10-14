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
    TOTAL_BIOLOGICAL_RATE_PER_DAY,
    fixed_allocation_rate_action,
    plant_only_actions,
    pool_fraction_to_rate,
)
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
