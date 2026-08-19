"""Public contracts for Transition-derived PPO learning semantics."""

import jax
import jax.numpy as jnp
import pytest

from mycormarl.algos.ppo import (
    ActorCritic,
    calculate_gae,
    discount_from_half_life,
    make_train,
    masked_mean,
    masked_normalize,
    PPOConfig,
    Trajectory,
    transition_to_ppo_fields,
)
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.transition import Transition


def test_trajectory_uses_explicit_ppo_boundary_controls_only():
    """Legacy done fields cannot compete with PPO bootstrap and trace masks."""
    assert "done" not in Trajectory._fields
    assert "terminal" not in Trajectory._fields
    assert "bootstrap_valid" in Trajectory._fields
    assert "gae_trace_continues" in Trajectory._fields


def _transition(
    *,
    operational_at_start=True,
    operational_at_end=True,
    allocation_executed=True,
    trade_executed=True,
    truncated=False,
):
    return Transition(
        requested_action=jnp.zeros(4, dtype=jnp.float32),
        realised_action=jnp.zeros(4, dtype=jnp.float32),
        operational_at_start=jnp.asarray(operational_at_start),
        operational_at_end=jnp.asarray(operational_at_end),
        allocation_executed=jnp.asarray(allocation_executed),
        trade_executed=jnp.asarray(trade_executed),
        truncated=jnp.asarray(truncated),
        final_observation=jnp.arange(5, dtype=jnp.float32),
    )


def test_ordinary_transition_is_valid_for_both_actor_factors_and_critic():
    fields = transition_to_ppo_fields(_transition())

    assert fields.critic_valid
    assert fields.allocation_actor_valid
    assert fields.trade_actor_valid
    assert not fields.terminated
    assert fields.bootstrap_valid
    assert fields.gae_trace_continues
    assert not fields.truncated
    assert jnp.array_equal(fields.bootstrap_observation, jnp.arange(5))


def test_death_transition_keeps_critic_sample_but_rejects_unexecuted_action():
    fields = transition_to_ppo_fields(
        _transition(
            operational_at_end=False,
            allocation_executed=False,
            trade_executed=False,
        )
    )

    assert fields.critic_valid
    assert not fields.allocation_actor_valid
    assert not fields.trade_actor_valid
    assert fields.terminated
    assert not fields.bootstrap_valid
    assert not fields.gae_trace_continues


def test_dead_or_absent_padding_is_invalid_for_all_learning():
    fields = transition_to_ppo_fields(
        _transition(
            operational_at_start=False,
            operational_at_end=False,
            allocation_executed=False,
            trade_executed=False,
        )
    )

    assert not fields.critic_valid
    assert not fields.allocation_actor_valid
    assert not fields.trade_actor_valid
    assert not fields.terminated
    assert not fields.bootstrap_valid
    assert not fields.gae_trace_continues


def test_truncated_survivor_bootstraps_from_final_observation_but_stops_trace():
    fields = transition_to_ppo_fields(_transition(truncated=True))

    assert fields.critic_valid
    assert not fields.terminated
    assert fields.truncated
    assert fields.bootstrap_valid
    assert not fields.gae_trace_continues
    assert jnp.array_equal(fields.bootstrap_observation, jnp.arange(5))


def test_cancelled_trade_does_not_mask_executed_allocation():
    fields = transition_to_ppo_fields(_transition(trade_executed=False))

    assert fields.critic_valid
    assert fields.allocation_actor_valid
    assert not fields.trade_actor_valid


@pytest.mark.parametrize("half_life_days", [30.0, 90.0, 365.0])
def test_physical_half_life_has_one_authoritative_per_step_discount_conversion(
    half_life_days,
):
    assert discount_from_half_life(0.025, None) == 1.0
    assert discount_from_half_life(0.025, jnp.inf) == 1.0
    assert discount_from_half_life(0.025, half_life_days) == jnp.exp(
        -jnp.log(2.0) * 0.025 / half_life_days
    )


@pytest.mark.parametrize(
    ("dt_days", "half_life_days"),
    [
        (0.0, 30.0),
        (jnp.nan, 30.0),
        (0.025, 0.0),
        (0.025, jnp.nan),
        (0.025, -jnp.inf),
    ],
)
def test_discount_conversion_rejects_nonphysical_inputs(dt_days, half_life_days):
    with pytest.raises(ValueError):
        discount_from_half_life(dt_days, half_life_days)


def test_truncation_bootstraps_from_final_value_without_admitting_reset_reward():
    advantages, targets = calculate_gae(
        rewards=jnp.array([5.0, 100.0]),
        values=jnp.array([2.0, 20.0]),
        bootstrap_values=jnp.array([7.0, 30.0]),
        critic_valid=jnp.array([True, True]),
        bootstrap_valid=jnp.array([True, True]),
        gae_trace_continues=jnp.array([False, True]),
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert jnp.allclose(advantages, jnp.array([6.5, 95.0]))
    assert jnp.allclose(targets, jnp.array([8.5, 115.0]))


def test_masked_reductions_exclude_padding_and_are_safe_when_empty():
    values = jnp.array([1.0, 3.0, 1000.0])
    mask = jnp.array([True, True, False])

    assert masked_mean(values, mask) == 2.0
    assert jnp.allclose(masked_normalize(values, mask), jnp.array([-1.0, 1.0, 0.0]))
    assert masked_mean(values, jnp.zeros(3, dtype=bool)) == 0.0
    assert jnp.array_equal(
        masked_normalize(values, jnp.zeros(3, dtype=bool)),
        jnp.zeros(3),
    )


def test_rollout_carries_per_species_transition_validity():
    environment = BaseMycorMarl(
        EnvConfig(
            max_steps=2,
            dt=0.05,
            consumer_mode="plant-only",
            soil_radius_cm=0.2,
            soil_depth_cm=0.2,
            radial_interval_cm=0.1,
            depth_interval_cm=0.1,
        ),
        SpeciesParams(plant=PlantTraits(), fungus=FungusTraits()),
    )
    output = jax.jit(
        make_train(
            environment,
            PPOConfig(
                TOTAL_TIMESTEPS=2,
                NUM_STEPS=2,
                NUM_ENVS=1,
                NUM_MINIBATCHES=1,
                UPDATE_EPOCHS=1,
                LR=0.0,
            ),
        )
    )(jax.random.PRNGKey(0))
    plant, fungus = output["trajectories"]

    assert plant.critic_valid.shape == (1, 2, 1)
    assert fungus.critic_valid.shape == (1, 2, 1)
    assert jnp.all(plant.critic_valid)
    assert jnp.all(plant.allocation_actor_valid)
    assert not jnp.any(plant.trade_actor_valid)
    assert not jnp.any(fungus.critic_valid)
    assert not jnp.any(fungus.allocation_actor_valid)
    assert not jnp.any(fungus.trade_actor_valid)
    train_states = output["runner_state"][0]
    assert train_states[PLANT].step == 1
    assert train_states[FUNGUS].step == 0
    plant_metrics = output["metrics"][PLANT]
    fungus_metrics = output["metrics"][FUNGUS]
    assert plant_metrics.critic_valid_count == 2
    assert plant_metrics.allocation_actor_valid_count == 2
    assert plant_metrics.trade_actor_valid_count == 0
    assert plant_metrics.allocation_actor_valid_fraction == 1.0
    assert fungus_metrics.critic_valid_count == 0
    assert fungus_metrics.allocation_actor_valid_count == 0
    assert fungus_metrics.trade_actor_valid_count == 0
    assert fungus_metrics.allocation_actor_valid_fraction == 0.0


def test_undiscounted_training_rejects_indefinitely_viable_configured_consumer():
    environment = BaseMycorMarl(
        EnvConfig(
            consumer_mode="plant-only",
            soil_radius_cm=0.2,
            soil_depth_cm=0.2,
            radial_interval_cm=0.1,
            depth_interval_cm=0.1,
        ),
        SpeciesParams(
            plant=PlantTraits(kappa_p=0.0),
            fungus=FungusTraits(),
        ),
    )

    with pytest.raises(ValueError, match="undiscounted.*finite lifetime"):
        make_train(environment, PPOConfig(DISCOUNT_HALF_LIFE_DAYS=None))


def test_rollout_distinguishes_death_transition_from_dead_padding():
    environment = BaseMycorMarl(
        EnvConfig(
            max_steps=4,
            dt=0.05,
            soil_radius_cm=0.2,
            soil_depth_cm=0.2,
            radial_interval_cm=0.1,
            depth_interval_cm=0.1,
        ),
        SpeciesParams(
            plant=PlantTraits(
                initial_biomass=1.0,
                initial_c_pool=0.0,
                initial_p_pool=0.0,
                kappa_c=100.0,
            ),
            fungus=FungusTraits(),
        ),
    )
    output = jax.jit(
        make_train(
            environment,
            PPOConfig(
                TOTAL_TIMESTEPS=2,
                NUM_STEPS=2,
                NUM_ENVS=1,
                NUM_MINIBATCHES=1,
                UPDATE_EPOCHS=1,
                LR=1e-2,
            ),
        )
    )(jax.random.PRNGKey(2))
    plant = output["trajectories"][0]

    assert jnp.array_equal(plant.critic_valid[0, :, 0], jnp.array([True, False]))
    assert not jnp.any(plant.allocation_actor_valid)
    assert not jnp.any(plant.trade_actor_valid)
    assert jnp.array_equal(plant.terminated[0, :, 0], jnp.array([True, False]))
    assert not jnp.any(plant.bootstrap_valid)
    assert not jnp.any(plant.gae_trace_continues)
    train_states = output["runner_state"][0]
    assert train_states[PLANT].step == 0
    assert train_states[FUNGUS].step == 1

    fungus_parameters = train_states[FUNGUS].params["params"]
    assert jnp.array_equal(
        fungus_parameters["trade_head"]["kernel"],
        jnp.zeros_like(fungus_parameters["trade_head"]["kernel"]),
    )
    assert jnp.allclose(
        fungus_parameters["trade_head"]["bias"],
        jnp.log(0.1 / 0.9),
    )
    assert jnp.array_equal(
        fungus_parameters["trade_log_std"],
        jnp.zeros_like(fungus_parameters["trade_log_std"]),
    )
    assert jnp.any(fungus_parameters["allocation_head"]["kernel"] != 0.0)


def test_jitted_truncation_targets_use_final_observation_and_stop_gae_trace():
    environment = BaseMycorMarl(
        EnvConfig(
            max_steps=1,
            dt=0.05,
            soil_radius_cm=0.2,
            soil_depth_cm=0.2,
            radial_interval_cm=0.1,
            depth_interval_cm=0.1,
        ),
        SpeciesParams(plant=PlantTraits(), fungus=FungusTraits()),
    )
    config = PPOConfig(
        TOTAL_TIMESTEPS=2,
        NUM_STEPS=2,
        NUM_ENVS=1,
        NUM_MINIBATCHES=1,
        UPDATE_EPOCHS=1,
        LR=0.0,
    )
    output = jax.jit(make_train(environment, config))(jax.random.PRNGKey(3))
    train_states = output["runner_state"][0]

    for agent, trajectory in zip(
        (PLANT, FUNGUS), output["trajectories"], strict=True
    ):
        _, final_values = ActorCritic(activation=config.ACTIVATION).apply(
            train_states[agent].params,
            trajectory.bootstrap_observation,
        )
        expected_advantages = trajectory.reward + final_values - trajectory.value

        assert jnp.all(trajectory.truncated)
        assert jnp.all(trajectory.bootstrap_valid)
        assert not jnp.any(trajectory.gae_trace_continues)
        assert jnp.allclose(output["advantages"][agent], expected_advantages)
        assert jnp.allclose(
            output["targets"][agent],
            trajectory.reward + final_values,
        )
