"""Public environment behavior for policy timing versus numerical timing."""

import jax
import jax.numpy as jnp
import pytest

from mycormarl.actions import physical_action
from mycormarl.algos.ppo import PPOConfig, make_train
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


def _environment(
    *, dt: float, decision_interval_days: float, initial_solution_p_um: float = 0.0
) -> PolicyIntervalMycorMarl:
    """Create a pool-only system with analytically known allocations."""
    numerical_environment = BaseMycorMarl(
        EnvConfig(
            max_steps=round(decision_interval_days / dt),
            dt=dt,
            soil_radius_cm=1.0,
            soil_depth_cm=1.0,
            radial_interval_cm=1.0,
            depth_interval_cm=1.0,
            initial_solution_p_um=initial_solution_p_um,
        ),
        SpeciesParams(
            plant=PlantTraits(
                initial_biomass=1.0,
                initial_c_pool=1.0,
                initial_p_pool=1.0,
                gamma_c=1.0,
                gamma_p=1.0,
                kappa_c=0.0,
                kappa_p=0.0,
                kleaf=0.0,
            ),
            fungus=FungusTraits(
                initial_biomass=1.0,
                initial_c_pool=1.0,
                initial_p_pool=1.0,
                gamma_c=1.0,
                gamma_p=1.0,
                kappa_c=0.0,
                kappa_p=0.0,
            ),
        ),
    )
    return PolicyIntervalMycorMarl(
        numerical_environment,
        decision_interval_days=decision_interval_days,
        max_episode_steps=1,
    )


def test_held_policy_action_has_compatible_fixed_horizon_effect_under_substepping():
    """Two numerical substeps do not execute a 50% allocation twice."""
    action = physical_action(0.0, 0.5, 0.0, 0.5)
    actions = {PLANT: action, FUNGUS: action}
    compatibility = _environment(dt=0.10, decision_interval_days=0.10)
    substepped = _environment(dt=0.05, decision_interval_days=0.10)

    _, compatibility_state = compatibility.reset(jax.random.PRNGKey(0))
    _, substepped_state = substepped.reset(jax.random.PRNGKey(0))
    _, compatibility_state, _, _, _ = compatibility.step_env(
        jax.random.PRNGKey(1), compatibility_state, actions
    )
    _, substepped_state, _, _, _ = substepped.step_env(
        jax.random.PRNGKey(1), substepped_state, actions
    )

    for field in (
        "plant_biomass",
        "fungus_biomass",
        "plant_c_pool",
        "plant_p_pool",
        "fungus_c_pool",
        "fungus_p_pool",
    ):
        assert getattr(substepped_state, field) == pytest.approx(
            getattr(compatibility_state, field)
        )
    assert compatibility_state.step == 1
    assert substepped_state.step == 2
    assert compatibility.max_episode_steps == substepped.max_episode_steps == 1


def test_decision_interval_equal_to_numerical_timestep_matches_base_environment():
    """The explicit compatibility configuration is biologically unchanged."""
    wrapped = _environment(dt=0.10, decision_interval_days=0.10, initial_solution_p_um=1.0)
    numerical = wrapped.numerical_environment
    actions = {
        PLANT: physical_action(0.25, 0.5, 0.25, 0.0),
        FUNGUS: physical_action(0.25, 0.5, 0.25, 0.0),
    }
    _, wrapped_state = wrapped.reset(jax.random.PRNGKey(0))
    _, numerical_state = numerical.reset(jax.random.PRNGKey(0))

    wrapped_result = wrapped.step_env(jax.random.PRNGKey(1), wrapped_state, actions)
    numerical_result = numerical.step_env(
        jax.random.PRNGKey(1), numerical_state, actions
    )

    for wrapped_leaf, numerical_leaf in zip(
        jax.tree.leaves(wrapped_result[:4]),
        jax.tree.leaves(numerical_result[:4]),
        strict=True,
    ):
        assert jnp.array_equal(wrapped_leaf, numerical_leaf)


def test_held_policy_action_preserves_nonzero_integrated_uptake_under_substepping():
    """A fixed policy schedule keeps physical phosphate fluxes comparable."""
    action = physical_action(0.0, 0.0, 0.0, 1.0)
    actions = {PLANT: action, FUNGUS: action}
    compatibility = _environment(
        dt=0.10, decision_interval_days=0.10, initial_solution_p_um=1.0
    )
    substepped = _environment(
        dt=0.05, decision_interval_days=0.10, initial_solution_p_um=1.0
    )

    _, compatibility_state = compatibility.reset(jax.random.PRNGKey(0))
    _, substepped_state = substepped.reset(jax.random.PRNGKey(0))
    _, compatibility_state, _, _, _ = compatibility.step_env(
        jax.random.PRNGKey(1), compatibility_state, actions
    )
    _, substepped_state, _, _, _ = substepped.step_env(
        jax.random.PRNGKey(1), substepped_state, actions
    )

    assert compatibility_state.cumulative_direct_plant_p_uptake_micromol[0] > 0.0
    assert substepped_state.cumulative_direct_plant_p_uptake_micromol[0] > 0.0
    for field in (
        "cumulative_direct_plant_p_uptake_micromol",
        "soil_labile_p",
        "plant_p_pool",
        "fungus_p_pool",
    ):
        assert jnp.allclose(
            getattr(substepped_state, field),
            getattr(compatibility_state, field),
            rtol=0.05,
            atol=1e-8,
        )


def test_policy_interval_supports_jitted_ppo_rollout():
    """PPO rolls out one decision while the wrapper executes two numerical steps."""
    environment = _environment(dt=0.05, decision_interval_days=0.10)
    config = PPOConfig(
        TOTAL_TIMESTEPS=2,
        NUM_STEPS=2,
        NUM_ENVS=1,
        NUM_MINIBATCHES=1,
        UPDATE_EPOCHS=1,
        LR=0.0,
        DISCOUNT_HALF_LIFE_DAYS=1.0,
    )
    output = jax.jit(make_train(environment, config))(jax.random.PRNGKey(1))

    for trajectory in output["trajectories"]:
        assert trajectory.reward.shape == (1, 2, 1)
        assert jnp.all(trajectory.truncated)
