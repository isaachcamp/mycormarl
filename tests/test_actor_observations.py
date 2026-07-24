import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mycormarl.actions import physical_action
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


def _environment(
    *,
    config: EnvConfig | None = None,
    plant: PlantTraits | None = None,
    fungus: FungusTraits | None = None,
) -> BaseMycorMarl:
    return BaseMycorMarl(
        config
        or EnvConfig(
            dt=1.0,
            max_steps=2,
            soil_radius_cm=1.0,
            soil_depth_cm=1.0,
            radial_interval_cm=0.5,
            depth_interval_cm=0.5,
            topsoil_depth_cm=0.5,
            initial_solution_p_um=0.0,
        ),
        SpeciesParams(
            plant=plant
            or PlantTraits(
                initial_biomass=10.0,
                initial_c_pool=20.0,
                initial_p_pool=10.0,
                biomass_cap=100.0,
                kappa_c=0.0,
                kappa_p=0.0,
            ),
            fungus=fungus
            or FungusTraits(
                initial_biomass=8.0,
                initial_c_pool=12.0,
                initial_p_pool=16.0,
                kappa_c=0.0,
                kappa_p=0.0,
            ),
        ),
    )


def test_reset_returns_stable_bounded_actor_observation_contract():
    env = _environment()

    observations, _ = env.reset(jax.random.PRNGKey(0))

    for agent in (PLANT, FUNGUS):
        observation = observations[agent]
        space = env.observation_spaces[agent]
        assert observation.shape == space.shape == (5,)
        assert observation.dtype == space.dtype == jnp.float32
        assert jnp.all(space.low == 0.0)
        assert jnp.all(space.high == 1.0)
        assert jnp.all(jnp.isfinite(observation))
        assert jnp.all((observation >= 0.0) & (observation <= 1.0))
        assert observation[3] == 0.0
        assert observation[4] == 1.0


def test_state_observations_use_agreed_feature_equations_and_order():
    """The synthetic fungal traits make radial-fill biomass 1 g."""
    env = _environment(
        plant=PlantTraits(
            initial_biomass=50.0,
            initial_c_pool=100.0,
            initial_p_pool=150.0,
            biomass_cap=100.0,
            gamma_c=2.0,
            gamma_p=3.0,
            kappa_c=0.0,
            kappa_p=0.0,
        ),
        fungus=FungusTraits(
            initial_biomass=0.5,
            initial_c_pool=0.5,
            initial_p_pool=1.0,
            gamma_c=1.0,
            gamma_p=2.0,
            kappa_c=0.0,
            kappa_p=0.0,
            saturation_density=3.0 / (2.0 * math.pi),
            hyphal_tissue_carbon_density=1.0 / math.pi,
            hyphal_radius=1.0,
        ),
    )
    _, state = env.reset(jax.random.PRNGKey(0))

    observations = env.get_obs(state)

    expected = np.array([0.5, 0.5, 0.5, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(observations[PLANT], expected)
    np.testing.assert_allclose(observations[FUNGUS], expected)


def test_received_trade_is_reconstructed_from_state_against_maintenance_need():
    env = _environment(
        plant=PlantTraits(
            initial_biomass=10.0,
            initial_c_pool=20.0,
            initial_p_pool=10.0,
            biomass_cap=100.0,
            kappa_p=0.2,
        ),
        fungus=FungusTraits(
            initial_biomass=8.0,
            initial_c_pool=12.0,
            initial_p_pool=16.0,
            kappa_c=0.5,
        ),
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    saved_state = state.replace(
        plant_last_p_received=jnp.array([2.0], dtype=jnp.float32),
        fungus_last_c_received=jnp.array([4.0], dtype=jnp.float32),
    )

    observations = env.get_obs(saved_state)

    assert observations[PLANT][3] == 0.5
    assert observations[FUNGUS][3] == 0.5
    leaves, tree = jax.tree_util.tree_flatten(saved_state)
    restored_state = jax.tree_util.tree_unflatten(tree, leaves)
    reconstructed = env.get_obs(restored_state)
    np.testing.assert_array_equal(reconstructed[PLANT], observations[PLANT])
    np.testing.assert_array_equal(reconstructed[FUNGUS], observations[FUNGUS])


def test_step_stores_realised_received_trade_before_reconstructing_observations():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(0))
    trade_everything = {
        PLANT: physical_action(1.0, 0.0, 0.0, 1.0),
        FUNGUS: physical_action(1.0, 0.0, 0.0, 1.0),
    }

    observations, next_state, _, _, _ = env.step_env(
        jax.random.PRNGKey(1), state, trade_everything
    )

    assert next_state.plant_last_p_received[0] == 16.0
    assert next_state.fungus_last_c_received[0] == 20.0
    reconstructed = env.get_obs(next_state)
    np.testing.assert_array_equal(reconstructed[PLANT], observations[PLANT])
    np.testing.assert_array_equal(reconstructed[FUNGUS], observations[FUNGUS])


def test_actor_observation_contract_has_no_raw_observation_switch():
    with pytest.raises(TypeError, match="norm_obs"):
        EnvConfig(norm_obs=False)


@pytest.mark.parametrize("pool_value", [0.0, 1e30])
def test_zero_and_extreme_pools_produce_finite_bounded_observations(pool_value):
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        plant_c_pool=jnp.array([pool_value]),
        plant_p_pool=jnp.array([pool_value]),
        fungus_c_pool=jnp.array([pool_value]),
        fungus_p_pool=jnp.array([pool_value]),
    )

    observations = env.get_obs(state)

    for agent in (PLANT, FUNGUS):
        assert jnp.all(jnp.isfinite(observations[agent]))
        assert jnp.all((observations[agent] >= 0.0) & (observations[agent] <= 1.0))


@pytest.mark.parametrize(
    ("mode", "absent_agent", "present_agent"),
    [
        ("plant-only", FUNGUS, PLANT),
        ("fungus-only", PLANT, FUNGUS),
    ],
)
def test_single_consumer_modes_zero_absent_observation_and_association(
    mode, absent_agent, present_agent
):
    config = EnvConfig(
        dt=1.0,
        max_steps=2,
        consumer_mode=mode,
        soil_radius_cm=1.0,
        soil_depth_cm=1.0,
        radial_interval_cm=0.5,
        depth_interval_cm=0.5,
        topsoil_depth_cm=0.5,
        initial_solution_p_um=0.0,
    )
    env = _environment(config=config)

    observations, _ = env.reset(jax.random.PRNGKey(0))

    np.testing.assert_array_equal(
        observations[absent_agent], np.zeros(5, dtype=np.float32)
    )
    assert observations[present_agent][4] == 0.0
    assert jnp.any(observations[present_agent][:3] > 0.0)


def test_partner_death_removes_association_and_dead_agent_observation():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(plant_dead=jnp.array([True]))

    observations = env.get_obs(state)

    np.testing.assert_array_equal(observations[PLANT], np.zeros(5, dtype=np.float32))
    assert observations[FUNGUS][4] == 0.0
    assert jnp.any(observations[FUNGUS][:3] > 0.0)


def test_no_trade_transition_clears_previous_received_trade_memory():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        plant_last_p_received=jnp.array([3.0]),
        fungus_last_c_received=jnp.array([4.0]),
    )
    no_trade = {
        PLANT: physical_action(0.0, 1.0, 0.0, 0.0),
        FUNGUS: physical_action(0.0, 1.0, 0.0, 0.0),
    }

    observations, next_state, _, _, _ = env.step_env(
        jax.random.PRNGKey(1), state, no_trade
    )

    assert next_state.plant_last_p_received[0] == 0.0
    assert next_state.fungus_last_c_received[0] == 0.0
    assert observations[PLANT][3] == 0.0
    assert observations[FUNGUS][3] == 0.0


def test_observations_reconstruct_under_jit_and_vectorised_environments():
    env = _environment()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)

    observations, states = jax.jit(jax.vmap(env.reset))(keys)
    reconstructed = jax.jit(jax.vmap(env.get_obs))(states)

    for agent in (PLANT, FUNGUS):
        assert observations[agent].shape == (2, 5)
        np.testing.assert_array_equal(reconstructed[agent], observations[agent])
