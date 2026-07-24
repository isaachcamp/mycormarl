import jax
import jax.numpy as jnp
import pytest

from mycormarl.actions import physical_action
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.transition import Transition


@pytest.fixture()
def species():
    return SpeciesParams(
        plant=PlantTraits(
            initial_biomass=10.0,
            initial_c_pool=20.0,
            initial_p_pool=10.0,
            gamma_c=2.0,
            gamma_p=1.0,
            kappa_c=0.0,
            kappa_p=0.0,
            death_fraction=0.2,
            biomass_cap=100.0,
            kleaf=1.0,
            amass=10.0,
        ),
        fungus=FungusTraits(
            initial_biomass=8.0,
            initial_c_pool=12.0,
            initial_p_pool=16.0,
            gamma_c=2.0,
            gamma_p=1.0,
            kappa_c=0.0,
            kappa_p=0.0,
            death_fraction=0.2,
        ),
    )


@pytest.fixture()
def config():
    return EnvConfig(
        dt=1.0,
        soil_radius_cm=1.0,
        radial_interval_cm=0.5,
        soil_depth_cm=1.0,
        depth_interval_cm=0.5,
        topsoil_depth_cm=0.5,
        initial_solution_p_um=0.0,
    )


@pytest.fixture()
def env(species, config):
    return BaseMycorMarl(config=config, species=species, max_episode_steps=3)


def reserve_actions(*, plant_trade=0.0, fungus_trade=0.0):
    return {
        PLANT: physical_action(plant_trade, 0.0, 0.0, 1.0),
        FUNGUS: physical_action(fungus_trade, 0.0, 0.0, 1.0),
    }


def test_ordinary_step_exposes_typed_transitions_separately_from_diagnostics(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = reserve_actions(plant_trade=0.25, fungus_trade=0.5)

    observations, _, _, _, info = env.step_env(
        jax.random.PRNGKey(1), state, actions
    )

    transitions = info["transitions"]
    assert set(transitions) == {PLANT, FUNGUS}
    assert isinstance(transitions[PLANT], Transition)
    assert isinstance(transitions[FUNGUS], Transition)
    assert info[PLANT] is not transitions[PLANT]
    assert info[FUNGUS] is not transitions[FUNGUS]

    for agent in (PLANT, FUNGUS):
        transition = transitions[agent]
        assert transition.operational_at_start
        assert transition.operational_at_end
        assert transition.allocation_executed
        assert transition.trade_executed
        assert not transition.truncated
        assert transition.requested_action == pytest.approx(actions[agent])
        assert transition.realised_action == pytest.approx(actions[agent])
        assert transition.final_observation == pytest.approx(observations[agent])


@pytest.mark.parametrize(
    ("consumer_mode", "operational_agent", "absent_agent"),
    [
        ("plant-only", PLANT, FUNGUS),
        ("fungus-only", FUNGUS, PLANT),
    ],
)
def test_single_consumer_step_preserves_fixed_transition_mapping(
    species, config, consumer_mode, operational_agent, absent_agent
):
    env = BaseMycorMarl(
        config=config.replace(consumer_mode=consumer_mode),
        species=species,
        max_episode_steps=3,
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = reserve_actions(plant_trade=0.25, fungus_trade=0.5)

    _, _, _, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)

    transitions = info["transitions"]
    assert set(transitions) == {PLANT, FUNGUS}
    assert transitions[operational_agent].operational_at_start
    assert transitions[operational_agent].operational_at_end
    assert transitions[operational_agent].allocation_executed
    assert not transitions[operational_agent].trade_executed
    assert not transitions[absent_agent].operational_at_start
    assert not transitions[absent_agent].operational_at_end
    assert not transitions[absent_agent].allocation_executed
    assert not transitions[absent_agent].trade_executed
    assert transitions[absent_agent].realised_action == pytest.approx(
        jnp.zeros(4)
    )


def test_maintenance_death_differs_from_dead_padding_and_preserves_survivor_allocation(
    env
):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        plant_biomass=jnp.array([1.0]),
        plant_history_max_biomass=jnp.array([1.0]),
        plant_c_pool=jnp.array([0.0]),
        plant_p_pool=jnp.array([0.0]),
    )
    env.species = env.species.replace(
        plant=env.species.plant.replace(
            kappa_c=1.8,
            death_fraction=0.2,
        )
    )
    actions = {
        PLANT: physical_action(1.0, 1.0, 0.0, 0.0),
        FUNGUS: physical_action(0.5, 1.0, 0.0, 0.0),
    }

    _, dead_state, _, dones, info = env.step_env(
        jax.random.PRNGKey(1), state, actions
    )

    death = info["transitions"]
    assert dones[PLANT]
    assert not dones["__all__"]
    assert death[PLANT].operational_at_start
    assert not death[PLANT].operational_at_end
    assert not death[PLANT].allocation_executed
    assert not death[PLANT].trade_executed
    assert death[PLANT].realised_action == pytest.approx(jnp.zeros(4))
    assert death[FUNGUS].operational_at_start
    assert death[FUNGUS].operational_at_end
    assert death[FUNGUS].allocation_executed
    assert not death[FUNGUS].trade_executed
    assert death[FUNGUS].realised_action == pytest.approx(
        jnp.array([0.0, 1.0, 0.0, 0.0])
    )

    _, _, _, _, padding_info = env.step_env(
        jax.random.PRNGKey(2), dead_state, actions
    )
    padding = padding_info["transitions"][PLANT]
    survivor = padding_info["transitions"][FUNGUS]
    assert not padding.operational_at_start
    assert not padding.operational_at_end
    assert not padding.allocation_executed
    assert not padding.trade_executed
    assert survivor.operational_at_start
    assert survivor.operational_at_end
    assert survivor.allocation_executed
    assert not survivor.trade_executed


def test_both_maintenance_deaths_are_biological_not_administrative(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        plant_biomass=jnp.array([1.0]),
        fungus_biomass=jnp.array([1.0]),
        plant_history_max_biomass=jnp.array([1.0]),
        fungus_history_max_biomass=jnp.array([1.0]),
        plant_c_pool=jnp.array([0.0]),
        plant_p_pool=jnp.array([0.0]),
        fungus_c_pool=jnp.array([0.0]),
        fungus_p_pool=jnp.array([0.0]),
    )
    env.species = env.species.replace(
        plant=env.species.plant.replace(kappa_c=1.8, death_fraction=0.2),
        fungus=env.species.fungus.replace(kappa_c=1.8, death_fraction=0.2),
    )

    _, _, _, dones, info = env.step_env(
        jax.random.PRNGKey(1), state, reserve_actions()
    )

    assert dones[PLANT]
    assert dones[FUNGUS]
    assert dones["__all__"]
    for transition in info["transitions"].values():
        assert transition.operational_at_start
        assert not transition.operational_at_end
        assert not transition.allocation_executed
        assert not transition.trade_executed
        assert not transition.truncated


def test_auto_reset_returns_reset_observations_but_preserves_final_observations(
    species, config
):
    env = BaseMycorMarl(config=config, species=species, max_episode_steps=1)
    initial_observations, state = env.reset(jax.random.PRNGKey(0))
    actions = reserve_actions(plant_trade=0.5, fungus_trade=0.5)
    expected_final, _, _, _, _ = env.step_env(
        jax.random.PRNGKey(1), state, actions
    )

    observations, reset_state, _, dones, info = env.step(
        jax.random.PRNGKey(2), state, actions
    )

    assert dones["__all__"]
    assert reset_state.step == 0
    for agent in (PLANT, FUNGUS):
        transition = info["transitions"][agent]
        assert transition.truncated
        assert observations[agent] == pytest.approx(initial_observations[agent])
        assert transition.final_observation == pytest.approx(
            expected_final[agent]
        )
        assert not jnp.allclose(
            transition.final_observation,
            observations[agent],
        )
    assert (
        info["transitions"][PLANT].truncated
        == info["transitions"][FUNGUS].truncated
    )


def test_transition_contract_is_stable_under_jit_and_vectorisation(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = reserve_actions(plant_trade=0.25, fungus_trade=0.5)

    _, _, _, _, jitted_info = jax.jit(env.step_env)(
        jax.random.PRNGKey(1), state, actions
    )
    assert isinstance(jitted_info["transitions"][PLANT], Transition)
    assert jitted_info["transitions"][PLANT].requested_action.shape == (4,)
    assert jitted_info["transitions"][PLANT].final_observation.shape == (5,)

    keys = jax.random.split(jax.random.PRNGKey(2), 2)
    states = jax.tree.map(lambda value: jnp.stack([value, value]), state)
    batched_actions = jax.tree.map(
        lambda value: jnp.stack([value, value]), actions
    )
    vectorised_step = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0)))

    _, _, _, _, batched_info = vectorised_step(
        keys, states, batched_actions
    )

    for transition in batched_info["transitions"].values():
        assert isinstance(transition, Transition)
        assert transition.operational_at_start.shape == (2,)
        assert transition.requested_action.shape == (2, 4)
        assert transition.final_observation.shape == (2, 5)
        assert transition.truncated.shape == (2,)
