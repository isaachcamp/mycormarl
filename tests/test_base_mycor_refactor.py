import jax
import jax.numpy as jnp
import pytest

from mycormarl.environments import base_mycor as env_mod
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


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
        norm_obs=False,
    )


@pytest.fixture()
def env(species, config):
    return BaseMycorMarl(config=config, species=species, max_episode_steps=3)


@pytest.fixture(autouse=True)
def simple_dynamics(monkeypatch):
    def zero_density(biomass, traits, r_edges, z_edges):
        return jnp.zeros((r_edges.shape[0] - 1, z_edges.shape[0] - 1))

    def no_soil_uptake(state, *soil_args):
        """Keep behavioural unit tests isolated from the soil transaction."""
        return state

    def pool_specific_growth(allocated_c, allocated_p, grow_c_cost, grow_p_cost, grow_type):
        return jnp.minimum(
            allocated_c / grow_c_cost,
            allocated_p / grow_p_cost,
        )

    monkeypatch.setattr(env_mod.plant, "density_field_from_biomass", zero_density)
    monkeypatch.setattr(env_mod.fungus, "density_field_from_biomass", zero_density)
    monkeypatch.setattr(env_mod, "evolve_soil_p", no_soil_uptake)
    monkeypatch.setattr(env_mod, "grow", pool_specific_growth)


def test_reset_uses_zero_trade_observations(env):
    obs, state = env.reset(jax.random.PRNGKey(0))

    assert state.step == 0
    assert obs[PLANT].shape == env.observation_spaces[PLANT].shape == (4,)
    assert obs[FUNGUS].shape == env.observation_spaces[FUNGUS].shape == (4,)
    assert obs[PLANT][-1] == 0.0
    assert obs[FUNGUS][-1] == 0.0


def test_step_trades_from_starting_pools_and_exposes_trade_in_obs(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        PLANT: jnp.array([1.0, 0.0, 0.0, 0.0]),
        FUNGUS: jnp.array([1.0, 0.0, 0.0, 0.0]),
    }

    obs, next_state, rewards, dones, infos = env.step_env(jax.random.PRNGKey(1), state, actions)

    expected_plant_c_fixed = 100.0
    assert next_state.plant_c_pool[0] == pytest.approx(expected_plant_c_fixed)
    assert next_state.plant_p_pool[0] == pytest.approx(26.0)
    assert next_state.fungus_c_pool[0] == pytest.approx(32.0)
    assert next_state.fungus_p_pool[0] == pytest.approx(0.0)
    assert obs[PLANT][-1] == pytest.approx(16.0)
    assert obs[FUNGUS][-1] == pytest.approx(20.0)
    assert rewards[PLANT] == pytest.approx(0.0)
    assert rewards[FUNGUS] == pytest.approx(0.0)
    assert dones["__all__"] == jnp.array(False)
    assert "growth" in infos[PLANT]
    assert "growth" in infos[FUNGUS]


def test_newly_fixed_carbon_is_not_available_for_same_step_growth(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        PLANT: jnp.array([0.0, 1.0, 0.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    _, next_state, _, _, infos = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert infos[PLANT]["growth"][0] == pytest.approx(10.0)
    assert next_state.plant_biomass[0] == pytest.approx(20.0)
    assert next_state.plant_c_pool[0] == pytest.approx(200.0)
    assert next_state.plant_p_pool[0] == pytest.approx(0.0)


def test_plant_growth_at_biomass_cap_charges_only_realised_structure(env):
    """Growth clipped by the biomass cap must not consume unrepresented C/P."""
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        plant_biomass=jnp.array([99.0]),
        plant_history_max_biomass=jnp.array([99.0]),
        plant_c_pool=jnp.array([20.0]),
        plant_p_pool=jnp.array([10.0]),
    )
    actions = {
        PLANT: jnp.array([0.0, 1.0, 0.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    _, next_state, _, _, infos = env.step_env(
        jax.random.PRNGKey(1), state, actions
    )

    assert infos[PLANT]["growth"][0] == pytest.approx(1.0)
    assert next_state.plant_biomass[0] == pytest.approx(100.0)
    assert next_state.plant_p_pool[0] == pytest.approx(9.0)
    assert state.plant_p_pool[0] - next_state.plant_p_pool[0] == pytest.approx(
        (next_state.plant_biomass[0] - state.plant_biomass[0])
        * env.species.plant.gamma_p
    )


def test_dead_plant_cannot_resume_biology_while_fungus_remains_alive(env):
    """Per-agent death is sticky and masks growth, fixation, trade, and uptake."""
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
            kappa_p=0.0,
            death_fraction=0.2,
        )
    )
    maintenance_failure = {
        PLANT: jnp.array([0.0, 0.0, 0.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }
    _, dead_state, _, dones, _ = env.step_env(
        jax.random.PRNGKey(1), state, maintenance_failure
    )
    assert dones[PLANT]
    assert not dones["__all__"]

    dead_state = dead_state.replace(
        plant_c_pool=jnp.array([20.0]),
        plant_p_pool=jnp.array([10.0]),
    )
    env.species = env.species.replace(
        plant=env.species.plant.replace(kappa_c=0.0)
    )
    attempted_recovery = {
        PLANT: jnp.array([1.0, 1.0, 0.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }
    biomass_before = dead_state.plant_biomass.copy()
    c_before = dead_state.plant_c_pool.copy()
    p_before = dead_state.plant_p_pool.copy()
    _, next_state, rewards, dones, infos = env.step_env(
        jax.random.PRNGKey(2), dead_state, attempted_recovery
    )

    assert dones[PLANT]
    assert infos[PLANT]["growth"][0] == pytest.approx(0.0)
    assert rewards[PLANT] == pytest.approx(0.0)
    assert next_state.plant_biomass == pytest.approx(biomass_before)
    assert next_state.plant_c_pool == pytest.approx(c_before)
    assert next_state.plant_p_pool == pytest.approx(p_before)
    assert jnp.all(next_state.root_length_density == 0.0)


def test_incoming_trade_is_not_available_for_same_step_growth(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        PLANT: jnp.array([0.0, 1.0, 0.0, 0.0]),
        FUNGUS: jnp.array([0.5, 0.0, 0.5, 0.0]),
    }

    _, next_state, _, _, infos = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert infos[PLANT]["growth"][0] == pytest.approx(10.0)
    assert next_state.plant_p_pool[0] == pytest.approx(8.0)


def test_reproduction_spends_pools_and_returns_reward(env):
    """Exports reproduced P and records it in the cumulative plant diagnostic."""
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        PLANT: jnp.array([0.0, 0.0, 0.0, 1.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    _, next_state, rewards, _, infos = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert infos[PLANT]["reproduction_c"][0] == pytest.approx(20.0)
    assert infos[PLANT]["reproduction_p"][0] == pytest.approx(10.0)
    assert rewards[PLANT] == pytest.approx(10.0)
    assert next_state.plant_c_pool[0] == pytest.approx(100.0)
    assert next_state.plant_p_pool[0] == pytest.approx(0.0)
    assert next_state.cumulative_plant_p_reproduction_export_mg[0] == pytest.approx(
        10.0
    )


def test_fungal_reproduction_export_is_accumulated(env):
    """Records fungal reproductive P separately from trade and living pools."""
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        cumulative_fungus_p_reproduction_export_mg=jnp.array([2.0])
    )
    actions = {
        PLANT: jnp.array([0.0, 0.0, 1.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 0.0, 1.0]),
    }

    _, next_state, _, _, infos = env.step_env(
        jax.random.PRNGKey(1), state, actions
    )

    assert infos[FUNGUS]["reproduction_p"][0] == pytest.approx(16.0)
    assert next_state.cumulative_fungus_p_reproduction_export_mg[0] == pytest.approx(
        18.0
    )


def test_reproduction_reward_uses_scaled_cobb_douglas(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        plant_c_pool=jnp.array([16.0]),
        plant_p_pool=jnp.array([9.0]),
    )
    env.species = env.species.replace(
        plant=env.species.plant.replace(
            gamma_c=4.0,
            gamma_p=1.0,
        )
    )

    actions = {
        PLANT: jnp.array([0.0, 0.0, 0.0, 1.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    _, _, rewards, _, infos = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert infos[PLANT]["reproduction_c"][0] == pytest.approx(16.0)
    assert infos[PLANT]["reproduction_p"][0] == pytest.approx(9.0)
    assert rewards[PLANT] == pytest.approx(6.0)


def test_excess_maintenance_allocation_is_not_wasted(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        plant_biomass=jnp.array([10.0]),
        plant_c_pool=jnp.array([20.0]),
        plant_p_pool=jnp.array([10.0]),
    )
    env.species = env.species.replace(
        plant=env.species.plant.replace(kappa_c=0.1, kappa_p=0.05)
    )

    actions = {
        PLANT: jnp.array([0.0, 0.0, 1.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    _, next_state, _, _, infos = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert infos[PLANT]["maint_c"][0] == pytest.approx(1.0)
    assert infos[PLANT]["maint_p"][0] == pytest.approx(0.5)
    assert infos[PLANT]["maint_c_used"][0] == pytest.approx(1.0)
    assert infos[PLANT]["maint_p_used"][0] == pytest.approx(0.5)
    assert infos[PLANT]["c_deficit"][0] == pytest.approx(0.0)
    assert infos[PLANT]["p_deficit"][0] == pytest.approx(0.0)
    assert next_state.plant_c_pool[0] == pytest.approx(119.0)
    assert next_state.plant_p_pool[0] == pytest.approx(9.5)


def test_terminal_after_max_episode_steps(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        PLANT: jnp.array([0.0, 0.0, 1.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    for step in range(3):
        _, state, _, dones, _ = env.step_env(jax.random.PRNGKey(step), state, actions)

    assert state.step == 3
    assert dones["__all__"] == jnp.array(True)
