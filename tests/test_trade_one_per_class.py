
import pytest

from omegaconf import OmegaConf, DictConfig
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from mycormarl.environments.base_mycor import BaseMycorMarl

# Create a test environment to verify that the environment can be initialized and stepped through without errors.
@pytest.fixture(scope="session")
def cfg():
    return OmegaConf.load("tests/test_params.yaml")

@pytest.fixture(scope="session")
def test_env(cfg: DictConfig):
    env = BaseMycorMarl(
        num_agents=2,
        agent_types={"plant": 1, "fungus": 1},
        growth_cost=cfg.control_params.GROWTH_COST,
        reproduction_cost=cfg.control_params.REPRODUCTION_COST,
        maintenance_cost_ratio=cfg.control_params.MAINTENANCE_COST_RATIO,
        p_uptake_max_rate=cfg.control_params.P_UPTAKE_MAX_RATE,
        fungus_p_uptake_efficiency=cfg.control_params.FUNGUS_P_UPTAKE_EFFICIENCY,
        plant_p_uptake_efficiency=cfg.control_params.PLANT_P_UPTAKE_EFFICIENCY,
        max_sugar_gen_rate=cfg.control_params.MAX_SUGAR_GEN_RATE,
        p_cost_per_sugar=cfg.control_params.P_COST_PER_SUGAR,
        trade_flow_constant=cfg.control_params.TRADE_FLOW_CONSTANT,
        max_episode_steps=5,
    )
    return env

def test_update_trade_flows_pos_val(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    p_trade = 10
    s_trade = 10

    state = test_env.update_trade_flows(state, 0, p_trade, s_trade)
    state = test_env.update_trade_flows(state, 1, p_trade, s_trade)

    assert (state.s_trades == jnp.array(
        [[0., 10.],
         [10., 0.]])
    ).all(), "Sugar trade flows not updated correctly."

    assert (state.p_trades == jnp.array(
        [[0., 10.],
         [10., 0.]]
    )).all(), "Phosphorus trade flows not updated correctly."

def test_update_trade_flows_zero_val(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    p_trade = 0
    s_trade = 0

    state = test_env.update_trade_flows(state, 0, p_trade, s_trade)
    state = test_env.update_trade_flows(state, 1, p_trade, s_trade)

    assert (state.s_trades == jnp.array(
        [[0., 0.],
         [0., 0.]])
    ).all(), "Sugar trade flows not updated correctly."

    assert (state.p_trades == jnp.array(
        [[0., 0.],
         [0., 0.]]
    )).all(), "Phosphorus trade flows not updated correctly."

def test_update_trade_flows_zero_connections(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Set adjacency matrix to zero to simulate no connections.
    state = jdc.replace(state, adj=jnp.zeros_like(state.adj))

    p_trade = 10
    s_trade = 10

    state = test_env.update_trade_flows(state, 0, p_trade, s_trade)
    state = test_env.update_trade_flows(state, 1, p_trade, s_trade)

    assert (state.s_trades == jnp.array(
        [[0., 0.],
         [0., 0.]])
    ).all(), "Sugar trade flows should be zero when there are no connections."

    assert (state.p_trades == jnp.array(
        [[0., 0.],
         [0., 0.]]
    )).all(), "Phosphorus trade flows should be zero when there are no connections."

def test_step_trade_pos_val(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions to trade all sugars and all phosphorus.
    actions = jnp.array([1., 1., 0., 0., 0.])
    constrained_actions = test_env.constrain_allocation(actions)
    allocated_actions = test_env.allocate_resources(state.agents[0], constrained_actions)

    p_trade = 100.
    s_trade = 50.

    state, _, _, _ = test_env.step_agent(key, 0, state, allocated_actions)
    state, _, _, _ = test_env.step_agent(key, 1, state, allocated_actions)

    # Trades subtracted from each agent before processing incoming trades, i.e.,
    # traded resources are only available next step.
    assert state.agents[0].phosphorus == 0., "Traded P incorrectly subtracted."
    assert state.agents[0].sugars == 0., "Traded Sugars incorrectly subtracted."

    # Fungus agent acquires 3 P from environment at 0.1 biomass.
    assert state.agents[1].phosphorus == 3., "Traded P incorrectly subtracted."
    assert state.agents[1].sugars == 0., "Traded Sugars incorrectly subtracted."

    state = test_env.step_trade(state, 0)
    state = test_env.step_trade(state, 1)

    assert state.agents[0].phosphorus == p_trade, "Traded P incorrectly added."
    assert state.agents[0].sugars == s_trade, "Traded Sugars incorrectly added."
    assert state.agents[1].phosphorus == p_trade + 3, "Traded P incorrectly added."
    assert state.agents[1].sugars == s_trade, "Traded Sugars incorrectly added."
