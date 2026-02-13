
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

def test_env_initialisation(test_env):
    key = jax.random.PRNGKey(0)

    assert test_env.growth_cost == jnp.array(100.), "Growth cost not initialized correctly."
    assert test_env.reproduction_cost == jnp.array(50.), "Reproduction cost not initialized correctly."
    assert test_env.maintenance_cost_ratio == jnp.array(0.5), "Maintenance cost not initialized correctly."

    assert test_env.p_uptake_max_rate == jnp.array(30.), "P uptake max rate not initialized correctly."
    assert test_env.fungus_p_uptake_efficiency == jnp.array(1.0), "Fungus P uptake efficiency not initialized correctly."
    assert test_env.plant_p_uptake_efficiency == jnp.array(0.0), "Plant P uptake efficiency not initialized correctly."

    assert test_env.max_sugar_gen_rate == jnp.array(10.), "Max sugar generation rate not initialized correctly."
    assert test_env.p_cost_per_sugar == jnp.array(3.), "P cost per sugar not initialized correctly."
    assert test_env.trade_flow_constant == jnp.array(100.), "Trade flow constant not initialized correctly."

    assert test_env.num_agents == 2, "Number of agents not initialized correctly."
    assert test_env.agents == ["agent_0", "agent_1"], "Agent names not initialized correctly."
    assert test_env.agent_types == [0, 1], "Agent types not initialized correctly."

    assert test_env.observation_spaces["agent_0"].shape == (6,), "Observation space shape not initialized correctly."
    assert test_env.action_spaces["agent_0"].shape == (5,), "Action space size not initialized correctly."

    assert test_env.observation_spaces["agent_1"].shape == (6,), "Observation space shape not initialized correctly."
    assert test_env.action_spaces["agent_1"].shape == (5,), "Action space size not initialized correctly."

    assert test_env.max_episode_steps == 5, "Max episode steps not initialized correctly."

def test_env_reset(test_env):
    key = jax.random.PRNGKey(0)
    obs, state = test_env.reset(key)

    assert state.step == 0, "Environment did not reset step counter correctly."
    assert len(state.agents) == 2, "Environment did not initialize correct number of agents on reset."
    assert state.agents[0].species_id == 0, "Agent 0 species ID not initialized correctly on reset."
    assert state.agents[1].species_id == 1, "Agent 1 species ID not initialized correctly on reset."

    assert state.adj.shape == (2, 2), "Adjacency matrix not initialized correctly on reset."
    assert state.s_trades.shape == (2, 2), "Sugar trade matrix not initialized correctly on reset."
    assert state.p_trades.shape == (2, 2), "Phosphorus trade matrix not initialized correctly on reset."

    assert obs["agent_0"].shape == (6,), "Observation for agent 0 not initialized correctly on reset."
    assert obs["agent_1"].shape == (6,), "Observation for agent 1 not initialized correctly on reset."

    assert (obs["agent_0"] == jnp.array([100.0, 0.1, 100.0, 50.0, 0.0, 0.0])).all(), "Observations incorrect upon reset."
    assert (obs["agent_1"] == jnp.array([100.0, 0.1, 100.0, 50.0, 0.0, 0.0])).all(), "Observations incorrect upon reset."

    assert not state.terminal, "Terminal state incorrectly reset."

def test_increment_step(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Sample random actions.
    actions = {agent: test_env.action_spaces[agent].sample(key) for agent in test_env.agents}

    _, state, _, _, _ = test_env.step(key, state, actions)

    assert state.step == 1, "Environment did not step correctly."

def test_zero_trade_matrices_after_step(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Sample random actions.
    actions = {agent: test_env.action_spaces[agent].sample(key) for agent in test_env.agents}

    _, state, _, _, _ = test_env.step(key, state, actions)

    assert (state.s_trades == jnp.zeros((2, 2))).all(), "s_trade matrix not reset to zero during step."
    assert (state.p_trades == jnp.zeros((2, 2))).all(), "p_trade matrix not reset to zero during step."

def test_env_max_steps_reset(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    done = {"__all__": False}
    step = 0
    while not done["__all__"]:
        key, action_key = jax.random.split(key)
        actions = {agent: test_env.action_spaces[agent].sample(action_key) for agent in test_env.agents}

        _, state, _, done, _ = test_env.step(key, state, actions)
        step += 1

    # Once max steps reached, environment should be reset.
    assert state.step == 0, "Environment did not end after max episode steps."
    assert not state.terminal, "Terminal state not reset after max episode steps."

def test_get_obs(test_env):
    key = jax.random.PRNGKey(0)
    obs, _ = test_env.reset(key)

    assert set(test_env.agents) == set(obs.keys()), "Observation keys do not match agent keys."

    assert obs["agent_0"].shape == (6,), "Observation for agent 0 not initialized correctly on reset."
    assert obs["agent_1"].shape == (6,), "Observation for agent 1 not initialized correctly on reset."

    assert (obs["agent_0"] == jnp.array([100.0, 0.1, 100.0, 50.0, 0.0, 0.0])).all(), "Observations incorrect upon reset."
    assert (obs["agent_1"] == jnp.array([100.0, 0.1, 100.0, 50.0, 0.0, 0.0])).all(), "Observations incorrect upon reset."

def test_is_terminal_manual(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Manually set terminal to True and check.
    state = jdc.replace(state, terminal=True)
    assert test_env.is_terminal(state), "Environment did not recognize terminal state."

def test_is_terminal_max_steps(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    state = jdc.replace(state, step=test_env.max_episode_steps)

    assert test_env.is_terminal(state), "Environment did not recognize terminal state after max episode steps."

def test_is_terminal_one_agent_dead(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Manually set one agent's health to 0 and check.
    with jdc.copy_and_mutate(state) as state:
        state.agents[0].health = jnp.array(0.0)

    assert not test_env.is_terminal(state), "Environment terminated after one agent's death."

def test_is_terminal_all_agents_dead(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Manually set one agent's health to 0 and check.
    with jdc.copy_and_mutate(state) as state:
        state.agents[0].health = jnp.array(0.0)
        state.agents[1].health = jnp.array(0.0)

    assert test_env.is_terminal(state), "Environment did not recognize terminal state after all agents' death."

def test_is_terminal_no_terminal_conditions(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    assert not test_env.is_terminal(state), "Terminal state recognised when no conditions met."

def test_create_adj_matrix_fc_interclass(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    assert state.adj.shape == (2, 2), "Adjacency matrix not the correct shape."
    assert (state.adj == jnp.array(
        [[0., 1.],
         [1., 0.]]
    )).all(), "Adjacency matrix not initialized correctly on reset."
