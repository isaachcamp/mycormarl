
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
        obs_size=4,
        action_size=5,
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


def test_constrain_allocations_sum_greater_than_one(test_env):
    # Create actions that exceed available sugars and phosphorus.
    actions = jnp.array([1.5, 1.0, 0., 0., 1.0])

    constrained_actions = test_env.constrain_allocation(actions)

    assert constrained_actions[0] == 1.0, "Phosphorus allocations sum to greater than one."
    assert constrained_actions[1:].sum() == 1.0, "Sugar allocations sum to greater than one."

    assert (constrained_actions == jnp.array([1., 0.5, 0., 0., 0.5])).all(), "Constrained actions not calculated correctly."

def test_constrain_allocations_sum_less_than_one(test_env):
    # Create actions that sum to less than one.
    actions = jnp.array([0.5, 0.2, 0., 0., 0.1])

    constrained_actions = test_env.constrain_allocation(actions)

    assert (constrained_actions == actions).all(), "Constrained actions should be unchanged when sums are less than one."

def test_allocate_resources_sum_less_than_one(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all resources to one agent.
    actions = jnp.array([0.5, 0.2, 0.2, 0.3, 0.3])

    allocated_actions = test_env.allocate_resources(state.agents[0], actions)

    assert allocated_actions[0] == 50.0, "Phosphorus allocation not calculated correctly."
    assert allocated_actions[1] == 10.0, "Sugar incorrectly allocated for trade."
    assert allocated_actions[2] == 10.0, "Sugar incorrectly allocated for growth."
    assert allocated_actions[3] == 15.0, "Sugar incorrectly allocated for maintenance."
    assert allocated_actions[4] == 15.0, "Sugar incorrectly allocated for reproduction."
    assert allocated_actions[1:].sum() == 50.0, "Sugar allocation not calculated correctly."

def test_allocate_resources_non_integer(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that are non-integer multiples of 0.1.
    actions = jnp.array([0.505, 0.25, 0.25, 0.25, 0.25])
    allocated_actions = test_env.allocate_resources(state.agents[0], actions)

    assert allocated_actions[0] == 50.0, "Non-integer phosphorus allocation not handled correctly."
    assert allocated_actions[1] == 12., "Non-integer sugar allocation not handled correctly."
    assert allocated_actions[2] == 12., "Non-integer sugar allocation not handled correctly."
    assert allocated_actions[3] == 12., "Non-integer sugar allocation not handled correctly."
    assert allocated_actions[4] == 12., "Non-integer sugar allocation not handled correctly."

    assert allocated_actions[1:].sum() == 48., "Non-integer sugar allocation not handled correctly."

def test_biomass_update_pos_val(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all sugars to growth.
    actions = {agent: jnp.array([0., 0., 1., 0., 0.]) for agent in test_env.agents}
    obs, state, _, _, _ = test_env.step_env(key, state, actions)

    # Both grow at same rate – equivalent efficiencies.
    growth_allocation = 50.0
    expected_biomass = (growth_allocation / test_env.growth_cost) + 0.1  # Add initial biomass of 0.1

    assert state.agents[0].biomass ==  expected_biomass, "Biomass incorrectly updated for plant."
    assert state.agents[1].biomass == expected_biomass, "Biomass incorrectly updated for fungus."

    assert obs["agent_0"][1] == expected_biomass, "Biomass observation incorrect for plant."
    assert obs["agent_1"][1] == expected_biomass, "Biomass observation incorrect for fungus."

def test_biomass_update_zero_val(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all sugars to growth.
    actions = {agent: jnp.array([0., 0., 0., 0., 0.]) for agent in test_env.agents}
    obs, state, _, _, _ = test_env.step_env(key, state, actions)

    # Both grow at same rate – equivalent efficiencies.
    expected_biomass = 0.1  # Initial biomass of 0.1

    assert state.agents[0].biomass == expected_biomass, "Biomass incorrectly updated for plant."
    assert state.agents[1].biomass == expected_biomass, "Biomass incorrectly updated for fungus."

    assert obs["agent_0"][1] == expected_biomass, "Biomass observation incorrect for plant."
    assert obs["agent_1"][1] == expected_biomass, "Biomass observation incorrect for fungus."

def test_biomass_update_all_vals(test_env):
    """Test no other updates interfere with biomass update."""
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all sugars to growth and all phosphorus to uptake.
    actions = {agent: jnp.array([0.5, 0.3, 0.3, 0.2, 0.2]) for agent in test_env.agents}
    obs, state, _, _, _ = test_env.step_env(key, state, actions)

    # Both grow at same rate – equivalent efficiencies.
    growth_allocation = 15.0
    expected_biomass = (growth_allocation / test_env.growth_cost) + 0.1  # Add initial biomass of 0.1

    assert state.agents[0].biomass == expected_biomass, "Biomass incorrectly updated for plant."
    assert state.agents[1].biomass == expected_biomass, "Biomass incorrectly updated for fungus."

    assert obs["agent_0"][1] == expected_biomass, "Biomass observation incorrect for plant."
    assert obs["agent_1"][1] == expected_biomass, "Biomass observation incorrect for fungus."

def test_sugar_generation_pos_val(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all phosphorus to uptake and no sugars.
    actions = {agent: jnp.array([0., 0., 0., 0., 0.]) for agent in test_env.agents}
    _, state, _, _, _ = test_env.step_env(key, state, actions)

    p_use = 100.0 // test_env.p_cost_per_sugar
    expected_sugar_gen = test_env.max_sugar_gen_rate * 0.1 * p_use  # Initial biomass of 0.1

    assert state.agents[0].sugars == expected_sugar_gen + 50., "Sugar generation incorrectly calculated for plant."

def test_sugar_generation_no_P(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all phosphorus to uptake and no sugars.
    actions = {agent: jnp.array([0., 0., 0., 0., 0.]) for agent in test_env.agents}

    with jdc.copy_and_mutate(state) as state:
        state.agents[0].phosphorus = jnp.array(0.)

    _, state, _, _, _ = test_env.step_env(key, state, actions)

    expected_sugar_gen = 0.0

    assert state.agents[0].sugars == expected_sugar_gen + 50., "Sugar generation incorrectly calculated for plant with zero phosphorus."

def test_sugar_generation_trade_all_P(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all phosphorus to uptake and no sugars.
    actions = {agent: jnp.array([1., 0., 0., 0., 0.]) for agent in test_env.agents}

    with jdc.copy_and_mutate(state) as state:
        state.agents[0].phosphorus = jnp.array(0.)

    _, state, _, _, _ = test_env.step_env(key, state, actions)

    expected_sugar_gen = 0.0

    assert state.agents[0].sugars == expected_sugar_gen + 50., "Sugar generation incorrectly calculated for plant with zero phosphorus."

def test_health_update_full_maintenance(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Growth cost * maintenance cost ratio * initial biomass
    required_sugars = 100.0 * 0.1 * test_env.maintenance_cost_ratio
    required_maintenance = required_sugars / 50.0  # Initial sugars of 50.

    # Create actions that allocate all sugars to maintenance.
    actions = {agent: jnp.array([0., 0., 0., required_maintenance, 0.]) for agent in test_env.agents}
    _, state, _, _, _ = test_env.step_env(key, state, actions)

    expected_health = 100.0  # should be the same as initial health

    assert state.agents[0].health == expected_health, "Health incorrectly updated for plant."
    assert state.agents[1].health == expected_health, "Health incorrectly updated for fungus."

def test_health_update_no_maintenance(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Growth cost * maintenance cost ratio * initial biomass
    required_sugars = 100.0 * 0.1 * test_env.maintenance_cost_ratio

    # Create actions that allocate no sugars to maintenance.
    actions = {agent: jnp.array([0., 0., 0., 0., 0.]) for agent in test_env.agents}
    _, state, _, _, _ = test_env.step_env(key, state, actions)

    expected_health = 100.0 - required_sugars  # Health should decrease by the amount of maintenance deficit.

    assert state.agents[0].health == expected_health, "Health incorrectly updated for plant with no maintenance."
    assert state.agents[1].health == expected_health, "Health incorrectly updated for fungus with no maintenance."

def test_health_update_partial_maintenance(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Growth cost * maintenance cost ratio * initial biomass
    required_sugars = 100.0 * 0.1 * test_env.maintenance_cost_ratio # 5 sugars
    required_maintenance = required_sugars / 50.0  # Initial sugars of 50.

    # Create actions that allocate half of required sugars to maintenance.
    actions = {agent: jnp.array([0., 0., 0., required_maintenance / 2, 0.]) for agent in test_env.agents}
    _, state, _, _, _ = test_env.step_env(key, state, actions)

    # Sugars allocated discretely, so allocated sugars here is 2 –> deficit is 3.
    expected_health = 100.0 - 3.  # Health should decrease by deficit.

    assert state.agents[0].health == expected_health, "Health should decrease by 3."
    assert state.agents[1].health == expected_health, "Health should decrease by 3."

def test_health_update_excess_maintenance(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Allocate 50 sugars to maintenance – more than required.
    actions = {agent: jnp.array([0., 0., 0., 1., 0.]) for agent in test_env.agents}
    _, state, _, _, _ = test_env.step_env(key, state, actions)

    expected_health = 100.0  # Health should not increase above initial value.

    assert state.agents[0].health <= expected_health, "Health cannot be greater than 100."
    assert state.agents[1].health <= expected_health, "Health cannot be greater than 100."

    assert state.agents[0].sugars == 33., "Sugars remaining should only be those generated this step for plant."
    assert state.agents[1].sugars == 0., "All sugars should be allocated to maintenance for fungus."

def test_health_recovery(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # First step with no maintenance to reduce health.
    actions = {agent: jnp.array([0., 0., 0., 0., 0.]) for agent in test_env.agents}
    _, state, _, _, _ = test_env.step_env(key, state, actions)

    # Then step with excess maintenance to recover health.
    # Allcates 10 sugars, 5 for maintenance and 5 for recovery back to 100 health.
    actions = {agent: jnp.array([0., 0., 0., 0.2, 0.]) for agent in test_env.agents}
    _, state, _, _, _ = test_env.step_env(key, state, actions)

    expected_health = 100.0  # Health should recover back to initial value.

    assert state.agents[0].health == expected_health, "Health should recover back to 100."
    assert state.agents[1].health == expected_health, "Health should recover back to 100."

def test_reproduction_whole_prop(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all sugars to reproduction.
    actions = {agent: jnp.array([0., 0., 0., 0., 1.]) for agent in test_env.agents}
    _, state, _, _, info = test_env.step_env(key, state, actions)

    assert info["agent_0"]["props_generated"] == 1.0, "Propagules generated incorrect for plant."
    assert info["agent_1"]["props_generated"] == 1.0, "Propagules generated incorrect for fungus."


def test_reproduction_greater_than_one_prop(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Expected props with 130 sugars is 2, with 30 sugars remaining.
    with jdc.copy_and_mutate(state) as state:
        state.agents[0].sugars = jnp.array(130.)

    # Create actions that allocate all sugars to reproduction.
    actions = {agent: jnp.array([0., 0., 0., 0., 1.]) for agent in test_env.agents}
    _, state, _, _, info = test_env.step_env(key, state, actions)

    assert info["agent_0"]["props_generated"] == 2.0, "Propagules generated incorrect for plant."
    assert info["agent_1"]["props_generated"] == 1.0, "Propagules generated incorrect for fungus."

    # 33 sugars generated from initial P.
    assert state.agents[0].sugars == 30. + 33., "Sugars incorrectly subtracted for plant."
    assert state.agents[1].sugars == 0., "Sugars incorrectly subtracted for fungus."

def test_reproduction_part_prop(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all sugars to reproduction.
    actions = {agent: jnp.array([0., 0., 0., 0., 0.5]) for agent in test_env.agents}
    _, state, _, _, info = test_env.step_env(key, state, actions)

    assert info["agent_0"]["props_generated"] == 0., "Propagules generated incorrect for plant."
    assert info["agent_1"]["props_generated"] == 0., "Propagules generated incorrect for fungus."

    # 33 sugars generated from initial P.
    assert state.agents[0].sugars == 50. + 33., "Sugars incorrectly subtracted for plant."
    assert state.agents[1].sugars == 50., "Sugars incorrectly subtracted for fungus."

def test_reproduction_zero_val(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all sugars to reproduction.
    actions = {agent: jnp.array([0., 0., 0., 0., 0.0]) for agent in test_env.agents}
    _, state, _, _, info = test_env.step_env(key, state, actions)

    assert info["agent_0"]["props_generated"] == 0., "Propagules generated incorrect for plant."
    assert info["agent_1"]["props_generated"] == 0., "Propagules generated incorrect for fungus."

    # 33 sugars generated from initial P.
    assert state.agents[0].sugars == 50. + 33., "Sugars incorrectly subtracted for plant."
    assert state.agents[1].sugars == 50., "Sugars incorrectly subtracted for fungus."

def test_rewards(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Assign all initial 50 sugars to reproduction – should yield a reward of 1.5.
    actions = {agent: jnp.array([0., 0., 0., 0., 1.]) for agent in test_env.agents}

    _, state, rewards, _, info = test_env.step(key, state, actions)

    assert isinstance(rewards, dict), "Rewards should be a dictionary."
    assert set(rewards.keys()) == set(test_env.agents), "Reward keys should match agent keys."
    assert all(isinstance(r, jnp.ndarray) for r in rewards.values()), "Rewards should be JAX arrays."

    assert rewards["agent_0"] == 1.5, "Reward for agent 0 not calculated correctly."
    assert rewards["agent_1"] == 1.5, "Reward for agent 1 not calculated correctly."

def test_all_state_vars_full_vals_set(test_env):
    key = jax.random.PRNGKey(0)
    _, state = test_env.reset(key)

    # Create actions that allocate all sugars to reproduction.
    actions = {agent: jnp.array([0.5, 0.2, 0.3, 0.1, 0.4]) for agent in test_env.agents}
    _, state, _, _, info = test_env.step_env(key, state, actions)

    assert state.agents[0].biomass == 0.1 + (15. / test_env.growth_cost), "Biomass incorrectly updated for plant."
    assert state.agents[1].biomass == 0.1 + (15. / test_env.growth_cost), "Biomass incorrectly updated for fungus."

    # Allocated 5 sugars to maintenance – enough to cover maintenance cost.
    assert state.agents[0].health == 100.0, "Health incorrectly updated for plant."
    assert state.agents[1].health == 100.0, "Health incorrectly updated for fungus."

    # Plant should use (50 // 3) * 3 = 48 P for sugar generation, not acquire any P,
    # and receive 50 from trade.
    assert state.agents[0].phosphorus == 50.0 + 2., "Phosphorus incorrectly updated for plant."
    # Fungus gives 50 P away, receives 50 P from trade and acquires 3 P from environment.
    p_acquired = test_env.p_uptake_max_rate * test_env.fungus_p_uptake_efficiency * 0.1  # Initial biomass of 0.1
    assert state.agents[1].phosphorus == 50.0 + 50.0 + p_acquired, "Phosphorus incorrectly updated for fungus."

    assert info["agent_0"]["props_generated"] == 0.0, "No propagules should be generated for plant."
    assert info["agent_1"]["props_generated"] == 0.0, "No propagules should be generated for fungus."

    # For plant, 16 sugars generated from initial P (50 P traded away),
    #           minus 5 for maintenance,
    #           minus 15 for growth,
    #           minus 10 for trade given, plus 10 for trade received.

    # For fungus, minus 5 for maintenance,
    #            minus 15 for growth,
    #            minus 10 for trade given, plus 10 for trade received.
    assert state.agents[0].sugars == 50. + 16. - 5. - 15. - 10. + 10., "Sugars incorrectly updated for plant."
    assert state.agents[1].sugars == 50. - 5. - 15. - 10. + 10., "Sugars incorrectly updated for fungus."
