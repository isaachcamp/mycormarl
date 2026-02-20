

from enum import IntEnum
from typing import Dict, List, Tuple
from functools import partial

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from mycormarl.agents.agent import AgentState


# TODO: add AgentState inits with configurable parameters (e.g. initial health, biomass, etc.)
# TODO: consider more complex allocation strategies based on past allocations.
# TODO: decouple constraints from the environment step function.


class AgentType(IntEnum):
    PLANT = 0
    FUNGUS = 1


@jdc.pytree_dataclass
class State:
    """
    Represents the full environment state with two AgentState instances for
    the Plant and Fungus.
    """
    agents: List[AgentState]
    adj: jax.Array # (n_agents, n_agents) adjacency matrix representing mycorrhizal network.
    s_trades: jax.Array # (n_agents, n_agents) Sugars traded by each agent in the current step
    p_trades: jax.Array # (n_agents, n_agents) Phosphorus traded by each agent in the current step
    step: jax.Array # To track episode length
    terminal: bool  # Flag to indicate if the episode is done

    # Solar irradiance for sugar production, can be modified for different scenarios
    # solar_irradiance: jax.Array = jdc.field(default_factory=lambda: jnp.array(400.0))


class BaseMycorMarl(MultiAgentEnv):
    def __init__(
        self,
        agent_types: Dict[str, int],
        growth_cost: float = 100.0,
        reproduction_cost: float = 50.0,
        maintenance_cost_ratio: float = 0.5,
        p_uptake_max_rate: float = 300.0,
        # p_availability: jnp.float32 = 1.0,
        fungus_p_uptake_efficiency: float = 1.0,
        plant_p_uptake_efficiency: float = 0.0,
        max_sugar_gen_rate: float = 100.0,
        p_cost_per_sugar: float = 3.0,
        trade_flow_constant: float = 100.0,
        max_episode_steps: int = 1000,
    ):
        self.num_agents: int = sum(agent_types.values())
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_types: List[AgentType] = [AgentType.PLANT] * agent_types.get("plant", 0) + \
                                            [AgentType.FUNGUS] * agent_types.get("fungus", 0)

        assert len(self.agent_types) == self.num_agents,\
            "Sum of agent types must equal num_agents"

        self.growth_cost = growth_cost
        self.reproduction_cost = reproduction_cost
        self.maintenance_cost_ratio = maintenance_cost_ratio
        self.p_uptake_max_rate = p_uptake_max_rate
        # self.p_availability = p_availability
        self.fungus_p_uptake_efficiency = fungus_p_uptake_efficiency
        self.plant_p_uptake_efficiency = plant_p_uptake_efficiency
        self.max_sugar_gen_rate = max_sugar_gen_rate
        self.p_cost_per_sugar = p_cost_per_sugar
        # self.trade_flow_constant = trade_flow_constant # Unused for now, can be used to scale the effect of trades on agent states in future iterations.

        obs_size = 4 + (self.num_agents - 1) * 2 # Add space for incoming trade flows in observations.
        action_size = 5 # [p_trade, s_trade, growth, maintenance, reproduction]

        self.observation_spaces = {agent: self.agent_obs_space(obs_size) for agent in self.agents}
        self.action_spaces = {agent: self.agent_action_space(action_size) for agent in self.agents}

        # Environment parameters
        self.max_episode_steps = max_episode_steps

    def _get_obs(self, state: State) -> Dict[str, jax.Array]:
        obs = {}
        for i, agent in enumerate(self.agents):
            agent_state = state.agents[i]
            # Example observation: [health, biomass, phosphorus, sugars]
            obs_vector = jnp.array([
                agent_state.health,
                agent_state.biomass,
                agent_state.phosphorus,
                agent_state.sugars
            ])

            # Add received resource trades to observations, select all but self.
            s_trades = state.s_trades[:, i].flatten() # Sugars received from other agents
            s_trades = jnp.delete(s_trades, i) # Remove self-trade
            p_trades = state.p_trades[:, i].flatten() # Phosphorus received from other agents
            p_trades = jnp.delete(p_trades, i) # Remove self-trade (always 0)
            obs_vector = jnp.concatenate((obs_vector, p_trades, s_trades))

            assert obs_vector.shape[0] == self.observation_spaces[agent].shape[0], \
                "Observation size mismatch"

            obs[agent] = obs_vector
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array) -> Tuple[Dict[str, jax.Array], State]:

        agents = [
            AgentState(
                species_id=jnp.array(species),
                p_uptake_efficiency=jnp.array(
                    self.fungus_p_uptake_efficiency
                    if species == AgentType.FUNGUS
                    else self.plant_p_uptake_efficiency
                )
            ) for species in self.agent_types
        ]

        # Begin with agents of one class connected to all agents of the other class.
        adj = self.create_adj_matrix_fc_interclass(self.agent_types)

        state = State(
            agents=agents,
            adj=adj,
            s_trades=jnp.zeros((self.num_agents, self.num_agents)),
            p_trades=jnp.zeros((self.num_agents, self.num_agents)),
            step=jnp.array(0, dtype=jnp.int32),
            terminal=False
        )

        obs = self._get_obs(state)
        return obs, state

    def step_env(
            self,
            key: jax.Array,
            state: State,
            actions: Dict[str, jax.Array]
        ) -> Tuple[Dict[str, jax.Array], State, Dict[str, float], Dict[str, bool], Dict]:

        rewards = {}
        shaped_rewards = {}
        dones = {agent: jnp.array(False) for agent in self.agents}
        infos = {}

        for i, agent in enumerate(self.agents):
            # Process and constrain actions for each agent.
            actions[agent] = self.constrain_allocation(actions[agent])
            extra_info = {"constrained_actions": actions[agent]}
            actions[agent] = self.allocate_resources(state.agents[i], actions[agent])
            extra_info["allocated_actions"] = actions[agent]

            # Step each agent and update state.
            state, reward, shaped_rewards, info = self.step_agent(
                key, i, state, actions[agent]
            )

            # Set done flag to True for agent if dead.
            dones[agent] = self.check_agent_is_dead(state.agents[i])

            rewards[agent] = jnp.array(reward)
            shaped_rewards[agent] = shaped_rewards
            infos[agent] = {**info, **extra_info} # combine extra_info with info if needed

        # After agents have stepped, process trade flows to update states based on trades.
        for i, agent in enumerate(self.agents):
            state = self.step_trade(state, i)

        # Update step and terminal state
        state = jdc.replace(state, step=state.step + 1)

        done = self.is_terminal(state)
        # dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        # Get observations for next state
        obs = self._get_obs(state)

        # Reset trade matrices for next step.
        state = jdc.replace(state, p_trades=jnp.zeros_like(state.p_trades))
        state = jdc.replace(state, s_trades=jnp.zeros_like(state.s_trades))

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            rewards,
            dones,
            infos
        )

    def step_agent(self, key: jax.Array, agt_id: int, state: State, action: jax.Array):
        """
        Step basic functions and additional agent-specific biology.
        action: [p_trade, s_trade, growth, maintenance, reproduction]
        """
        # Check if agent is already dead before processing actions, zero out actions if so.
        already_dead = self.check_agent_is_dead(state.agents[agt_id])
        state, action = self.handle_agent_death(already_dead, state, agt_id, action)

        [p_trade, s_trade, growth, maintenance, reproduction] = action

        # List of agent-specific step functions for lax.switch
        step_fns = [self.step_plant, self.step_fungus]

        # --- Calculate resource usage ---
        # Calculate number of propagules produced based on reproduction energy.
        props_generated = reproduction // self.reproduction_cost

        # Recalculate sugars used with effective reproduction cost (discrete
        # no. of propagules).
        effective_reproduction_cost = props_generated * self.reproduction_cost

        # Growth + maintenance + effective_reproduction_cost
        s_used = growth + maintenance + effective_reproduction_cost

        # --- Update trade flows ---
        state = self.update_trade_flows(state, agt_id, p_trade, s_trade)

        # ---  Resource absorption  ---
        # Phosphorus absorption based on biomass and max uptake rate, scaled by efficiency.
        p_acquired = jnp.floor(
            self.p_uptake_max_rate \
            * state.agents[agt_id].p_uptake_efficiency \
            * state.agents[agt_id].biomass \
            * (1 - already_dead) # If agent is dead, no P is acquired.
        )

        # --- Update agent state ---
        # Update health proportional to maintenance deficit.
        required_maintenance = self.maintenance_cost_ratio * state.agents[agt_id].biomass \
                               * self.growth_cost
        health_deficit = maintenance - required_maintenance
        new_health = jnp.clip(state.agents[agt_id].health + health_deficit, 0, 100.0)

        # Update biomass based on growth / cost.
        new_biomass = jnp.clip(state.agents[agt_id].biomass + growth / self.growth_cost, 0, jnp.inf) # Clip biomass to prevent exponential growth.

        # Execute agent-specific function for extra biology, returns modified agent state.
        agent_mod = jax.lax.switch(agt_id, step_fns, key, state, action, state.agents[agt_id])

        # Update state with agent and trade values
        with jdc.copy_and_mutate(state) as state:
            state.agents[agt_id].health = new_health
            state.agents[agt_id].biomass = new_biomass
            state.agents[agt_id].phosphorus = agent_mod.phosphorus + p_acquired
            state.agents[agt_id].sugars = agent_mod.sugars - s_used

        # Check if agent is dead after health update.
        now_dead = self.check_agent_is_dead(state.agents[agt_id]).astype(jnp.float32)
        dead_this_step = now_dead - already_dead.astype(jnp.float32)

        info = {
            "props_generated": props_generated,
            "growth": growth,
            "maintenance": maintenance,
            "reproduction": reproduction,
            "s_trade": s_trade,
            "p_trade": p_trade,
            "phosphorus_acquired": p_acquired,
            "sugars_generated": agent_mod.sugars - state.agents[agt_id].sugars,
            "health": new_health,
            "biomass": state.agents[agt_id].biomass,
            "avail_sugars": state.agents[agt_id].sugars,
            "avail_phosphorus": state.agents[agt_id].phosphorus
        }

        # Rewards for agent based on allocation.
        reward = 0.0
        shaped_rewards = {}
        reward += props_generated * 1.5 # Reward for each seed produced
        # reward += -100 * dead_this_step # Large penalty for death

        return state, reward, shaped_rewards, info

    def step_plant(self, key: jax.Array, state, action: jax.Array, agent: AgentState):
        # Check agent is alive before processing plant-specific logic.
        is_dead = self.check_agent_is_dead(agent)
        p_trade = action[0]

        # Sugars generated from sunlight, constrained by available phosphorus.
        # If agent is dead, no sugars are generated and phosphorus is not used.
        p_use = agent.phosphorus * (1 - is_dead) - p_trade
        s_gen = jnp.clip(self.max_sugar_gen_rate * agent.biomass, 0, p_use // self.p_cost_per_sugar)

        with jdc.copy_and_mutate(agent) as agent_mod:
            agent_mod.sugars += s_gen
            agent_mod.phosphorus -= s_gen * self.p_cost_per_sugar # Remove P based on sugars generated.

        return agent_mod

    def step_fungus(self, key: jax.Array, state, action: jax.Array, agent: AgentState):
        # Placeholder implementation for stepping a fungus agent.
        return agent

    @staticmethod
    def update_trade_flows(
            state: State,
            agt_id: int,
            p_trade: jax.Array,
            s_trade: jax.Array
        ) -> State:
        """Update state trade matrices for agt_id based on unilateral allocations."""

        # Splits total allocation evenly to each partner.
        # If agent has zero connections, the trade allocation has no effect.
        no_of_partners = state.adj[agt_id].sum()
        no_of_partners = jnp.where(no_of_partners > 0, no_of_partners, 1.0) # Avoid division by 0.

        outgoing_p = (p_trade * state.adj[agt_id]) / no_of_partners
        outgoing_s = (s_trade * state.adj[agt_id]) / no_of_partners

        with jdc.copy_and_mutate(state) as new_state:
            # Set outgoing trades for agent (only works since the adj is undirected).
            new_state.p_trades = state.p_trades.at[agt_id, :].add(outgoing_p)
            new_state.s_trades = state.s_trades.at[agt_id, :].add(outgoing_s)

        return new_state

    @staticmethod
    def step_trade(state: State, agt_id: int) -> State:
        """Update agent states based on trade flows."""
        p_trade_in = state.p_trades[:, agt_id].sum()
        s_trade_in = state.s_trades[:, agt_id].sum()

        p_trade_out = state.p_trades[agt_id, :].sum()
        s_trade_out = state.s_trades[agt_id, :].sum()

        with jdc.copy_and_mutate(state) as new_state:
            new_state.agents[agt_id].phosphorus += p_trade_in - p_trade_out
            new_state.agents[agt_id].sugars += s_trade_in - s_trade_out

        return new_state

    @staticmethod
    def create_adj_matrix_fc_interclass(agent_types: List[AgentType]) -> jax.Array:
        """Create an fully connected interclass adjacency matrix."""
        # All agents of fungi class connected with all agents of plant
        # class, no connections within class.
        agent_types_array = jnp.array(agent_types)
        adj = jnp.not_equal(
            agent_types_array[:, None],
            agent_types_array[None, :]
        ).astype(jnp.float32)

        return adj

    @staticmethod
    def constrain_allocation(actions: jax.Array) -> jax.Array:
        """Constrain resource allocations to ensure they sum to 1 or less."""
        def constrain_resource_allocation(*args):
            total = sum(args)
            return jnp.array([x / total for x in args])

        # Prevent negative values.
        actions = jnp.clip(actions, 0)
        p_trade = jnp.clip(actions[0], 0, 1)

        # Constrain sugar allocations to sum to 1 or less.
        constrained_allocations = jax.lax.cond(
            sum(actions[1:]) > 1,
            lambda x: constrain_resource_allocation(*x),
            lambda x: x,
            (actions[1:])
        )

        # Update actions with constrained values
        return jnp.array([p_trade, *constrained_allocations])

    @staticmethod
    def allocate_resources(agent: AgentState, actions: jax.Array) -> jax.Array:
        """Allocate resources based on agent budget (ratios) allocation actions."""
        p_trade = jnp.floor(actions[0] * agent.phosphorus)
        s_allocations = jnp.floor(actions[1:] * agent.sugars)

        return jnp.array([p_trade, *s_allocations])

    @staticmethod
    def handle_agent_death(
            is_dead: bool, state: State, agt_id: int, action: jax.Array
        ) -> Tuple[State, jax.Array]:
        """Set agent state to dead and remove connections if health <= 0."""
        with jdc.copy_and_mutate(state) as new_state:
            # Remove connections if agent is dead.
            new_state.adj = new_state.adj.at[agt_id, :].multiply(1 - is_dead)
            new_state.adj = new_state.adj.at[:, agt_id].multiply(1 - is_dead)

            # Remove any incoming trades from agents already stepped.
            new_state.p_trades = new_state.p_trades.at[:].multiply(new_state.adj)
            new_state.s_trades = new_state.s_trades.at[:].multiply(new_state.adj)

        action = jax.lax.cond(
            is_dead,
            lambda x: jnp.zeros_like(x), # If dead, zero out all actions.
            lambda x: x, # If alive, keep actions unchanged.
            action
        )
        return new_state, action

    def agent_obs_space(self, size: int) -> spaces.Space:
        return spaces.Box(-jnp.inf, jnp.inf, shape=(size,), dtype=jnp.float32)

    def agent_action_space(self, size: int) -> spaces.Space:
        return spaces.Box(0., 1.0, shape=(size,), dtype=jnp.float32)

    def is_terminal(self, state: State) -> jax.Array:
        return state.terminal | \
               (state.step >= self.max_episode_steps) | \
               jnp.all(jnp.array([agent.health <= 0.0 for agent in state.agents]))

    def check_agent_is_dead(self, agent: AgentState) -> jax.Array:
        return jnp.array(agent.health <= 0.0)
