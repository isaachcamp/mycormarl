"""PPO-facing policy scheduler over a numerical MycorMARL environment."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from mycormarl.environments.base_mycor import AGENTS, FUNGUS, PLANT, BaseMycorMarl
from mycormarl.state import State
from mycormarl.transition import Transition


class PolicyIntervalMycorMarl(MultiAgentEnv):
    """Hold one policy decision across integral numerical timesteps.

    ``BaseMycorMarl`` remains the numerical environment. This wrapper is the
    policy-facing seam: it holds one Rate action constant while the numerical
    environment integrates it over each numerical step, then returns one
    transition to PPO for the complete policy interval.
    """

    def __init__(
        self,
        numerical_environment: BaseMycorMarl,
        *,
        decision_interval_days: float,
        max_episode_steps: int,
    ) -> None:
        super().__init__(num_agents=2)
        if not math.isfinite(decision_interval_days) or decision_interval_days <= 0.0:
            raise ValueError("decision_interval_days must be finite and positive")
        ratio = decision_interval_days / numerical_environment.config.dt
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("decision_interval_days must contain whole numerical timesteps")
        self.numerical_environment = numerical_environment
        self.decision_interval_days = decision_interval_days
        self.numerical_substeps_per_decision = round(ratio)
        self.max_episode_steps = max_episode_steps
        self.agents = numerical_environment.agents
        self.action_spaces = numerical_environment.action_spaces
        self.observation_spaces = numerical_environment.observation_spaces

    @property
    def agent_classes(self) -> dict:
        return self.numerical_environment.agent_classes

    @property
    def config(self):
        return self.numerical_environment.config

    @property
    def species(self):
        return self.numerical_environment.species

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        return self.numerical_environment.reset(key)

    def step_env(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]):
        initial_state = state
        def numerical_substep(carry, _):
            key, state, rewards, trade_executed = carry
            key, substep_key = jax.random.split(key)
            observations, state, substep_rewards, dones, infos = self.numerical_environment.step_env(
                substep_key, state, actions
            )
            rewards = {agent: rewards[agent] + substep_rewards[agent] for agent in AGENTS}
            trade_executed = {
                agent: trade_executed[agent] | infos["transitions"][agent].trade_executed
                for agent in AGENTS
            }
            return (key, state, rewards, trade_executed), (observations, dones, infos)

        (_, state, rewards, trade_executed), (observations, dones, infos) = jax.lax.scan(
            numerical_substep,
            (key, state, {PLANT: jnp.asarray(0.0), FUNGUS: jnp.asarray(0.0)},
             {PLANT: jnp.asarray(False), FUNGUS: jnp.asarray(False)}),
            xs=None,
            length=self.numerical_substeps_per_decision,
        )
        observations = jax.tree.map(lambda values: values[-1], observations)
        dones = jax.tree.map(lambda values: values[-1], dones)
        infos = jax.tree.map(lambda values: values[-1], infos)
        for agent, active in ((PLANT, self.numerical_environment.plant_active), (FUNGUS, self.numerical_environment.fungus_active)):
            start = jnp.logical_and(active, ~getattr(initial_state, f"{agent}_dead")).squeeze()
            end = jnp.logical_and(active, ~getattr(state, f"{agent}_dead")).squeeze()
            infos["transitions"][agent] = Transition(
                requested_action=actions[agent],
                realised_action=jnp.where(end, actions[agent], jnp.zeros_like(actions[agent])),
                operational_at_start=start,
                operational_at_end=end,
                allocation_executed=end,
                trade_executed=trade_executed[agent],
                truncated=state.step >= self.numerical_environment.max_episode_steps,
                final_observation=observations[agent],
            )
        return observations, state, rewards, dones, infos
