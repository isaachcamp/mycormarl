"""Reproduction diagnostics agree with rewards at both timestep boundaries."""

import jax
import jax.numpy as jnp
import pytest

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_base_and_held_interval_reward_decomposition(alpha):
    environment = BaseMycorMarl(
        EnvConfig(max_steps=6, dt=0.05, alpha=alpha, soil_radius_cm=0.2,
                  soil_depth_cm=0.2, radial_interval_cm=0.1, depth_interval_cm=0.1),
        SpeciesParams(
            PlantTraits(initial_c_pool=1.0, initial_p_pool=1.0, kappa_c=0.0, kappa_p=0.0),
            FungusTraits(initial_c_pool=1.0, initial_p_pool=1.0, kappa_c=0.0, kappa_p=0.0),
        ),
    )
    actions = {agent: jnp.asarray([0.0, 0.1, 0.3, 0.2]) for agent in environment.agents}
    _, initial = environment.reset(jax.random.PRNGKey(17))
    _, _, rewards, _, infos = environment.step_env(jax.random.PRNGKey(19), initial, actions)
    for agent in environment.agents:
        traits = getattr(environment.species, agent)
        expected = ((infos[agent]["reproduction_c"] / traits.gamma_c) ** alpha
                    * (infos[agent]["reproduction_p"] / traits.gamma_p) ** (1.0 - alpha))
        assert float(infos[agent]["reproductive_fitness_from_components"].squeeze()) == pytest.approx(float(expected.squeeze()), rel=1e-6)
        assert float(rewards[agent]) == pytest.approx(float(expected.squeeze()), rel=1e-6)

    held = PolicyIntervalMycorMarl(environment, decision_interval_days=0.1, max_episode_steps=3)
    _, _, held_rewards, _, held_infos = held.step_env(jax.random.PRNGKey(19), initial, actions)
    state, key = initial, jax.random.PRNGKey(19)
    sums = dict.fromkeys(environment.agents, 0.0)
    for _ in range(2):
        key, step_key = jax.random.split(key)
        _, state, substep_rewards, _, _ = environment.step_env(step_key, state, actions)
        for agent in environment.agents:
            sums[agent] += float(substep_rewards[agent])
    for agent in environment.agents:
        assert float(held_rewards[agent]) == pytest.approx(sums[agent], rel=1e-6)
        assert float(held_infos[agent]["reproductive_fitness_from_components"].squeeze()) == pytest.approx(sums[agent], rel=1e-6)
