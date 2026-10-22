"""Reproduction diagnostics agree with rewards at both timestep boundaries."""

import jax
import jax.numpy as jnp
import pytest
import math

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


def test_obsolete_cobb_douglas_configuration_is_rejected():
    with pytest.raises(TypeError,match='alpha'):
        EnvConfig(alpha=.8)


@pytest.mark.parametrize("scale", [0.5, 1., 2.])
def test_base_and_held_interval_reward_decomposition(scale):
    environment = BaseMycorMarl(
        EnvConfig(max_steps=6, dt=0.05, soil_radius_cm=0.2,
                  soil_depth_cm=0.2, radial_interval_cm=0.1, depth_interval_cm=0.1),
        SpeciesParams(
            PlantTraits(initial_c_pool=1.0, initial_p_pool=1.0, kappa_c=0.0, kappa_p=0.0,reproduction_cost_scale=scale),
            FungusTraits(initial_c_pool=1.0, initial_p_pool=1.0, kappa_c=0.0, kappa_p=0.0,reproduction_cost_scale=scale),
        ),
    )
    actions = {agent: jnp.asarray([0.0, 0.1, 0.3]) for agent in environment.agents}
    _, initial = environment.reset(jax.random.PRNGKey(17))
    _, _, rewards, _, infos = environment.step_env(jax.random.PRNGKey(19), initial, actions)
    for agent in environment.agents:
        traits = getattr(environment.species, agent)
        # Independent isolated first-order solution over the first allocation.
        q0 = min(1./traits.gamma_c,1./traits.gamma_p)
        expected = .75 * (1.-math.exp(-.4*.05)) * q0 / scale
        assert float(infos[agent]["reproductive_fitness_from_components"].squeeze()) == pytest.approx(expected, rel=1e-6)
        assert float(rewards[agent]) == pytest.approx(expected, rel=1e-6)

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
