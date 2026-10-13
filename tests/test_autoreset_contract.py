"""An installed MycorMARL owns its reset boundary without dependency patches."""

import jax
import jax.numpy as jnp
import pytest

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.fungus.traits import FungusTraits


@pytest.mark.parametrize("held", [False, True])
def test_autoreset_matches_explicit_keys_and_preserves_terminal_transition(held):
    base = BaseMycorMarl(
        EnvConfig(max_steps=1, dt=0.05, soil_radius_cm=0.2, soil_depth_cm=0.2,
                  radial_interval_cm=0.1, depth_interval_cm=0.1),
        SpeciesParams(PlantTraits(kappa_c=0.0, kappa_p=0.0),
                      FungusTraits(kappa_c=0.0, kappa_p=0.0)),
    )
    env = PolicyIntervalMycorMarl(base, decision_interval_days=0.05,
                                 max_episode_steps=1) if held else base
    _, initial = env.reset(jax.random.PRNGKey(3))
    key = jax.random.PRNGKey(17)
    actions = {agent: jnp.asarray([0.0, 0.1, 0.3, 0.2]) for agent in env.agents}
    step_key, reset_key = jax.random.split(key)
    _, terminal, expected_rewards, _, expected_infos = env.step_env(step_key, initial, actions)
    expected_observations, expected_state = env.reset(reset_key)
    observations, state, rewards, dones, infos = env.step(key, initial, actions)
    assert bool(dones["__all__"])
    assert int(terminal.step) == 1 and int(state.step) == 0
    for actual, expected in zip(jax.tree.leaves((observations, state, rewards, infos)),
                                jax.tree.leaves((expected_observations, expected_state,
                                                 expected_rewards, expected_infos)), strict=True):
        if jnp.issubdtype(actual.dtype, jnp.floating):
            # Nested XLA compilation may fuse arithmetic differently from
            # separately compiled step_env; bound that roundoff in dtype eps.
            assert jnp.allclose(actual, expected,
                                rtol=4 * jnp.finfo(actual.dtype).eps, atol=0.0)
        else:
            assert jnp.array_equal(actual, expected)
    for agent in env.agents:
        assert bool(infos["transitions"][agent].truncated)
