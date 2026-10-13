"""MycorMARL auto-reset boundary implemented with the supported JAX tree API.

JaxMARL 0.0.2's inherited step uses the removed ``jax.tree_map`` symbol.
Keep its key splitting and pre-reset diagnostic semantics explicitly here,
without changing installed dependencies or the biological transition.
"""

from functools import partial

import jax


class AutoResetMixin:
    """Select reset observations/state only at the global episode boundary."""

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions):
        step_key, reset_key = jax.random.split(key)
        observations, next_state, rewards, dones, infos = self.step_env(
            step_key, state, actions
        )
        reset_observations, reset_state = self.reset(reset_key)
        select = lambda reset, current: jax.lax.select(dones["__all__"], reset, current)
        return (
            jax.tree.map(select, reset_observations, observations),
            jax.tree.map(select, reset_state, next_state),
            rewards,
            dones,
            infos,
        )
