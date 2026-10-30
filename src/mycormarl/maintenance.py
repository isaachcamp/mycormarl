
import chex
import jax.numpy as jnp


def _update_history_max(current: chex.Array, hist_max: chex.Array) -> chex.Array:
    return jnp.maximum(hist_max, current)

def _death_mask(biomass: chex.Array, hist_max: chex.Array, death_fraction: float) -> chex.Array:
    return biomass < (death_fraction * jnp.maximum(hist_max, 1e-8))

def remove_density_from_biomass(
        biomass_t2: chex.Array, biomass_t1: float, density: chex.Array
    ) -> chex.Array:
    """Scale density by the ratio of new biomass to old biomass."""
    return density * (biomass_t2 / biomass_t1)
