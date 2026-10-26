
import chex
import jax.numpy as jnp


def remove_density_from_biomass(
        biomass_t2: chex.Array, biomass_t1: float, density: chex.Array
    ) -> chex.Array:
    """Scale density by the ratio of new biomass to old biomass."""
    return density * (biomass_t2 / biomass_t1)
