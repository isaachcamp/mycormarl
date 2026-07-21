
from __future__ import annotations

import chex
import jax.numpy as jnp


def _grow_biomass_essential_resources(
        allocated_c: chex.Array,
        allocated_p: chex.Array,
        grow_c_cost: float,
        grow_p_cost: float
    ) -> chex.Array:
    """Grow biomass with essential resources."""
    delta_biomass = jnp.minimum(
        (allocated_c) / grow_c_cost,
        (allocated_p) / grow_p_cost
    )

    return delta_biomass

def grow(
        allocated_c: chex.Array,
        allocated_p: chex.Array,
        grow_c_cost: float,
        grow_p_cost: float,
        grow_type: str
    ) -> chex.Array:
    """Grow the plant and fungus based on the action allocation."""
    if grow_type == "essential":
        delta_biomass = _grow_biomass_essential_resources(
            allocated_c=allocated_c,
            allocated_p=allocated_p,
            grow_c_cost=grow_c_cost,
            grow_p_cost=grow_p_cost
        )
    else:
        # Placeholder for other growth types.
        delta_biomass = jnp.array(0.0)

    return delta_biomass
