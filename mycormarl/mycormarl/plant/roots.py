from __future__ import annotations

import jax.numpy as jnp


def hypha_sa_from_density(density, hyphal_radius, volume):
    return 2 * jnp.pi * hyphal_radius * density * volume

def root_length_increase_by_biomass(delta_biomass, kroot, specific_root_length):
    return delta_biomass * kroot * specific_root_length
