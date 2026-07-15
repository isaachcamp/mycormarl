from __future__ import annotations

import jax.numpy as jnp


def hypha_sa_from_density(density, hyphal_radius, volume):
    return 2 * jnp.pi * hyphal_radius * density * volume

def hyphal_density_from_biomass(biomass, fungal_carbon_fraction, volume):
    return biomass / (fungal_carbon_fraction * volume)