from __future__ import annotations

import chex

from mycormarl.state import MycorMarlState as State
from mycormarl.plant import PlantTraits
from mycormarl.params import EnvConfig

def _photosynthesise_simple(biomass: chex.Array, kleaf: float, amass: float, dt: float) -> chex.Array:
    """Carbon fixed in one time step, ignoring light."""
    return kleaf * biomass * amass * dt

def photosynthesise(state: State, traits: PlantTraits, config: EnvConfig) -> chex.Array:
    """Returns mass of carbon fixed in one time step."""

    return _photosynthesise_simple(state.plant_biomass, traits.kleaf, traits.amass, config.dt)

def plant_maintenance_demand(biomass: chex.Array, kappa_c: float, kappa_p: float, dt: float) -> tuple[chex.Array, chex.Array]:
    return kappa_c * biomass * dt, kappa_p * biomass * dt


if __name__ == "__main__":
    pass
