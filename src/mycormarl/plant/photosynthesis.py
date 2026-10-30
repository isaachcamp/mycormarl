from __future__ import annotations

import chex

from mycormarl.state import State
from mycormarl.plant import PlantTraits
from mycormarl.params import EnvConfig


def photosynthesise(state: State, traits: PlantTraits, config: EnvConfig) -> chex.Array:
    """Spread a reference day's apparent-gross carbon input through a step."""
    return traits.kleaf * state.plant_biomass * traits.amass * config.dt
