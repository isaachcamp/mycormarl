
from flax import struct
import math

from mycormarl.plant.traits import PlantTraits
from mycormarl.fungus.traits import FungusTraits

# To implement: implement function to read params from YAML file.

@struct.dataclass
class SpeciesParams:
    """Container for species-specific parameters."""
    plant: PlantTraits
    fungus: FungusTraits


@struct.dataclass
class EnvConfig:
    """Environment-wide controls.

    The defaults intentionally keep the environment lightweight. More detailed
    mechanisms can be swapped in later without changing the JaxMARL-facing API.
    """

    max_steps: int = 256
    dt: float = 0.05
    soil_radius_cm: float = math.sqrt(10_000.0 / math.pi)
    soil_depth_cm: float = 100.0
    n_radial_cells: int = 564
    n_depth_cells: int = 1000
    topsoil_depth_cm: float = 25.0
    initial_solution_p_um: float = 1.0
    soil_diffusion: float = 0.0
    buffer_power: float = 239.0
    soil_impedence: float = 1.0
    theta_water: float = 0.3
    alpha: float = 0.5
    reward_scaling: float = 1.0
    norm_obs: bool = True
