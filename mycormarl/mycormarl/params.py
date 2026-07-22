
from flax import struct

from mycormarl.plant.traits import PlantTraits
from mycormarl.fungus.traits import FungusTraits


@struct.dataclass
class SpeciesParams:
    """Container for species-specific parameters."""
    plant: PlantTraits
    fungus: FungusTraits


@struct.dataclass
class EnvConfig:
    """Environment-wide controls.

    The defaults describe the selected production phosphate grid. Smaller
    explicit domains should be used for tests and development runs.
    Spatial values use cm. Radial and depth intervals must divide their
    corresponding extents into uniform cells; invalid requests are rejected
    with the nearest valid interval. ``dt`` uses days, the phosphate diffusion
    coefficient uses cm² s⁻¹, and its dimensionless impedance factor and CFL
    safety are applied by the soil scheduler. The uptake-regime reference time
    uses days; its positive transition exponent is dimensionless.
    """

    max_steps: int = 14600
    dt: float = 0.025
    consumer_mode: str = "mixed"
    soil_radius_cm: float = 50.0
    soil_depth_cm: float = 100.0
    radial_interval_cm: float = 0.1
    depth_interval_cm: float = 0.1
    topsoil_depth_cm: float = 25.0
    initial_solution_p_um: float = 1.0
    phosphate_diffusion_coefficient_cm2_s: float = 1e-5
    b_p: float = 239.0  # linear volumetric P buffer power
    phosphate_impedance_factor: float = 0.308
    diffusion_cfl_safety: float = 0.8
    uptake_reference_time_days: float = 1.0
    uptake_transition_exponent: float = 2.0
    theta_water: float = 0.3
    alpha: float = 0.5
    norm_obs: bool = True
