
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class PlantTraits:
    """Static functional traits for the plant partner."""

    initial_biomass: float = 1.0
    initial_c_pool: float = 0.5
    initial_p_pool: float = 0.05
    kleaf: float = 0.30  # biomass fraction dedicated to photosynthesis
    kroot: float = 0.25  # biomass fraction dedicated to roots
    amass: float = 1.0  # photosynthetic rate per unit leaf dry mass
    jmax: float = 0.05
    km: float = 0.1
    root_radius: float = 0.01
    specific_root_length: float = 100.0
    root_length_density: float = 1.0
    beta_root_distribution: float = 0.96
    carbon_per_growth: float = 0.45
    phosphorus_per_growth: float = 0.05
    maintenance_kappa_c: float = 0.02
    maintenance_kappa_p: float = 0.002
    death_fraction: float = 0.20
    biomass_cap: float = 100.0
