
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class FungusTraits:
    """Static functional traits for the fungus partner."""

    carbon_per_growth: float = 0.45
    phosphorus_per_growth: float = 0.05
    maintenance_kappa_c: float = 0.03
    maintenance_kappa_p: float = 0.003
    death_fraction: float = 0.05
    hyphal_radius: float = 0.002
    saturation_density: float = 1.0
    fungal_carbon_fraction: float = 0.45
    jmax: float = 0.05
    km: float = 0.1
