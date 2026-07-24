
import math

from flax import struct

@struct.dataclass
class FungusTraits:
    """Static functional traits for the fungus partner.

    Biomass is in grams dry mass. ``gamma_c`` is g C per g dry biomass,
    ``gamma_p`` is mg P per g dry biomass, hyphal radius is in cm,
    ``hyphal_tissue_carbon_density`` is g C per cm³ living fungal tissue, and
    saturation density is cm hypha per cm³ bulk soil. ``jmax`` is µmol P
    cm^-2 s^-1 and ``km`` is µmol P cm^-3.
    """

    initial_biomass: float = 1.0
    initial_c_pool: float = 0.5
    initial_p_pool: float = 0.05
    gamma_c: float = 0.5
    gamma_p: float = 40.0
    kappa_c: float = 0.03
    kappa_p: float = 0.003
    death_fraction: float = 0.05
    hyphal_radius: float = 5e-4
    hyphal_tissue_carbon_density: float = 0.1155
    saturation_density: float = 168.75
    jmax: float = 3.26e-6  # µmol P cm^-2 s^-1
    km: float = 5.8e-3  # µmol P cm^-3


def validate_fungus_growth_geometry_traits(traits: FungusTraits) -> None:
    """Validate every fungal trait that forms state or controls a rate."""
    for name in (
        "gamma_c",
        "gamma_p",
        "hyphal_radius",
        "hyphal_tissue_carbon_density",
        "saturation_density",
    ):
        value = getattr(traits, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"fungus {name} must be finite and greater than zero")
    if not math.isfinite(traits.initial_biomass) or traits.initial_biomass < 0.0:
        raise ValueError("fungus initial_biomass must be finite and non-negative")
    for name in ("initial_c_pool", "initial_p_pool", "kappa_c", "kappa_p"):
        value = getattr(traits, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"fungus {name} must be finite and non-negative")
    if not math.isfinite(traits.death_fraction) or not (
        0.0 <= traits.death_fraction <= 1.0
    ):
        raise ValueError("fungus death_fraction must be finite and within [0, 1]")
    if not math.isfinite(traits.jmax) or traits.jmax < 0.0:
        raise ValueError("fungus jmax must be finite and non-negative")
    if not math.isfinite(traits.km) or traits.km <= 0.0:
        raise ValueError("fungus km must be finite and greater than zero")
