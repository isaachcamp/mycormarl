
import math

from flax import struct

@struct.dataclass
class PlantTraits:
    """Static functional traits for the plant partner.

    Biomass is in grams dry mass. ``gamma_c`` is g C per g dry biomass,
    ``gamma_p`` is mg P per g dry biomass, root radius is in cm, and specific
    root length is cm root per g root dry mass. ``root_length_density`` is the
    uniform within-disc ``lambda_root`` in cm root per cm³ bulk soil.
    ``jmax`` is µmol P cm^-2 s^-1 and ``km`` is µmol P cm^-3.
    """

    initial_biomass: float = 1.0
    initial_c_pool: float = 0.5
    initial_p_pool: float = 0.05
    kleaf: float = 0.30  # biomass fraction dedicated to photosynthesis
    kroot: float = 0.62  # dry-biomass fraction assigned to roots
    amass: float = 1.0  # photosynthetic rate per unit leaf dry mass
    jmax: float = 3.26e-6  # µmol P cm^-2 s^-1
    km: float = 5.8e-3  # µmol P cm^-3
    root_radius: float = 0.01
    specific_root_length: float = 25_434.3
    root_length_density: float = 1.0
    beta_root_distribution: float = 0.96
    max_rooting_depth_cm: float = 150.0
    gamma_c: float = 0.402
    gamma_p: float = 1.92
    kappa_c: float = 0.02
    kappa_p: float = 0.002
    death_fraction: float = 0.20
    biomass_cap: float = 100.0


def validate_plant_growth_geometry_traits(traits: PlantTraits) -> None:
    """Validate every plant trait that forms state or controls a rate."""
    for name in (
        "gamma_c",
        "gamma_p",
        "root_radius",
        "specific_root_length",
        "root_length_density",
        "max_rooting_depth_cm",
    ):
        value = getattr(traits, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"plant {name} must be finite and greater than zero")
    if not math.isfinite(traits.kroot) or not 0.0 <= traits.kroot <= 1.0:
        raise ValueError("plant kroot must be finite and within [0, 1]")
    if not math.isfinite(traits.beta_root_distribution) or not (
        0.0 < traits.beta_root_distribution < 1.0
    ):
        raise ValueError("plant beta_root_distribution must be finite and within (0, 1)")
    if not math.isfinite(traits.initial_biomass) or traits.initial_biomass < 0.0:
        raise ValueError("plant initial_biomass must be finite and non-negative")
    if traits.initial_biomass > traits.biomass_cap:
        raise ValueError("plant initial_biomass must not exceed biomass_cap")
    for name in ("initial_c_pool", "initial_p_pool", "kappa_c", "kappa_p", "amass"):
        value = getattr(traits, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"plant {name} must be finite and non-negative")
    if not math.isfinite(traits.kleaf) or not 0.0 <= traits.kleaf <= 1.0:
        raise ValueError("plant kleaf must be finite and within [0, 1]")
    if not math.isfinite(traits.death_fraction) or not (
        0.0 <= traits.death_fraction <= 1.0
    ):
        raise ValueError("plant death_fraction must be finite and within [0, 1]")
    if not math.isfinite(traits.biomass_cap) or traits.biomass_cap <= 0.0:
        raise ValueError("plant biomass_cap must be finite and greater than zero")
    if not math.isfinite(traits.jmax) or traits.jmax < 0.0:
        raise ValueError("plant jmax must be finite and non-negative")
    if not math.isfinite(traits.km) or traits.km <= 0.0:
        raise ValueError("plant km must be finite and greater than zero")
