
import math

from flax import struct

@struct.dataclass
class PlantTraits:
    """Static functional traits for the plant partner.

    Biomass is in grams dry mass. ``kfroot`` is represented fine-root mass
    divided by whole-plant mass. ``gamma_c`` is g C per g dry biomass,
    ``gamma_p`` is mg P per g dry biomass, root radius is in cm, and specific
    root length is cm root per g fine-root dry mass. ``root_length_density`` is the
    uniform within-disc ``lambda_root`` in cm root per cm³ bulk soil.
    ``amass`` is apparent-gross g C fixed per g leaf dry biomass for one
    reference day, spread uniformly through time by the current model.
    ``kappa_c`` is g C maintenance per g whole-plant dry biomass per day.
    ``kappa_p`` is a lumped irreversible free-P loss in mg P per g
    whole-plant dry biomass per day; it abstracts small losses such as
    herbivory and unrecovered turnover rather than measured P maintenance.
    Structural P is represented only by ``gamma_p``. ``jmax`` is µmol P
    cm^-2 s^-1 and ``km`` is µmol P cm^-3. Initial free C and P pools each
    contain one structural-biomass equivalent at the configured initial
    biomass. ``biomass_cap`` is a numerical growth guard in g dry mass;
    ``biomass_observation_reference`` is the independent g dry mass scale used
    by the bounded actor observation.
    """

    initial_biomass: float = 0.01
    initial_c_pool: float = 0.00402
    initial_p_pool: float = 0.0192
    kleaf: float = 0.30  # biomass fraction dedicated to photosynthesis
    kfroot: float = 0.18  # fine-root dry-mass fraction of whole-plant dry mass
    amass: float = 0.05
    jmax: float = 3.26e-6  # µmol P cm^-2 s^-1
    km: float = 5.8e-3  # µmol P cm^-3
    root_radius: float = 0.01
    specific_root_length: float = 25_434.3
    root_length_density: float = 1.0
    beta_root_distribution: float = 0.96
    max_rooting_depth_cm: float = 150.0
    gamma_c: float = 0.402
    gamma_p: float = 1.92
    kappa_c: float = 0.007
    kappa_p: float = 0.001
    death_fraction: float = 0.20
    biomass_cap: float = 50.0
    biomass_observation_reference: float = 50.0

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
    if not math.isfinite(traits.kfroot) or not 0.0 <= traits.kfroot <= 1.0:
        raise ValueError("plant kfroot must be finite and within [0, 1]")
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
    if (
        not math.isfinite(traits.biomass_observation_reference)
        or traits.biomass_observation_reference <= 0.0
    ):
        raise ValueError(
            "plant biomass_observation_reference must be finite and greater than zero"
        )
    if not math.isfinite(traits.jmax) or traits.jmax < 0.0:
        raise ValueError("plant jmax must be finite and non-negative")
    if not math.isfinite(traits.km) or traits.km <= 0.0:
        raise ValueError("plant km must be finite and greater than zero")
