import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest

from mycormarl.fungus.mycelium import (
    axisymmetric_density_from_biomass,
    colony_radius_from_length_axisymmetric,
    hyphal_length_from_fungal_biomass,
)
from mycormarl.fungus.traits import FungusTraits
from mycormarl.plant.roots import root_disc_radii_from_biomass
from mycormarl.plant.traits import PlantTraits

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "geometry_growth_video.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("geometry_growth_video", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
geometry_growth_video = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(geometry_growth_video)

fungal_biomass_for_colony_radius = (
    geometry_growth_video.fungal_biomass_for_colony_radius
)
growth_radii_from_grid = geometry_growth_video.growth_radii_from_grid
root_biomass_for_max_disc_radius = (
    geometry_growth_video.root_biomass_for_max_disc_radius
)


def test_growth_radii_advance_by_radial_grid_interval():
    """Uses successive radial edges as frames, including a shortened final cell."""
    radii = growth_radii_from_grid(
        max_radius_cm=0.35,
        radial_interval_cm=0.1,
    )

    assert jnp.allclose(radii, jnp.array([0.1, 0.2, 0.3, 0.35]))


def test_growth_radii_reject_more_than_one_thousand_frames():
    """Enforces the requested hard upper bound on video length."""
    with pytest.raises(ValueError, match="1,000"):
        growth_radii_from_grid(
            max_radius_cm=100.1,
            radial_interval_cm=0.1,
        )


def test_fungal_biomass_inverts_the_p2_colony_radius_pipeline():
    """Targets each frame radius through the implemented biomass conversion."""
    traits = FungusTraits()
    target_radius = 1.2

    biomass = fungal_biomass_for_colony_radius(target_radius, traits)
    length = hyphal_length_from_fungal_biomass(
        biomass,
        traits.gamma_c,
        traits.hyphal_tissue_carbon_density,
        traits.hyphal_radius,
    )
    recovered_radius = colony_radius_from_length_axisymmetric(
        length,
        traits.saturation_density,
    )
    density = axisymmetric_density_from_biomass(
        jnp.array([biomass]),
        traits,
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([0.0, 1.0, 2.0]),
    )

    assert recovered_radius == pytest.approx(target_radius, rel=1e-6)
    assert jnp.any(density > 0.0)


def test_root_biomass_inverts_maximum_depth_specific_disc_radius():
    """Targets the shallowest disc while preserving beta-dependent radii."""
    traits = PlantTraits()
    z_edges = jnp.linspace(0.0, 25.0, 251)
    target_radius = 2.0

    biomass = root_biomass_for_max_disc_radius(
        target_radius_cm=target_radius,
        traits=traits,
        z_edges=z_edges,
    )
    radii = root_disc_radii_from_biomass(
        jnp.atleast_1d(biomass),
        traits,
        z_edges,
    )

    assert jnp.max(radii) == pytest.approx(target_radius, rel=1e-6)
    assert radii[-1] < radii[0]
