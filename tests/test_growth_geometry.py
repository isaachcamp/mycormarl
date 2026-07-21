import jax
import jax.numpy as jnp
import pytest

from mycormarl.environments import base_mycor as env_mod
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.mycelium import (
    _volume_under_sphere_within_radius,
    axisymmetric_density_from_biomass,
    axisymmetric_hemisphere_cell_fractions,
    hyphal_length_from_fungal_biomass,
)
from mycormarl.fungus.traits import FungusTraits
from mycormarl.plant.roots import (
    axisymmetric_stacked_disc_root_density,
    root_disc_radii_from_biomass,
    root_length_from_plant_biomass,
)
from mycormarl.plant.traits import PlantTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.soil.phosphate_grid import axisymmetric_cylindrical_cell_volumes


def test_provisional_growth_geometry_trait_defaults():
    """Locks agreed plant and fungal growth-geometry parameters and units."""
    plant = PlantTraits()
    fungus = FungusTraits()

    assert plant.gamma_c == pytest.approx(0.402)
    assert plant.gamma_p == pytest.approx(1.92)
    assert plant.kroot == pytest.approx(0.62)
    assert plant.specific_root_length == pytest.approx(25_434.3)
    assert plant.root_radius == pytest.approx(0.01)
    assert plant.root_length_density == pytest.approx(1.0)

    assert fungus.gamma_c == pytest.approx(0.5)
    assert fungus.gamma_p == pytest.approx(40.0)
    assert fungus.hyphal_radius == pytest.approx(5e-4)
    assert fungus.hyphal_tissue_carbon_density == pytest.approx(0.1155)
    assert fungus.saturation_density == pytest.approx(168.75)


def test_one_gram_biomass_has_expected_root_and_hyphal_length():
    """Checks the independently calculated per-gram P2 reference lengths."""
    root_length = root_length_from_plant_biomass(
        biomass_g=1.0,
        root_mass_fraction=0.62,
        specific_root_length_cm_g=25_434.3,
    )
    hyphal_length = hyphal_length_from_fungal_biomass(
        biomass_g=1.0,
        gamma_c_g_c_per_g=0.5,
        tissue_carbon_density_g_c_cm3=0.1155,
        hyphal_radius_cm=5e-4,
    )

    assert root_length == pytest.approx(15_769.266, rel=1e-6)
    assert hyphal_length == pytest.approx(5_511_859.501, rel=1e-6)


def test_length_conversions_have_physical_zero_and_nonnegative_limits():
    """Prevents zero or defensive negative biomass from creating structure."""
    root = root_length_from_plant_biomass(
        biomass_g=jnp.array([0.0, -1.0]),
        root_mass_fraction=0.62,
        specific_root_length_cm_g=25_434.3,
    )
    hypha = hyphal_length_from_fungal_biomass(
        biomass_g=jnp.array([0.0, -1.0]),
        gamma_c_g_c_per_g=0.5,
        tissue_carbon_density_g_c_cm3=0.1155,
        hyphal_radius_cm=5e-4,
    )

    assert jnp.all(root == 0.0)
    assert jnp.all(hypha == 0.0)


def test_length_conversions_are_jittable():
    """Keeps biomass-to-length conversion usable in the JIT environment step."""
    root_fn = jax.jit(root_length_from_plant_biomass)
    hypha_fn = jax.jit(hyphal_length_from_fungal_biomass)

    root = root_fn(jnp.array([1.0]), 0.62, 25_434.3)
    hypha = hypha_fn(jnp.array([1.0]), 0.5, 0.1155, 5e-4)

    assert jnp.all(jnp.isfinite(root))
    assert jnp.all(jnp.isfinite(hypha))


def test_root_density_conserves_length_on_shortened_boundary_cells():
    """Integrates the stacked-disc field back to its biomass-implied length."""
    traits = PlantTraits(
        kroot=0.25,
        specific_root_length=100.0,
        root_length_density=100.0,
        beta_root_distribution=0.5,
    )
    r_edges = jnp.array([0.0, 1.0, 2.0, 2.5])
    z_edges = jnp.array([0.0, 1.0, 2.0, 2.5])
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)

    density = axisymmetric_stacked_disc_root_density(
        biomass=jnp.array([10.0]),
        traits=traits,
        r_edges=r_edges,
        z_edges=z_edges,
    )

    assert density.shape == (3, 3)
    assert jnp.sum(density * volumes) == pytest.approx(250.0, rel=1e-6)


def test_root_disc_radii_decrease_with_beta_weighted_depth():
    """Uses one uniform density while deeper beta-weighted discs grow slower."""
    traits = PlantTraits(
        kroot=1.0,
        specific_root_length=10.0,
        root_length_density=2.0,
        beta_root_distribution=0.5,
    )
    z_edges = jnp.array([0.0, 1.0, 2.0, 3.0])

    radii = root_disc_radii_from_biomass(
        biomass=jnp.array([1.0]),
        traits=traits,
        z_edges=z_edges,
    )

    expected_weights = jnp.array([0.5, 0.25, 0.125]) / 0.875
    expected = jnp.sqrt(10.0 * expected_weights / (jnp.pi * 2.0))
    assert jnp.allclose(radii, expected, rtol=1e-6)
    assert jnp.all(jnp.diff(radii) < 0.0)


def test_root_density_is_uniform_inside_depth_specific_discs():
    """Keeps occupied cell density at lambda_root across all depth layers."""
    traits = PlantTraits(
        kroot=1.0,
        specific_root_length=1.0,
        root_length_density=1.0,
        beta_root_distribution=0.5,
    )
    r_edges = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    z_edges = jnp.array([0.0, 1.0, 2.0])

    density = axisymmetric_stacked_disc_root_density(
        biomass=jnp.array([1.0]),
        traits=traits,
        r_edges=r_edges,
        z_edges=z_edges,
    )

    assert jnp.all(density >= 0.0)
    assert jnp.all(density <= traits.root_length_density)
    assert jnp.any(density == traits.root_length_density)
    assert jnp.any((density > 0.0) & (density < traits.root_length_density))


def test_root_geometry_clips_each_layer_at_radial_domain_boundary():
    """Represents the analytical disc-domain intersection at lambda_root."""
    traits = PlantTraits(
        kroot=0.25,
        specific_root_length=100.0,
        root_length_density=1.0,
        beta_root_distribution=0.5,
    )
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0, 2.0])
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)

    radii = root_disc_radii_from_biomass(
        biomass=jnp.array([10.0]),
        traits=traits,
        z_edges=z_edges,
    )
    density = axisymmetric_stacked_disc_root_density(
        biomass=jnp.array([10.0]),
        traits=traits,
        r_edges=r_edges,
        z_edges=z_edges,
    )

    dz = jnp.diff(z_edges)
    clipped_radii = jnp.minimum(radii, r_edges[-1])
    expected_represented = jnp.sum(
        traits.root_length_density * jnp.pi * clipped_radii**2 * dz
    )
    assert jnp.any(radii > r_edges[-1])
    assert jnp.sum(density * volumes) == pytest.approx(
        expected_represented, rel=1e-6
    )


def test_sphere_cylinder_intersection_helper_returns_physical_volume():
    """Includes pi in the internal intersection helper's cm³ result."""
    volume = _volume_under_sphere_within_radius(
        radial_limit=jnp.array(1.0),
        z_inner=jnp.array(0.0),
        z_outer=jnp.array(1.0),
        colony_radius=jnp.array(2.0),
    )

    assert volume == pytest.approx(jnp.pi)


def test_fungal_density_conserves_length_with_partial_front_cells():
    """Integrates a contained volume-averaged hemisphere to its total length."""
    saturation_density = 4.0
    colony_radius = 1.5
    expected_length = (
        (2.0 / 3.0) * jnp.pi * colony_radius**3 * saturation_density
    )
    gamma_c = 0.5
    tissue_carbon_density = 0.2
    hyphal_radius = 0.1
    biomass = (
        expected_length
        * tissue_carbon_density
        * jnp.pi
        * hyphal_radius**2
        / gamma_c
    )
    traits = FungusTraits(
        gamma_c=gamma_c,
        hyphal_tissue_carbon_density=tissue_carbon_density,
        hyphal_radius=hyphal_radius,
        saturation_density=saturation_density,
    )
    r_edges = jnp.array([0.0, 0.7, 1.4, 2.1])
    z_edges = jnp.array([0.0, 0.8, 1.6, 2.4])
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)

    density = axisymmetric_density_from_biomass(
        biomass=jnp.array([biomass]),
        traits=traits,
        r_edges=r_edges,
        z_edges=z_edges,
    )

    assert density.shape == (3, 3)
    assert jnp.sum(density * volumes) == pytest.approx(expected_length, rel=1e-5)
    assert jnp.any((density > 0.0) & (density < saturation_density))


def test_fungal_extent_is_monotonic_with_biomass():
    """Ensures increasing fungal structure cannot reduce occupied fractions."""
    r_edges = jnp.array([0.0, 0.5, 1.0, 1.5])
    z_edges = jnp.array([0.0, 0.5, 1.0, 1.5])

    smaller = axisymmetric_hemisphere_cell_fractions(r_edges, z_edges, 0.75)
    larger = axisymmetric_hemisphere_cell_fractions(r_edges, z_edges, 1.25)

    assert jnp.all(larger >= smaller)
    assert jnp.sum(larger) > jnp.sum(smaller)


def test_fungal_density_clips_at_saturated_domain_capacity():
    """Makes out-of-domain fungal length explicit without exceeding saturation."""
    traits = FungusTraits(
        gamma_c=1.0,
        hyphal_tissue_carbon_density=1.0,
        hyphal_radius=0.1,
        saturation_density=2.0,
    )
    r_edges = jnp.array([0.0, 0.5, 1.0])
    z_edges = jnp.array([0.0, 0.5, 1.0])
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)
    domain_capacity = traits.saturation_density * jnp.sum(volumes)

    density = axisymmetric_density_from_biomass(
        biomass=jnp.array([10.0]),
        traits=traits,
        r_edges=r_edges,
        z_edges=z_edges,
    )

    implied_length = hyphal_length_from_fungal_biomass(
        10.0,
        traits.gamma_c,
        traits.hyphal_tissue_carbon_density,
        traits.hyphal_radius,
    )
    represented_length = jnp.sum(density * volumes)
    assert implied_length > domain_capacity
    assert represented_length == pytest.approx(domain_capacity, rel=1e-6)
    assert jnp.allclose(density, traits.saturation_density)


def test_zero_biomass_produces_zero_spatial_density():
    """Checks the spatial constructors preserve the conversion zero limit."""
    r_edges = jnp.array([0.0, 1.0])
    z_edges = jnp.array([0.0, 1.0])

    root = axisymmetric_stacked_disc_root_density(
        jnp.array([0.0]), PlantTraits(), r_edges, z_edges
    )
    hypha = axisymmetric_density_from_biomass(
        jnp.array([0.0]), FungusTraits(), r_edges, z_edges
    )

    assert jnp.all(root == 0.0)
    assert jnp.all(hypha == 0.0)


def _geometry_species(**trait_overrides):
    """Build small integration traits whose initial fungal colony fits."""
    plant_overrides = trait_overrides.get("plant", {})
    fungus_overrides = trait_overrides.get("fungus", {})
    return SpeciesParams(
        plant=PlantTraits(**{
            "initial_biomass": 0.01,
            "initial_c_pool": 0.402,
            "initial_p_pool": 1.92,
            "kappa_c": 0.0,
            "kappa_p": 0.0,
            "biomass_cap": 10.0,
            "root_length_density": 100.0,
            **plant_overrides,
        }),
        fungus=FungusTraits(**{
            "initial_biomass": 1e-5,
            "initial_c_pool": 0.5,
            "initial_p_pool": 40.0,
            "kappa_c": 0.0,
            "kappa_p": 0.0,
            **fungus_overrides,
        }),
    )


def _geometry_config():
    """Build a cheap axisymmetric domain for P2 environment tests."""
    return EnvConfig(
        dt=1.0,
        soil_radius_cm=2.0,
        soil_depth_cm=3.0,
        radial_interval_cm=1.0,
        depth_interval_cm=1.0,
        topsoil_depth_cm=1.5,
        initial_solution_p_um=0.0,
        norm_obs=False,
    )


def test_reset_initialises_geometry_from_initial_structural_biomass():
    """Makes initial roots and hyphae available before the first soil step."""
    species = _geometry_species()
    env = BaseMycorMarl(config=_geometry_config(), species=species)

    _, state = env.reset(jax.random.PRNGKey(0))

    root_length = jnp.sum(state.root_length_density * env.cell_volumes)
    hyphal_length = jnp.sum(state.hyphae_length_density * env.cell_volumes)
    expected_root = root_length_from_plant_biomass(
        species.plant.initial_biomass,
        species.plant.kroot,
        species.plant.specific_root_length,
    )
    expected_hypha = hyphal_length_from_fungal_biomass(
        species.fungus.initial_biomass,
        species.fungus.gamma_c,
        species.fungus.hyphal_tissue_carbon_density,
        species.fungus.hyphal_radius,
    )

    assert root_length == pytest.approx(expected_root, rel=1e-5)
    assert hyphal_length == pytest.approx(expected_hypha, rel=1e-5)


def test_realised_growth_updates_geometry_before_soil_stage(monkeypatch):
    """Checks immediate growth credit and the agreed environment-step order."""
    species = _geometry_species()
    env = BaseMycorMarl(config=_geometry_config(), species=species)
    _, state = env.reset(jax.random.PRNGKey(0))
    observed = {}

    def observe_geometry(soil_state, *soil_args):
        """Capture the geometry visible at the soil-stage boundary."""
        observed["root"] = soil_state.root_length_density
        observed["hypha"] = soil_state.hyphae_length_density
        return soil_state

    monkeypatch.setattr(env_mod, "evolve_soil_p", observe_geometry)
    actions = {
        PLANT: jnp.array([0.0, 1.0, 0.0, 0.0]),
        FUNGUS: jnp.array([0.0, 1.0, 0.0, 0.0]),
    }

    _, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)

    expected_root = axisymmetric_stacked_disc_root_density(
        next_state.plant_biomass,
        species.plant,
        env.r_edges,
        env.z_edges,
    )
    expected_hypha = axisymmetric_density_from_biomass(
        next_state.fungus_biomass,
        species.fungus,
        env.r_edges,
        env.z_edges,
    )
    assert jnp.allclose(observed["root"], expected_root)
    assert jnp.allclose(observed["hypha"], expected_hypha)
    assert jnp.allclose(next_state.root_length_density, expected_root)
    assert jnp.allclose(next_state.hyphae_length_density, expected_hypha)


def test_maintenance_biomass_loss_contracts_geometry(monkeypatch):
    """Ensures density is derived from surviving rather than historical biomass."""
    species = _geometry_species(
        plant={"kappa_c": 1.0, "kappa_p": 1.0},
        fungus={"kappa_c": 1.0, "kappa_p": 1.0},
    )
    env = BaseMycorMarl(config=_geometry_config(), species=species)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        plant_c_pool=jnp.array([0.0]),
        plant_p_pool=jnp.array([0.0]),
        fungus_c_pool=jnp.array([0.0]),
        fungus_p_pool=jnp.array([0.0]),
    )
    def no_soil_uptake(soil_state, *soil_args):
        """Keep the maintenance test focused on geometry contraction."""
        return soil_state

    monkeypatch.setattr(env_mod, "evolve_soil_p", no_soil_uptake)
    actions = {
        PLANT: jnp.array([0.0, 0.0, 1.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    _, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert next_state.plant_biomass[0] == pytest.approx(0.0)
    assert next_state.fungus_biomass[0] == pytest.approx(0.0)
    assert jnp.all(next_state.root_length_density == 0.0)
    assert jnp.all(next_state.hyphae_length_density == 0.0)


@pytest.mark.parametrize(
    ("consumer", "field", "value"),
    [
        ("plant", "gamma_c", 0.0),
        ("plant", "gamma_p", -1.0),
        ("plant", "kroot", 1.1),
        ("plant", "specific_root_length", -1.0),
        ("plant", "root_length_density", 0.0),
        ("plant", "jmax", -1.0),
        ("plant", "km", 0.0),
        ("plant", "beta_root_distribution", 1.0),
        ("fungus", "gamma_c", 0.0),
        ("fungus", "gamma_p", -1.0),
        ("fungus", "hyphal_radius", 0.0),
        ("fungus", "hyphal_tissue_carbon_density", 0.0),
        ("fungus", "saturation_density", 0.0),
        ("fungus", "jmax", float("nan")),
        ("fungus", "km", -1.0),
    ],
)
def test_environment_rejects_invalid_growth_geometry_traits(
    consumer, field, value
):
    """Fails before JAX kernels can create invalid or non-finite geometry."""
    species = _geometry_species()
    invalid_traits = getattr(species, consumer).replace(**{field: value})
    species = species.replace(**{consumer: invalid_traits})

    with pytest.raises(ValueError, match=field):
        BaseMycorMarl(config=_geometry_config(), species=species)
