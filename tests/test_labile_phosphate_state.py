import jax
import jax.numpy as jnp
import pytest

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_grid import (
    axisymmetric_profile_p_concentration,
    initial_labile_p_from_micromolar,
    labile_amount_to_solution_concentration,
    solution_concentration_to_labile_amount,
)


def test_relative_depth_profile_is_linearly_interpolated_and_clamped_above_anchor():
    """A 5-cm anchor uses the selected grassland-relative profile exactly."""
    concentration = axisymmetric_profile_p_concentration(
        r_edges=jnp.array([0.0, 1.0]),
        z_edges=jnp.array([0.0, 5.0, 10.0, 15.0, 30.0, 60.0, 100.0]),
        surface_concentration_um=1.0,
        depth_profile=((5.0, 1.0), (15.0, 0.345), (30.0, 0.170), (60.0, 0.103), (100.0, 0.069)),
    )

    assert concentration.shape == (1, 6)
    # Cell centres: 2.5, 7.5, 12.5, 22.5, 45, and 80 cm.
    assert concentration[0] == pytest.approx(
        jnp.array([1.0, 0.83625, 0.50875, 0.2575, 0.1365, 0.086]) * 1e-3,
        rel=1e-6,
    )


def test_absent_depth_profile_initializes_every_depth_cell_at_the_declared_concentration():
    """No profile is the explicit uniform-through-depth treatment."""
    env = BaseMycorMarl(
        EnvConfig(
            soil_radius_cm=1.0,
            soil_depth_cm=3.0,
            radial_interval_cm=1.0,
            depth_interval_cm=1.0,
            initial_solution_p_um=2.0,
        ),
        SpeciesParams(PlantTraits(), FungusTraits()),
    )

    _, state = env.reset(jax.random.PRNGKey(0))
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p, env.cell_volumes, env.config.theta_water, env.config.b_p
    )

    assert jnp.allclose(concentration, 2e-3, rtol=1e-6, atol=1e-10)


def test_depth_profile_is_the_only_opt_in_for_initial_vertical_heterogeneity():
    """Profile knots scale the declared µM concentration by cell-centre depth."""
    env = BaseMycorMarl(
        EnvConfig(
            soil_radius_cm=1.0,
            soil_depth_cm=2.0,
            radial_interval_cm=1.0,
            depth_interval_cm=1.0,
            initial_solution_p_um=2.0,
            initial_solution_p_depth_profile=((0.0, 1.0), (2.0, 0.5)),
        ),
        SpeciesParams(PlantTraits(), FungusTraits()),
    )

    _, state = env.reset(jax.random.PRNGKey(0))
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p, env.cell_volumes, env.config.theta_water, env.config.b_p
    )

    assert jnp.allclose(concentration, jnp.array([[1.75e-3, 1.25e-3]]), rtol=1e-6)


@pytest.fixture()
def species():
    """Provide distinct pools so reset tests can detect biological changes."""
    return SpeciesParams(
        plant=PlantTraits(
            initial_biomass=2.0,
            initial_c_pool=3.0,
            initial_p_pool=4.0,
        ),
        fungus=FungusTraits(
            initial_biomass=5.0,
            initial_c_pool=6.0,
            initial_p_pool=7.0,
        ),
    )


@pytest.fixture()
def small_config():
    """Provide a cheap grid for reset and buffering checks."""
    return EnvConfig(
        soil_radius_cm=2.0,
        soil_depth_cm=3.0,
        radial_interval_cm=1.0,
        depth_interval_cm=1.0,
        initial_solution_p_um=2.0,
        theta_water=0.3,
        b_p=239.0,
    )


def test_buffered_amount_concentration_round_trip():
    """Verifies canonical amount can recover its driving solution concentration."""
    volumes = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    concentration = jnp.array([[0.0, 1e-3], [3e-3, 1e-2]])

    amount = solution_concentration_to_labile_amount(
        concentration,
        volumes,
        theta_water=0.3,
        b_p=239.0,
    )
    recovered = labile_amount_to_solution_concentration(
        amount,
        volumes,
        theta_water=0.3,
        b_p=239.0,
    )

    assert jnp.allclose(recovered, concentration, rtol=1e-6, atol=1e-10)
    assert jnp.all(amount >= 0.0)


def test_default_domain_initial_inventory_matches_configured_extents():
    """The default uniform treatment fills the configured cylinder."""
    r_edges = jnp.linspace(0.0, 50.0, 501)
    z_edges = jnp.linspace(0.0, 100.0, 1001)

    amount = initial_labile_p_from_micromolar(
        r_edges,
        z_edges,
        concentration_um=1.0,
        theta_water=0.3,
        b_p=239.0,
    )

    expected = jnp.pi * 50.0**2 * 100.0 * 0.001 * (0.3 + 239.0)
    assert amount.shape == (500, 1000)
    # A 500,000-cell float32 reduction incurs about 1 ppm rounding error.
    assert jnp.sum(amount) == pytest.approx(expected, rel=2e-6)
    assert jnp.all(amount > 0.0)


def test_uniform_amount_uses_each_cell_volume():
    """Uniform Pi is converted to canonical amount in every cell."""
    amount = initial_labile_p_from_micromolar(
        r_edges=jnp.array([0.0, 1.0]),
        z_edges=jnp.array([0.0, 1.0, 3.0]),
        concentration_um=2.0,
        theta_water=0.3,
        b_p=0.0,
    )

    assert amount[0, 0] == pytest.approx(jnp.pi * 0.3 * 0.002)
    assert amount[0, 1] == pytest.approx(2.0 * jnp.pi * 0.3 * 0.002)


def test_buffered_transformations_are_jittable():
    """Ensures amount/concentration transformations can run inside soil kernels."""
    to_amount = jax.jit(solution_concentration_to_labile_amount)
    to_concentration = jax.jit(labile_amount_to_solution_concentration)
    volumes = jnp.array([[1.0, 2.0]])
    concentration = jnp.array([[1e-3, 2e-3]])

    amount = to_amount(concentration, volumes, 0.3, 239.0)
    recovered = to_concentration(amount, volumes, 0.3, 239.0)

    assert jnp.allclose(recovered, concentration, rtol=1e-6)


def test_default_config_describes_reference_domain_and_initial_condition():
    """Locks user-facing defaults to the agreed physical reference scenario."""
    config = EnvConfig()

    assert config.dt == pytest.approx(0.025)
    assert config.max_steps == 14600
    assert config.soil_radius_cm == pytest.approx(50.0)
    assert config.soil_depth_cm == pytest.approx(100.0)
    assert config.radial_interval_cm == pytest.approx(0.1)
    assert config.depth_interval_cm == pytest.approx(0.1)
    assert config.initial_solution_p_depth_profile is None
    assert config.initial_solution_p_um == pytest.approx(1.0)
    assert config.theta_water == pytest.approx(0.3)
    assert config.b_p == pytest.approx(239.0)
    assert not hasattr(config, "buffer_power")
    assert config.phosphate_diffusion_coefficient_cm2_s == pytest.approx(1e-5)
    assert config.phosphate_impedance_factor == pytest.approx(0.308)
    assert config.diffusion_cfl_safety == pytest.approx(0.8)


def test_reset_stores_only_axisymmetric_labile_amount_and_zero_diagnostics(
    species, small_config
):
    """Checks reset state shape, canonical storage, geometry, and diagnostics."""
    env = BaseMycorMarl(config=small_config, species=species)

    _, state = env.reset(jax.random.PRNGKey(0))

    assert state.soil_labile_p.shape == (2, 3)
    assert state.root_length_density.shape == (2, 3)
    assert state.hyphae_length_density.shape == (2, 3)
    assert not hasattr(state, "soil_p")
    assert env.r_edges.shape == (3,)
    assert env.z_edges.shape == (4,)
    assert env.cell_volumes.shape == (2, 3)
    assert env.radial_face_areas.shape == (3, 3)
    assert env.vertical_face_areas.shape == (2, 4)
    assert state.cumulative_plant_p_mortality_loss_mg[0] == pytest.approx(0.0)
    assert state.cumulative_fungus_p_mortality_loss_mg[0] == pytest.approx(0.0)
    assert state.cumulative_plant_p_reproduction_export_mg[0] == pytest.approx(0.0)
    assert state.cumulative_fungus_p_reproduction_export_mg[0] == pytest.approx(0.0)


def test_reset_recovers_configured_solution_field_and_preserves_biological_pools(
    species, small_config
):
    """Checks reset composition without altering initial organism resources."""
    env = BaseMycorMarl(config=small_config, species=species)

    _, state = env.reset(jax.random.PRNGKey(0))
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p,
        env.cell_volumes,
        small_config.theta_water,
        small_config.b_p,
    )

    assert jnp.allclose(concentration, 2e-3)
    assert state.plant_p_pool[0] == pytest.approx(4.0)
    assert state.fungus_p_pool[0] == pytest.approx(7.0)
    assert state.plant_c_pool[0] == pytest.approx(3.0)
    assert state.fungus_c_pool[0] == pytest.approx(6.0)


@pytest.mark.parametrize(
    "config",
    [
        EnvConfig(radial_interval_cm=0.0),
        EnvConfig(theta_water=0.0),
        EnvConfig(b_p=-1.0),
        EnvConfig(initial_solution_p_um=-1.0),
        EnvConfig(dt=0.0),
        EnvConfig(dt=float("inf")),
        EnvConfig(max_steps=0),
        EnvConfig(phosphate_diffusion_coefficient_cm2_s=-1.0),
        EnvConfig(phosphate_impedance_factor=1.1),
        EnvConfig(diffusion_cfl_safety=0.0),
        EnvConfig(uptake_reference_time_days=0.0),
        EnvConfig(uptake_reference_time_days=float("inf")),
        EnvConfig(uptake_transition_exponent=0.0),
        EnvConfig(uptake_transition_exponent=float("nan")),
    ],
)
def test_environment_rejects_invalid_soil_configuration(config, species):
    """Checks invalid physical configuration fails before grid construction."""
    with pytest.raises(ValueError):
        BaseMycorMarl(config=config, species=species)


def test_soil_step_updates_canonical_amount_without_storing_concentration(
    species, small_config
):
    """Allows redistribution and uptake while retaining the labile-state contract."""
    env = BaseMycorMarl(config=small_config, species=species)
    _, state = env.reset(jax.random.PRNGKey(0))

    next_state = env.step_phosphorus_field(state)

    assert not hasattr(next_state, "soil_p")
    assert jnp.all(next_state.soil_labile_p >= 0.0)
    assert jnp.sum(next_state.soil_labile_p) <= jnp.sum(state.soil_labile_p)
