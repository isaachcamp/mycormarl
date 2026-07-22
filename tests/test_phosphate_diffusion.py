"""Numerical contracts for conservative axisymmetric phosphate diffusion."""

import math

import jax
import jax.numpy as jnp
import pytest

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_diffusion import (
    apparent_diffusivity_cm2_s,
    axisymmetric_diffusion_conductances,
    diffuse_labile_amount,
    explicit_diffusion_cfl_seconds,
    required_diffusion_substeps,
    validate_diffusion_parameters,
)
from mycormarl.soil.phosphate_grid import (
    axisymmetric_cylindrical_cell_volumes,
    axisymmetric_radial_face_areas,
    axisymmetric_vertical_face_areas,
    solution_concentration_to_labile_amount,
)
from mycormarl.soil.soil import (
    evolve_soil_p,
    soil_diffusion_uptake_substep,
    uptake_geometry_coefficients,
)


def _p5_coefficients(env, state):
    """Reproduce coefficients cached once by the integrated soil step."""
    return uptake_geometry_coefficients(
        state,
        env.species,
        env.config.phosphate_diffusion_coefficient_cm2_s,
        env.config.theta_water,
        env.config.phosphate_impedance_factor,
        env.config.buffer_power,
        env.config.uptake_reference_time_days,
        env.config.uptake_transition_exponent,
    )


def test_diffusion_defaults_match_schnepf_roose_parameters():
    """Locks molecular diffusion, impedance, and explicit safety defaults."""
    config = EnvConfig()

    assert config.phosphate_diffusion_coefficient_cm2_s == pytest.approx(1e-5)
    assert config.phosphate_impedance_factor == pytest.approx(0.308)
    assert config.diffusion_cfl_safety == pytest.approx(0.8)
    assert config.uptake_reference_time_days == pytest.approx(1.0)
    assert config.uptake_transition_exponent == pytest.approx(2.0)
    assert not hasattr(config, "soil_diffusion")
    assert not hasattr(config, "soil_impedence")


def test_apparent_diffusivity_matches_buffered_reference():
    """Reproduces D_app = D_l theta f_l / (theta + B)."""
    apparent = apparent_diffusivity_cm2_s(
        diffusion_coefficient_cm2_s=1e-5,
        theta_water=0.3,
        impedance_factor=0.308,
        buffer_power=239.0,
    )

    expected = 1e-5 * 0.3 * 0.308 / 239.3
    assert apparent == pytest.approx(expected, rel=1e-6)
    assert apparent == pytest.approx(3.861263e-9, rel=1e-6)


def test_conductances_use_actual_shortened_cell_centre_distances():
    """Checks G = D_l theta f_l A/d on nonuniform final grid cells."""
    r_edges = jnp.array([0.0, 1.0, 2.5])
    z_edges = jnp.array([0.0, 2.0, 3.0])
    radial_areas = axisymmetric_radial_face_areas(r_edges, z_edges)
    vertical_areas = axisymmetric_vertical_face_areas(r_edges, z_edges)

    radial, vertical = axisymmetric_diffusion_conductances(
        r_edges,
        z_edges,
        radial_areas,
        vertical_areas,
        diffusion_coefficient_cm2_s=1.0,
        theta_water=1.0,
        impedance_factor=1.0,
    )

    radial_distance = 1.25
    vertical_distance = 1.5
    assert radial.shape == (1, 2)
    assert vertical.shape == (2, 1)
    assert jnp.allclose(radial, radial_areas[1:-1] / radial_distance)
    assert jnp.allclose(vertical, vertical_areas[:, 1:-1] / vertical_distance)


def test_exact_cfl_matches_two_cell_radial_reference():
    """Independently verifies min[V(theta+B)/sum(G)] for two annuli."""
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0])
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)
    radial, vertical = axisymmetric_diffusion_conductances(
        r_edges,
        z_edges,
        axisymmetric_radial_face_areas(r_edges, z_edges),
        axisymmetric_vertical_face_areas(r_edges, z_edges),
        diffusion_coefficient_cm2_s=1.0,
        theta_water=1.0,
        impedance_factor=1.0,
    )

    cfl = explicit_diffusion_cfl_seconds(
        volumes,
        theta_water=1.0,
        buffer_power=0.0,
        radial_conductance=radial,
        vertical_conductance=vertical,
    )

    assert cfl == pytest.approx(0.5, rel=1e-6)


def test_required_substeps_apply_safety_ceiling_exactly():
    """Uses ceil(dt_bio/[safety dt_CFL]) and retains at least one substep."""
    one_second_days = 1.0 / 86_400.0

    assert required_diffusion_substeps(one_second_days, 0.5, 0.8) == 3
    assert required_diffusion_substeps(0.1 / 86_400.0, 0.5, 0.8) == 1


def test_zero_diffusion_has_infinite_cfl_and_one_substep():
    """Allows diffusion to be disabled without an empty or invalid schedule."""
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0])
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)
    radial, vertical = axisymmetric_diffusion_conductances(
        r_edges,
        z_edges,
        axisymmetric_radial_face_areas(r_edges, z_edges),
        axisymmetric_vertical_face_areas(r_edges, z_edges),
        diffusion_coefficient_cm2_s=0.0,
        theta_water=0.3,
        impedance_factor=0.308,
    )
    cfl = explicit_diffusion_cfl_seconds(
        volumes, 0.3, 239.0, radial, vertical
    )

    assert jnp.all(radial == 0.0)
    assert jnp.all(vertical == 0.0)
    assert math.isinf(float(cfl))
    assert required_diffusion_substeps(1.0, float(cfl), 0.8) == 1


@pytest.mark.parametrize(
    ("diffusion", "theta", "impedance", "safety"),
    [
        (-1.0, 0.3, 0.308, 0.8),
        (float("nan"), 0.3, 0.308, 0.8),
        (1e-5, 0.0, 0.308, 0.8),
        (1e-5, 0.3, -0.1, 0.8),
        (1e-5, 0.3, 1.1, 0.8),
        (1e-5, 0.3, 0.308, 0.0),
        (1e-5, 0.3, 0.308, 1.1),
    ],
)
def test_invalid_diffusion_parameters_fail_fast(
    diffusion, theta, impedance, safety
):
    """Rejects invalid transport and stability scalars before JAX execution."""
    with pytest.raises(ValueError):
        validate_diffusion_parameters(diffusion, theta, impedance, safety)


def _diffusion_geometry(r_edges, z_edges, diffusion=1.0, theta=1.0, impedance=1.0):
    """Build volumes and internal conductances for compact diffusion fixtures."""
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)
    radial, vertical = axisymmetric_diffusion_conductances(
        r_edges,
        z_edges,
        axisymmetric_radial_face_areas(r_edges, z_edges),
        axisymmetric_vertical_face_areas(r_edges, z_edges),
        diffusion,
        theta,
        impedance,
    )
    return volumes, radial, vertical


def test_uniform_solution_concentration_is_invariant_on_nonuniform_grid():
    """Ensures unequal annular volumes do not create spurious diffusion."""
    r_edges = jnp.array([0.0, 1.0, 2.5])
    z_edges = jnp.array([0.0, 1.0, 3.0])
    volumes, radial, vertical = _diffusion_geometry(r_edges, z_edges)
    concentration = jnp.full(volumes.shape, 2.0)
    amount = solution_concentration_to_labile_amount(
        concentration, volumes, theta_water=1.0, buffer_power=2.0
    )

    updated = diffuse_labile_amount(
        amount,
        volumes,
        theta_water=1.0,
        buffer_power=2.0,
        radial_conductance=radial,
        vertical_conductance=vertical,
        dt_seconds=0.01,
    )

    assert jnp.allclose(updated, amount, rtol=1e-6, atol=1e-7)


def test_two_cell_radial_diffusion_matches_independent_transfer():
    """Checks signed radial transfer and equal-and-opposite amount updates."""
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0])
    volumes, radial, vertical = _diffusion_geometry(r_edges, z_edges)
    amount = solution_concentration_to_labile_amount(
        jnp.array([[2.0], [0.0]]), volumes, 1.0, 0.0
    )

    updated = diffuse_labile_amount(
        amount, volumes, 1.0, 0.0, radial, vertical, dt_seconds=0.1
    )

    transfer = 1.0 * (2.0 * jnp.pi) / 1.0 * (2.0 - 0.0) * 0.1
    assert updated[0, 0] == pytest.approx(2.0 * jnp.pi - transfer, rel=1e-6)
    assert updated[1, 0] == pytest.approx(transfer, rel=1e-6)
    assert jnp.sum(updated) == pytest.approx(jnp.sum(amount), rel=1e-6)


def test_two_cell_vertical_diffusion_matches_independent_transfer():
    """Checks actual vertical centre distance and conservative signed transfer."""
    r_edges = jnp.array([0.0, 1.0])
    z_edges = jnp.array([0.0, 1.0, 3.0])
    volumes, radial, vertical = _diffusion_geometry(r_edges, z_edges)
    amount = solution_concentration_to_labile_amount(
        jnp.array([[2.0, 0.0]]), volumes, 1.0, 0.0
    )

    updated = diffuse_labile_amount(
        amount, volumes, 1.0, 0.0, radial, vertical, dt_seconds=0.1
    )

    transfer = 1.0 * jnp.pi / 1.5 * (2.0 - 0.0) * 0.1
    assert updated[0, 0] == pytest.approx(2.0 * jnp.pi - transfer, rel=1e-6)
    assert updated[0, 1] == pytest.approx(transfer, rel=1e-6)
    assert jnp.sum(updated) == pytest.approx(jnp.sum(amount), rel=1e-6)


def test_single_cell_closed_domain_has_no_boundary_flux():
    """Confirms surface, bottom, outer radius, and central axis are all closed."""
    r_edges = jnp.array([0.0, 1.0])
    z_edges = jnp.array([0.0, 1.0])
    volumes, radial, vertical = _diffusion_geometry(r_edges, z_edges)
    amount = jnp.array([[7.0]])

    updated = diffuse_labile_amount(
        amount, volumes, 1.0, 0.0, radial, vertical, dt_seconds=100.0
    )

    assert radial.shape == (0, 1)
    assert vertical.shape == (1, 0)
    assert jnp.array_equal(updated, amount)


def test_diffusion_conserves_amount_and_is_nonnegative_at_safety_limit():
    """Checks closed-domain mass balance and positivity at 0.8 dt_CFL,min."""
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0, 2.0])
    volumes, radial, vertical = _diffusion_geometry(r_edges, z_edges)
    concentration = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    amount = solution_concentration_to_labile_amount(
        concentration, volumes, 1.0, 0.0
    )
    cfl = explicit_diffusion_cfl_seconds(volumes, 1.0, 0.0, radial, vertical)

    updated = diffuse_labile_amount(
        amount, volumes, 1.0, 0.0, radial, vertical, dt_seconds=0.8 * cfl
    )

    assert jnp.all(updated >= 0.0)
    assert jnp.sum(updated) == pytest.approx(jnp.sum(amount), rel=1e-6)


def test_diffusion_amount_update_is_jittable():
    """Keeps the finite-volume update compatible with compiled soil substeps."""
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0])
    volumes, radial, vertical = _diffusion_geometry(r_edges, z_edges)
    step = jax.jit(diffuse_labile_amount)

    updated = step(
        jnp.array([[1.0], [0.0]]),
        volumes,
        1.0,
        0.0,
        radial,
        vertical,
        0.1,
    )

    assert jnp.all(jnp.isfinite(updated))
    assert jnp.all(updated >= 0.0)


def _diffusion_species(jmax=0.0):
    """Return valid consumers with optional uptake for transport integration."""
    return SpeciesParams(
        plant=PlantTraits(initial_biomass=0.0, initial_p_pool=0.0, jmax=jmax),
        fungus=FungusTraits(initial_biomass=0.0, initial_p_pool=0.0, jmax=0.0),
    )


def _subcycling_config(dt_seconds=1.0):
    """Return a two-annulus unbuffered fixture with a 0.5-second exact CFL."""
    return EnvConfig(
        dt=dt_seconds / 86_400.0,
        soil_radius_cm=2.0,
        soil_depth_cm=1.0,
        radial_interval_cm=1.0,
        depth_interval_cm=1.0,
        topsoil_depth_cm=1.0,
        initial_solution_p_um=0.0,
        theta_water=1.0,
        buffer_power=0.0,
        phosphate_diffusion_coefficient_cm2_s=1.0,
        phosphate_impedance_factor=1.0,
        diffusion_cfl_safety=0.8,
        norm_obs=False,
    )


def test_environment_caches_exact_cfl_and_substep_schedule():
    """Builds conductances once and exposes the expected three-step schedule."""
    env = BaseMycorMarl(_subcycling_config(), _diffusion_species())

    assert env.radial_diffusion_conductance.shape == (1, 1)
    assert env.vertical_diffusion_conductance.shape == (2, 0)
    assert env.diffusion_cfl_seconds == pytest.approx(0.5, rel=1e-6)
    assert env.soil_substeps == 3
    assert env.soil_substep_days * env.soil_substeps == pytest.approx(
        env.config.dt, rel=1e-12
    )


def test_one_substep_evolution_matches_direct_diffusion_uptake_substep():
    """Keeps the no-subcycling path identical to one complete soil transaction."""
    env = BaseMycorMarl(_subcycling_config(dt_seconds=0.1), _diffusion_species())
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(soil_labile_p=jnp.array([[2.0 * jnp.pi], [0.0]]))
    root_resistance, fungus_resistance, weight = _p5_coefficients(env, state)

    integrated = env.step_phosphorus_field(state)
    direct = soil_diffusion_uptake_substep(
        state,
        env.soil_substep_days,
        env.species,
        env.cell_volumes,
        env.config.theta_water,
        env.config.buffer_power,
        env.radial_diffusion_conductance,
        env.vertical_diffusion_conductance,
        root_resistance,
        fungus_resistance,
        weight,
    )

    assert env.soil_substeps == 1
    assert jnp.allclose(integrated.soil_labile_p, direct.soil_labile_p)
    assert jnp.allclose(integrated.plant_p_pool, direct.plant_p_pool)


def test_multi_substep_evolution_matches_three_explicit_repetitions():
    """Checks equal substeps sum to dt and repeatedly update concentration."""
    env = BaseMycorMarl(_subcycling_config(), _diffusion_species(jmax=1e-3))
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        soil_labile_p=jnp.array([[2.0 * jnp.pi], [0.0]]),
        root_length_density=jnp.array([[0.0], [1.0]]),
    )
    root_resistance, fungus_resistance, weight = _p5_coefficients(env, state)

    integrated = env.step_phosphorus_field(state)
    repeated = state
    for _ in range(3):
        repeated = soil_diffusion_uptake_substep(
            repeated,
            env.soil_substep_days,
            env.species,
            env.cell_volumes,
            env.config.theta_water,
            env.config.buffer_power,
            env.radial_diffusion_conductance,
            env.vertical_diffusion_conductance,
            root_resistance,
            fungus_resistance,
            weight,
        )

    assert env.soil_substeps == 3
    assert jnp.allclose(integrated.soil_labile_p, repeated.soil_labile_p, rtol=1e-6)
    assert jnp.allclose(integrated.plant_p_pool, repeated.plant_p_pool, rtol=1e-6)
    assert integrated.plant_p_pool[0] > state.plant_p_pool[0]


def test_subcycled_diffusion_only_conserves_closed_domain_amount():
    """Extends pairwise conservation through the environment's counted loop."""
    env = BaseMycorMarl(_subcycling_config(), _diffusion_species(jmax=0.0))
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(soil_labile_p=jnp.array([[2.0 * jnp.pi], [0.0]]))

    next_state = env.step_phosphorus_field(state)

    assert env.soil_substeps == 3
    assert jnp.all(next_state.soil_labile_p >= 0.0)
    assert jnp.sum(next_state.soil_labile_p) == pytest.approx(
        jnp.sum(state.soil_labile_p), rel=1e-6
    )
    assert jnp.array_equal(next_state.plant_p_pool, state.plant_p_pool)
    assert jnp.array_equal(next_state.fungus_p_pool, state.fungus_p_pool)


def test_diffusion_precedes_uptake_and_refreshes_receiver_concentration():
    """Allows roots in an initially empty neighbour to absorb newly diffused P."""
    env = BaseMycorMarl(_subcycling_config(dt_seconds=0.1), _diffusion_species(jmax=1.0))
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        soil_labile_p=jnp.array([[2.0 * jnp.pi], [0.0]]),
        root_length_density=jnp.array([[0.0], [1.0]]),
    )

    next_state = env.step_phosphorus_field(state)

    assert state.soil_labile_p[1, 0] == pytest.approx(0.0)
    assert next_state.plant_p_pool[0] > state.plant_p_pool[0]


def test_subcycled_soil_evolution_is_jittable():
    """Compiles a dynamic counted loop over complete diffusion–uptake substeps."""
    env = BaseMycorMarl(_subcycling_config(), _diffusion_species(jmax=1e-3))
    _, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(evolve_soil_p)

    next_state = step(
        state,
        env.config.dt,
        env.species,
        env.cell_volumes,
        env.config.theta_water,
        env.config.buffer_power,
        env.radial_diffusion_conductance,
        env.vertical_diffusion_conductance,
        env.soil_substeps,
        env.config.phosphate_diffusion_coefficient_cm2_s,
        env.config.phosphate_impedance_factor,
        env.config.uptake_reference_time_days,
        env.config.uptake_transition_exponent,
    )

    assert jnp.all(jnp.isfinite(next_state.soil_labile_p))
    assert jnp.all(next_state.soil_labile_p >= 0.0)
