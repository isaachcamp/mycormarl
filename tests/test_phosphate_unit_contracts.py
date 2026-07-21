import jax
import jax.numpy as jnp
import pytest

from mycormarl.soil.phosphate_units import (
    MICROMOLAR_TO_MICROMOL_PER_CM3,
    MICROMOL_P_TO_MG_P,
    SECONDS_PER_DAY,
    cylindrical_lateral_area,
    days_to_seconds,
    dissolved_labile_fraction,
    labile_capacity_factor,
    michaelis_menten_surface_flux,
    micromolar_to_micromol_per_cm3,
    micromol_p_to_mg_p,
    retardation_factor,
    validate_linear_buffer_parameters,
    validate_michaelis_menten_parameters,
)


REL_TOL = 1e-6


def test_phosphate_conversion_constants_have_expected_values():
    """Locks the three cross-boundary conversion constants used by the pipeline."""
    assert MICROMOLAR_TO_MICROMOL_PER_CM3 == pytest.approx(1e-3)
    assert MICROMOL_P_TO_MG_P == pytest.approx(0.0309738)
    assert SECONDS_PER_DAY == pytest.approx(86_400.0)


def test_configuration_units_convert_to_physical_kernel_units():
    """Checks user-facing µM/days and soil µmol convert to kernel/pool units."""
    assert micromolar_to_micromol_per_cm3(1.0) == pytest.approx(1e-3)
    assert micromol_p_to_mg_p(1.0) == pytest.approx(0.0309738)
    assert days_to_seconds(0.05) == pytest.approx(4_320.0)


def test_schnepf_roose_linear_buffer_identities():
    """Guards the strong-buffer reference capacity, retardation, and fraction."""
    theta_water = 0.3
    buffer_power = 239.0

    capacity = labile_capacity_factor(theta_water, buffer_power)
    retardation = retardation_factor(theta_water, buffer_power)
    dissolved_fraction = dissolved_labile_fraction(theta_water, buffer_power)

    assert capacity == pytest.approx(239.3, rel=REL_TOL)
    assert retardation == pytest.approx(797.6666667, rel=REL_TOL)
    assert dissolved_fraction == pytest.approx(0.0012536565, rel=REL_TOL)
    assert dissolved_fraction == pytest.approx(1.0 / retardation, rel=REL_TOL)


def test_michaelis_menten_reference_at_one_micromolar():
    """Anchors uptake kinetics at the nominal 1 µM initial concentration."""
    concentration = micromolar_to_micromol_per_cm3(1.0)
    j_max = 3.26e-6
    k_m = 5.8e-3

    flux = michaelis_menten_surface_flux(concentration, j_max, k_m)

    assert flux / j_max == pytest.approx(1.0 / 6.8, rel=REL_TOL)
    assert flux == pytest.approx(4.794117647e-7, rel=REL_TOL)


@pytest.mark.parametrize(
    ("radius_cm", "expected_area_cm2"),
    [
        (5e-4, 3.141592654e-3),
        (1e-2, 6.283185307e-2),
    ],
)
def test_one_centimetre_cylinder_lateral_area_excludes_end_caps(
    radius_cm, expected_area_cm2
):
    """Verifies root/hyphal length becomes lateral absorbing area only."""
    area = cylindrical_lateral_area(length_cm=1.0, radius_cm=radius_cm)

    assert area == pytest.approx(expected_area_cm2, rel=REL_TOL)


def test_zero_concentration_and_zero_length_have_zero_flux_and_area():
    """Checks physical zero limits before flux and area enter uptake requests."""
    assert michaelis_menten_surface_flux(0.0, 3.26e-6, 5.8e-3) == pytest.approx(0.0)
    assert cylindrical_lateral_area(0.0, 5e-4) == pytest.approx(0.0)


def test_array_helpers_are_jittable():
    """Ensures conversion, kinetics, and area helpers survive JAX compilation."""
    convert_concentration = jax.jit(micromolar_to_micromol_per_cm3)
    surface_flux = jax.jit(michaelis_menten_surface_flux)
    lateral_area = jax.jit(cylindrical_lateral_area)

    concentrations_um = jnp.array([0.0, 1.0, 10.0])
    concentrations = convert_concentration(concentrations_um)
    fluxes = surface_flux(concentrations, 3.26e-6, 5.8e-3)
    areas = lateral_area(jnp.array([0.0, 1.0, 2.0]), 5e-4)

    assert jnp.allclose(concentrations, jnp.array([0.0, 1e-3, 1e-2]))
    assert jnp.all(jnp.isfinite(fluxes))
    assert jnp.all(fluxes >= 0.0)
    assert jnp.allclose(
        areas,
        jnp.array([0.0, jnp.pi * 1e-3, jnp.pi * 2e-3]),
    )


@pytest.mark.parametrize(
    ("theta_water", "buffer_power"),
    [
        (0.0, 239.0),
        (-0.1, 239.0),
        (float("nan"), 239.0),
        (0.3, -1.0),
        (0.3, float("inf")),
    ],
)
def test_invalid_linear_buffer_parameters_fail_fast(theta_water, buffer_power):
    """Prevents invalid buffering from reaching amount/concentration kernels."""
    with pytest.raises(ValueError):
        validate_linear_buffer_parameters(theta_water, buffer_power)


@pytest.mark.parametrize(
    ("j_max", "k_m"),
    [
        (-1.0, 5.8e-3),
        (float("nan"), 5.8e-3),
        (3.26e-6, 0.0),
        (3.26e-6, -1.0),
        (3.26e-6, float("inf")),
    ],
)
def test_invalid_michaelis_menten_parameters_fail_fast(j_max, k_m):
    """Prevents invalid uptake kinetics from reaching transformed kernels."""
    with pytest.raises(ValueError):
        validate_michaelis_menten_parameters(j_max, k_m)
