"""Scientific contracts for the P5 sparse-to-continuous uptake closure.

These tests isolate the analytical model from the environment. They verify
the geometry and units behind depletion-zone overlap, the stable physical
root for surface concentration, and request blending before shared supply is
allocated. Integrated soil-pipeline behaviour remains in the environment and
diffusion test modules.
"""

import jax
import jax.numpy as jnp
import pytest

from mycormarl.soil.phosphate_diffusion import apparent_diffusivity_cm2_s
from mycormarl.soil.phosphate_uptake import (
    blend_uptake_requests,
    continuous_regime_weight,
    effective_uptake_radius_cm,
    hyphal_overlap_time_seconds,
    sparse_surface_concentration,
    sparse_uptake_request,
    sparse_uptake_resistance,
    territory_radius_cm,
)
from mycormarl.soil.phosphate_units import days_to_seconds


def test_territory_radius_matches_cylindrical_area_partition():
    """One unit length times its circular territory must occupy 1/lambda."""
    density = jnp.array([1.0 / jnp.pi, 4.0 / jnp.pi, 0.0])

    radius = territory_radius_cm(density)

    assert radius[0] == pytest.approx(1.0)
    assert radius[1] == pytest.approx(0.5)
    assert jnp.isinf(radius[2])


def test_nominal_hyphal_overlap_time_is_about_five_and_a_half_days():
    """Locks the agreed Schnepf–Roose-based saturated-density estimate."""
    d_app = apparent_diffusivity_cm2_s(1e-5, 0.3, 0.308, 239.0)

    overlap_seconds = hyphal_overlap_time_seconds(
        jnp.asarray(168.75),
        hyphal_radius_cm=5e-4,
        apparent_diffusivity_cm2_s=d_app,
    )

    assert overlap_seconds / days_to_seconds(1.0) == pytest.approx(5.5, rel=0.03)


def test_overlap_weight_handles_absent_dense_and_zero_diffusion_limits():
    """No hyphae stay sparse; a closed gap becomes continuous if P diffuses."""
    d_app = apparent_diffusivity_cm2_s(1e-5, 0.3, 0.308, 239.0)
    density = jnp.array([0.0, 168.75, 2e6])
    times = hyphal_overlap_time_seconds(density, 5e-4, d_app)
    weights = continuous_regime_weight(times, reference_time_days=1.0, exponent=2.0)

    assert jnp.isinf(times[0])
    assert weights[0] == pytest.approx(0.0)
    assert weights[1] == pytest.approx(1.0 / (1.0 + 5.5**2), rel=0.06)
    assert times[2] == pytest.approx(0.0)
    assert weights[2] == pytest.approx(1.0)

    no_diffusion = hyphal_overlap_time_seconds(density[2], 5e-4, 0.0)
    assert jnp.isinf(no_diffusion)
    assert continuous_regime_weight(no_diffusion, 1.0, 2.0) == pytest.approx(0.0)


def test_effective_radius_is_bounded_by_territory_and_diffusion_distance():
    """The analytical annulus cannot extend beyond either physical bound."""
    radius = 0.01
    d_app = 1e-6
    t_ref_seconds = float(days_to_seconds(1.0))
    propagation = (d_app * t_ref_seconds) ** 0.5

    sparse = effective_uptake_radius_cm(0.01, radius, d_app, 1.0)
    crowded = effective_uptake_radius_cm(1e8, radius, d_app, 1.0)

    assert sparse == pytest.approx(radius + propagation)
    assert crowded == pytest.approx(radius)


@pytest.mark.parametrize("bulk", [0.0, 1e-12, 1e-3, 1.0])
@pytest.mark.parametrize("resistance", [0.0, 1e-8, 0.2, 1e6, jnp.inf])
def test_surface_concentration_is_bounded_and_satisfies_quadratic(bulk, resistance):
    """The stable root remains physical across zero and extreme resistance."""
    k_m = 5.8e-3
    surface = sparse_surface_concentration(bulk, k_m, resistance)

    assert jnp.isfinite(surface)
    assert 0.0 <= surface <= bulk
    if jnp.isfinite(resistance):
        residual = surface**2 - (bulk - k_m - resistance) * surface - bulk * k_m
        scale = max(bulk * k_m, surface**2, 1e-20)
        assert residual == pytest.approx(0.0, abs=2e-5 * scale)
    if resistance == 0.0:
        assert surface == pytest.approx(bulk)
    if jnp.isinf(resistance):
        assert surface == pytest.approx(0.0)


def test_surface_concentration_avoids_cancellation_for_large_positive_a():
    """The piecewise quadratic evaluation also stays stable when C_b is huge."""
    bulk = jnp.asarray(1e8)

    surface = sparse_surface_concentration(bulk, 5.8e-3, 1e-8)

    assert jnp.isfinite(surface)
    assert surface > 0.99 * bulk


def test_sparse_request_matches_surface_flux_area_time_and_zero_limits():
    """Sparse uptake uses represented length and its depleted surface value."""
    concentration = jnp.array([1e-3, 1e-3, 0.0])
    density = jnp.array([2.0, 0.0, 2.0])
    volumes = jnp.array([3.0, 3.0, 3.0])
    resistance = jnp.array([0.2, 0.2, 0.2])
    radius = 0.01
    j_max = 3.26e-6
    k_m = 5.8e-3
    dt_days = 0.05
    surface = sparse_surface_concentration(concentration, k_m, resistance)

    request = sparse_uptake_request(
        concentration,
        density,
        volumes,
        radius,
        j_max,
        k_m,
        dt_days,
        resistance,
    )
    expected_first = (
        2.0
        * jnp.pi
        * radius
        * density[0]
        * volumes[0]
        * j_max
        * surface[0]
        / (k_m + surface[0])
        * days_to_seconds(dt_days)
    )

    assert request[0] == pytest.approx(expected_first, rel=1e-6)
    assert request[1] == pytest.approx(0.0)
    assert request[2] == pytest.approx(0.0)


def test_zero_diffusive_supply_gives_infinite_resistance_and_zero_sparse_request():
    """A disabled transport coefficient cannot feed the analytical absorber."""
    resistance = sparse_uptake_resistance(
        jnp.array([2.0]),
        absorber_radius_cm=0.01,
        j_max=3.26e-6,
        amount_flux_diffusivity_cm2_s=0.0,
        apparent_diffusivity_cm2_s=0.0,
        reference_time_days=1.0,
    )
    request = sparse_uptake_request(
        jnp.array([1e-3]),
        jnp.array([2.0]),
        jnp.array([3.0]),
        0.01,
        3.26e-6,
        5.8e-3,
        0.05,
        resistance,
    )

    assert jnp.isinf(resistance[0])
    assert request[0] == pytest.approx(0.0)


def test_blending_uses_one_weight_and_recovers_both_limits():
    """P5 interpolates alternative requests rather than summing two sinks."""
    sparse = jnp.array([2.0, 4.0, 6.0])
    continuous = jnp.array([10.0, 12.0, 14.0])
    weight = jnp.array([0.0, 0.25, 1.0])

    blended = blend_uptake_requests(sparse, continuous, weight)

    assert jnp.allclose(blended, jnp.array([2.0, 6.0, 14.0]))


def test_sparse_primitives_are_jittable():
    """All concentration-dependent P5 calculations remain JAX-transformable."""
    kernel = jax.jit(
        lambda concentration, resistance: sparse_uptake_request(
            concentration,
            jnp.ones_like(concentration),
            jnp.ones_like(concentration),
            0.01,
            3.26e-6,
            5.8e-3,
            0.05,
            resistance,
        )
    )

    result = kernel(jnp.array([0.0, 1e-3]), jnp.array([0.2, 0.2]))

    assert jnp.all(jnp.isfinite(result))
    assert jnp.all(result >= 0.0)
