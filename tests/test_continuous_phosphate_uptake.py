"""Scientific contracts for P3 continuous phosphate uptake and competition."""

import jax
import jax.numpy as jnp
import pytest

from mycormarl.fungus.traits import FungusTraits
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_uptake import (
    allocate_competing_uptake,
    continuous_uptake_request,
)


REL_TOL = 1e-6


def test_continuous_kinetic_defaults_match_agreed_reference():
    """Locks the shared Tinker–Nye kinetic values and centimetre-based units."""
    plant = PlantTraits()
    fungus = FungusTraits()

    assert plant.jmax == pytest.approx(3.26e-6)
    assert fungus.jmax == pytest.approx(3.26e-6)
    assert plant.km == pytest.approx(5.8e-3)
    assert fungus.km == pytest.approx(5.8e-3)


def test_continuous_request_matches_independent_flux_area_time_product():
    """Verifies µmol/cell request from concentration, represented length, and dt."""
    concentration = jnp.array([[1e-3]])
    density = jnp.array([[3.0]])
    volume = jnp.array([[2.0]])

    request = continuous_uptake_request(
        concentration,
        density,
        volume,
        absorber_radius_cm=0.01,
        j_max=3.26e-6,
        k_m=5.8e-3,
        dt_days=0.05,
    )

    flux = 3.26e-6 * 1e-3 / (5.8e-3 + 1e-3)
    length = 3.0 * 2.0
    area = 2.0 * jnp.pi * 0.01 * length
    expected = flux * area * (0.05 * 86_400.0)
    assert request[0, 0] == pytest.approx(expected, rel=REL_TOL)


def test_continuous_request_has_physical_zero_and_monotonic_limits():
    """Checks absent P or absorber gives zero and more P/density raises demand."""
    concentrations = jnp.array([0.0, 1e-3, 2e-3])
    densities = jnp.array([4.0, 4.0, 8.0])

    request = continuous_uptake_request(
        concentrations,
        densities,
        jnp.ones_like(concentrations),
        absorber_radius_cm=0.01,
        j_max=3.26e-6,
        k_m=5.8e-3,
        dt_days=0.05,
    )
    no_absorber = continuous_uptake_request(
        1e-3,
        0.0,
        1.0,
        absorber_radius_cm=0.01,
        j_max=3.26e-6,
        k_m=5.8e-3,
        dt_days=0.05,
    )

    assert request[0] == pytest.approx(0.0)
    assert request[2] > request[1] > request[0]
    assert no_absorber == pytest.approx(0.0)


def test_continuous_request_is_jittable():
    """Keeps the P3 request calculation usable inside the compiled environment."""
    request_fn = jax.jit(continuous_uptake_request)

    request = request_fn(
        jnp.array([[1e-3, 2e-3]]),
        jnp.array([[1.0, 2.0]]),
        jnp.array([[3.0, 4.0]]),
        0.01,
        3.26e-6,
        5.8e-3,
        0.05,
    )

    assert jnp.all(jnp.isfinite(request))
    assert jnp.all(request > 0.0)


def test_competition_leaves_uncapped_requests_unchanged():
    """Accepts both consumers in full when their combined request is available."""
    remaining, root, fungus = allocate_competing_uptake(
        labile_amount_micromol=jnp.array([10.0]),
        root_request_micromol=jnp.array([2.0]),
        fungus_request_micromol=jnp.array([3.0]),
    )

    assert root[0] == pytest.approx(2.0)
    assert fungus[0] == pytest.approx(3.0)
    assert remaining[0] == pytest.approx(5.0)


def test_oversubscribed_competition_preserves_request_shares():
    """Scales simultaneous root/fungal demand once against shared inventory."""
    remaining, root, fungus = allocate_competing_uptake(
        labile_amount_micromol=jnp.array([3.0]),
        root_request_micromol=jnp.array([4.0]),
        fungus_request_micromol=jnp.array([8.0]),
    )

    assert root[0] == pytest.approx(1.0)
    assert fungus[0] == pytest.approx(2.0)
    assert remaining[0] == pytest.approx(0.0)


def test_competition_handles_zero_inventory_and_zero_demand_without_nan():
    """Protects empty cells and absorber-free cells with an exact zero guard."""
    remaining, root, fungus = allocate_competing_uptake(
        labile_amount_micromol=jnp.array([0.0, 5.0]),
        root_request_micromol=jnp.array([1.0, 0.0]),
        fungus_request_micromol=jnp.array([2.0, 0.0]),
    )

    assert jnp.all(jnp.isfinite(remaining))
    assert jnp.all(jnp.isfinite(root))
    assert jnp.all(jnp.isfinite(fungus))
    assert jnp.allclose(remaining, jnp.array([0.0, 5.0]))
    assert jnp.all(root == 0.0)
    assert jnp.all(fungus == 0.0)


def test_competition_is_symmetric_and_cellwise_conservative_under_jit():
    """Checks identical consumers split supply equally and exactly account for loss."""
    allocation_fn = jax.jit(allocate_competing_uptake)
    available = jnp.array([1.0, 10.0])
    request = jnp.array([2.0, 3.0])

    remaining, root, fungus = allocation_fn(available, request, request)

    assert jnp.allclose(root, fungus, rtol=REL_TOL)
    assert jnp.all(remaining >= 0.0)
    assert jnp.all(root <= request)
    assert jnp.all(fungus <= request)
    assert jnp.allclose(available - remaining, root + fungus, rtol=REL_TOL)
