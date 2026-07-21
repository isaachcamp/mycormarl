
import math
from typing import Optional

import chex
import jax.numpy as jnp

from mycormarl.soil.phosphate_units import (
    labile_capacity_factor,
    micromolar_to_micromol_per_cm3,
)


def validate_axisymmetric_grid_parameters(
        radius_cm: float,
        depth_cm: float,
        radial_interval_cm: float,
        depth_interval_cm: float,
    ) -> None:
    """Reject invalid physical extents or intervals before grid allocation.

    This protects every downstream geometry, amount conversion, and numerical
    kernel from empty grids and non-finite or non-positive dimensions.
    """
    for name, value in (
        ("radius_cm", radius_cm),
        ("depth_cm", depth_cm),
        ("radial_interval_cm", radial_interval_cm),
        ("depth_interval_cm", depth_interval_cm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero")


def _edges_from_interval(maximum_cm: float, interval_cm: float) -> chex.Array:
    """Return zero-based edges at the interval, ending exactly at the maximum."""
    ratio = maximum_cm / interval_cm
    nearest_integer = round(ratio)
    if math.isclose(ratio, nearest_integer, rel_tol=1e-12, abs_tol=1e-12):
        n_cells = int(nearest_integer)
    else:
        n_cells = math.ceil(ratio)

    edges = jnp.arange(n_cells + 1, dtype=jnp.float32) * interval_cm
    return edges.at[-1].set(maximum_cm)


def axisymmetric_edges_from_intervals(
        radius_cm: float,
        depth_cm: float,
        radial_interval_cm: float,
        depth_interval_cm: float,
    ) -> tuple[chex.Array, chex.Array]:
    """Generate radial and depth boundaries from requested centimetre spacing.

    These edges define the shared geometry for phosphate amount, length
    density, diffusion faces, and spatial uptake. Each final cell is
    shortened when necessary so its outer edge equals the configured maximum.
    """
    validate_axisymmetric_grid_parameters(
        radius_cm, depth_cm, radial_interval_cm, depth_interval_cm
    )
    return (
        _edges_from_interval(radius_cm, radial_interval_cm),
        _edges_from_interval(depth_cm, depth_interval_cm),
    )


def axisymmetric_cylindrical_cell_volumes(
        r_edges: chex.Array,
        z_edges: chex.Array,
    ) -> chex.Array:
    """Return annular cell volumes for an axisymmetric cylindrical r-z grid.

    ``r_edges`` and ``z_edges`` define radial and depth boundaries. The
    ``(n_r, n_z)`` result converts concentration to conserved amount and later
    integrates root/hyphal length density.

    ### Note var[:, None] adds a new axis e.g., 
    ### var[:, None].shape = (n, 1) and var[None, :].shape = (1, n)
    """
    r_edges = jnp.asarray(r_edges, dtype=jnp.float32)
    z_edges = jnp.asarray(z_edges, dtype=jnp.float32)

    annular_areas = jnp.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    dz = z_edges[1:] - z_edges[:-1]
    return annular_areas[:, None] * dz[None, :]


def axisymmetric_radial_face_areas(
        r_edges: chex.Array,
        z_edges: chex.Array,
    ) -> chex.Array:
    """Return radial-face areas ``2πr dz`` with shape ``(n_r + 1, n_z)``.

    Diffusion fluxes use these areas for conservative radial finite-volume 
    fluxes; the central face is naturally zero and enforces cylindrical symmetry.
    """
    r_edges = jnp.asarray(r_edges, dtype=jnp.float32)
    z_edges = jnp.asarray(z_edges, dtype=jnp.float32)
    dz = z_edges[1:] - z_edges[:-1]
    return 2.0 * jnp.pi * r_edges[:, None] * dz[None, :]


def axisymmetric_vertical_face_areas(
        r_edges: chex.Array,
        z_edges: chex.Array,
    ) -> chex.Array:
    """Return annular horizontal-face areas with shape ``(n_r, n_z + 1)``.
    
    Annular areas are constant with depth, so broadcast used to repeat areas 
    across depth dimension.
    Diffusion fluxes use repeated annular areas for conservative vertical fluxes
    and closed top/bottom boundary conditions.
    """
    r_edges = jnp.asarray(r_edges, dtype=jnp.float32)
    z_edges = jnp.asarray(z_edges, dtype=jnp.float32)
    annular_areas = jnp.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    return jnp.broadcast_to(
        annular_areas[:, None],
        (annular_areas.shape[0], z_edges.shape[0]),
    )


def axisymmetric_topsoil_fractions(
        z_edges: chex.Array,
        topsoil_depth_cm: float,
    ) -> chex.Array:
    """Return the fraction of every depth cell lying above the topsoil limit.

    Fractional occupancy prevents initial inventory from changing when the
    physical topsoil boundary cuts through a finite grid cell.
    
    ### This can be avoided if the topsoil depth is an integer multiple of the
    depth interval.
    """
    z_edges = jnp.asarray(z_edges, dtype=jnp.float32)
    dz = z_edges[1:] - z_edges[:-1]
    return jnp.clip((topsoil_depth_cm - z_edges[:-1]) / dz, 0.0, 1.0)


def axisymmetric_uniform_p_conc(
        r_edges: chex.Array,
        z_edges: chex.Array,
        conc: float,
        topsoil_depth: Optional[float] = None,
    ) -> chex.Array:
    """Construct a uniform phosphorus concentration on an r-z cylindrical grid.

    When ``topsoil_depth`` is provided, subsoil cells are zero and a crossed
    cell stores its volume-averaged concentration. This is an intermediate
    reset field only: the concentration is immediately converted to canonical
    labile amount and is not stored in environment state.
    """
    r_edges = jnp.asarray(r_edges, dtype=jnp.float32)
    z_edges = jnp.asarray(z_edges, dtype=jnp.float32)

    soil_p = jnp.full((r_edges.shape[0] - 1, z_edges.shape[0] - 1), conc, dtype=jnp.float32)
    if topsoil_depth is None:
        return soil_p

    topsoil_fraction = axisymmetric_topsoil_fractions(z_edges, topsoil_depth)
    return soil_p * topsoil_fraction[None, :]


def solution_concentration_to_labile_amount(
        concentration_micromol_cm3: chex.Array,
        cell_volumes_cm3: chex.Array,
        theta_water: float,
        buffer_power: float,
    ) -> chex.Array:
    """Convert solution concentration to canonical labile P in µmol/cell.

    Implements ``M = C * V * (theta + B)``. Diffusion and uptake will mutate
    this conserved amount rather than subtracting amounts from concentration.
    """
    return (
        jnp.asarray(concentration_micromol_cm3)
        * jnp.asarray(cell_volumes_cm3)
        * labile_capacity_factor(theta_water, buffer_power)
    )


def labile_amount_to_solution_concentration(
        labile_amount_micromol: chex.Array,
        cell_volumes_cm3: chex.Array,
        theta_water: float,
        buffer_power: float,
    ) -> chex.Array:
    """Derive solution concentration from canonical labile amount.

    Implements ``C = M / (V * (theta + B))``. Diffusion gradients and
    Michaelis–Menten uptake use this derived field on every soil update.
    """
    return jnp.asarray(labile_amount_micromol) / (
        jnp.asarray(cell_volumes_cm3)
        * labile_capacity_factor(theta_water, buffer_power)
    )


def initial_labile_p_from_micromolar(
        r_edges: chex.Array,
        z_edges: chex.Array,
        concentration_um: float,
        topsoil_depth_cm: Optional[float],
        theta_water: float,
        buffer_power: float,
    ) -> chex.Array:
    """Build the reset-time canonical labile-P field from configured µM.

    It takes in a uniform concentration in micromolar, converts to 
    canonical labile amount, and returns a 2D array of shape (n_r, n_z).

    The function composes unit conversion, partial-topsoil concentration,
    annular cell volumes, and linear buffering. Its output is stored directly
    as ``State.soil_labile_p``.
    """
    concentration = axisymmetric_uniform_p_conc(
        r_edges,
        z_edges,
        conc=micromolar_to_micromol_per_cm3(concentration_um),
        topsoil_depth=topsoil_depth_cm,
    )
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)
    return solution_concentration_to_labile_amount(
        concentration,
        volumes,
        theta_water,
        buffer_power,
    )
