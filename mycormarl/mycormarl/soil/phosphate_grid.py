
import math

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
    """Reject invalid physical extents or non-uniform interval requests.

    This protects every downstream geometry, amount conversion, and numerical
    kernel from empty grids and non-finite or non-positive dimensions. An
    interval must divide its corresponding extent into a whole number of cells;
    otherwise the error suggests the closest interval of the form ``extent/n``.
    """
    for name, value in (
        ("radius_cm", radius_cm),
        ("depth_cm", depth_cm),
        ("radial_interval_cm", radial_interval_cm),
        ("depth_interval_cm", depth_interval_cm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero")

    # Check whether each interval divides its extent into a whole number of cells.
    invalid_intervals = []
    for extent_name, extent, interval_name, interval in (
        ("radius_cm", radius_cm, "radial_interval_cm", radial_interval_cm),
        ("depth_cm", depth_cm, "depth_interval_cm", depth_interval_cm),
    ):
        ratio = extent / interval
        nearest_integer = round(ratio)

        # Only allow exact integer ratios; otherwise suggest the nearest valid
        # uniform interval.
        if nearest_integer >= 1 and math.isclose(
            ratio, nearest_integer, rel_tol=1e-12, abs_tol=1e-12
        ):
            continue

        # Suggest the nearest valid uniform interval of the form ``extent/n``.
        lower_cells = max(1, math.floor(ratio))
        upper_cells = max(1, math.ceil(ratio))
        candidate_cells = {lower_cells, upper_cells}
        suggested_cells = min(
            candidate_cells,
            key=lambda n: (
                abs(extent / n - interval),
                extent / n > interval,
            ),
        )
        suggested_interval = extent / suggested_cells
        invalid_intervals.append(
            f"{interval_name}={interval:.12g} cm does not divide "
            f"{extent_name}={extent:.12g} cm; nearest valid uniform interval "
            f"is {suggested_interval:.12g} cm ({suggested_cells} cells)"
        )

    if invalid_intervals:
        raise ValueError(". ".join(invalid_intervals))


def _edges_from_interval(maximum_cm: float, interval_cm: float) -> chex.Array:
    """Return exact requested uniform edges after public validation."""
    n_cells = int(round(maximum_cm / interval_cm))
    edges = jnp.arange(n_cells + 1, dtype=jnp.float32) * interval_cm
    return edges.at[-1].set(maximum_cm)


def axisymmetric_edges_from_intervals(
        radius_cm: float,
        depth_cm: float,
        radial_interval_cm: float,
        depth_interval_cm: float,
    ) -> tuple[chex.Array, chex.Array]:
    """Generate radial and depth boundaries from explicit uniform spacing.

    These edges define the shared geometry for phosphate amount, length
    density, diffusion faces, and spatial uptake. Each requested interval must
    divide its extent exactly; invalid requests fail with the nearest valid
    uniform interval rather than silently creating a shortened boundary cell.
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


def axisymmetric_profile_p_concentration(
        r_edges: chex.Array,
        z_edges: chex.Array,
        surface_concentration_um: float,
        depth_profile: tuple[tuple[float, float], ...] | list[list[float]],
    ) -> chex.Array:
    """Return a radial-uniform, linearly interpolated solution-P field.

    ``depth_profile`` contains ``(depth_cm, relative_factor)`` knots. The
    configured surface concentration is the value at the first knot; shallower
    cell centres retain that first factor. Domain construction validates that
    the final knot covers the represented depth, so this helper never silently
    extrapolates into unobserved subsoil.
    """
    knots = jnp.asarray(depth_profile, dtype=jnp.float32)
    z_centres = (jnp.asarray(z_edges[:-1]) + jnp.asarray(z_edges[1:])) / 2.0
    factors = jnp.interp(
        z_centres,
        knots[:, 0],
        knots[:, 1],
        left=knots[0, 1],
        right=knots[-1, 1],
    )
    concentration = (
        jnp.asarray(surface_concentration_um, dtype=jnp.float32)
        * factors
        * 1e-3
    )
    return jnp.broadcast_to(
        concentration[None, :], (len(r_edges) - 1, len(z_centres))
    )


def solution_concentration_to_labile_amount(
        concentration_micromol_cm3: chex.Array,
        cell_volumes_cm3: chex.Array,
        theta_water: float,
        b_p: float,
    ) -> chex.Array:
    """Convert solution concentration to canonical labile P in µmol/cell.

    Implements ``M = C * V * (theta + b_p)``. Diffusion and uptake will mutate
    this conserved amount rather than subtracting amounts from concentration.
    """
    return (
        jnp.asarray(concentration_micromol_cm3)
        * jnp.asarray(cell_volumes_cm3)
        * labile_capacity_factor(theta_water, b_p)
    )


def labile_amount_to_solution_concentration(
        labile_amount_micromol: chex.Array,
        cell_volumes_cm3: chex.Array,
        theta_water: float,
        b_p: float,
    ) -> chex.Array:
    """Derive solution concentration from canonical labile amount.

    Implements ``C = M / (V * (theta + b_p))``. Diffusion gradients and
    Michaelis–Menten uptake use this derived field on every soil update.
    """
    return jnp.asarray(labile_amount_micromol) / (
        jnp.asarray(cell_volumes_cm3)
        * labile_capacity_factor(theta_water, b_p)
    )


def initial_labile_p_from_micromolar(
        r_edges: chex.Array,
        z_edges: chex.Array,
        concentration_um: float,
        theta_water: float,
        b_p: float,
        depth_profile: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
    ) -> chex.Array:
    """Build the reset-time canonical labile-P field from configured µM.

    With no depth profile it applies one uniform concentration in micromolar
    to every cell; otherwise it scales that concentration by the profile.
    It returns canonical labile amount with shape ``(n_r, n_z)``.

    The function composes unit conversion, depth treatment, annular cell
    volumes, and linear buffering. Its output is stored directly
    as ``State.soil_labile_p``.
    """
    concentration = (
        axisymmetric_profile_p_concentration(
            r_edges,
            z_edges,
            concentration_um,
            depth_profile,
        )
        if depth_profile is not None
        else jnp.full(
            (len(r_edges) - 1, len(z_edges) - 1),
            micromolar_to_micromol_per_cm3(concentration_um),
        )
    )
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)
    return solution_concentration_to_labile_amount(
        concentration,
        volumes,
        theta_water,
        b_p,
    )
