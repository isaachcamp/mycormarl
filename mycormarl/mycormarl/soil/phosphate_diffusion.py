"""Static geometry and stability helpers for axisymmetric P diffusion.

The canonical state stores total labile amount, while diffusion is driven by
soil-solution concentration. This module constructs the internal-face
conductances used by the conservative finite-volume update and calculates its
exact cellwise explicit stability ceiling.
"""

from __future__ import annotations

import math

import chex
import jax.numpy as jnp

from mycormarl.soil.phosphate_grid import labile_amount_to_solution_concentration
from mycormarl.soil.phosphate_units import SECONDS_PER_DAY, labile_capacity_factor


def apparent_diffusivity_cm2_s(
    diffusion_coefficient_cm2_s: float,
    theta_water: float,
    impedance_factor: float,
    buffer_power: float,
) -> chex.Array:
    """Return buffered apparent diffusivity ``D_l theta f_l/(theta+B)``.

    This diagnostic describes propagation of a disturbance in the complete
    labile inventory. The conservative amount-flux kernel must instead use
    ``D_l theta f_l`` because buffering is already present in the conversion
    between stored amount and solution concentration.
    """
    return (
        jnp.asarray(diffusion_coefficient_cm2_s)
        * jnp.asarray(theta_water)
        * jnp.asarray(impedance_factor)
        / labile_capacity_factor(theta_water, buffer_power)
    )


def axisymmetric_diffusion_conductances(
    r_edges: chex.Array,
    z_edges: chex.Array,
    radial_face_areas_cm2: chex.Array,
    vertical_face_areas_cm2: chex.Array,
    diffusion_coefficient_cm2_s: float,
    theta_water: float,
    impedance_factor: float,
) -> tuple[chex.Array, chex.Array]:
    """Return internal radial and vertical conductances in cm³ s⁻¹.

    Each conductance is ``D_l theta f_l A/d`` using the distance between the
    actual neighbouring cell centres. External faces are omitted, enforcing
    closed boundaries; the zero-area central radial face is omitted as well.
    This handles a shortened final grid cell without assuming uniform spacing.
    """
    r_edges = jnp.asarray(r_edges)
    z_edges = jnp.asarray(z_edges)
    radial_face_areas_cm2 = jnp.asarray(radial_face_areas_cm2)
    vertical_face_areas_cm2 = jnp.asarray(vertical_face_areas_cm2)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    d_flux = (
        jnp.asarray(diffusion_coefficient_cm2_s)
        * jnp.asarray(theta_water)
        * jnp.asarray(impedance_factor)
    )
    radial = (
        d_flux
        * radial_face_areas_cm2[1:-1, :]
        / jnp.diff(r_centres)[:, None]
    )
    vertical = (
        d_flux
        * vertical_face_areas_cm2[:, 1:-1]
        / jnp.diff(z_centres)[None, :]
    )
    return radial, vertical


def cell_outgoing_diffusion_conductance(
    cell_volumes_cm3: chex.Array,
    radial_conductance: chex.Array,
    vertical_conductance: chex.Array,
) -> chex.Array:
    """Sum all internal-face conductances incident on every soil cell."""
    outgoing = jnp.zeros_like(jnp.asarray(cell_volumes_cm3))
    outgoing = outgoing.at[:-1, :].add(radial_conductance)
    outgoing = outgoing.at[1:, :].add(radial_conductance)
    outgoing = outgoing.at[:, :-1].add(vertical_conductance)
    outgoing = outgoing.at[:, 1:].add(vertical_conductance)
    return outgoing


def diffuse_labile_amount(
    labile_amount_micromol: chex.Array,
    cell_volumes_cm3: chex.Array,
    theta_water: float,
    buffer_power: float,
    radial_conductance: chex.Array,
    vertical_conductance: chex.Array,
    dt_seconds: float,
) -> chex.Array:
    """Advance canonical labile amount by one conservative explicit step.

    Internal-face transfer is positive from the lower-index cell toward the
    higher-index cell when its solution concentration is larger. Every face
    amount is subtracted from one cell and added to its neighbour, so closed
    boundaries conserve domain-total µmol P. Callers must keep ``dt_seconds``
    at or below the configured safety fraction of the exact CFL ceiling.
    """
    amount = jnp.asarray(labile_amount_micromol)
    concentration = labile_amount_to_solution_concentration(
        amount, cell_volumes_cm3, theta_water, buffer_power
    )
    radial_transfer = (
        jnp.asarray(radial_conductance)
        * (concentration[:-1, :] - concentration[1:, :])
        * jnp.asarray(dt_seconds)
    )
    vertical_transfer = (
        jnp.asarray(vertical_conductance)
        * (concentration[:, :-1] - concentration[:, 1:])
        * jnp.asarray(dt_seconds)
    )
    delta = jnp.zeros_like(amount)
    delta = delta.at[:-1, :].add(-radial_transfer)
    delta = delta.at[1:, :].add(radial_transfer)
    delta = delta.at[:, :-1].add(-vertical_transfer)
    delta = delta.at[:, 1:].add(vertical_transfer)
    return amount + delta


def explicit_diffusion_cfl_seconds(
    cell_volumes_cm3: chex.Array,
    theta_water: float,
    buffer_power: float,
    radial_conductance: chex.Array,
    vertical_conductance: chex.Array,
) -> chex.Array:
    """Return the minimum exact cellwise positivity ceiling in seconds.

    For cell ``i`` the explicit finite-volume limit is
    ``V_i(theta+B)/sum_j(G_ij)``. Cells with no diffusive neighbours have an
    infinite ceiling and therefore cannot restrict the global timestep.
    """
    volumes = jnp.asarray(cell_volumes_cm3)
    outgoing = cell_outgoing_diffusion_conductance(
        volumes, radial_conductance, vertical_conductance
    )
    capacity = volumes * labile_capacity_factor(theta_water, buffer_power)
    safe_outgoing = jnp.where(outgoing > 0.0, outgoing, 1.0)
    cell_limits = jnp.where(outgoing > 0.0, capacity / safe_outgoing, jnp.inf)
    return jnp.min(cell_limits)


def required_diffusion_substeps(
    dt_days: float,
    cfl_seconds: float,
    safety_factor: float,
) -> int:
    """Return ``max(1, ceil(dt_seconds/(safety*dt_CFL)))``.

    The calculation runs once at environment construction because geometry,
    buffering, and transport parameters are static in the initial model.
    Infinite CFL denotes disabled diffusion and requires only one soil step.
    """
    if math.isinf(cfl_seconds):
        return 1
    allowed_seconds = safety_factor * cfl_seconds
    return max(1, math.ceil(dt_days * SECONDS_PER_DAY / allowed_seconds))


def validate_diffusion_parameters(
    diffusion_coefficient_cm2_s: float,
    theta_water: float,
    impedance_factor: float,
    safety_factor: float,
) -> None:
    """Reject invalid transport scalars before conductance arrays are built."""
    if not math.isfinite(diffusion_coefficient_cm2_s) or (
        diffusion_coefficient_cm2_s < 0.0
    ):
        raise ValueError(
            "phosphate_diffusion_coefficient_cm2_s must be finite and non-negative"
        )
    if not math.isfinite(theta_water) or theta_water <= 0.0:
        raise ValueError("theta_water must be finite and greater than zero")
    if not math.isfinite(impedance_factor) or not 0.0 <= impedance_factor <= 1.0:
        raise ValueError("phosphate_impedance_factor must be finite and within [0, 1]")
    if not math.isfinite(safety_factor) or not 0.0 < safety_factor <= 1.0:
        raise ValueError("diffusion_cfl_safety must be finite and within (0, 1]")
