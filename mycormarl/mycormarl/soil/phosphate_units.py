"""Dimensional contract and pure helpers for soil-phosphate calculations.

Configuration may express solution concentration in micromolar and biological
time in days. Physical uptake kernels use ``µmol P cm^-3`` and seconds, while
organism phosphorus pools use milligrams. This module makes those boundaries
explicit without owning or migrating environment state.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


MICROMOLAR_TO_MICROMOL_PER_CM3 = 1e-3
"""Multiply a concentration in µM by this to obtain µmol cm^-3."""

MICROMOL_P_TO_MG_P = 0.0309738
"""Phosphorus mass in milligrams per micromole of elemental P."""

SECONDS_PER_DAY = 86_400.0
"""Physical seconds in one model day."""


def micromolar_to_micromol_per_cm3(concentration_um):
    """Convert configured solution P from µM to ``µmol P cm^-3``.

    User-facing concentrations enter the pipeline in µM; buffering, diffusion,
    and uptake kernels require centimetre-based concentration units.
    """
    return jnp.asarray(concentration_um) * MICROMOLAR_TO_MICROMOL_PER_CM3


def micromol_p_to_mg_p(amount_micromol):
    """Convert soil uptake from µmol P to organism-pool milligrams.

    This is the sole intended unit boundary between the soil uptake
    calculation and plant/fungal phosphorus pools.
    """
    return jnp.asarray(amount_micromol) * MICROMOL_P_TO_MG_P


def days_to_seconds(days):
    """Convert the biological timestep in days to physical seconds.

    Environment scheduling remains day-based while physical diffusion and
    surface-flux rates are expressed per second.
    """
    return jnp.asarray(days) * SECONDS_PER_DAY


def labile_capacity_factor(theta_water, b_p):
    """Return the linear-buffer storage factor ``theta_water + b_p``.

    Multiplying this factor by bulk cell volume and solution concentration
    gives the total reversibly labile amount stored in that cell.
    """
    return jnp.asarray(theta_water) + jnp.asarray(b_p)


def retardation_factor(theta_water, b_p):
    """Return ``(theta + b_p) / theta``, the buffering retardation factor.

    Diagnostic used to relate solute-only diffusion to the slower
    apparent propagation of the complete labile inventory.
    """
    return labile_capacity_factor(theta_water, b_p) / jnp.asarray(theta_water)


def dissolved_labile_fraction(theta_water, b_p):
    """Return ``theta / (theta + b_p)``, the dissolved labile-P fraction.

    This diagnostic explains why solution concentration can be low while the
    reversibly available cell inventory is much larger.
    """
    return jnp.asarray(theta_water) / labile_capacity_factor(theta_water, b_p)


def michaelis_menten_surface_flux(concentration, j_max, k_m):
    """Return uptake flux in µmol P cm^-2 s^-1.

    ``concentration`` and ``k_m`` must both use µmol P cm^-3. This supplies the
    concentration-dependent flux used by both sparse and continuous uptake
    closures. Validation belongs at configuration boundaries so the function
    remains JAX-transformable; physical callers must supply non-negative
    concentration.
    """
    concentration = jnp.asarray(concentration)
    return jnp.asarray(j_max) * concentration / (jnp.asarray(k_m) + concentration)


def cylindrical_lateral_area(length_cm, radius_cm):
    """Return lateral absorbing area in cm² for a cylindrical length.

    Length density is first integrated over cell volume to obtain length; this
    helper then converts that length to uptake-active surface. End caps are
    intentionally excluded because roots and hyphae are represented as long
    cylindrical structures.

    """
    return 2.0 * jnp.pi * jnp.asarray(radius_cm) * jnp.asarray(length_cm)


def validate_linear_buffer_parameters(theta_water: float, b_p: float) -> None:
    """Reject invalid buffer configuration before JAX arrays are constructed.

    Keeping scalar validation outside model kernels avoids traced Python
    branches while preventing zero denominators and negative storage capacity.
    """
    if not math.isfinite(theta_water) or theta_water <= 0.0:
        raise ValueError("theta_water must be finite and greater than zero")
    if not math.isfinite(b_p) or b_p < 0.0:
        raise ValueError("b_p must be finite and non-negative")


def validate_michaelis_menten_parameters(j_max: float, k_m: float) -> None:
    """Reject invalid uptake kinetics at the configuration boundary.

    ``j_max`` may be zero to disable a consumer; ``k_m`` must remain positive
    so the later surface-flux calculation has a valid denominator.
    """
    if not math.isfinite(j_max) or j_max < 0.0:
        raise ValueError("j_max must be finite and non-negative")
    if not math.isfinite(k_m) or k_m <= 0.0:
        raise ValueError("k_m must be finite and greater than zero")
