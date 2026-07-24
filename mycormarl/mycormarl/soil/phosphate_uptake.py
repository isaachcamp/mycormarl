"""Sparse/continuous phosphate uptake and shared inventory competition.

The module owns both analytical uptake requests and their amount-conservative
competition transaction. All spatial inputs use centimetres, concentration
uses µmol P cm^-3, time enters in model days, and returned cell amounts use
µmol P. Geometry-only uptake coefficients can be calculated once per biological
step while concentration-dependent requests are refreshed after diffusion.
"""

from __future__ import annotations

from typing import NamedTuple

import chex
import jax.numpy as jnp

from mycormarl.params import SpeciesParams
from mycormarl.soil.phosphate_units import (
    cylindrical_lateral_area,
    days_to_seconds,
    michaelis_menten_surface_flux,
)


def continuous_uptake_request(
    solution_concentration_micromol_cm3: chex.Array,
    length_density_cm_cm3: chex.Array,
    cell_volumes_cm3: chex.Array,
    absorber_radius_cm: float,
    j_max: float,
    k_m: float,
    dt_days: float,
) -> chex.Array:
    """Return a continuous-regime P request in µmol for every soil cell.

    Length density is integrated over the bulk cell volume, converted to the
    lateral area of cylindrical absorbers, and multiplied by the
    Michaelis–Menten surface flux and physical timestep. The calculation uses
    the cell's derived bulk solution concentration as its surface
    concentration; the integrated pipeline blends it with the sparse closure.
    """
    concentration = jnp.maximum(
        jnp.asarray(solution_concentration_micromol_cm3), 0.0
    )
    represented_length = (
        jnp.maximum(jnp.asarray(length_density_cm_cm3), 0.0)
        * jnp.asarray(cell_volumes_cm3)
    )
    absorbing_area = cylindrical_lateral_area(
        represented_length, absorber_radius_cm
    )
    surface_flux = michaelis_menten_surface_flux(concentration, j_max, k_m)
    return surface_flux * absorbing_area * days_to_seconds(dt_days)


def territory_radius_cm(length_density_cm_cm3: chex.Array) -> chex.Array:
    """Return cylindrical soil-territory radius for a local length density.

    The circular cross-sectional territory per unit absorber length is
    ``1 / lambda``. A non-positive density has no finite territory and returns
    infinity; downstream requests remain zero because represented length is
    independently clipped to zero.
    """
    density = jnp.maximum(jnp.asarray(length_density_cm_cm3), 0.0)
    safe_density = jnp.where(density > 0.0, density, 1.0)
    radius = 1.0 / jnp.sqrt(jnp.pi * safe_density)
    return jnp.where(density > 0.0, radius, jnp.inf)


def effective_uptake_radius_cm(
    length_density_cm_cm3: chex.Array,
    absorber_radius_cm: float,
    apparent_diffusivity_cm2_s: float,
    reference_time_days: float,
) -> chex.Array:
    """Return the outer radius reached by sparse depletion over ``T_ref``.

    The depleted annulus grows no farther than either the absorber's assigned
    soil territory or the apparent buffered diffusion distance. Its inner
    boundary is the absorbing surface, so crowded territories safely collapse
    to the absorber radius.
    """
    absorber_radius = jnp.asarray(absorber_radius_cm)
    territory_radius = territory_radius_cm(length_density_cm_cm3)
    territory_gap = jnp.maximum(territory_radius - absorber_radius, 0.0)
    propagation_distance = jnp.sqrt(
        jnp.maximum(jnp.asarray(apparent_diffusivity_cm2_s), 0.0)
        * days_to_seconds(reference_time_days)
    )
    return absorber_radius + jnp.minimum(propagation_distance, territory_gap)


def sparse_uptake_resistance(
    length_density_cm_cm3: chex.Array,
    absorber_radius_cm: float,
    j_max: float,
    amount_flux_diffusivity_cm2_s: float,
    apparent_diffusivity_cm2_s: float,
    reference_time_days: float,
) -> chex.Array:
    """Return geometry-dependent sparse uptake resistance ``k``.

    ``D_flux`` controls steady amount supply, whereas buffered ``D_app`` only
    controls how far the depletion profile propagates. Keeping these roles
    separate avoids counting sorption retardation twice. With no diffusive
    supply the resistance is infinite and sparse uptake is therefore zero.
    """
    absorber_radius = jnp.asarray(absorber_radius_cm)
    effective_radius = effective_uptake_radius_cm(
        length_density_cm_cm3,
        absorber_radius,
        apparent_diffusivity_cm2_s,
        reference_time_days,
    )
    log_radius_ratio = jnp.log(effective_radius / absorber_radius)
    d_flux = jnp.asarray(amount_flux_diffusivity_cm2_s)
    safe_d_flux = jnp.where(d_flux > 0.0, d_flux, 1.0)
    resistance = (
        absorber_radius * jnp.asarray(j_max) * log_radius_ratio / safe_d_flux
    )
    return jnp.where(d_flux > 0.0, resistance, jnp.inf)


def hyphal_overlap_time_seconds(
    hyphal_length_density_cm_cm3: chex.Array,
    hyphal_radius_cm: float,
    apparent_diffusivity_cm2_s: float,
) -> chex.Array:
    """Return the static time for neighbouring hyphal depletion zones to meet.

    Only hyphal density drives this shared-regime diagnostic. No hyphae, or no
    apparent diffusion, gives infinite overlap time. A territory boundary at
    or inside the hyphal surface gives zero time when diffusion is active.
    """
    density = jnp.maximum(jnp.asarray(hyphal_length_density_cm_cm3), 0.0)
    territory = territory_radius_cm(density)
    gap = jnp.maximum(territory - jnp.asarray(hyphal_radius_cm), 0.0)
    d_app = jnp.asarray(apparent_diffusivity_cm2_s)
    safe_d_app = jnp.where(d_app > 0.0, d_app, 1.0)
    overlap_time = gap**2 / safe_d_app
    has_overlap_physics = (density > 0.0) & (d_app > 0.0)
    return jnp.where(has_overlap_physics, overlap_time, jnp.inf)


def continuous_regime_weight(
    overlap_time_seconds: chex.Array,
    reference_time_days: float,
    exponent: float,
) -> chex.Array:
    """Return the smooth shared continuous-regime weight in ``[0, 1]``.

    This stable form, ``1 / (1 + (t_diff/T_ref)^p)``, is algebraically
    equivalent to the agreed Omega expression but directly handles zero and
    infinite overlap times without an ``inf / inf`` intermediate.
    """
    time_ratio = jnp.maximum(jnp.asarray(overlap_time_seconds), 0.0) / (
        days_to_seconds(reference_time_days)
    )
    return 1.0 / (1.0 + time_ratio ** jnp.asarray(exponent))


def sparse_surface_concentration(
    bulk_concentration_micromol_cm3: chex.Array,
    k_m: float,
    resistance: chex.Array,
) -> chex.Array:
    """Return the stable physical root for sparse absorber concentration.

    The direct positive quadratic root loses precision when ``a`` is negative,
    while its rationalised form loses precision when ``a`` is positive. The
    implementation selects between both equivalent forms by the sign of
    ``a = C_b - K_m - k``. Infinite resistance maps explicitly to zero, and
    clipping suppresses only round-off excursions outside ``[0, C_b]``.
    """
    bulk = jnp.maximum(jnp.asarray(bulk_concentration_micromol_cm3), 0.0)
    resistance = jnp.maximum(jnp.asarray(resistance), 0.0)
    k_m_array = jnp.asarray(k_m)
    a = bulk - k_m_array - resistance
    discriminant_root = jnp.sqrt(a * a + 4.0 * bulk * k_m_array)
    denominator = discriminant_root - a
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    rationalised_root = 2.0 * bulk * k_m_array / safe_denominator
    direct_root = 0.5 * (a + discriminant_root)
    surface = jnp.where(a >= 0.0, direct_root, rationalised_root)
    surface = jnp.where(jnp.isfinite(resistance), surface, 0.0)
    surface = jnp.where(resistance == 0.0, bulk, surface)
    return jnp.clip(surface, 0.0, bulk)


def sparse_uptake_request(
    solution_concentration_micromol_cm3: chex.Array,
    length_density_cm_cm3: chex.Array,
    cell_volumes_cm3: chex.Array,
    absorber_radius_cm: float,
    j_max: float,
    k_m: float,
    dt_days: float,
    resistance: chex.Array,
) -> chex.Array:
    """Return a sparse-regime P request in µmol for every soil cell.

    The geometry-dependent resistance supplies a temporary depleted surface
    concentration. Michaelis–Menten flux at that surface is then integrated
    over represented cylindrical length and physical timestep.
    """
    surface_concentration = sparse_surface_concentration(
        solution_concentration_micromol_cm3, k_m, resistance
    )
    represented_length = (
        jnp.maximum(jnp.asarray(length_density_cm_cm3), 0.0)
        * jnp.asarray(cell_volumes_cm3)
    )
    absorbing_area = cylindrical_lateral_area(
        represented_length, absorber_radius_cm
    )
    surface_flux = michaelis_menten_surface_flux(
        surface_concentration, j_max, k_m
    )
    return surface_flux * absorbing_area * days_to_seconds(dt_days)


def blend_uptake_requests(
    sparse_request_micromol: chex.Array,
    continuous_request_micromol: chex.Array,
    continuous_weight: chex.Array,
) -> chex.Array:
    """Interpolate sparse and continuous requests with one bounded weight.

    This operation occurs before shared inventory allocation. The closures are
    alternatives, so they are blended rather than added as independent sinks.
    """
    weight = jnp.clip(jnp.asarray(continuous_weight), 0.0, 1.0)
    sparse = jnp.maximum(jnp.asarray(sparse_request_micromol), 0.0)
    continuous = jnp.maximum(jnp.asarray(continuous_request_micromol), 0.0)
    return (1.0 - weight) * sparse + weight * continuous


class PhosphateUptakeDiagnostics(NamedTuple):
    """Cellwise observables from one blended uptake transaction.

    Ratios and weights are dimensionless; every request field is µmol P per
    cell for the supplied timestep. ``capped`` identifies cells whose final
    blended root-plus-fungal demand exceeded canonical labile inventory.
    """

    root_surface_ratio: chex.Array
    fungus_surface_ratio: chex.Array
    continuous_weight: chex.Array
    root_sparse_request_micromol: chex.Array
    fungus_sparse_request_micromol: chex.Array
    root_continuous_request_micromol: chex.Array
    fungus_continuous_request_micromol: chex.Array
    root_request_micromol: chex.Array
    fungus_request_micromol: chex.Array
    capped: chex.Array


def blended_uptake_transaction(
    labile_amount_micromol: chex.Array,
    solution_concentration_micromol_cm3: chex.Array,
    root_length_density_cm_cm3: chex.Array,
    hyphal_length_density_cm_cm3: chex.Array,
    cell_volumes_cm3: chex.Array,
    species: SpeciesParams,
    dt_days: float,
    root_sparse_resistance: chex.Array,
    fungus_sparse_resistance: chex.Array,
    continuous_weight: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array, PhosphateUptakeDiagnostics]:
    """Calculate, blend, cap, and diagnose one simultaneous uptake event.

    This is the single scientific implementation shared by production soil
    evolution and offline qualification. It returns remaining soil amount, accepted
    root amount, accepted fungal amount, and cellwise diagnostics. Bulk-zero
    cells report a surface ratio of one (no defined depletion from zero), while
    their uptake requests remain exactly zero.
    """
    concentration = jnp.maximum(
        jnp.asarray(solution_concentration_micromol_cm3), 0.0
    )
    root_continuous = continuous_uptake_request(
        concentration,
        root_length_density_cm_cm3,
        cell_volumes_cm3,
        species.plant.root_radius,
        species.plant.jmax,
        species.plant.km,
        dt_days,
    )
    fungus_continuous = continuous_uptake_request(
        concentration,
        hyphal_length_density_cm_cm3,
        cell_volumes_cm3,
        species.fungus.hyphal_radius,
        species.fungus.jmax,
        species.fungus.km,
        dt_days,
    )
    root_surface = sparse_surface_concentration(
        concentration, species.plant.km, root_sparse_resistance
    )
    fungus_surface = sparse_surface_concentration(
        concentration, species.fungus.km, fungus_sparse_resistance
    )
    root_sparse = sparse_uptake_request(
        concentration,
        root_length_density_cm_cm3,
        cell_volumes_cm3,
        species.plant.root_radius,
        species.plant.jmax,
        species.plant.km,
        dt_days,
        root_sparse_resistance,
    )
    fungus_sparse = sparse_uptake_request(
        concentration,
        hyphal_length_density_cm_cm3,
        cell_volumes_cm3,
        species.fungus.hyphal_radius,
        species.fungus.jmax,
        species.fungus.km,
        dt_days,
        fungus_sparse_resistance,
    )
    root_request = blend_uptake_requests(
        root_sparse, root_continuous, continuous_weight
    )
    fungus_request = blend_uptake_requests(
        fungus_sparse, fungus_continuous, continuous_weight
    )
    remaining, accepted_root, accepted_fungus = allocate_competing_uptake(
        labile_amount_micromol, root_request, fungus_request
    )
    safe_concentration = jnp.where(concentration > 0.0, concentration, 1.0)
    root_ratio = jnp.where(
        concentration > 0.0, root_surface / safe_concentration, 1.0
    )
    fungus_ratio = jnp.where(
        concentration > 0.0, fungus_surface / safe_concentration, 1.0
    )
    diagnostics = PhosphateUptakeDiagnostics(
        root_surface_ratio=root_ratio,
        fungus_surface_ratio=fungus_ratio,
        continuous_weight=jnp.clip(jnp.asarray(continuous_weight), 0.0, 1.0),
        root_sparse_request_micromol=root_sparse,
        fungus_sparse_request_micromol=fungus_sparse,
        root_continuous_request_micromol=root_continuous,
        fungus_continuous_request_micromol=fungus_continuous,
        root_request_micromol=root_request,
        fungus_request_micromol=fungus_request,
        capped=(root_request + fungus_request)
        > jnp.maximum(jnp.asarray(labile_amount_micromol), 0.0),
    )
    return remaining, accepted_root, accepted_fungus, diagnostics


def allocate_competing_uptake(
    labile_amount_micromol: chex.Array,
    root_request_micromol: chex.Array,
    fungus_request_micromol: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Cap simultaneous requests once against each shared cell inventory.

    Returns ``(remaining, accepted_root, accepted_fungus)`` in µmol P per
    cell. When demand exceeds supply, accepted total uptake equals the
    available amount and is divided in proportion to the two requests. A
    zero-demand guard avoids division by zero, and assigning the fungal share
    as the accepted-total remainder keeps the transaction conservative under
    finite-precision arithmetic.
    """
    available = jnp.maximum(jnp.asarray(labile_amount_micromol), 0.0)
    root_request = jnp.maximum(jnp.asarray(root_request_micromol), 0.0)
    fungus_request = jnp.maximum(jnp.asarray(fungus_request_micromol), 0.0)
    total_request = root_request + fungus_request
    accepted_total = jnp.minimum(available, total_request)
    safe_total = jnp.where(total_request > 0.0, total_request, 1.0)
    root_fraction = jnp.where(total_request > 0.0, root_request / safe_total, 0.0)
    accepted_root = accepted_total * root_fraction
    accepted_fungus = accepted_total - accepted_root
    remaining = available - accepted_total
    return remaining, accepted_root, accepted_fungus
