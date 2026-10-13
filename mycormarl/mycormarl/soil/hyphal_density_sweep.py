"""Time-dependent finite-inventory hyphal-density sweep."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig
from mycormarl.soil.phosphate_diffusion import apparent_diffusivity_cm2_s
from mycormarl.soil.phosphate_units import (
    SECONDS_PER_DAY,
    micromolar_to_micromol_per_cm3,
)
from mycormarl.soil.phosphate_uptake import (
    blend_uptake_requests,
    continuous_regime_weight,
    continuous_uptake_request,
    hyphal_overlap_time_seconds,
    sparse_uptake_request,
    sparse_uptake_resistance,
    territory_radius_cm,
)


def _step_sizes(dt_days: float, horizon_days: float):
    count = math.ceil(horizon_days / dt_days)
    for index in range(count):
        yield min(dt_days, horizon_days - index * dt_days)


def run_hyphal_density_sweep(
    *,
    densities_cm_cm3=range(25, 3001, 25),
    horizon_days: float = 30.0,
    dt_days: float | None = None,
    cell_volume_cm3: float = 1.0,
    absorber_radius_cm: float | None = None,
    initial_solution_p_um: float | None = None,
    config: EnvConfig | None = None,
    fungus_traits: FungusTraits | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run one fixed-cell, finite-P sweep with a dynamic depletion gradient.

    ``time_series`` contains one row per integration step and density;
    ``summaries`` contains one endpoint row per density. Elapsed global
    experiment time controls both the sparse resistance and the closure blend.
    """
    config = EnvConfig() if config is None else config
    traits = FungusTraits() if fungus_traits is None else fungus_traits
    dt_days = config.dt if dt_days is None else float(dt_days)
    radius = traits.hyphal_radius if absorber_radius_cm is None else float(absorber_radius_cm)
    initial_p_um = (
        config.initial_solution_p_um
        if initial_solution_p_um is None
        else float(initial_solution_p_um)
    )
    densities = np.asarray([float(value) for value in densities_cm_cm3], dtype=float)
    if densities.size == 0:
        raise ValueError("densities_cm_cm3 must not be empty")
    if not math.isfinite(horizon_days) or horizon_days <= 0.0:
        raise ValueError("horizon_days must be finite and greater than zero")
    if not math.isfinite(dt_days) or dt_days <= 0.0:
        raise ValueError("dt_days must be finite and greater than zero")
    if not math.isfinite(cell_volume_cm3) or cell_volume_cm3 <= 0.0:
        raise ValueError("cell_volume_cm3 must be finite and greater than zero")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("absorber_radius_cm must be finite and greater than zero")
    if not math.isfinite(initial_p_um) or initial_p_um <= 0.0:
        raise ValueError("initial_solution_p_um must be finite and greater than zero")
    if np.any(~np.isfinite(densities) | (densities <= 0.0)):
        raise ValueError("densities must be finite and greater than zero")

    bulk_initial = float(micromolar_to_micromol_per_cm3(initial_p_um))
    capacity = cell_volume_cm3 * (config.theta_water + config.b_p)
    initial_labile = bulk_initial * capacity
    d_flux = (
        config.phosphate_diffusion_coefficient_cm2_s
        * config.theta_water
        * config.phosphate_impedance_factor
    )
    d_app = float(
        apparent_diffusivity_cm2_s(
            config.phosphate_diffusion_coefficient_cm2_s,
            config.theta_water,
            config.phosphate_impedance_factor,
            config.b_p,
        )
    )

    time_series: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    territory = np.asarray(territory_radius_cm(densities), dtype=float)
    valid = radius < territory
    represented_length = densities * cell_volume_cm3
    construction_carbon = (
        represented_length * math.pi * radius**2
        * traits.hyphal_tissue_carbon_density
    )
    overlap_seconds = np.asarray(
        hyphal_overlap_time_seconds(densities, radius, d_app), dtype=float
    )
    t_diff_days = overlap_seconds / SECONDS_PER_DAY
    remaining = np.full(densities.shape, initial_labile, dtype=float)
    cumulative = np.zeros(densities.shape, dtype=float)
    elapsed = 0.0

    for step_days in _step_sizes(dt_days, horizon_days):
        bulk_before = remaining / capacity
        resistance = sparse_uptake_resistance(
            jnp.asarray(densities),
            radius,
            traits.jmax,
            d_flux,
            d_app,
            elapsed,
        )
        weight = continuous_regime_weight(
            jnp.asarray(overlap_seconds),
            elapsed,
            config.uptake_transition_exponent,
        )
        sparse = sparse_uptake_request(
            jnp.asarray(bulk_before),
            jnp.asarray(densities),
            cell_volume_cm3,
            radius,
            traits.jmax,
            traits.km,
            step_days,
            resistance,
        )
        continuous = continuous_uptake_request(
            jnp.asarray(bulk_before),
            jnp.asarray(densities),
            cell_volume_cm3,
            radius,
            traits.jmax,
            traits.km,
            step_days,
        )
        requested = np.asarray(blend_uptake_requests(sparse, continuous, weight), dtype=float)
        weight = np.asarray(weight, dtype=float)
        accepted = np.minimum(np.maximum(requested, 0.0), np.maximum(remaining, 0.0))
        accepted = np.where(valid, accepted, 0.0)
        remaining = np.maximum(remaining - accepted, 0.0)
        cumulative += accepted
        elapsed += step_days
        for index, density in enumerate(densities):
            time_series.append(
                {
                    "density_cm_cm3": float(density),
                    "initial_solution_p_um": initial_p_um,
                    "time_days": elapsed,
                    "step_days": step_days,
                    "absorber_radius_cm": radius,
                    "cell_volume_cm3": cell_volume_cm3,
                    "represented_length_cm": float(represented_length[index]),
                    "territory_radius_cm": float(territory[index]),
                    "geometry_valid": bool(valid[index]),
                    "t_diff_days": float(t_diff_days[index]),
                    "continuous_weight": float(weight[index]),
                    "bulk_before_micromol_cm3": float(bulk_before[index]),
                    "bulk_after_micromol_cm3": float(remaining[index] / capacity),
                    "requested_uptake_micromol": float(requested[index]),
                    "accepted_uptake_micromol": float(accepted[index]),
                    "cumulative_uptake_micromol": float(cumulative[index]),
                    "construction_carbon_g": float(construction_carbon[index]),
                    "uptake_per_construction_carbon_micromol_g_c": float(
                        cumulative[index] / construction_carbon[index]
                    ),
                    "apparent_diffusivity_cm2_s": d_app,
                    "amount_flux_diffusivity_cm2_s": d_flux,
                }
            )

    for index, density in enumerate(densities):
        summaries.append(
            {
                "density_cm_cm3": float(density),
                "initial_solution_p_um": initial_p_um,
                "horizon_days": horizon_days,
                "absorber_radius_cm": radius,
                "cell_volume_cm3": cell_volume_cm3,
                "represented_length_cm": float(represented_length[index]),
                "territory_radius_cm": float(territory[index]),
                "geometry_valid": bool(valid[index]),
                "t_diff_days": float(t_diff_days[index]),
                "initial_labile_p_micromol": initial_labile,
                "final_labile_p_micromol": float(remaining[index]),
                "total_uptake_micromol": float(cumulative[index]),
                "construction_carbon_g": float(construction_carbon[index]),
                "uptake_per_construction_carbon_micromol_g_c": (
                    float(cumulative[index] / construction_carbon[index])
                ),
                "conservation_error_micromol": float(
                    initial_labile - remaining[index] - cumulative[index]
                ),
            }
        )
    return time_series, summaries
