"""Fixed-reservoir diagnostic for time-dependent sparse depletion gradients."""

from __future__ import annotations

import math

from mycormarl.params import EnvConfig
from mycormarl.fungus.traits import FungusTraits
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_diffusion import apparent_diffusivity_cm2_s
from mycormarl.soil.phosphate_units import (
    SECONDS_PER_DAY,
    micromolar_to_micromol_per_cm3,
)
from mycormarl.soil.phosphate_uptake import (
    blend_uptake_requests,
    continuous_regime_weight,
    continuous_uptake_request,
    effective_uptake_radius_cm,
    hyphal_overlap_time_seconds,
    sparse_surface_concentration,
    sparse_uptake_request,
    sparse_uptake_resistance,
    territory_radius_cm,
)


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")


def run_time_dependent_depletion_gradient_diagnostic(
    *,
    times_days,
    absorber_radii_cm=(1e-2, 5e-4),
    length_densities_cm_cm3=(1.0, 100.0, 2_000.0),
    cell_volume_cm3: float = 1.0,
    config: EnvConfig | None = None,
    plant_traits: PlantTraits | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return fixed-geometry closure trajectories and their endpoint totals.

    Elapsed experiment time replaces ``T_ref`` only for the sparse effective
    radius.  This isolated diagnostic does not evolve a soil state or alter
    production uptake semantics.
    """
    config = EnvConfig() if config is None else config
    traits = PlantTraits() if plant_traits is None else plant_traits
    times = [float(time) for time in times_days]
    radii = [float(radius) for radius in absorber_radii_cm]
    densities = [float(density) for density in length_densities_cm_cm3]
    if not times:
        raise ValueError("times_days must not be empty")
    if any(not math.isfinite(time) or time < 0.0 for time in times):
        raise ValueError("times_days must be finite and non-negative")
    if times != sorted(times):
        raise ValueError("times_days must be sorted in ascending order")
    if not radii or not densities:
        raise ValueError("absorber radii and length densities must not be empty")
    _validate_positive("cell_volume_cm3", cell_volume_cm3)
    for radius in radii:
        _validate_positive("absorber_radius_cm", radius)
    for density in densities:
        _validate_positive("length_density_cm_cm3", density)

    bulk = float(micromolar_to_micromol_per_cm3(config.initial_solution_p_um))
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
    rows = []
    for radius in radii:
        for density in densities:
            represented_length = density * cell_volume_cm3
            territory = float(territory_radius_cm(density))
            overlap_seconds = float(
                hyphal_overlap_time_seconds(density, radius, d_app)
            )
            overlap_days = overlap_seconds / SECONDS_PER_DAY
            geometry_times = sorted(
                set(times + ([overlap_days] if overlap_days <= times[-1] else []))
            )
            for time_days in geometry_times:
                effective_radius = float(
                    effective_uptake_radius_cm(density, radius, d_app, time_days)
                )
                resistance = float(
                    sparse_uptake_resistance(
                        density, radius, traits.jmax, d_flux, d_app, time_days
                    )
                )
                sparse_surface = float(
                    sparse_surface_concentration(bulk, traits.km, resistance)
                )
                sparse_rate = float(
                    sparse_uptake_request(
                        bulk,
                        density,
                        cell_volume_cm3,
                        radius,
                        traits.jmax,
                        traits.km,
                        1.0 / SECONDS_PER_DAY,
                        resistance,
                    )
                )
                continuous_rate = float(
                    continuous_uptake_request(
                        bulk,
                        density,
                        cell_volume_cm3,
                        radius,
                        traits.jmax,
                        traits.km,
                        1.0 / SECONDS_PER_DAY,
                    )
                )
                continuous_weight = (
                    0.0
                    if time_days == 0.0
                    else float(
                        continuous_regime_weight(
                            overlap_seconds,
                            time_days,
                            config.uptake_transition_exponent,
                        )
                    )
                )
                blended_rate = float(
                    blend_uptake_requests(
                        sparse_rate, continuous_rate, continuous_weight
                    )
                )
                shared = {
                    "time_days": time_days,
                    "absorber_radius_cm": radius,
                    "length_density_cm_cm3": density,
                    "cell_volume_cm3": cell_volume_cm3,
                    "represented_length_cm": represented_length,
                    "bulk_concentration_micromol_cm3": bulk,
                    "reference_time_days": config.uptake_reference_time_days,
                    "plant_jmax_micromol_cm2_s": traits.jmax,
                    "plant_km_micromol_cm3": traits.km,
                    "territory_radius_cm": territory,
                    "diffusion_overlap_time_days": overlap_days,
                    "is_diffusion_overlap_time": math.isclose(
                        time_days, overlap_days, rel_tol=0.0, abs_tol=1e-12
                    ),
                    "effective_radius_cm": effective_radius,
                    "sparse_resistance": resistance,
                    "continuous_weight": continuous_weight,
                    "amount_flux_diffusivity_cm2_s": d_flux,
                    "apparent_diffusivity_cm2_s": d_app,
                }
                rows.append(
                    {
                        **shared,
                        "closure": "blended_time_dependent",
                        "surface_concentration_micromol_cm3": sparse_surface,
                        "sparse_uptake_rate_micromol_s": sparse_rate,
                        "continuous_uptake_rate_micromol_s": continuous_rate,
                        "uptake_rate_per_length_micromol_cm_s": blended_rate
                        / represented_length,
                        "total_uptake_rate_micromol_s": blended_rate,
                    }
                )
    for radius in radii:
        for density in densities:
            selected = [
                row
                for row in rows
                if row["absorber_radius_cm"] == radius
                and row["length_density_cm_cm3"] == density
            ]
            cumulative = 0.0
            selected[0]["cumulative_uptake_micromol"] = cumulative
            for left, right in zip(selected, selected[1:]):
                cumulative += (
                    0.5
                    * (
                        left["total_uptake_rate_micromol_s"]
                        + right["total_uptake_rate_micromol_s"]
                    )
                    * (right["time_days"] - left["time_days"])
                    * SECONDS_PER_DAY
                )
                right["cumulative_uptake_micromol"] = cumulative

    summaries = []
    for radius in radii:
        for density in densities:
            selected = [
                row
                for row in rows
                if row["absorber_radius_cm"] == radius
                and row["length_density_cm_cm3"] == density
            ]
            summaries.append(
                {
                    "absorber_radius_cm": radius,
                    "length_density_cm_cm3": density,
                    "closure": "blended_time_dependent",
                    "end_time_days": times[-1],
                    "cumulative_uptake_micromol": selected[-1][
                        "cumulative_uptake_micromol"
                    ],
                }
            )
    return rows, summaries


def run_native_geometry_closure_comparisons(
    *,
    times_days,
    cell_volume_cm3: float = 1.0,
    config: EnvConfig | None = None,
    plant_traits: PlantTraits | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compare closure limits and blend clocks at native plant/fungus geometry.

    Both organism geometries use the same plant kinetic baseline. This makes
    their separation a geometry comparison, rather than a trait-bundle one.
    """
    config = EnvConfig() if config is None else config
    traits = PlantTraits() if plant_traits is None else plant_traits
    fungus_traits = FungusTraits()
    times = [float(time) for time in times_days]
    if not times or times[0] != 0.0:
        raise ValueError("times_days must begin at zero")
    if any(not math.isfinite(time) or time < 0.0 for time in times):
        raise ValueError("times_days must be finite and non-negative")
    if times != sorted(times):
        raise ValueError("times_days must be sorted in ascending order")
    _validate_positive("cell_volume_cm3", cell_volume_cm3)

    treatments = (
        ("closure_limits", "sparse_only"),
        ("closure_limits", "continuous_only"),
        ("blend_time_reference", "fixed_t_ref"),
        ("blend_time_reference", "simulation_time"),
    )
    bulk = float(micromolar_to_micromol_per_cm3(config.initial_solution_p_um))
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
    transition_overlap_days = 10.0
    transition_radius = fungus_traits.hyphal_radius
    transition_territory_radius = transition_radius + math.sqrt(
        d_app * transition_overlap_days * SECONDS_PER_DAY
    )
    transition_density = 1.0 / (math.pi * transition_territory_radius**2)
    geometries = (
        ("plant_default", traits.root_radius, traits.root_length_density),
        (
            "fungus_default",
            fungus_traits.hyphal_radius,
            fungus_traits.saturation_density,
        ),
        ("transition_scale", transition_radius, transition_density),
    )
    rows = []
    for organism_geometry, radius, density in geometries:
        represented_length = density * cell_volume_cm3
        overlap_seconds = float(hyphal_overlap_time_seconds(density, radius, d_app))
        overlap_days = overlap_seconds / SECONDS_PER_DAY
        geometry_times = sorted(
            set(times + ([overlap_days] if overlap_days <= times[-1] else []))
        )
        for time_days in geometry_times:
            resistance = float(
                sparse_uptake_resistance(
                    density, radius, traits.jmax, d_flux, d_app, time_days
                )
            )
            sparse_rate = float(
                sparse_uptake_request(
                    bulk,
                    density,
                    cell_volume_cm3,
                    radius,
                    traits.jmax,
                    traits.km,
                    1.0 / SECONDS_PER_DAY,
                    resistance,
                )
            )
            continuous_rate = float(
                continuous_uptake_request(
                    bulk,
                    density,
                    cell_volume_cm3,
                    radius,
                    traits.jmax,
                    traits.km,
                    1.0 / SECONDS_PER_DAY,
                )
            )
            simulation_weight = (
                0.0
                if time_days == 0.0
                else float(
                    continuous_regime_weight(
                        overlap_seconds,
                        time_days,
                        config.uptake_transition_exponent,
                    )
                )
            )
            fixed_weight = float(
                continuous_regime_weight(
                    overlap_seconds,
                    config.uptake_reference_time_days,
                    config.uptake_transition_exponent,
                )
            )
            for panel, treatment in treatments:
                if treatment == "sparse_only":
                    total_rate, weight = sparse_rate, 0.0
                elif treatment == "continuous_only":
                    total_rate, weight = continuous_rate, 1.0
                elif treatment == "fixed_t_ref":
                    total_rate, weight = float(
                        blend_uptake_requests(sparse_rate, continuous_rate, fixed_weight)
                    ), fixed_weight
                else:
                    total_rate, weight = float(
                        blend_uptake_requests(
                            sparse_rate, continuous_rate, simulation_weight
                        )
                    ), simulation_weight
                rows.append(
                    {
                        "comparison_panel": panel,
                        "treatment": treatment,
                        "organism_geometry": organism_geometry,
                        "time_days": time_days,
                        "absorber_radius_cm": radius,
                        "length_density_cm_cm3": density,
                        "represented_length_cm": represented_length,
                        "reference_time_days": config.uptake_reference_time_days,
                        "diffusion_overlap_time_days": overlap_seconds
                        / SECONDS_PER_DAY,
                        "is_diffusion_overlap_time": math.isclose(
                            time_days, overlap_days, rel_tol=0.0, abs_tol=1e-12
                        ),
                        "sparse_resistance": resistance,
                        "continuous_weight": weight,
                        "sparse_uptake_rate_micromol_s": sparse_rate,
                        "continuous_uptake_rate_micromol_s": continuous_rate,
                        "total_uptake_rate_micromol_s": total_rate,
                    }
                )
    for panel, treatment in treatments:
        for organism_geometry, *_ in geometries:
            trajectory = [
                row
                for row in rows
                if row["comparison_panel"] == panel
                and row["treatment"] == treatment
                and row["organism_geometry"] == organism_geometry
            ]
            cumulative = 0.0
            trajectory[0]["cumulative_uptake_micromol"] = cumulative
            for left, right in zip(trajectory, trajectory[1:]):
                cumulative += (
                    0.5
                    * (
                        left["total_uptake_rate_micromol_s"]
                        + right["total_uptake_rate_micromol_s"]
                    )
                    * (right["time_days"] - left["time_days"])
                    * SECONDS_PER_DAY
                )
                right["cumulative_uptake_micromol"] = cumulative
    summaries = [
        {
            "comparison_panel": panel,
            "treatment": treatment,
            "organism_geometry": organism_geometry,
            "end_time_days": times[-1],
            "cumulative_uptake_micromol": next(
                row["cumulative_uptake_micromol"]
                for row in reversed(rows)
                if row["comparison_panel"] == panel
                and row["treatment"] == treatment
                and row["organism_geometry"] == organism_geometry
            ),
        }
        for panel, treatment in treatments
        for organism_geometry, *_ in geometries
    ]
    return rows, summaries
