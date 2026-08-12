"""Isolated absorber qualification and construction-economics helpers."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from mycormarl.fungus.mycelium import fungal_biomass_from_hyphal_length
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_diffusion import (
    apparent_diffusivity_cm2_s,
    validate_diffusion_parameters,
)
from mycormarl.soil.phosphate_units import (
    micromolar_to_micromol_per_cm3,
    validate_linear_buffer_parameters,
    validate_michaelis_menten_parameters,
)
from mycormarl.soil.phosphate_uptake import (
    blend_uptake_requests,
    continuous_regime_weight,
    continuous_uptake_request,
    hyphal_overlap_time_seconds,
    sparse_surface_concentration,
    sparse_uptake_request,
    sparse_uptake_resistance,
)


_SCIENTIFIC_RESULT_FIELDS = (
    "construction_carbon_g",
    "integrated_uptake_micromol",
    "maximum_instantaneous_uptake_rate_micromol_s",
    "integrated_uptake_per_construction_carbon_micromol_g_c",
    "maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s",
    "sparse_resistance_micromol_cm3",
    "continuous_weight",
    "initial_bulk_concentration_micromol_cm3",
    "final_bulk_concentration_micromol_cm3",
    "initial_surface_concentration_micromol_cm3",
    "final_surface_concentration_micromol_cm3",
    "minimum_surface_concentration_micromol_cm3",
    "initial_labile_p_micromol",
    "final_labile_p_micromol",
    "conservation_error_micromol",
    "t_1_percent_days",
    "t_1_percent_reached",
)


def _apply_geometry_validity(row: dict[str, object]) -> dict[str, object]:
    """Add territory fields and blank results outside the closure domain."""
    density = float(row["length_density_cm_cm3"])
    territory_radius = (
        math.inf if density == 0.0 else 1.0 / math.sqrt(math.pi * density)
    )
    valid = float(row["absorber_radius_cm"]) < territory_radius
    row["territory_radius_cm"] = territory_radius
    row["geometry_valid"] = valid
    if not valid:
        for field in _SCIENTIFIC_RESULT_FIELDS:
            row[field] = None
    return row


def root_tissue_carbon_density_g_cm3(traits: PlantTraits) -> float:
    """Infer root-tissue structural-carbon density from reference root traits."""
    for name, value in (
        ("specific_root_length", traits.specific_root_length),
        ("root_radius", traits.root_radius),
        ("gamma_c", traits.gamma_c),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"plant {name} must be finite and greater than zero")
    return traits.gamma_c / (
        traits.specific_root_length * math.pi * traits.root_radius**2
    )


def plant_construction_carbon_g(
    absorber_length_cm: float,
    absorber_radius_cm: float,
    traits: PlantTraits,
) -> float:
    """Return structural root-tissue carbon for a candidate cylinder."""
    if not math.isfinite(absorber_length_cm) or absorber_length_cm < 0.0:
        raise ValueError("absorber_length_cm must be finite and non-negative")
    if not math.isfinite(absorber_radius_cm) or absorber_radius_cm <= 0.0:
        raise ValueError("absorber_radius_cm must be finite and greater than zero")
    return (
        absorber_length_cm
        * math.pi
        * absorber_radius_cm**2
        * root_tissue_carbon_density_g_cm3(traits)
    )


def fungal_construction_carbon_g(
    absorber_length_cm: float,
    absorber_radius_cm: float,
    traits: FungusTraits,
) -> float:
    """Return fungal structural carbon represented by an external-hyphal length."""
    if not math.isfinite(absorber_length_cm) or absorber_length_cm < 0.0:
        raise ValueError("absorber_length_cm must be finite and non-negative")
    for name, value in (
        ("gamma_c", traits.gamma_c),
        ("hyphal_tissue_carbon_density", traits.hyphal_tissue_carbon_density),
        ("absorber_radius_cm", absorber_radius_cm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"fungus {name} must be finite and greater than zero")
    biomass_g = fungal_biomass_from_hyphal_length(
        absorber_length_cm,
        traits.gamma_c,
        traits.hyphal_tissue_carbon_density,
        absorber_radius_cm,
    )
    return float(biomass_g) * traits.gamma_c


def _step_sizes(dt_days: float, horizon_days: float):
    """Yield positive timesteps that cover the requested horizon exactly."""
    step_count = math.ceil(horizon_days / dt_days)
    for index in range(step_count):
        remaining_days = horizon_days - index * dt_days
        yield min(dt_days, remaining_days)


def _depletion_event_times_days(
    initial_bulk,
    capacity,
    cell_volume_cm3,
    length_density,
    absorber_radius,
    traits,
    resistance,
    continuous_weight_value,
):
    """Integrate time until sparse surface concentration reaches one percent."""
    bulk_0 = np.asarray(initial_bulk, dtype=float)
    density = np.asarray(length_density, dtype=float)
    radius = np.asarray(absorber_radius, dtype=float)
    resistance_array = np.asarray(resistance, dtype=float)
    weight = np.asarray(continuous_weight_value, dtype=float)

    surface_0 = np.asarray(
        sparse_surface_concentration(bulk_0, traits.km, resistance_array),
        dtype=float,
    )
    surface_target = 0.01 * surface_0
    bulk_target = surface_target + (
        resistance_array * surface_target / (traits.km + surface_target)
    )

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(64)
    half_span = 0.5 * (bulk_0 - bulk_target)
    midpoint = 0.5 * (bulk_0 + bulk_target)
    concentrations = midpoint[..., None] + half_span[..., None] * nodes
    represented_length = density * float(cell_volume_cm3)
    area = 2.0 * math.pi * radius * represented_length

    a = concentrations - traits.km - resistance_array[..., None]
    root = np.sqrt(a * a + 4.0 * concentrations * traits.km)
    direct_surface = 0.5 * (a + root)
    rational_surface = (
        2.0 * concentrations * traits.km / np.maximum(root - a, 1e-300)
    )
    sparse_surface = np.where(a >= 0.0, direct_surface, rational_surface)
    sparse_flux = traits.jmax * sparse_surface / (traits.km + sparse_surface)
    continuous_flux = traits.jmax * concentrations / (traits.km + concentrations)
    uptake_rate = area[..., None] * (
        (1.0 - weight[..., None]) * sparse_flux
        + weight[..., None] * continuous_flux
    )
    integral = half_span * np.sum(
        quadrature_weights / np.maximum(uptake_rate, 1e-300), axis=-1
    )
    event_days = float(capacity) * integral / 86_400.0
    valid_event = (density > 0.0) & (surface_0 > 0.0) & np.isfinite(event_days)
    return np.where(valid_event, event_days, np.nan)


def _fixed_reservoir_row(
    absorber_radius_cm: float,
    length_density_cm_cm3: float,
    dt_days: float,
    reference_time_days: float,
    cell_volume_cm3: float,
    config: EnvConfig,
    traits,
    *,
    construction_carbon_fn=None,
    record_type: str = "surface",
    marker_label: str = "",
    economics_mode: str = "plant",
    uptake_traits: str = "plant",
) -> dict[str, object]:
    """Run one fixed-geometry plant absorber against a fixed reservoir."""
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
    resistance = float(
        sparse_uptake_resistance(
            length_density_cm_cm3,
            absorber_radius_cm,
            traits.jmax,
            d_flux,
            d_app,
            reference_time_days,
        )
    )
    weight = float(
        continuous_regime_weight(
            hyphal_overlap_time_seconds(
                length_density_cm_cm3,
                absorber_radius_cm,
                d_app,
            ),
            reference_time_days,
            config.uptake_transition_exponent,
        )
    )
    surface = float(sparse_surface_concentration(bulk, traits.km, resistance))
    initial_sparse_rate = sparse_uptake_request(
        bulk,
        length_density_cm_cm3,
        cell_volume_cm3,
        absorber_radius_cm,
        traits.jmax,
        traits.km,
        1.0 / 86_400.0,
        resistance,
    )
    initial_continuous_rate = continuous_uptake_request(
        bulk,
        length_density_cm_cm3,
        cell_volume_cm3,
        absorber_radius_cm,
        traits.jmax,
        traits.km,
        1.0 / 86_400.0,
    )
    maximum_rate = float(
        blend_uptake_requests(
            initial_sparse_rate, initial_continuous_rate, weight
        )
    )
    integrated = 0.0
    for step_days in _step_sizes(dt_days, reference_time_days):
        sparse = sparse_uptake_request(
            bulk,
            length_density_cm_cm3,
            cell_volume_cm3,
            absorber_radius_cm,
            traits.jmax,
            traits.km,
            step_days,
            resistance,
        )
        continuous = continuous_uptake_request(
            bulk,
            length_density_cm_cm3,
            cell_volume_cm3,
            absorber_radius_cm,
            traits.jmax,
            traits.km,
            step_days,
        )
        accepted = float(blend_uptake_requests(sparse, continuous, weight))
        integrated += accepted

    represented_length = length_density_cm_cm3 * cell_volume_cm3
    if construction_carbon_fn is None:
        construction_carbon_fn = plant_construction_carbon_g
    construction_carbon = construction_carbon_fn(
        represented_length, absorber_radius_cm, traits
    )
    initial_labile = bulk * cell_volume_cm3 * (
        config.theta_water + config.b_p
    )
    return {
        "record_type": record_type,
        "marker_label": marker_label,
        "marker_metric": "",
        "marker_solve_status": "not_applicable",
        "target_metric_value": None,
        "experiment_mode": "fixed_reservoir",
        "economics_mode": economics_mode,
        "uptake_traits": uptake_traits,
        "dt_days": dt_days,
        "reference_time_days": reference_time_days,
        "amount_flux_diffusivity_cm2_s": d_flux,
        "apparent_diffusivity_cm2_s": d_app,
        "absorber_radius_cm": absorber_radius_cm,
        "length_density_cm_cm3": length_density_cm_cm3,
        "cell_volume_cm3": cell_volume_cm3,
        "represented_length_cm": represented_length,
        "root_tissue_carbon_density_g_cm3": (
            root_tissue_carbon_density_g_cm3(traits)
            if economics_mode == "plant"
            else None
        ),
        "construction_carbon_g": construction_carbon,
        "integrated_uptake_micromol": integrated,
        "maximum_instantaneous_uptake_rate_micromol_s": maximum_rate,
        "integrated_uptake_per_construction_carbon_micromol_g_c": (
            integrated / construction_carbon if construction_carbon > 0.0 else 0.0
        ),
        "maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s": (
            maximum_rate / construction_carbon
            if construction_carbon > 0.0
            else 0.0
        ),
        "sparse_resistance_micromol_cm3": resistance,
        "continuous_weight": weight,
        "initial_bulk_concentration_micromol_cm3": bulk,
        "final_bulk_concentration_micromol_cm3": bulk,
        "initial_surface_concentration_micromol_cm3": surface,
        "final_surface_concentration_micromol_cm3": surface,
        "minimum_surface_concentration_micromol_cm3": surface,
        "initial_labile_p_micromol": initial_labile,
        "final_labile_p_micromol": initial_labile,
        "conservation_error_micromol": None,
        "t_1_percent_days": None,
        "t_1_percent_reached": False,
    }


def _finite_inventory_row(
    absorber_radius_cm: float,
    length_density_cm_cm3: float,
    dt_days: float,
    reference_time_days: float,
    cell_volume_cm3: float,
    config: EnvConfig,
    traits,
    *,
    construction_carbon_fn=None,
    record_type: str = "surface",
    marker_label: str = "",
    economics_mode: str = "plant",
    uptake_traits: str = "plant",
) -> dict[str, object]:
    """Run one fixed-geometry plant absorber against finite labile P."""
    initial_bulk = float(
        micromolar_to_micromol_per_cm3(config.initial_solution_p_um)
    )
    initial_labile = (
        initial_bulk * cell_volume_cm3 * (config.theta_water + config.b_p)
    )
    remaining = initial_labile
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
    resistance = float(
        sparse_uptake_resistance(
            length_density_cm_cm3,
            absorber_radius_cm,
            traits.jmax,
            d_flux,
            d_app,
            reference_time_days,
        )
    )
    weight = float(
        continuous_regime_weight(
            hyphal_overlap_time_seconds(
                length_density_cm_cm3,
                absorber_radius_cm,
                d_app,
            ),
            reference_time_days,
            config.uptake_transition_exponent,
        )
    )
    initial_surface = float(
        sparse_surface_concentration(initial_bulk, traits.km, resistance)
    )
    final_bulk = initial_bulk
    final_surface = initial_surface
    minimum_surface = initial_surface
    initial_sparse_rate = sparse_uptake_request(
        initial_bulk,
        length_density_cm_cm3,
        cell_volume_cm3,
        absorber_radius_cm,
        traits.jmax,
        traits.km,
        1.0 / 86_400.0,
        resistance,
    )
    initial_continuous_rate = continuous_uptake_request(
        initial_bulk,
        length_density_cm_cm3,
        cell_volume_cm3,
        absorber_radius_cm,
        traits.jmax,
        traits.km,
        1.0 / 86_400.0,
    )
    maximum_rate = float(
        blend_uptake_requests(
            initial_sparse_rate, initial_continuous_rate, weight
        )
    )
    integrated = 0.0
    for step_days in _step_sizes(dt_days, reference_time_days):
        bulk = remaining / (
            cell_volume_cm3 * (config.theta_water + config.b_p)
        )
        sparse = sparse_uptake_request(
            bulk,
            length_density_cm_cm3,
            cell_volume_cm3,
            absorber_radius_cm,
            traits.jmax,
            traits.km,
            step_days,
            resistance,
        )
        continuous = continuous_uptake_request(
            bulk,
            length_density_cm_cm3,
            cell_volume_cm3,
            absorber_radius_cm,
            traits.jmax,
            traits.km,
            step_days,
        )
        request = float(blend_uptake_requests(sparse, continuous, weight))
        accepted = min(remaining, request)
        remaining -= accepted
        integrated += accepted
        final_bulk = remaining / (
            cell_volume_cm3 * (config.theta_water + config.b_p)
        )
        final_surface = float(
            sparse_surface_concentration(final_bulk, traits.km, resistance)
        )
        minimum_surface = min(minimum_surface, final_surface)

    threshold_time = float(
        _depletion_event_times_days(
            initial_bulk,
            cell_volume_cm3 * (config.theta_water + config.b_p),
            cell_volume_cm3,
            length_density_cm_cm3,
            absorber_radius_cm,
            traits,
            resistance,
            weight,
        )
    )
    if not math.isfinite(threshold_time):
        threshold_time = None

    represented_length = length_density_cm_cm3 * cell_volume_cm3
    if construction_carbon_fn is None:
        construction_carbon_fn = plant_construction_carbon_g
    construction_carbon = construction_carbon_fn(
        represented_length, absorber_radius_cm, traits
    )
    return {
        "record_type": record_type,
        "marker_label": marker_label,
        "marker_metric": "",
        "marker_solve_status": "not_applicable",
        "target_metric_value": None,
        "experiment_mode": "finite_inventory",
        "economics_mode": economics_mode,
        "uptake_traits": uptake_traits,
        "dt_days": dt_days,
        "reference_time_days": reference_time_days,
        "amount_flux_diffusivity_cm2_s": d_flux,
        "apparent_diffusivity_cm2_s": d_app,
        "absorber_radius_cm": absorber_radius_cm,
        "length_density_cm_cm3": length_density_cm_cm3,
        "cell_volume_cm3": cell_volume_cm3,
        "represented_length_cm": represented_length,
        "root_tissue_carbon_density_g_cm3": (
            root_tissue_carbon_density_g_cm3(traits)
            if economics_mode == "plant"
            else None
        ),
        "construction_carbon_g": construction_carbon,
        "integrated_uptake_micromol": integrated,
        "maximum_instantaneous_uptake_rate_micromol_s": maximum_rate,
        "integrated_uptake_per_construction_carbon_micromol_g_c": (
            integrated / construction_carbon if construction_carbon > 0.0 else 0.0
        ),
        "maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s": (
            maximum_rate / construction_carbon
            if construction_carbon > 0.0
            else 0.0
        ),
        "sparse_resistance_micromol_cm3": resistance,
        "continuous_weight": weight,
        "initial_bulk_concentration_micromol_cm3": initial_bulk,
        "final_bulk_concentration_micromol_cm3": final_bulk,
        "initial_surface_concentration_micromol_cm3": initial_surface,
        "final_surface_concentration_micromol_cm3": final_surface,
        "minimum_surface_concentration_micromol_cm3": minimum_surface,
        "initial_labile_p_micromol": initial_labile,
        "final_labile_p_micromol": remaining,
        "conservation_error_micromol": initial_labile - remaining - integrated,
        "t_1_percent_days": threshold_time,
        "t_1_percent_reached": threshold_time is not None,
    }


def _surface_rows(
    radii: list[float],
    densities: list[float],
    mode: str,
    dt_days: float,
    reference_time_days: float,
    cell_volume_cm3: float,
    config: EnvConfig,
    traits: PlantTraits,
) -> list[dict[str, object]]:
    """Evaluate an entire plant-economics surface with array closure calls."""
    radius_grid = jnp.asarray(radii)[:, None]
    density_grid = jnp.asarray(densities)[None, :]
    geometry_shape = (len(radii), len(densities))
    radius_grid = jnp.broadcast_to(radius_grid, geometry_shape)
    density_grid = jnp.broadcast_to(density_grid, geometry_shape)
    bulk = float(micromolar_to_micromol_per_cm3(config.initial_solution_p_um))
    capacity = cell_volume_cm3 * (config.theta_water + config.b_p)
    initial_labile = bulk * capacity
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
    resistance = sparse_uptake_resistance(
        density_grid,
        radius_grid,
        traits.jmax,
        d_flux,
        d_app,
        reference_time_days,
    )
    weight = continuous_regime_weight(
        hyphal_overlap_time_seconds(density_grid, radius_grid, d_app),
        reference_time_days,
        config.uptake_transition_exponent,
    )
    initial_surface = sparse_surface_concentration(bulk, traits.km, resistance)
    represented_length = density_grid * cell_volume_cm3
    root_carbon_density = root_tissue_carbon_density_g_cm3(traits)
    construction_carbon = (
        represented_length * jnp.pi * radius_grid**2 * root_carbon_density
    )
    initial_sparse_rate = sparse_uptake_request(
        bulk,
        density_grid,
        cell_volume_cm3,
        radius_grid,
        traits.jmax,
        traits.km,
        1.0 / 86_400.0,
        resistance,
    )
    initial_continuous_rate = continuous_uptake_request(
        bulk,
        density_grid,
        cell_volume_cm3,
        radius_grid,
        traits.jmax,
        traits.km,
        1.0 / 86_400.0,
    )
    maximum_rate = blend_uptake_requests(
        initial_sparse_rate, initial_continuous_rate, weight
    )
    integrated = jnp.zeros(geometry_shape)
    remaining = jnp.full(geometry_shape, initial_labile)
    final_bulk = jnp.full(geometry_shape, bulk)
    final_surface = initial_surface
    minimum_surface = initial_surface
    threshold_time = jnp.full(geometry_shape, jnp.nan)

    for step_days in _step_sizes(dt_days, reference_time_days):
        current_bulk = bulk if mode == "fixed_reservoir" else remaining / capacity
        sparse = sparse_uptake_request(
            current_bulk,
            density_grid,
            cell_volume_cm3,
            radius_grid,
            traits.jmax,
            traits.km,
            step_days,
            resistance,
        )
        continuous = continuous_uptake_request(
            current_bulk,
            density_grid,
            cell_volume_cm3,
            radius_grid,
            traits.jmax,
            traits.km,
            step_days,
        )
        request = blend_uptake_requests(sparse, continuous, weight)
        accepted = request if mode == "fixed_reservoir" else jnp.minimum(remaining, request)
        integrated = integrated + accepted
        if mode == "finite_inventory":
            remaining = remaining - accepted
            final_bulk = remaining / capacity
            final_surface = sparse_surface_concentration(
                final_bulk, traits.km, resistance
            )
            minimum_surface = jnp.minimum(minimum_surface, final_surface)

    if mode == "fixed_reservoir":
        remaining = jnp.full(geometry_shape, initial_labile)
        final_bulk = jnp.full(geometry_shape, bulk)
    else:
        threshold_time = _depletion_event_times_days(
            bulk,
            capacity,
            cell_volume_cm3,
            density_grid,
            radius_grid,
            traits,
            resistance,
            weight,
        )

    arrays = {
        "absorber_radius_cm": radius_grid,
        "length_density_cm_cm3": density_grid,
        "represented_length_cm": represented_length,
        "construction_carbon_g": construction_carbon,
        "integrated_uptake_micromol": integrated,
        "maximum_instantaneous_uptake_rate_micromol_s": maximum_rate,
        "integrated_uptake_per_construction_carbon_micromol_g_c": jnp.where(
            construction_carbon > 0.0, integrated / construction_carbon, 0.0
        ),
        "maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s": jnp.where(
            construction_carbon > 0.0, maximum_rate / construction_carbon, 0.0
        ),
        "sparse_resistance_micromol_cm3": resistance,
        "continuous_weight": weight,
        "final_bulk_concentration_micromol_cm3": final_bulk,
        "initial_surface_concentration_micromol_cm3": initial_surface,
        "final_surface_concentration_micromol_cm3": final_surface,
        "minimum_surface_concentration_micromol_cm3": minimum_surface,
        "final_labile_p_micromol": remaining,
        "t_1_percent_days": threshold_time,
    }
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    rows = []
    for radius_index in range(len(radii)):
        for density_index in range(len(densities)):
            index = (radius_index, density_index)
            threshold = float(arrays["t_1_percent_days"][index])
            threshold_reached = math.isfinite(threshold)
            final_labile = float(arrays["final_labile_p_micromol"][index])
            uptake = float(arrays["integrated_uptake_micromol"][index])
            rows.append(
                {
                    "record_type": "surface",
                    "marker_label": "",
                    "marker_metric": "",
                    "marker_solve_status": "not_applicable",
                    "target_metric_value": None,
                    "experiment_mode": mode,
                    "economics_mode": "plant",
                    "uptake_traits": "plant",
                    "dt_days": dt_days,
                    "reference_time_days": reference_time_days,
                    "amount_flux_diffusivity_cm2_s": d_flux,
                    "apparent_diffusivity_cm2_s": d_app,
                    "absorber_radius_cm": radii[radius_index],
                    "length_density_cm_cm3": densities[density_index],
                    "cell_volume_cm3": cell_volume_cm3,
                    "represented_length_cm": float(arrays["represented_length_cm"][index]),
                    "root_tissue_carbon_density_g_cm3": root_carbon_density,
                    "construction_carbon_g": float(arrays["construction_carbon_g"][index]),
                    "integrated_uptake_micromol": uptake,
                    "maximum_instantaneous_uptake_rate_micromol_s": float(arrays["maximum_instantaneous_uptake_rate_micromol_s"][index]),
                    "integrated_uptake_per_construction_carbon_micromol_g_c": float(arrays["integrated_uptake_per_construction_carbon_micromol_g_c"][index]),
                    "maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s": float(arrays["maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s"][index]),
                    "sparse_resistance_micromol_cm3": float(arrays["sparse_resistance_micromol_cm3"][index]),
                    "continuous_weight": float(arrays["continuous_weight"][index]),
                    "initial_bulk_concentration_micromol_cm3": bulk,
                    "final_bulk_concentration_micromol_cm3": float(arrays["final_bulk_concentration_micromol_cm3"][index]),
                    "initial_surface_concentration_micromol_cm3": float(arrays["initial_surface_concentration_micromol_cm3"][index]),
                    "final_surface_concentration_micromol_cm3": float(arrays["final_surface_concentration_micromol_cm3"][index]),
                    "minimum_surface_concentration_micromol_cm3": float(arrays["minimum_surface_concentration_micromol_cm3"][index]),
                    "initial_labile_p_micromol": initial_labile,
                    "final_labile_p_micromol": final_labile,
                    "conservation_error_micromol": (
                        None
                        if mode == "fixed_reservoir"
                        else initial_labile - final_labile - uptake
                    ),
                    "t_1_percent_days": threshold if threshold_reached else None,
                    "t_1_percent_reached": threshold_reached,
                }
            )
            _apply_geometry_validity(rows[-1])
    return rows


def _solve_equivalent_marker(
    row_factory,
    metric: str,
    target_row: dict[str, object],
    radii: list[float],
    densities: list[float],
    density: float,
    dt_days: float,
    reference_time_days: float,
    cell_volume_cm3: float,
    config: EnvConfig,
    plant_traits: PlantTraits,
) -> dict[str, object]:
    """Solve a valid-domain plant radius matching one fungal P-per-C target."""
    target = float(target_row[metric])
    territory = 1.0 / math.sqrt(math.pi * density)
    lower = min(radii)
    upper = min(max(radii), math.nextafter(territory, 0.0))

    def evaluate(radius: float) -> tuple[float, dict[str, object]]:
        row = row_factory(
            radius,
            density,
            dt_days,
            reference_time_days,
            cell_volume_cm3,
            config,
            plant_traits,
            record_type="marker",
            marker_label="fungus_equivalent_plant_geometry",
            economics_mode="plant",
            uptake_traits="plant",
        )
        return float(row[metric]) - target, row

    bracket = None
    if min(densities) <= density <= max(densities) and lower < upper:
        previous_radius = None
        previous_difference = None
        for candidate in np.geomspace(lower, upper, 129):
            difference, _ = evaluate(float(candidate))
            if difference == 0.0:
                bracket = (float(candidate), float(candidate))
                break
            if (
                previous_difference is not None
                and difference * previous_difference < 0.0
            ):
                bracket = (previous_radius, float(candidate))
                break
            previous_radius = float(candidate)
            previous_difference = difference

    if bracket is None:
        unavailable = dict(target_row)
        unavailable.update(
            {
                "marker_label": "fungus_equivalent_plant_geometry",
                "marker_metric": metric,
                "marker_solve_status": "unavailable",
                "target_metric_value": target,
                "economics_mode": "plant",
                "uptake_traits": "plant",
                "absorber_radius_cm": None,
                "length_density_cm_cm3": density,
                "territory_radius_cm": territory,
                "geometry_valid": False,
            }
        )
        for field in _SCIENTIFIC_RESULT_FIELDS:
            unavailable[field] = None
        return unavailable

    low, high = bracket
    if low != high:
        low_difference, _ = evaluate(low)
        for _ in range(64):
            midpoint = math.sqrt(low * high)
            middle_difference, _ = evaluate(midpoint)
            if middle_difference == 0.0:
                low = high = midpoint
                break
            if low_difference * middle_difference <= 0.0:
                high = midpoint
            else:
                low = midpoint
                low_difference = middle_difference
        solved_radius = math.sqrt(low * high)
    else:
        solved_radius = low
    _, solved = evaluate(solved_radius)
    solved.update(
        {
            "marker_metric": metric,
            "marker_solve_status": "solved",
            "target_metric_value": target,
        }
    )
    return _apply_geometry_validity(solved)


def run_absorber_geometry_sweep(
    *,
    absorber_radii_cm=None,
    length_densities_cm_cm3=None,
    radius_count: int = 40,
    density_count: int = 40,
    modes=("fixed_reservoir", "finite_inventory"),
    dt_days: float | None = None,
    reference_time_days: float | None = None,
    cell_volume_cm3: float = 1.0,
    config: EnvConfig | None = None,
    plant_traits: PlantTraits | None = None,
    fungus_traits: FungusTraits | None = None,
    include_markers: bool = True,
) -> list[dict[str, object]]:
    """Return tabular rows for an isolated absorber geometry sweep."""
    config = EnvConfig() if config is None else config
    traits = PlantTraits() if plant_traits is None else plant_traits
    fungus = FungusTraits() if fungus_traits is None else fungus_traits
    if not isinstance(radius_count, int) or radius_count < 2:
        raise ValueError("radius_count must be an integer of at least two")
    if not isinstance(density_count, int) or density_count < 2:
        raise ValueError("density_count must be an integer of at least two")
    validate_linear_buffer_parameters(config.theta_water, config.b_p)
    validate_diffusion_parameters(
        config.phosphate_diffusion_coefficient_cm2_s,
        config.theta_water,
        config.phosphate_impedance_factor,
        config.diffusion_cfl_safety,
    )
    validate_michaelis_menten_parameters(traits.jmax, traits.km)
    for name, value in (
        ("plant specific_root_length", traits.specific_root_length),
        ("plant root_radius", traits.root_radius),
        ("plant gamma_c", traits.gamma_c),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero")
    if include_markers:
        validate_michaelis_menten_parameters(fungus.jmax, fungus.km)
        for name, value in (
            ("fungus gamma_c", fungus.gamma_c),
            ("fungus hyphal_radius", fungus.hyphal_radius),
            (
                "fungus hyphal_tissue_carbon_density",
                fungus.hyphal_tissue_carbon_density,
            ),
            ("fungus saturation_density", fungus.saturation_density),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
    if (
        not math.isfinite(config.initial_solution_p_um)
        or config.initial_solution_p_um < 0.0
    ):
        raise ValueError("initial_solution_p_um must be finite and non-negative")
    if (
        not math.isfinite(config.uptake_transition_exponent)
        or config.uptake_transition_exponent <= 0.0
    ):
        raise ValueError(
            "uptake_transition_exponent must be finite and greater than zero"
        )
    if absorber_radii_cm is None:
        absorber_radii_cm = jnp.geomspace(1e-4, 3e-2, radius_count)
    if length_densities_cm_cm3 is None:
        length_densities_cm_cm3 = jnp.geomspace(1e-1, 1e4, density_count)
    radii = [float(value) for value in absorber_radii_cm]
    densities = [float(value) for value in length_densities_cm_cm3]
    modes = tuple(modes)
    dt_days = config.dt if dt_days is None else dt_days
    reference_time_days = (
        config.uptake_reference_time_days
        if reference_time_days is None
        else reference_time_days
    )
    if not radii:
        raise ValueError("absorber_radii_cm must not be empty")
    if not densities:
        raise ValueError("length_densities_cm_cm3 must not be empty")
    if not modes:
        raise ValueError("modes must not be empty")
    for radius in radii:
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                "absorber_radius_cm must be finite and greater than zero"
            )
    for density in densities:
        if not math.isfinite(density) or density < 0.0:
            raise ValueError(
                "length_density_cm_cm3 must be finite and non-negative"
            )
    for name, value in (
        ("dt_days", dt_days),
        ("reference_time_days", reference_time_days),
        ("cell_volume_cm3", cell_volume_cm3),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero")
    rows = []
    for mode in modes:
        if mode not in ("fixed_reservoir", "finite_inventory"):
            raise ValueError(f"unsupported experiment mode: {mode}")
        row_factory = (
            _fixed_reservoir_row
            if mode == "fixed_reservoir"
            else _finite_inventory_row
        )
        rows.extend(
            _surface_rows(
                radii,
                densities,
                mode,
                dt_days,
                reference_time_days,
                cell_volume_cm3,
                config,
                traits,
            )
        )
        if include_markers:
            plant_native = row_factory(
                traits.root_radius,
                traits.root_length_density,
                dt_days,
                reference_time_days,
                cell_volume_cm3,
                config,
                traits,
                record_type="marker",
                marker_label="plant_native",
                economics_mode="plant",
                uptake_traits="plant",
            )
            fungus_native = row_factory(
                fungus.hyphal_radius,
                fungus.saturation_density,
                dt_days,
                reference_time_days,
                cell_volume_cm3,
                config,
                fungus,
                construction_carbon_fn=fungal_construction_carbon_g,
                record_type="marker",
                marker_label="fungus_native",
                economics_mode="fungus",
                uptake_traits="fungus",
            )
            fungus_geometry_plant_economics = row_factory(
                fungus.hyphal_radius,
                fungus.saturation_density,
                dt_days,
                reference_time_days,
                cell_volume_cm3,
                config,
                traits,
                record_type="marker",
                marker_label="fungus_geometry_plant_economics",
                economics_mode="plant",
                uptake_traits="plant",
            )
            plant_native = _apply_geometry_validity(plant_native)
            fungus_native = _apply_geometry_validity(fungus_native)
            fungus_geometry_plant_economics = _apply_geometry_validity(
                fungus_geometry_plant_economics
            )
            rows.extend(
                (plant_native, fungus_native, fungus_geometry_plant_economics)
            )
            for metric in (
                "integrated_uptake_per_construction_carbon_micromol_g_c",
                "maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s",
            ):
                rows.append(
                    _solve_equivalent_marker(
                        row_factory,
                        metric,
                        fungus_native,
                        radii,
                        densities,
                        fungus.saturation_density,
                        dt_days,
                        reference_time_days,
                        cell_volume_cm3,
                        config,
                        traits,
                    )
                )
    return rows
