"""Deterministic plant-only, high-P growth-scale qualification."""

from __future__ import annotations

import math
from typing import Any, Iterable


_CHECKPOINT_DAYS = (40, 60, 80, 100, 120)
_FORTO_BIOMASS = (0.225, 0.836, 3.046, 10.042, 23.26)
_FORTO_RGR = (0.066, 0.065, 0.060, 0.042)


def _validate_kleaf_values(values: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("kleaf values must be finite values within [0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("kleaf values must be unique")
    return values


def _case(kleaf: float, *, amass: float, dt: float, cap: float) -> dict[str, Any]:
    biomass = 0.01
    # Match BaseMycorMarl.reset: the initial free pools fund one maintenance
    # payment, rather than an additional biomass equivalent.
    free_c = 0.007 * biomass * dt
    free_p = 0.001 * biomass * dt
    initial_c = free_c
    initial_p = free_p
    checkpoints: dict[str, dict[str, float]] = {}
    gross_c = maintenance_c = structural_c = 0.0
    p_uptake = 0.0
    cap_contact = False
    biomass_by_day = {0: biomass}
    accounting_by_day: dict[int, tuple[float, float, float, float]] = {}

    for step in range(1, round(120 / dt) + 1):
        required_c = 0.007 * biomass * dt
        required_p = 0.001 * biomass * dt
        maintenance = min(free_c, required_c)
        p_maintenance = min(free_p, required_p)
        c_deficit = required_c - maintenance
        p_deficit = required_p - p_maintenance
        biomass -= min(biomass, max(c_deficit / 0.402, p_deficit / 1.92))
        free_c -= maintenance
        free_p -= p_maintenance
        maintenance_c += maintenance

        growth = min(free_c / 0.402, free_p / 1.92)
        growth = min(growth, max(cap - biomass, 0.0))
        free_c -= growth * 0.402
        free_p -= growth * 1.92
        biomass += growth
        structural_c += growth * 0.402
        if biomass >= cap:
            cap_contact = True
            biomass = cap

        # Runtime adds fixed C and soil uptake after allocation. The high-P
        # control preloads exactly enough P for the following transition's
        # maintenance and carbon-limited growth allocation.
        gross = biomass * kleaf * amass * dt
        free_c += gross
        gross_c += gross
        next_required_c = 0.007 * biomass * dt
        next_required_p = 0.001 * biomass * dt
        next_growth_c = max(free_c - next_required_c, 0.0)
        supplied_p = max(
            0.0,
            next_required_p + next_growth_c * 1.92 / 0.402 - free_p,
        )
        free_p += supplied_p
        p_uptake += supplied_p
        day = round(step * dt)
        if day in _CHECKPOINT_DAYS and abs(step * dt - day) < 1e-9:
            biomass_by_day[day] = biomass
            accounting_by_day[day] = (
                gross_c,
                maintenance_c,
                structural_c,
                free_c - initial_c,
            )

    for index, day in enumerate(_CHECKPOINT_DAYS):
        current = biomass_by_day[day]
        previous = biomass_by_day[_CHECKPOINT_DAYS[index - 1]] if index else None
        gross_at_day, maintenance_at_day, structural_at_day, free_change_at_day = accounting_by_day[day]
        checkpoints[str(day)] = {
            "biomass_g_dm": current,
            "windowed_rgr_per_day": (
                math.log(current / previous) / 20.0 if previous else None
            ),
            "gross_fixation_g_c": gross_at_day,
            "maintenance_g_c": maintenance_at_day,
            "structural_growth_c_g": structural_at_day,
            "free_pool_change_g_c": free_change_at_day,
            "initial_pool_contribution_g_c": initial_c,
        }

    endpoint = biomass_by_day[120]
    mismatch = endpoint < 25.0 or endpoint > 35.0
    return {
        "kleaf": kleaf,
        "amass": amass,
        "effective_fixation_g_c_per_g_dm_day": kleaf * amass,
        "biomass_g_dm": {str(day): biomass_by_day[day] for day in _CHECKPOINT_DAYS},
        "checkpoints": checkpoints,
        "gross_fixation_g_c": gross_c,
        "maintenance_g_c": maintenance_c,
        "structural_growth_c_g": structural_c,
        "free_pool_change_g_c": free_c - initial_c,
        "initial_pool_contribution_g_c": initial_c + initial_p * 0.402 / 1.92,
        "p_uptake_mg": p_uptake,
        "realized_limitation": "carbon",
        "cap_contact": cap_contact,
        "material_growth_scale_mismatch": mismatch,
        "status": "failed" if cap_contact or mismatch else "passed",
    }


def run_plant_growth_qualification(
    *,
    kleaf_values: Iterable[float] = (0.30, 0.45, 0.50, 0.60, 0.65, 0.675, 0.68, 0.70),
    amass: float = 0.05,
    timestep_days: float = 0.025,
    biomass_cap: float = 50.0,
) -> dict[str, Any]:
    """Run the deterministic high-P plant-only qualification and return JSON data."""
    values = _validate_kleaf_values(kleaf_values)
    if not math.isfinite(amass) or amass <= 0 or not math.isfinite(timestep_days) or timestep_days <= 0:
        raise ValueError("amass and timestep_days must be finite and positive")
    if biomass_cap <= 0 or not math.isfinite(biomass_cap):
        raise ValueError("biomass_cap must be finite and positive")
    selected_kleaf = 0.68
    net_fixation = selected_kleaf * amass - 0.007
    cases = {
        f"kleaf_{value:.3f}": _case(
            value, amass=amass, dt=timestep_days, cap=biomass_cap
        )
        for value in values
    }
    return {
        "stage": "plant-growth-qualification",
        "selected_kleaf": selected_kleaf,
        "policy": {"consumer_mode": "plant-only", "trade": 0.0, "growth": 1.0, "reproduction": 0.0, "reserve": 0.0},
        "horizon_days": 120,
        "timestep_days": timestep_days,
        "cases": cases,
        "analytical_carbon_only": {
            "effective_fixation_g_c_per_g_dm_day": selected_kleaf * amass,
            "net_carbon_rate_g_c_per_g_dm_day": net_fixation,
            "sustained_rgr_per_day": net_fixation / 0.402,
            "upper_bound_endpoint_g_dm": 0.01 * (
                1.0 + (net_fixation / 0.402) * timestep_days
            ) ** (round(120 / timestep_days) - 1),
        },
        "reference": {
            "forto_endpoint_g_dm": 23.26,
            "favourable_trajectory_range_g_dm": [25.0, 35.0],
            "rgr_windows_per_day": list(_FORTO_RGR),
            "forto_biomass_g_dm": dict(zip(map(str, _CHECKPOINT_DAYS), _FORTO_BIOMASS)),
        },
        "status": "failed" if any(case["status"] == "failed" for case in cases.values()) else "passed",
    }
