"""Pure trace aggregation for resource-pressure screen analyses.

The screen stores realised growth in ``used_c_normalized``.  It has units of
grams dry mass per transition, so cumulative values reconstruct the living
biomass trajectory from the recorded initial biomass.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


TRACE_FACTORS = (
    "plant_kappa_c",
    "fungus_kappa_c",
    "fungus_gamma_p",
    "fungus_initial_biomass",
    "initial_solution_p_micromolar",
    "plant_trade",
    "fungus_trade",
)


def reconstruct_biomass_trajectory(entry: dict[str, Any], agent: str) -> list[tuple[float, float]]:
    """Return ``(day, biomass_g)`` from the initial biomass and realised growth."""
    biomass = float(entry["initial_biomass"][agent])
    trajectory = []
    for step in entry["limitation_trace"]:
        biomass += float(step["agents"][agent]["used_c_normalized"])
        trajectory.append((float(step["day"]), biomass))
    return trajectory


def daily_amf_trace_rows(entries: list[dict[str, Any]], day_width: float = 1.0) -> list[dict[str, float | str]]:
    """Aggregate AMF timing diagnostics into non-overlapping day windows.

    Rates retain their recorded per-day units: fungal P uptake and transfer are
    mg P day⁻¹, and realised fungal growth is g dry mass day⁻¹.  The plant
    indirect-P fraction is P received from the fungus divided by the sum of
    that transfer and direct soil-P acquisition in the same window.
    """
    if day_width <= 0:
        raise ValueError("day_width must be positive")
    rows: list[dict[str, float | str]] = []
    for entry in entries:
        accumulated_biomass = {
            "plant": float(entry["initial_biomass"]["plant"]),
            "fungus": float(entry["initial_biomass"]["fungus"]),
        }
        windows: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for step in entry["limitation_trace"]:
            day = float(step["day"])
            window = max(1, math.ceil(day / day_width - 1e-12))
            fungus = step["agents"]["fungus"]
            plant = step["agents"]["plant"]
            fungal_growth = float(fungus["used_c_normalized"])
            accumulated_biomass["fungus"] += fungal_growth
            accumulated_biomass["plant"] += float(plant["used_c_normalized"])
            values = windows[window]
            values["fungus_growth_g"] += fungal_growth
            values["fungus_p_uptake_mg"] += float(fungus["acquired_p"])
            values["fungus_p_transfer_mg"] += float(fungus["trade_out_raw"])
            values["plant_direct_p_uptake_mg"] += float(plant["acquired_p"])
            values["plant_indirect_p_uptake_mg"] += float(plant["trade_in_raw"])
            values["fungus_biomass"] = accumulated_biomass["fungus"]
            values["plant_biomass"] = accumulated_biomass["plant"]
        for window, values in sorted(windows.items()):
            direct = values["plant_direct_p_uptake_mg"]
            indirect = values["plant_indirect_p_uptake_mg"]
            total = direct + indirect
            row: dict[str, float | str] = {
                "id": str(entry["id"]),
                "day": window * day_width,
                "fungus_biomass": values["fungus_biomass"],
                "plant_biomass": values["plant_biomass"],
                "fungus_growth_rate_g_per_day": values["fungus_growth_g"] / day_width,
                "fungus_p_uptake_mg_per_day": values["fungus_p_uptake_mg"] / day_width,
                "fungus_p_transfer_mg_per_day": values["fungus_p_transfer_mg"] / day_width,
                "plant_indirect_p_fraction": indirect / total if total > 0 else 0.0,
            }
            row.update({factor: float(entry["factors"][factor]) for factor in TRACE_FACTORS})
            rows.append(row)
    return rows
