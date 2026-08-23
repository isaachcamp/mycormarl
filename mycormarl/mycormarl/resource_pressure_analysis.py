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


def _plant_limitation_response(entry: dict[str, Any]) -> dict[str, float | int | None]:
    """Return plant limitation fractions over realised-growth timesteps only."""
    trace = [step["agents"]["plant"] for step in entry["limitation_trace"]]
    realised = [step for step in trace if not step["no_realized_growth"]]
    if not realised:
        return {
            "realised_growth_steps": 0,
            "plant_p_limited_fraction": None,
            "plant_c_limited_fraction": None,
        }
    count = len(realised)
    return {
        "realised_growth_steps": count,
        "plant_p_limited_fraction": sum(
            step["limiting_resource"] == "phosphate" for step in realised
        ) / count,
        "plant_c_limited_fraction": sum(
            step["limiting_resource"] == "carbon" for step in realised
        ) / count,
    }


def factorial_plant_boundary_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Estimate observed plant P/C boundaries from factorial checkpoint entries.

    A threshold is calculated only within the first adjacent sampled initial-P
    pair that brackets a P-limited fraction of 0.5.  No response curve is fit
    and censoring preserves outcomes outside the 0.5--1.3 µM test interval.
    """
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for entry in entries:
        factors = entry["factors"]
        key = (float(factors["plant_kappa_c"]), float(factors["plant_trade"]))
        grouped.setdefault(key, []).append(entry)

    rows = []
    for (plant_kappa_c, plant_trade), cell_entries in sorted(grouped.items()):
        response = []
        for entry in sorted(
            cell_entries,
            key=lambda item: float(item["factors"]["initial_solution_p_micromolar"]),
        ):
            response.append({
                "initial_solution_p_micromolar": float(
                    entry["factors"]["initial_solution_p_micromolar"]
                ),
                "final_plant_biomass_g": float(entry["biomass"]["plant"]),
                "final_fungus_biomass_g": float(entry["biomass"]["fungus"]),
                **_plant_limitation_response(entry),
            })

        valid = [point for point in response if point["plant_p_limited_fraction"] is not None]
        threshold = None
        status = "insufficient-realised-growth"
        if valid:
            exact = next(
                (point for point in valid if point["plant_p_limited_fraction"] == 0.5), None
            )
            if exact is not None:
                threshold = exact["initial_solution_p_micromolar"]
                status = "observed-level"
            else:
                for lower, upper in zip(response, response[1:]):
                    if (
                        lower["plant_p_limited_fraction"] is None
                        or upper["plant_p_limited_fraction"] is None
                    ):
                        continue
                    lower_fraction = float(lower["plant_p_limited_fraction"])
                    upper_fraction = float(upper["plant_p_limited_fraction"])
                    if (lower_fraction - 0.5) * (upper_fraction - 0.5) < 0.0:
                        lower_p = float(lower["initial_solution_p_micromolar"])
                        upper_p = float(upper["initial_solution_p_micromolar"])
                        threshold = lower_p + (0.5 - lower_fraction) * (
                            upper_p - lower_p
                        ) / (upper_fraction - lower_fraction)
                        status = "observed-crossing"
                        break
                if threshold is None:
                    fractions = [float(point["plant_p_limited_fraction"]) for point in valid]
                    if all(fraction > 0.5 for fraction in fractions):
                        status = "upper-censored"
                    elif all(fraction < 0.5 for fraction in fractions):
                        status = "lower-censored"
                    else:
                        status = "unbracketed"

        fractions = [float(point["plant_p_limited_fraction"]) for point in valid]
        rows.append({
            "plant_kappa_c": plant_kappa_c,
            "plant_trade": plant_trade,
            "threshold_initial_p_micromolar": threshold,
            "threshold_status": status,
            "response_is_monotonic_nonincreasing": all(
                left >= right for left, right in zip(fractions, fractions[1:])
            ),
            "mean_final_plant_biomass_g": sum(
                point["final_plant_biomass_g"] for point in response
            ) / len(response),
            "mean_final_fungus_biomass_g": sum(
                point["final_fungus_biomass_g"] for point in response
            ) / len(response),
            "p_response": response,
        })
    return rows


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
