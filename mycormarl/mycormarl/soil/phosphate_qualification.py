"""Pure comparison and cost helpers for phosphate-model qualification."""

from __future__ import annotations

import math


def reference_relative_change(
    candidate: float,
    reference: float,
    absolute_floor: float,
) -> float:
    """Return candidate error relative to reference with a near-zero guard.

    The P6 selection rule compares a coarser/larger-step candidate with its
    finer/smaller-step reference. If both magnitudes are below the declared
    scientific reporting floor, their relative difference is defined as zero
    rather than amplified into a meaningless convergence failure.
    """
    if max(abs(candidate), abs(reference)) <= absolute_floor:
        return 0.0
    denominator = max(abs(reference), absolute_floor)
    return abs(candidate - reference) / denominator


def annual_runtime_projection(
    dt_days: float,
    seconds_per_step: float,
) -> dict[str, float | int]:
    """Project warmed fixed-step runtime for a 365-day simulation.

    The ceiling ensures a partial final interval still covers the full year.
    Compilation time is intentionally excluded and should be reported beside
    this projection as a separate one-off cost.
    """
    if not math.isfinite(dt_days) or dt_days <= 0.0:
        raise ValueError("dt_days must be finite and greater than zero")
    if not math.isfinite(seconds_per_step) or seconds_per_step < 0.0:
        raise ValueError("seconds_per_step must be finite and non-negative")
    steps_per_year = math.ceil(365.0 / dt_days)
    runtime_seconds = steps_per_year * seconds_per_step
    return {
        "steps_per_year": steps_per_year,
        "runtime_seconds": runtime_seconds,
        "runtime_hours": runtime_seconds / 3600.0,
    }
