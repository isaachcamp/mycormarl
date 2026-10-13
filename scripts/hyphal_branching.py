"""Shared illustrative branching used by both sections of the uptake video."""
import numpy as np
from manim import VMobject

TEAL = "#7AE6CF"
LATERAL_COUNT_RANGE = (8, 13)  # Inclusive 8–12, shared by both colony sections.


def organic_filament(start, end, rng, width=1.2):
    """A slender, gently wandering hypha with exact attachment endpoints."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = end - start
    normal = np.array([-delta[1], delta[0], 0.0])
    normal /= max(np.linalg.norm(normal), 1e-10)
    points = []
    bend = rng.uniform(-0.07, 0.07) * np.linalg.norm(delta)
    for t in np.linspace(0, 1, 6):
        deviation = bend * np.sin(np.pi * t) + 0.015 * np.linalg.norm(delta) * np.sin(2 * np.pi * t)
        points.append(start + t * delta + normal * deviation)
    return VMobject(stroke_color=TEAL, stroke_width=width).set_points_smoothly(points)


def side_branches(parent, rng, count, max_length, bounds, order=1):
    """Irregular laterals with abundant finer twigs, attached to curved parents."""
    result = []
    for fraction in np.sort(rng.uniform(0.15, 0.86, count)):
        start = parent.point_from_proportion(float(fraction))
        tangent = parent.point_from_proportion(min(float(fraction) + 0.025, 1)) - start
        heading = np.arctan2(tangent[1], tangent[0])
        sign = rng.choice([-1, 1])
        angle = heading + sign * rng.uniform(0.55, 1.35)
        length = max_length * rng.uniform(0.3, 1.0)
        end = start + length * np.array([np.cos(angle), np.sin(angle), 0])
        end[:2] = np.clip(end[:2], bounds[0], bounds[1])
        if np.linalg.norm(end - start) < 0.015:
            continue
        child = organic_filament(start, end, rng, width=1.05 if order == 1 else 0.65)
        result.append((child, parent, float(fraction), order))
        if order == 1:
            result.extend(side_branches(child, rng, int(rng.integers(2, 5)), max_length * 0.32, bounds, order=2))
    return result


def growth_timings(records, rng):
    """Opening-section tip speeds and delayed parent-contact births."""
    timings = {}
    for curve, parent, fraction, order in records:
        duration = curve.get_arc_length() / (0.70 if order == 0 else 0.42)
        if parent is None:
            birth = 0.0
        else:
            parent_birth, parent_duration = timings[id(parent)]
            birth = parent_birth + parent_duration * fraction + rng.uniform(0.08, 0.25)
        timings[id(curve)] = (birth, duration)
    return timings
