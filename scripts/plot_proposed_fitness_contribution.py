"""Quickly visualise the proposed weighted-input contribution function."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STUDY = Path("outputs/historical-grid-trade-only-pilot/historical-grid-trade-only-pilot")
P_LEVELS = (3.0, 5.0, 10.0)
COLOURS = {3.0: "#2878b5", 5.0: "#1b9e77", 10.0: "#d95f02"}
ALPHA = 0.5


def main() -> None:
    bundle = json.loads((STUDY / "result-bundle.json").read_text())
    points = []
    for entry in bundle["entries"]:
        p_level = float(entry["initial_p_micromolar"])
        if entry["mode"] != "mixed" or p_level not in P_LEVELS:
            continue
        evaluation = json.loads((STUDY / entry["evaluation_artifacts"][-1]).read_text())
        trace = evaluation["episodes"][0]["trace"]
        c = sum(row["reproductive_resources"]["plant"]["carbon"] for row in trace)
        p = sum(row["reproductive_resources"]["plant"]["phosphorus"] for row in trace)
        share = (1 - ALPHA) * p / (ALPHA * c + (1 - ALPHA) * p)
        points.append((p_level, int(entry["seed"]), p / max(c, 1e-12), 100 * share))

    ratio = np.logspace(-2, 2, 400)
    curve = 100 * ((1 - ALPHA) * ratio) / (ALPHA + (1 - ALPHA) * ratio)
    figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    axis.plot(ratio, curve, color="#222222", linewidth=1.8, label="Proposed function")
    for p_level in P_LEVELS:
        subset = [point for point in points if point[0] == p_level]
        axis.scatter([point[2] for point in subset], [point[3] for point in subset],
                     color=COLOURS[p_level], edgecolor="white", linewidth=0.6,
                     s=62, label=f"Observed P = {p_level:g} µM")
        for _, seed, x, y in subset:
            axis.annotate(str(seed), (x, y), xytext=(4, 3), textcoords="offset points",
                          fontsize=8, color=COLOURS[p_level])
    axis.set_xscale("log")
    axis.set_xlabel("Cumulative reproductive P / C input")
    axis.set_ylabel("Normalised phosphorus contribution (%)")
    axis.set_ylim(0, 100)
    axis.grid(color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_title("Proposed weighted-input contribution function\n"
                   r"$s_P=100(1-\alpha)P/[\alpha C+(1-\alpha)P]$, with $\alpha=0.5$")
    axis.legend(frameon=True, fontsize=8)
    output = STUDY / "figures" / "trade-only-ippo-proposed-fitness-contribution.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    print(output)


if __name__ == "__main__":
    main()
