"""Plot the carbon-for-fungal-growth feedback diagnostic for trade-only runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


P_LEVELS = (0.3, 0.75, 1.5, 3.0, 5.0, 10.0)
HIGH_P_LEVELS = (3.0, 5.0, 10.0)
COLOURS = {3.0: "#2878b5", 5.0: "#1b9e77", 10.0: "#d95f02"}
FRACTION_INTERVAL_DAYS = 0.25


def mean_pool_fraction(trace: list[dict], agent: str) -> float:
    """Return mean fraction of the current pool traded per policy interval."""
    fractions = [
        float(-np.expm1(-step["actions"][agent][0] * FRACTION_INTERVAL_DAYS))
        for step in trace
    ]
    return float(np.mean(fractions))


def load_rows(study_dir: Path) -> tuple[list[dict], dict[float, float]]:
    """Load endpoint and cumulative-flow diagnostics from mixed evaluations."""
    bundle = json.loads((study_dir / "result-bundle.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for entry in bundle["entries"]:
        if entry["mode"] != "mixed" or not entry.get("evaluation_artifacts"):
            continue
        evaluation_path = study_dir / entry["evaluation_artifacts"][-1]
        episode = json.loads(evaluation_path.read_text(encoding="utf-8"))["episodes"][0]
        trace = episode["trace"]
        rows.append({
            "p": float(entry["initial_p_micromolar"]),
            "seed": int(entry["seed"]),
            "plant_fitness": float(episode["summary"]["cumulative_reproductive_fitness"]["plant"]),
            "plant_trade_fraction": mean_pool_fraction(trace, "plant"),
            "fungus_trade_fraction": mean_pool_fraction(trace, "fungus"),
            "plant_carbon_export": float(sum(step["transfers"]["plant"]["out"] for step in trace)),
            "fungus_p_export": float(sum(step["transfers"]["fungus"]["out"] for step in trace)),
            "fungus_biomass": float(trace[-1]["state"]["living_biomass"]["fungus"]),
        })

    plant_only = {
        float(entry["initial_p_micromolar"]): float(
            entry["cumulative_reproductive_fitness"]["plant"]
        )
        for entry in bundle["entries"]
        if entry["mode"] == "plant-only"
    }
    if {row["p"] for row in rows} != set(P_LEVELS) or len(rows) != 30:
        raise ValueError("expected five mixed seeds at each of six P levels")
    if set(plant_only) != set(P_LEVELS):
        raise ValueError("expected six plant-only controls")
    return rows, plant_only


def plot(rows: list[dict], plant_only: dict[float, float], output: Path) -> None:
    """Plot the four relationships relevant to the proposed feedback loop."""
    rows = [row for row in rows if row["p"] in HIGH_P_LEVELS]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    axes = axes.ravel()

    panels = (
        ("plant_trade_fraction", "plant_fitness", "Plant→fungus fraction per 0.25-day interval",
         "Plant fitness", "Plant trade can be costly"),
        ("plant_carbon_export", "fungus_biomass", "C exported by plant (g C)",
         "Final living fungal biomass (g)", "Carbon supply and fungal growth"),
        ("fungus_biomass", "fungus_p_export", "Final living fungal biomass (g)",
         "Cumulative fungus→plant P transfer (mg P)", "Fungal biomass and P return"),
        ("fungus_trade_fraction", "fungus_p_export", "Fungus→plant fraction per 0.25-day interval",
         "Cumulative fungus→plant P transfer (mg P)", "Rate is not total P transfer"),
    )

    for panel_label, axis, (x_key, y_key, x_label, y_label, _) in zip(
        ("a.", "b.", "c.", "d."), axes, panels, strict=True
    ):
        for p in HIGH_P_LEVELS:
            subset = [row for row in rows if row["p"] == p]
            colour = COLOURS[p]
            axis.scatter(
                [row[x_key] for row in subset], [row[y_key] for row in subset],
                color=colour, edgecolor="white", linewidth=0.6, s=58,
                alpha=0.82, label=f"P = {p:g} µM",
            )
            for row in subset:
                axis.annotate(str(row["seed"]), (row[x_key], row[y_key]),
                              xytext=(4, 3), textcoords="offset points", fontsize=7,
                              color=colour)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.text(
            -0.16, 1.02, panel_label, transform=axis.transAxes,
            ha="right", va="bottom", fontsize=10, clip_on=False,
        )
        axis.grid(axis="both", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)

    # The first panel is the direct comparison with the biologically relevant
    # no-trade control; controls are P-specific, so use horizontal segments.
    for p in HIGH_P_LEVELS:
        subset = [row for row in rows if row["p"] == p]
        colour = COLOURS[p]
        x_values = [row["plant_trade_fraction"] for row in subset]
        axes[0].hlines(plant_only[p], min(x_values), max(x_values),
                       color=colour, linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].legend(frameon=True, fontsize=8, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.study_dir / "figures" / "trade-only-feedback-diagnostic.png"
    rows, plant_only = load_rows(args.study_dir)
    plot(rows, plant_only, output)
    print(output)


if __name__ == "__main__":
    main()
