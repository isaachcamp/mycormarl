"""Plot plant phosphorus intake and the plant Cobb--Douglas components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


P_LEVELS = (0.3, 0.75, 1.5, 3.0, 5.0, 10.0)
PLANT_COLOUR = "#2878b5"
FUNGUS_COLOUR = "#d95f02"
TOTAL_COLOUR = "#222222"


def load_rows(study_dir: Path) -> list[dict]:
    bundle = json.loads((study_dir / "result-bundle.json").read_text(encoding="utf-8"))
    rows = []
    for entry in bundle["entries"]:
        if entry["mode"] != "mixed" or not entry.get("evaluation_artifacts"):
            continue
        path = study_dir / entry["evaluation_artifacts"][-1]
        episode = json.loads(path.read_text(encoding="utf-8"))["episodes"][0]
        trace = episode["trace"]
        alpha = 0.5
        c = np.asarray([step["reproductive_resources"]["plant"]["carbon"] for step in trace])
        p = np.asarray([step["reproductive_resources"]["plant"]["phosphorus"] for step in trace])
        reward = np.asarray([step["reproductive_fitness"]["plant"] for step in trace])
        component_fitness = np.asarray([
            step["reproductive_fitness_from_components"]["plant"] for step in trace
        ])
        if not np.allclose(reward, component_fitness, rtol=2e-5, atol=2e-8):
            raise ValueError("stored resource components do not reproduce interval fitness")
        direct = sum(step["uptake_mg"]["plant"] for step in trace)
        indirect = sum(step["transfers"]["plant"]["in"] for step in trace)
        rows.append({
            "p": float(entry["initial_p_micromolar"]),
            "seed": int(entry["seed"]),
            "direct_p": float(direct),
            "indirect_p": float(indirect),
            "total_p": float(direct + indirect),
            "reproductive_c": float(np.sum(c)),
            "reproductive_p": float(np.sum(p)),
            "fitness": float(np.sum(reward)),
        })
    if len(rows) != 30 or {row["p"] for row in rows} != set(P_LEVELS):
        raise ValueError("expected five mixed seeds at each of six P levels")
    return rows


def grouped(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray([[row[key] for row in rows if row["p"] == p] for p in P_LEVELS])
    return np.mean(values, axis=1), np.quantile(values, 0.25, axis=1), np.quantile(values, 0.75, axis=1)


def plot_intake(rows: list[dict], output: Path) -> None:
    x = np.arange(len(P_LEVELS))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    for key, colour, label in (
        ("direct_p", PLANT_COLOUR, "Direct soil P uptake"),
        ("indirect_p", FUNGUS_COLOUR, "Indirect fungal P transfer"),
        ("total_p", TOTAL_COLOUR, "Total plant P intake"),
    ):
        mean, lower, upper = grouped(rows, key)
        axes[0].plot(x, mean, marker="o", color=colour, label=label)
        axes[0].fill_between(x, lower, upper, color=colour, alpha=0.14, linewidth=0)
    mean, lower, upper = grouped(rows, "indirect_p")
    total_mean, _, _ = grouped(rows, "total_p")
    axes[1].plot(x, mean / total_mean, marker="o", color=FUNGUS_COLOUR)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Indirect fraction of total plant P intake")
    axes[1].set_title("Fungal contribution to plant P intake")
    axes[0].set_ylabel("Cumulative plant P intake (mg P)")
    axes[0].set_title("Direct and indirect plant phosphorus intake")
    axes[0].legend(frameon=False, loc="upper left")
    for axis in axes:
        axis.set_xticks(x, [f"{p:g}" for p in P_LEVELS])
        axis.set_xlabel("Initial solution P (µM)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
    fig.suptitle("Plant phosphorus acquisition in mixed trade-only runs\nLines: seed means; bands: seed IQR")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_fitness_components(rows: list[dict], output: Path) -> None:
    x = np.asarray(P_LEVELS)
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True, constrained_layout=True)
    for axis, key, label, colour in (
        (axes[0], "reproductive_c", "Reproductive C\n(biomass-equivalent)", PLANT_COLOUR),
        (axes[1], "reproductive_p", "Reproductive P\n(biomass-equivalent)", FUNGUS_COLOUR),
        (axes[2], "fitness", "Cobb--Douglas\nreproductive fitness", TOTAL_COLOUR),
    ):
        mean, lower, upper = grouped(rows, key)
        axis.plot(x, mean, marker="o", color=colour, linewidth=1.4)
        axis.fill_between(x, lower, upper, color=colour, alpha=0.16, linewidth=0)
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[-1].set_xlabel("Initial solution P (µM)")
    axes[-1].set_xscale("log")
    axes[-1].set_xticks(x, [f"{p:g}" for p in x])
    fig.suptitle(
        "Plant fitness resource components in mixed trade-only runs\n"
        r"Per-step fitness is $C_{rep}^{0.5}P_{rep}^{0.5}$; lines: seed means; bands: seed IQR"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.study_dir)
    output = args.study_dir / "figures"
    plot_intake(rows, output / "trade-only-ippo-plant-phosphorus-intake.png")
    plot_fitness_components(rows, output / "trade-only-ippo-fitness-resource-components.png")
    print(f"wrote resource-accounting figures to {output}")


if __name__ == "__main__":
    main()
