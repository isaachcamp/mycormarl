"""Create the four surviving plots for the paired low-P fungal gamma_P screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GAMMA = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
P_BINS = (("0.1–0.2 µM", 0.1, 0.2), ("0.2–0.3 µM", 0.2, 0.3))
FACTORS = (("plant_kappa_c", "Plant κC", True), ("fungus_kappa_c", "Fungus κC", True),
           ("fungus_initial_biomass", "Initial AMF biomass", True), ("plant_trade", "Plant→fungus trade", False),
           ("fungus_trade", "Fungus→plant trade", False))
COLORS = {P_BINS[0][0]: "#2878b5", P_BINS[1][0]: "#d95f02"}


def load_rows(directory: Path) -> list[dict[str, float]]:
    rows = []
    for path in sorted(directory.glob("condition-*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        trace = [step["agents"]["fungus"] for step in entry.get("limitation_trace", [])]
        active = [not step["no_realized_growth"] for step in trace]
        denominator = max(sum(active), 1)
        factors = entry["factors"]
        row = {name: float(factors[name]) for name, _, _ in FACTORS}
        row.update({"gamma_p": float(factors["fungus_gamma_p"]), "initial_p": float(factors["initial_solution_p_micromolar"]),
                    "plant_biomass": float(entry["biomass"]["plant"]), "fungus_biomass": float(entry["biomass"]["fungus"]),
                    "fungus_p_limited": sum(step["limiting_resource"] == "phosphate" and on for step, on in zip(trace, active)) / denominator})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no checkpoints found in {directory}")
    return rows


def groups(rows: list[dict[str, float]]) -> list[list[dict[str, float]]]:
    grouped = {}
    for index, row in enumerate(rows):
        grouped.setdefault(index // len(GAMMA), []).append(row)
    result = [sorted(value, key=lambda row: row["gamma_p"]) for _, value in sorted(grouped.items())]
    if any(len(group) != len(GAMMA) for group in result):
        raise ValueError("paired checkpoints do not contain exactly five gamma_P levels per base condition")
    return result


def p_band(row: dict[str, float]) -> str:
    return P_BINS[0][0] if row["initial_p"] < 0.2 else P_BINS[1][0]


def threshold(group: list[dict[str, float]], cutoff: float = 0.5) -> float:
    values = np.array([row["fungus_p_limited"] for row in group])
    for index, (low, high) in enumerate(zip(values[:-1], values[1:])):
        if low <= cutoff <= high:
            if low == high:
                return float(GAMMA[index])
            return float(GAMMA[index] + (cutoff - low) / (high - low) * (GAMMA[index + 1] - GAMMA[index]))
    return float(GAMMA[0] if values[0] > cutoff else GAMMA[-1])


def threshold_records(base_groups):
    records = []
    for group in base_groups:
        row = group[0]
        records.append({name: row[name] for name, _, _ in FACTORS} | {"p_band": p_band(row), "transition_gamma": threshold(group)})
    return records


def gamma_edges():
    edges = np.empty(len(GAMMA) + 1)
    edges[1:-1] = (GAMMA[:-1] + GAMMA[1:]) / 2
    edges[0] = GAMMA[0] - (edges[1] - GAMMA[0])
    edges[-1] = GAMMA[-1] + (GAMMA[-1] - edges[-2])
    return edges


def trade_edges():
    return np.arange(0.50, 0.80 + 0.025, 0.05)


def quantile_groups(values, bins=4):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)


def interaction_grid(rows, lower, upper):
    selected = [row for row in rows if lower <= row["initial_p"] < upper]
    y_edges = trade_edges()
    grid = np.full((len(y_edges) - 1, len(GAMMA)), np.nan)
    for column, gamma in enumerate(GAMMA):
        current = [row for row in selected if row["gamma_p"] == gamma]
        bins = np.clip(np.digitize([row["fungus_trade"] for row in current], y_edges[1:-1]), 0, len(y_edges) - 2)
        for index in range(len(y_edges) - 1):
            values = [row["fungus_p_limited"] for row, value in zip(current, bins) if value == index]
            if values:
                grid[index, column] = np.median(values)
    return grid, y_edges


def plot_thresholds(records, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, constrained_layout=True)
    for ax, (name, label, logarithmic) in zip(axes.flat, FACTORS):
        for band, _, _ in P_BINS:
            values = [row for row in records if row["p_band"] == band]
            ax.scatter([row[name] for row in values], [row["transition_gamma"] for row in values], color=COLORS[band], s=28, alpha=.75, label=band)
        if logarithmic:
            ax.set_xscale("log")
        ax.set_xlabel(label); ax.set_ylim(.45, 2.05); ax.grid(color="#ddd", linewidth=.5)
    axes[0, 0].set_ylabel("Transition γP (fungal P-limitation fraction = 0.5)")
    axes[1, 0].set_ylabel("Transition γP"); axes[0, 0].legend(frameon=False, fontsize=9); axes.flat[-1].axis("off")
    fig.suptitle("Fungal P-limitation transition threshold by sampled factor"); fig.savefig(output, dpi=180); plt.close(fig)


def plot_probability_curves(base_groups, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(.15, .9, 4))
    for ax, (name, label, _) in zip(axes.flat, FACTORS):
        for band_index, (_, lower, upper) in enumerate(P_BINS):
            subset = [group for group in base_groups if lower <= group[0]["initial_p"] < upper]
            quartiles = quantile_groups(np.array([group[0][name] for group in subset]))
            for quartile in range(4):
                selected = [group for group, value in zip(subset, quartiles) if value == quartile]
                probability = [np.mean([next(row for row in group if row["gamma_p"] == gamma)["fungus_p_limited"] < .5 for group in selected]) for gamma in GAMMA]
                ax.plot(GAMMA, probability, color=colors[quartile], linestyle="-" if band_index == 0 else "--", marker="o", markersize=3, linewidth=1.2)
        ax.set_title(label); ax.set_xlabel("Fungal γP"); ax.set_ylim(-.03, 1.03); ax.grid(color="#ddd", linewidth=.5)
    axes[0, 0].set_ylabel("Functional fungal regime probability"); axes[1, 0].set_ylabel("Functional regime probability")
    axes[0, 0].plot([], [], color="black", label="0.1–0.2 µM P"); axes[0, 0].plot([], [], color="black", linestyle="--", label="0.2–0.3 µM P")
    axes[0, 0].plot([], [], color=colors[0], label="Q1 factor"); axes[0, 0].plot([], [], color=colors[-1], label="Q4 factor")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2); axes.flat[-1].axis("off")
    fig.suptitle("Probability of a functional fungal regime by factor quartile"); fig.savefig(output, dpi=180); plt.close(fig)


def plot_interaction(rows, output, bands=P_BINS):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharey=True, constrained_layout=True); image = None
    for ax, (_, lower, upper) in zip(axes, bands):
        grid, y_edges = interaction_grid(rows, lower, upper); centres = (y_edges[:-1] + y_edges[1:]) / 2
        image = ax.pcolormesh(gamma_edges(), y_edges, np.ma.masked_invalid(grid), cmap="magma", vmin=0, vmax=1, shading="flat")
        try: ax.contour(GAMMA, centres, grid, levels=[.5], colors="white", linewidths=2)
        except ValueError: pass
        ax.set_title(f"{lower:.1f}–{upper:.1f} µM initial P"); ax.set_xlabel("Fungal γP"); ax.set_xticks(GAMMA); ax.set_yticks(y_edges); ax.grid(color="white", linewidth=.4, alpha=.45)
    axes[0].set_ylabel("Fungus→plant trade fraction"); axes[0].text(.02, .04, "white contour: 0.5 P-limited", transform=axes[0].transAxes, color="white", fontsize=9)
    fig.colorbar(image, ax=axes.tolist(), label="Median fungal P-limited timestep fraction", shrink=.88); fig.suptitle("Fungal γP × fungus→plant trade interaction"); fig.savefig(output, dpi=180); plt.close(fig)


def plot_summary(rows, base_groups, output):
    records = threshold_records(base_groups)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True); biomass, limitation, threshold_ax, heatmap = axes.flat
    for label, lower, upper in P_BINS:
        for organism, key, linestyle, marker in (("plant", "plant_biomass", "-", "o"), ("fungus", "fungus_biomass", "--", "s")):
            values = [np.median([row[key] for row in rows if lower <= row["initial_p"] < upper and row["gamma_p"] == gamma]) for gamma in GAMMA]
            biomass.plot(GAMMA, values, color=COLORS[label], linestyle=linestyle, marker=marker, label=f"{label}, {organism}")
        limited = [np.median([row["fungus_p_limited"] for row in rows if lower <= row["initial_p"] < upper and row["gamma_p"] == gamma]) for gamma in GAMMA]
        limitation.plot(GAMMA, limited, color=COLORS[label], marker="o", label=label)
    biomass.set_yscale("log"); biomass.set_xticks(GAMMA); biomass.set_xlabel("Fungal γP"); biomass.set_ylabel("Median final biomass (g; log scale)"); biomass.set_title("Matched median biomass"); biomass.grid(axis="y", which="both", color="#ddd", linewidth=.5); biomass.legend(frameon=False, fontsize=8)
    limitation.set_xticks(GAMMA); limitation.set_ylim(0, 1); limitation.set_xlabel("Fungal γP"); limitation.set_ylabel("Median fungal P-limited fraction"); limitation.set_title("Fungal P limitation"); limitation.grid(axis="y", color="#ddd", linewidth=.5); limitation.legend(frameon=False, fontsize=8)
    for band in (P_BINS[0][0], P_BINS[1][0]):
        values = [row for row in records if row["p_band"] == band]
        threshold_ax.scatter([row["fungus_trade"] for row in values], [row["transition_gamma"] for row in values], color=COLORS[band], s=25, alpha=.8, label=band)
    threshold_ax.set_xlabel("Fungus→plant trade fraction"); threshold_ax.set_ylabel("Transition γP (fungal P-limited fraction = 0.5)"); threshold_ax.set_ylim(.45, 2.05); threshold_ax.grid(color="#ddd", linewidth=.5); threshold_ax.set_title("Fungus→plant trade transition threshold"); threshold_ax.legend(frameon=False, fontsize=8)
    grid, y_edges = interaction_grid(rows, .2, .3); image = heatmap.pcolormesh(gamma_edges(), y_edges, np.ma.masked_invalid(grid), cmap="magma", vmin=0, vmax=1, shading="flat")
    try: heatmap.contour(GAMMA, (y_edges[:-1] + y_edges[1:]) / 2, grid, levels=[.5], colors="white", linewidths=1.8)
    except ValueError: pass
    heatmap.set_xticks(GAMMA); heatmap.set_yticks(y_edges); heatmap.set_xlabel("Fungal γP"); heatmap.set_ylabel("Fungus→plant trade fraction"); heatmap.set_title("Higher-P trade × γP interaction"); heatmap.grid(color="white", linewidth=.35, alpha=.45); fig.colorbar(image, ax=heatmap, label="Median fungal P-limited fraction", shrink=.85)
    for axis, label in zip(axes.flat, "ABCD"): axis.text(-.12, 1.05, label, transform=axis.transAxes, fontsize=14, fontweight="bold", va="top")
    fig.suptitle("Paired low-P fungal γP experiment: biomass and fungal P-limitation summary", fontsize=15); fig.savefig(output, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("checkpoint_dir", type=Path); parser.add_argument("output_dir", type=Path); args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.checkpoint_dir); base_groups = groups(rows); records = threshold_records(base_groups)
    plot_thresholds(records, args.output_dir / "resource-pressure-low-p-gamma-regime-thresholds.png")
    plot_probability_curves(base_groups, args.output_dir / "resource-pressure-low-p-gamma-regime-probability-curves.png")
    plot_interaction(rows, args.output_dir / "resource-pressure-low-p-gamma-trade-limitation-interaction.png")
    plot_summary(rows, base_groups, args.output_dir / "resource-pressure-low-p-gamma-summary.png")
    print(f"wrote four surviving figures from {len(base_groups)} matched base conditions")


if __name__ == "__main__": main()
