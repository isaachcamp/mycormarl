"""Generate all surviving trade-only study figures from one self-contained analysis script."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

_AGENTS = ("plant", "fungus")
_COLOURS = ("#2878b5", "#d95f02", "#1b9e77")
_LABELS = ("Plant", "Fungus", "Combined")


def load_fitness(study_dir: Path) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, dict[float, dict[str, float]]]]:
    """Load IPPO seed fitness and the two fixed-control fitness references."""
    values: dict[float, dict[str, list[float]]] = {}
    for path in sorted(study_dir.glob("mixed/p-*/seed-*/evaluation.json")):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        p_level = float(evaluation["initial_p_micromolar"])
        seed = int(evaluation["seed"])
        fitness = evaluation["evaluation"]["episodes"][0]["summary"][
            "cumulative_reproductive_fitness"
        ]
        record = values.setdefault(p_level, {agent: [] for agent in (*_AGENTS, "combined")})
        if seed in record.setdefault("seeds", []):
            raise ValueError(f"duplicate evaluation for P={p_level:g}, seed={seed}")
        record["seeds"].append(seed)
        for agent in _AGENTS:
            record[agent].append(float(fitness[agent]))
        record["combined"].append(float(fitness["plant"]) + float(fitness["fungus"]))

    if len(values) != 6:
        raise ValueError(f"expected six P levels, found {len(values)}")
    for p_level, record in values.items():
        if sorted(record["seeds"]) != [0, 1, 2, 3, 4]:
            raise ValueError(f"P={p_level:g} does not contain exactly seeds 0--4")
        for metric in (*_AGENTS, "combined"):
            record[metric] = np.asarray(record[metric], dtype=float)
    baseline_payload = json.loads(
        (study_dir / "fixed-allocation-baseline.json").read_text(encoding="utf-8")
    )
    baselines: dict[str, dict[float, dict[str, float]]] = {"plant-only": {}, "mixed": {}}
    for entry in baseline_payload["entries"]:
        mode = entry.get("mode")
        if mode not in baselines or entry.get("status") != "completed":
            continue
        p_level = float(entry["initial_p_micromolar"])
        plant = float(entry["cumulative_reproductive_fitness"]["plant"])
        fungus = float(entry["cumulative_reproductive_fitness"].get("fungus", 0.0))
        baselines[mode][p_level] = {
            "plant": plant, "fungus": fungus, "combined": plant + fungus,
        }
    if any(set(baselines[mode]) != set(values) for mode in baselines):
        raise ValueError("fixed baselines and IPPO P conditions do not match")
    return values, baselines


def plot_fitness_legacy(
    records: dict[float, dict[str, np.ndarray]],
    baselines: dict[str, dict[float, dict[str, float]]],
    output: Path,
) -> None:
    """Plot seed means with sample-standard-deviation error bars and seed points."""
    p_levels = sorted(records)
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.0), sharey=False, constrained_layout=True)
    x = np.arange(3, dtype=float)
    offsets = (-0.18, 0.0, 0.18)
    rng = np.random.default_rng(2048)
    for axis, p_level in zip(axes.flat, p_levels, strict=True):
        record = records[p_level]
        for index, (metric, colour, label, offset) in enumerate(
            zip(("plant", "fungus", "combined"), _COLOURS, _LABELS, offsets, strict=True)
        ):
            seed_values = record[metric]
            mean = float(np.mean(seed_values))
            spread = float(np.std(seed_values, ddof=1))
            jitter = rng.uniform(-0.055, 0.055, size=seed_values.size)
            axis.scatter(
                np.full(seed_values.size, x[index] + offset) + jitter,
                seed_values,
                color=colour, alpha=0.28, edgecolor="white", linewidth=0.35,
                s=38, zorder=2,
            )
            axis.errorbar(
                x[index] + offset, mean, yerr=spread, fmt="o", color=colour,
                markeredgecolor="white", markeredgewidth=0.5, markersize=7,
                capsize=3.5, elinewidth=1.2, zorder=3,
            )
            for mode, marker in (("plant-only", "s"), ("mixed", "^")):
                axis.scatter(
                    x[index] + offset, baselines[mode][p_level][metric],
                    color=colour, marker=marker, edgecolor="white", linewidth=0.5,
                    s=58, zorder=4,
                )
        axis.set_title(f"Initial solution P = {p_level:g} µM")
        axis.set_xticks(x, _LABELS)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_xlim(-0.5, 2.5)

    for axis in axes[:, 0]:
        axis.set_ylabel("Cumulative reproductive fitness")
    for axis in axes[-1]:
        axis.set_xlabel("Fitness component")
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], color="#555555", marker="s", linestyle="None", markersize=7,
               label="Fixed plant-only"),
        Line2D([], [], color="#555555", marker="^", linestyle="None", markersize=7,
               label="Fixed mixed trade"),
    ]
    axes[0, 0].legend(legend_handles, [handle.get_label() for handle in legend_handles],
                      loc="upper left", frameon=True, fontsize=9)
    fig.suptitle("Plant, fungal, and combined fitness by initial solution P\nmean ± sample SD; translucent points are individual seeds")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main_fitness() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path, nargs="?", default=Path("outputs/trade-only-ippo"))
    parser.add_argument(
        "--pareto-overlay-output", type=Path,
        default=Path("outputs/trade-only-ippo/figures/trade-only-ippo-fitness-pareto-overlay.png"),
    )
    parser.add_argument(
        "--pareto-baseline-output", type=Path,
        default=Path("outputs/trade-only-ippo/figures/trade-only-ippo-fitness-pareto-fixed-trade-reference.png"),
    )
    args = parser.parse_args()
    args.pareto_overlay_output.parent.mkdir(parents=True, exist_ok=True)
    records, baselines = load_fitness(args.study_dir)
    plot_pareto_overlay(records, baselines, args.pareto_overlay_output)
    plot_pareto_fixed_reference(records, baselines, args.pareto_baseline_output)
    print(args.pareto_overlay_output)
    print(args.pareto_baseline_output)


def plot_by_component(
    records: dict[float, dict[str, np.ndarray]],
    baselines: dict[str, dict[float, dict[str, float]]],
    output: Path,
) -> None:
    """Compare all P levels per fitness component on independent log y axes."""
    p_levels = sorted(records)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.0), sharex=True, constrained_layout=True)
    x = np.asarray(p_levels, dtype=float)
    offsets = (-0.025, 0.0, 0.025)
    for axis, metric, colour, label in zip(
        axes, ("plant", "fungus", "combined"), _COLOURS, _LABELS, strict=True
    ):
        positive_values = []
        for p_level in p_levels:
            positive_values.extend(records[p_level][metric][records[p_level][metric] > 0])
            for mode in baselines:
                value = baselines[mode][p_level][metric]
                if value > 0:
                    positive_values.append(value)
        floor = max(min(positive_values) * 0.5, 1e-8)
        for index, p_level in enumerate(p_levels):
            seed_values = records[p_level][metric]
            mean = float(np.mean(seed_values))
            spread = float(np.std(seed_values, ddof=1))
            lower_spread = min(spread, mean - floor)
            axis.errorbar(
                p_level * (1.0 + offsets[0]), mean,
                yerr=np.asarray([[max(lower_spread, 0.0)], [spread]]),
                fmt="o", color=colour, markersize=6.5, capsize=3, elinewidth=1.1,
                markeredgecolor="white", markeredgewidth=0.45, zorder=3,
            )
            jitter = np.linspace(-0.012, 0.012, seed_values.size)
            axis.scatter(
                p_level * (1.0 + jitter), seed_values, color=colour, alpha=0.28,
                edgecolor="white", linewidth=0.3, s=34, zorder=2,
            )
            for mode, marker, offset in (("plant-only", "s", offsets[1]), ("mixed", "^", offsets[2])):
                value = baselines[mode][p_level][metric]
                if value > 0:
                    axis.scatter(
                        p_level * (1.0 + offset), value, color=colour, marker=marker,
                        edgecolor="white", linewidth=0.45, s=55, zorder=4,
                    )
        axis.set_yscale("log")
        axis.set_ylabel(f"{label}\nfitness")
        axis.set_title(label)
        axis.grid(axis="y", which="both", color="#dddddd", linewidth=0.6)
        axis.set_ylim(bottom=floor)
    axes[-1].set_xlabel("Initial solution P (µM; logarithmic scale)")
    axes[-1].set_xscale("log")
    axes[-1].set_xticks(x, [f"{p:g}" for p in x])
    from matplotlib.lines import Line2D
    axes[0].legend(
        [Line2D([], [], color="#555555", marker="s", linestyle="None", markersize=7),
         Line2D([], [], color="#555555", marker="^", linestyle="None", markersize=7)],
        ["Fixed plant-only", "Fixed mixed trade"], loc="upper left", frameon=True, fontsize=9,
    )
    fig.suptitle(
        "Fitness across P levels\nIPPO mean ± sample SD; translucent points are seeds; zero baselines omitted on log axes"
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_pareto(
    records: dict[float, dict[str, np.ndarray]],
    baselines: dict[str, dict[float, dict[str, float]]],
    output: Path,
) -> None:
    """Plot absolute combined fitness against plant/fungal fitness by P level."""
    p_levels = sorted(records)
    colours = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(p_levels)))
    fig, axes = plt.subplots(3, 4, figsize=(14.0, 9.2), sharey=False, constrained_layout=True)
    axes = np.asarray(axes).reshape(3, 4)

    for index, (p_level, colour) in enumerate(zip(p_levels, colours, strict=True)):
        row = index // 2
        column_offset = (index % 2) * 2
        record = records[p_level]
        for column, (component, component_label) in enumerate(
            (("plant", "Plant fitness"), ("fungus", "Fungal fitness"))
        ):
            axis = axes[row, column_offset + column]
            x_values = record[component]
            y_values = record["combined"]
            axis.scatter(x_values, y_values, color=colour, alpha=0.30,
                         edgecolor="white", linewidth=0.35, s=38, zorder=2)
            x_median = float(np.median(x_values))
            y_median = float(np.median(y_values))
            x_q25, x_q75 = np.quantile(x_values, (0.25, 0.75))
            y_q25, y_q75 = np.quantile(y_values, (0.25, 0.75))
            axis.errorbar(
                x_median, y_median,
                xerr=np.asarray([[x_median - x_q25], [x_q75 - x_median]]),
                yerr=np.asarray([[y_median - y_q25], [y_q75 - y_median]]),
                fmt="o", color=colour, markeredgecolor="white", markeredgewidth=0.6,
                markersize=7, capsize=3.5, elinewidth=1.2, zorder=3,
            )
            for mode, marker in (("plant-only", "s"), ("mixed", "^")):
                if component == "fungus" and mode == "plant-only":
                    continue
                baseline = baselines[mode][p_level]
                axis.scatter(baseline[component], baseline["combined"], color=colour,
                             marker=marker, edgecolor="white", linewidth=0.55,
                             s=66, zorder=4)
            axis.set_title(f"P = {p_level:g} µM — {component_label}")
            axis.set_xlabel(component_label)
            axis.set_ylabel("Combined fitness")
            axis.grid(color="#dddddd", linewidth=0.6)
            axis.set_axisbelow(True)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    p_handles = [
        Patch(facecolor=colour, edgecolor="none", label=f"P = {p_level:g} µM")
        for p_level, colour in zip(p_levels, colours, strict=True)
    ]
    mode_handles = [
        Line2D([], [], color="#555555", marker="s", linestyle="None", markersize=7,
               label="Fixed plant-only"),
        Line2D([], [], color="#555555", marker="^", linestyle="None", markersize=7,
               label="Fixed mixed trade"),
    ]
    axes[0, 3].legend(handles=[*p_handles, *mode_handles], loc="best", frameon=True, fontsize=8)
    fig.suptitle(
        "Fitness trade-offs stratified by initial P level\n"
        "Absolute fitness; IPPO medians ± IQR; translucent points are individual seeds"
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_pareto_overlay(
    records: dict[float, dict[str, np.ndarray]],
    baselines: dict[str, dict[float, dict[str, float]]],
    output: Path,
) -> None:
    """Overlay plant and fungal fitness using separate x-axes per P panel."""
    p_levels = sorted(records)
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 11.0), sharey=False, constrained_layout=True)
    axes = np.asarray(axes).reshape(3, 2)
    plant_colour, fungus_colour = "#2878b5", "#d95f02"
    for axis, p_level in zip(axes.flat, p_levels, strict=True):
        record = records[p_level]
        fungus_axis = axis.twiny()
        for component, colour, target_axis in (
            ("plant", plant_colour, axis), ("fungus", fungus_colour, fungus_axis)
        ):
            x_values = record[component]
            y_values = record["combined"]
            target_axis.scatter(x_values, y_values, color=colour, alpha=0.8,
                                edgecolor="white", linewidth=0.35, s=38, zorder=2)
            for mode, marker in (("plant-only", "s"), ("mixed", "^")):
                if component == "fungus" and mode == "plant-only":
                    continue
                baseline = baselines[mode][p_level]
                target_axis.scatter(
                    baseline[component], baseline["combined"], color=colour, alpha=0.8,
                    marker=marker, edgecolor="white", linewidth=0.55, s=66, zorder=4,
                )
        axis.set_title(f"Initial solution P = {p_level:g} µM")
        axis.set_xlabel("Plant fitness", color=plant_colour)
        axis.set_ylabel("Combined fitness")
        axis.tick_params(axis="x", colors=plant_colour)
        fungus_axis.set_xlabel("Fungal fitness", color=fungus_colour)
        fungus_axis.tick_params(axis="x", colors=fungus_colour)
        axis.grid(color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)

    from matplotlib.lines import Line2D
    axes[0, 1].legend(
        handles=[
            Line2D([], [], color=plant_colour, marker="o", linestyle="None", markersize=7,
                   label="Plant fitness"),
            Line2D([], [], color=fungus_colour, marker="o", linestyle="None", markersize=7,
                   label="Fungal fitness"),
            Line2D([], [], color="#555555", marker="s", linestyle="None", markersize=7,
                   label="Fixed plant-only"),
            Line2D([], [], color="#555555", marker="^", linestyle="None", markersize=7,
                   label="Fixed mixed trade"),
        ], loc="best", frameon=True, fontsize=8,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_pareto_fixed_reference(
    records: dict[float, dict[str, np.ndarray]],
    baselines: dict[str, dict[float, dict[str, float]]],
    output: Path,
) -> None:
    """Plot plant/fungal fitness deviations from the fixed mixed baseline."""
    p_levels = sorted(records)
    colours = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(p_levels)))
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 11.0), sharey=False, constrained_layout=True)
    axes = np.asarray(axes).reshape(3, 2)
    for axis, p_level, colour in zip(axes.flat, p_levels, colours, strict=True):
        reference = baselines["mixed"][p_level]
        x_values = records[p_level]["fungus"] - reference["fungus"]
        y_values = records[p_level]["plant"] - reference["plant"]
        axis.scatter(x_values, y_values, color=colour, alpha=0.8,
                     edgecolor="white", linewidth=0.35, s=42, zorder=2)
        plant_only = baselines["plant-only"][p_level]
        axis.scatter(
            -reference["fungus"], plant_only["plant"] - reference["plant"],
            color="#555555", alpha=0.8, marker="s", edgecolor="white",
            linewidth=0.55, s=70, zorder=4,
        )
        axis.scatter(0.0, 0.0, color="#555555", alpha=0.8, marker="^",
                     edgecolor="white", linewidth=0.55, s=70, zorder=4)
        k = float(max(
            np.max(np.abs(x_values)), np.max(np.abs(y_values)),
            abs(plant_only["plant"] - reference["plant"]),
            abs(reference["fungus"]), 1e-12,
        )) * 1.12
        axis.set_xlim(-k, k)
        axis.set_ylim(-k, k)
        axis.set_aspect("equal", adjustable="box")
        axis.axhline(0.0, color="#888888", linewidth=0.8, zorder=1)
        axis.axvline(0.0, color="#888888", linewidth=0.8, zorder=1)
        axis.set_title(f"Initial solution P = {p_level:g} µM")
        axis.set_xlabel("Fungal fitness − fixed mixed")
        axis.set_ylabel("Plant fitness − fixed mixed")
        axis.grid(color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)

    from matplotlib.lines import Line2D
    axes[0, 1].legend(
        handles=[
            Line2D([], [], color="#555555", marker="o", linestyle="None", alpha=0.8,
                   markersize=7, label="IPPO seed"),
            Line2D([], [], color="#555555", marker="s", linestyle="None", alpha=0.8,
                   markersize=7, label="Fixed plant-only"),
            Line2D([], [], color="#555555", marker="^", linestyle="None", alpha=0.8,
                   markersize=7, label="Fixed mixed reference"),
        ], loc="best", frameon=True, fontsize=8,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)

_P_MAJOR_TICKS = (0.3, 0.75, 1.5, 3.0, 5.0, 10.0)
_BLUE = "#2878b5"
_ORANGE = "#d95f02"
_GREEN = "#1b9e77"


def load_records(study_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Load P-matched fixed controls and learned mixed IPPO endpoints."""
    baseline = json.loads(
        (study_dir / "fixed-allocation-baseline.json").read_text(encoding="utf-8")
    )
    controls: dict[str, dict[float, dict[str, float]]] = {
        "plant-only": {}, "mixed": {},
    }
    for entry in baseline["entries"]:
        mode = entry.get("mode")
        if mode not in controls or entry.get("status") != "completed":
            continue
        p = float(entry["initial_p_micromolar"])
        controls[mode][p] = {
            "biomass": float(entry["final_living_biomass"]["plant"]),
            "fitness": float(entry["cumulative_reproductive_fitness"]["plant"]),
        }

    learned: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: {"biomass": [], "fitness": []}
    )
    for path in sorted(study_dir.glob("mixed/p-*/seed-*/evaluation.json")):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        p = float(evaluation["initial_p_micromolar"])
        summary = evaluation["evaluation"]["episodes"][0]["summary"]
        learned[p]["biomass"].append(float(summary["final_living_biomass"]["plant"]))
        learned[p]["fitness"].append(
            float(summary["cumulative_reproductive_fitness"]["plant"])
        )

    if set(controls["plant-only"]) != set(learned) or set(controls["mixed"]) != set(learned):
        raise ValueError(
            "fixed controls and learned mixed P conditions do not match: "
            f"plant-only={sorted(controls['plant-only'])}, "
            f"mixed={sorted(controls['mixed'])}, learned={sorted(learned)}"
        )
    if not all(controls["plant-only"][p][metric] > 0
               for p in controls["plant-only"] for metric in controls["plant-only"][p]):
        raise ValueError("all plant-only endpoints must be positive for Delta_AM ratios")

    return {
        metric: {
            "p": sorted(controls["plant-only"]),
            "plant_only": [controls["plant-only"][p][metric]
                           for p in sorted(controls["plant-only"])],
            "fixed_mixed": [controls["mixed"][p][metric]
                            for p in sorted(controls["plant-only"])],
            "learned": [learned[p][metric] for p in sorted(controls["plant-only"])],
        }
        for metric in ("biomass", "fitness")
    }


def _format_p_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator(_P_MAJOR_TICKS))
    axis.xaxis.set_major_formatter(FixedFormatter([f"{p:g}" for p in _P_MAJOR_TICKS]))
    axis.tick_params(axis="x", which="minor", length=3)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)


def _plot_learned_medians(
    axis: plt.Axes, p: list[float], values: list[list[float]]
) -> None:
    """Plot learned-IPPO medians with asymmetric seed interquartile ranges."""
    quantiles = np.asarray([np.quantile(seed_values, [0.25, 0.5, 0.75])
                            for seed_values in values])
    lower, median, upper = quantiles.T
    axis.errorbar(
        p,
        median,
        yerr=np.vstack((median - lower, upper - median)),
        fmt="o",
        color=_BLUE,
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=7.5,
        capsize=3.5,
        elinewidth=1.15,
        zorder=3,
    )


def _plot_learned_seed_points(
    axis: plt.Axes, p: list[float], values: list[list[float]]
) -> None:
    """Plot the individual seed outcomes behind the summary statistic."""
    axis.scatter(
        np.repeat(p, [len(seed_values) for seed_values in values]),
        np.concatenate(values),
        color=_BLUE,
        edgecolor="white",
        linewidth=0.35,
        s=42,
        alpha=0.28,
        zorder=2,
    )


def plot_ippo(records: dict[str, dict[str, list[float]]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharex="col", constrained_layout=True)
    endpoints = (
        ("biomass", "Final living plant biomass (g)", "Plant biomass (g)"),
        ("fitness", "Cumulative reproductive fitness", "Plant fitness"),
    )
    for row, (metric, absolute_label, ratio_label) in enumerate(endpoints):
        p = records[metric]["p"]
        plant_only = records[metric]["plant_only"]
        fixed_mixed = records[metric]["fixed_mixed"]
        learned = records[metric]["learned"]
        ratios = [[value / baseline for value in seed_values]
                  for baseline, seed_values in zip(plant_only, learned, strict=True)]
        fixed_mixed_ratios = [value / baseline
                              for baseline, value in zip(plant_only, fixed_mixed, strict=True)]

        ratio_axis, absolute_axis = axes[row]
        ratio_axis.axhline(1.0, color="#555555", linewidth=0.9, zorder=1)
        _plot_learned_seed_points(ratio_axis, p, ratios)
        _plot_learned_medians(ratio_axis, p, ratios)
        ratio_axis.scatter(
            p, fixed_mixed_ratios, color=_GREEN, marker="^", edgecolor="white",
            linewidth=0.55, s=70, label="Fixed mixed trade control", zorder=4,
        )
        ratio_axis.set_ylabel(rf"{ratio_label}\n$\Delta_{{AM}}$ (mixed / plant-only)")
        ratio_axis.set_title(
            r"$\Delta_{AM}$ ratio (IPPO median + IQR)" if row == 0 else "",
            fontsize=14,
        )

        _plot_learned_seed_points(absolute_axis, p, learned)
        _plot_learned_medians(absolute_axis, p, learned)
        absolute_axis.scatter(
            p, plant_only,
            color=_ORANGE,
            marker="s",
            edgecolor="white",
            linewidth=0.55,
            s=62,
            label="Fixed plant-only control",
            zorder=4,
        )
        absolute_axis.scatter(
            p, fixed_mixed, color=_GREEN, marker="^", edgecolor="white",
            linewidth=0.55, s=70, label="Fixed mixed trade control", zorder=4,
        )
        absolute_axis.set_ylabel(absolute_label)
        absolute_axis.set_title(
            "Absolute (IPPO seeds + median/IQR; control)"
            if row == 0 else "",
            fontsize=14,
        )

        for axis in (ratio_axis, absolute_axis):
            _format_p_axis(axis)

    for axis in axes[-1]:
        axis.set_xlabel("Initial solution P (µM)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(
        [Line2D([], [], color=_BLUE, marker="o", linestyle="-", markersize=6), *handles],
        ["IPPO seeds; median/IQR", *labels],
        loc="upper left", frameon=True, fontsize=9,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main_ippo() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path, nargs="?", default=Path("outputs/trade-only-ippo"))
    parser.add_argument(
        "output", type=Path, nargs="?",
        default=Path("outputs/trade-only-ippo/figures/trade-only-ippo-plant-performance.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(load_records(args.study_dir), args.output)
    print(args.output)

def load_episode_rates(study_dir: Path) -> dict[float, dict[str, np.ndarray]]:
    rates: dict[float, dict[str, list[float]]] = {}
    for path in sorted(study_dir.glob("mixed/p-*/seed-*/evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        p_level = float(payload["initial_p_micromolar"])
        trace = payload["evaluation"]["episodes"][0]["trace"]
        record = rates.setdefault(p_level, {"plant_to_fungus": [], "fungus_to_plant": []})
        record["plant_to_fungus"].append(float(np.mean([step["actions"]["plant"][0] for step in trace])))
        record["fungus_to_plant"].append(float(np.mean([step["actions"]["fungus"][0] for step in trace])))
    return {p: {key: np.asarray(value) for key, value in record.items()} for p, record in rates.items()}


def load_baseline_rates(study_dir: Path) -> dict[str, dict[float, dict[str, float]]]:
    payload = json.loads((study_dir / "fixed-allocation-baseline.json").read_text(encoding="utf-8"))
    result = {"plant-only": {}, "mixed": {}}
    for entry in payload["entries"]:
        mode = entry.get("mode")
        if mode not in result or entry.get("status") != "completed":
            continue
        p_level = float(entry["initial_p_micromolar"])
        result[mode][p_level] = {
            "plant_to_fungus": float(entry["commanded_rate_actions"]["plant"][0]),
            "fungus_to_plant": float(entry["commanded_rate_actions"]["fungus"][0]),
        }
    return result


def plot_trade_fitness_legacy(records, baselines, rates, direction: str, output: Path) -> None:
    p_levels = sorted(records)
    colours = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(p_levels)))
    fig, axes = plt.subplots(3, 4, figsize=(14.0, 9.2), sharey=False, constrained_layout=True)
    axes = np.asarray(axes).reshape(3, 4)
    for index, (p_level, colour) in enumerate(zip(p_levels, colours, strict=True)):
        row, offset = divmod(index, 2)
        col0 = offset * 2
        for col, component, label in ((col0, "plant", "Plant fitness"), (col0 + 1, "fungus", "Fungal fitness")):
            axis = axes[row, col]
            x = records[p_level][component]
            y = rates[p_level][direction]
            axis.scatter(x, y, color=colour, alpha=0.30, edgecolor="white", linewidth=0.35, s=38, zorder=2)
            xm, ym = float(np.median(x)), float(np.median(y))
            x25, x75 = np.quantile(x, (0.25, 0.75)); y25, y75 = np.quantile(y, (0.25, 0.75))
            axis.errorbar(xm, ym, xerr=[[xm-x25], [x75-xm]], yerr=[[ym-y25], [y75-ym]],
                          fmt="o", color=colour, markeredgecolor="white", markeredgewidth=0.6,
                          markersize=7, capsize=3.5, elinewidth=1.2, zorder=3)
            for mode, marker in (("plant-only", "s"), ("mixed", "^")):
                if component == "fungus" and mode == "plant-only":
                    continue
                baseline = baselines[mode][p_level]
                axis.scatter(baseline["fitness_" + component], baseline[direction],
                             color=colour, marker=marker,
                             edgecolor="white", linewidth=0.55, s=66, zorder=4)
            axis.set_title(f"P = {p_level:g} µM — {label}")
            axis.set_xlabel(label)
            axis.set_ylabel("Mean commanded trade rate (day⁻¹)")
            axis.grid(color="#dddddd", linewidth=0.6); axis.set_axisbelow(True)
    # Baseline fitness values are attached by main after loading.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="none", label=f"P = {p:g} µM") for p, c in zip(p_levels, colours, strict=True)]
    handles += [Line2D([], [], color="#555", marker="s", linestyle="None", markersize=7, label="Fixed plant-only"),
                Line2D([], [], color="#555", marker="^", linestyle="None", markersize=7, label="Fixed mixed trade")]
    axes[0, 3].legend(handles=handles, loc="best", fontsize=8)
    direction_label = "Fungal → plant" if direction == "fungus_to_plant" else "Plant → fungal"
    fig.suptitle(f"{direction_label} trade rate versus fitness, stratified by initial P\nAbsolute fitness; IPPO medians ± IQR; translucent points are individual seeds")
    fig.savefig(output, dpi=220); plt.close(fig)


def plot_overlay(records, baselines, rates, direction: str, output: Path) -> None:
    """Plot both fitness components against one trade-rate axis per P panel."""
    p_levels = sorted(records)
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 11.0), sharey=False, constrained_layout=True)
    axes = np.asarray(axes).reshape(3, 2)
    plant_colour, fungus_colour = "#2878b5", "#d95f02"
    for axis, p_level in zip(axes.flat, p_levels, strict=True):
        record = records[p_level]
        fungus_axis = axis.twiny()
        for component, colour, target_axis in (
            ("plant", plant_colour, axis), ("fungus", fungus_colour, fungus_axis)
        ):
            x_values = record[component]
            y_values = rates[p_level][direction]
            target_axis.scatter(x_values, y_values, color=colour, alpha=0.8,
                                edgecolor="white", linewidth=0.35, s=38, zorder=2)
            for mode, marker in (("plant-only", "s"), ("mixed", "^")):
                if component == "fungus" and mode == "plant-only":
                    continue
                baseline = baselines[mode][p_level]
                target_axis.scatter(
                    baseline["fitness_" + component], baseline[direction], color=colour, alpha=0.8,
                    marker=marker, edgecolor="white", linewidth=0.55, s=66, zorder=4,
                )
        axis.set_title(f"Initial solution P = {p_level:g} µM")
        axis.set_xlabel("Plant fitness", color=plant_colour)
        direction_label = "Fungal → plant" if direction == "fungus_to_plant" else "Plant → fungus"
        axis.set_ylabel(f"{direction_label}\nmean commanded trade rate (day⁻¹)")
        axis.tick_params(axis="x", colors=plant_colour)
        fungus_axis.set_xlabel("Fungal fitness", color=fungus_colour)
        fungus_axis.tick_params(axis="x", colors=fungus_colour)
        axis.grid(color="#dddddd", linewidth=0.6); axis.set_axisbelow(True)

    from matplotlib.lines import Line2D
    axes[0, 1].legend(
        handles=[
            Line2D([], [], color=plant_colour, marker="o", linestyle="None", markersize=7,
                   label="Plant fitness"),
            Line2D([], [], color=fungus_colour, marker="o", linestyle="None", markersize=7,
                   label="Fungal fitness"),
            Line2D([], [], color="#555555", marker="s", linestyle="None", markersize=7,
                   label="Fixed plant-only"),
            Line2D([], [], color="#555555", marker="^", linestyle="None", markersize=7,
                   label="Fixed mixed trade"),
        ], loc="best", frameon=True, fontsize=8,
    )
    fig.savefig(output, dpi=220); plt.close(fig)


def plot_fixed_reference_overlay(records, baselines, rates, direction: str, output: Path) -> None:
    """Plot rate and fitness differences relative to the fixed mixed control."""
    p_levels = sorted(records)
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 11.0), sharey=False, constrained_layout=True)
    axes = np.asarray(axes).reshape(3, 2)
    plant_colour, fungus_colour = "#2878b5", "#d95f02"
    rate_label = "Fungal → plant" if direction == "fungus_to_plant" else "Plant → fungus"
    for axis, p_level in zip(axes.flat, p_levels, strict=True):
        reference = baselines["mixed"][p_level]
        reference_rate = reference[direction]
        fungus_axis = axis.twiny()
        for component, colour, target_axis in (("plant", plant_colour, axis), ("fungus", fungus_colour, fungus_axis)):
            x_values = records[p_level][component] - reference["fitness_" + component]
            y_values = rates[p_level][direction] - reference_rate
            target_axis.scatter(x_values, y_values, color=colour, alpha=0.8,
                                edgecolor="white", linewidth=0.35, s=38, zorder=2)
            for mode, marker in (("plant-only", "s"), ("mixed", "^")):
                if component == "fungus" and mode == "plant-only":
                    continue
                baseline = baselines[mode][p_level]
                target_axis.scatter(
                    baseline["fitness_" + component] - reference["fitness_" + component],
                    baseline[direction] - reference_rate,
                    color=colour, alpha=0.8, marker=marker,
                    edgecolor="white", linewidth=0.55, s=66, zorder=4,
                )
        axis.set_title(f"Initial solution P = {p_level:g} µM")
        axis.set_xlabel("Plant fitness − fixed mixed", color=plant_colour)
        axis.set_ylabel(f"{rate_label}\nmean commanded trade rate − fixed mixed (day⁻¹)")
        axis.tick_params(axis="x", colors=plant_colour)
        fungus_axis.set_xlabel("Fungal fitness − fixed mixed", color=fungus_colour)
        fungus_axis.tick_params(axis="x", colors=fungus_colour)
        axis.axhline(0.0, color="#888888", linewidth=0.8, zorder=1)
        axis.axvline(0.0, color="#888888", linewidth=0.8, zorder=1)
        axis.grid(color="#dddddd", linewidth=0.6); axis.set_axisbelow(True)

    from matplotlib.lines import Line2D
    axes[0, 1].legend(handles=[
        Line2D([], [], color=plant_colour, marker="o", linestyle="None", markersize=7, label="Plant fitness"),
        Line2D([], [], color=fungus_colour, marker="o", linestyle="None", markersize=7, label="Fungal fitness"),
        Line2D([], [], color="#555555", marker="s", linestyle="None", markersize=7, label="Fixed plant-only"),
        Line2D([], [], color="#555555", marker="^", linestyle="None", markersize=7, label="Fixed mixed reference"),
    ], loc="best", frameon=True, fontsize=8)
    fig.savefig(output, dpi=220); plt.close(fig)


def main_trade_fitness() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study_dir", type=Path, nargs="?", default=Path("outputs/trade-only-ippo"))
    args = parser.parse_args()
    output_dir = args.study_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    records, baselines = load_fitness(args.study_dir)
    rates = load_episode_rates(args.study_dir)
    baseline_rates = load_baseline_rates(args.study_dir)
    # Add baseline fitness coordinates to the same structure used for plotting.
    for mode in baselines:
        for p_level, values in baselines[mode].items():
            values.update({f"fitness_{metric}": values[metric] for metric in ("plant", "fungus")})
            values.pop("plant", None); values.pop("fungus", None); values.pop("combined", None)
            values.update(baseline_rates[mode][p_level])
    for direction, name in (("fungus_to_plant", "fungal-to-plant"), ("plant_to_fungus", "plant-to-fungus")):
        plot_overlay(records, baselines, rates, direction,
                     output_dir / f"trade-only-ippo-{name}-rate-vs-fitness-overlay.png")

_PLANT_COLOUR = "#2878b5"
_FUNGUS_COLOUR = "#d95f02"


def load_trade_rates(study_dir: Path) -> dict[float, dict[str, np.ndarray]]:
    """Load daily commanded trade-rate trajectories, grouped by P condition."""
    trajectories: dict[float, list[dict[str, np.ndarray]]] = defaultdict(list)
    for path in sorted(study_dir.glob("mixed/p-*/seed-*/evaluation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        trace = record["evaluation"]["episodes"][0]["trace"]
        trajectories[float(record["initial_p_micromolar"])].append({
            "seed": record["seed"],
            "time_days": np.asarray([step["time_days"] for step in trace]),
            "plant": np.asarray([step["actions"]["plant"][0] for step in trace]),
            "fungus": np.asarray([step["actions"]["fungus"][0] for step in trace]),
            "plant_c_export": np.asarray([
                step["transfers"]["plant"]["out"] for step in trace
            ]),
            "fungus_p_export": np.asarray([
                step["transfers"]["fungus"]["out"] for step in trace
            ]),
        })

    if not trajectories:
        raise ValueError(f"no learned-IPPO evaluations found under {study_dir}")
    result = {}
    for p_level, rows in trajectories.items():
        time_days = rows[0]["time_days"]
        if any(not np.array_equal(row["time_days"], time_days) for row in rows[1:]):
            raise ValueError(f"evaluation time grids do not match at {p_level:g} µM P")
        result[p_level] = {
            "seeds": np.asarray([row["seed"] for row in rows]),
            "time_days": time_days,
            "plant": np.asarray([row["plant"] for row in rows]),
            "fungus": np.asarray([row["fungus"] for row in rows]),
            "plant_c_export": np.asarray([row["plant_c_export"] for row in rows]),
            "fungus_p_export": np.asarray([row["fungus_p_export"] for row in rows]),
        }
    return result


def _plot_rate(
    axis: plt.Axes, time_days: np.ndarray, values: np.ndarray, colour: str, label: str
) -> None:
    mean = np.mean(values, axis=0)
    lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
    axis.fill_between(time_days, lower, upper, color=colour, alpha=0.18, linewidth=0)
    axis.plot(
        time_days,
        mean,
        color=colour,
        marker="o",
        markersize=2.0,
        linewidth=1.05,
        label=label,
        zorder=3,
    )


def plot_trade_rates(records: dict[float, dict[str, np.ndarray]], output: Path) -> None:
    p_levels = sorted(records)
    if len(p_levels) != 6:
        raise ValueError(f"expected six P conditions, found {len(p_levels)}")
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for axis, p_level in zip(axes.flat, p_levels, strict=True):
        record = records[p_level]
        _plot_rate(axis, record["time_days"], record["plant"], _PLANT_COLOUR,
                   "Plant C-trade rate")
        _plot_rate(axis, record["time_days"], record["fungus"], _FUNGUS_COLOUR,
                   "Fungal P-trade rate")
        axis.set_title(f"Initial solution P = {p_level:g} µM")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_xlim(record["time_days"][0], record["time_days"][-1])

    for axis in axes[:, 0]:
        axis.set_ylabel("Commanded trade rate (day⁻¹)")
    for axis in axes[-1]:
        axis.set_xlabel("Time (days; one point per day)")
    axes[0, 0].legend(
        loc="center", frameon=False,
        title="Line: seed mean; band: seed IQR",
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_realised_transfers(records: dict[float, dict[str, np.ndarray]], output: Path) -> None:
    """Plot mean/IQR daily realised exports, retaining separate C and P axes."""
    p_levels = sorted(records)
    if len(p_levels) != 6:
        raise ValueError(f"expected six P conditions, found {len(p_levels)}")
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), sharex=True,
                             constrained_layout=True)
    fungal_axes = []
    for index, (axis, p_level) in enumerate(zip(axes.flat, p_levels, strict=True)):
        _, column = divmod(index, 2)
        record = records[p_level]
        fungal_axis = axis.twinx()
        fungal_axes.append(fungal_axis)
        _plot_rate(axis, record["time_days"], record["plant_c_export"], _PLANT_COLOUR,
                   "Plant C exported")
        _plot_rate(fungal_axis, record["time_days"], record["fungus_p_export"],
                   _FUNGUS_COLOUR, "Fungal P exported")
        axis.set_title(f"Initial solution P = {p_level:g} µM")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_xlim(record["time_days"][0], record["time_days"][-1])
        axis.tick_params(axis="y", colors=_PLANT_COLOUR)
        fungal_axis.tick_params(axis="y", colors=_FUNGUS_COLOUR)
        if column == 0:
            axis.set_ylabel("Plant C exported\n(per daily interval)", color=_PLANT_COLOUR)
        if column == 1:
            fungal_axis.set_ylabel("Fungal P exported\n(per daily interval)",
                                   color=_FUNGUS_COLOUR)

    for axis in axes[-1]:
        axis.set_xlabel("Time (days; one point per day)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fungal_handles, fungal_labels = fungal_axes[0].get_legend_handles_labels()
    axes[0, 0].legend(
        handles + fungal_handles, labels + fungal_labels, loc="lower right", frameon=False,
        title="Line: seed mean; band: seed IQR",
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_seed_episodes(records: dict[float, dict[str, np.ndarray]], output_dir: Path) -> list[Path]:
    """Write one two-panel, all-seed evaluation-episode figure per P level."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colours = plt.get_cmap("tab10").colors
    for p_level in sorted(records):
        record = records[p_level]
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True)
        for index, seed in enumerate(record["seeds"]):
            colour = colours[index]
            for axis, agent, label in (
                (axes[0], "plant", "Plant C-trade rate"),
                (axes[1], "fungus", "Fungal P-trade rate"),
            ):
                axis.plot(
                    record["time_days"], record[agent][index], color=colour,
                    marker="o", markersize=2.0, linewidth=1.0, label=f"Seed {seed}",
                )
                axis.set_ylabel(f"{label}\n(day⁻¹)")
                axis.grid(axis="y", color="#dddddd", linewidth=0.6)
                axis.set_xlim(record["time_days"][0], record["time_days"][-1])
        axes[1].set_xlabel("Time (days; one point per day)")
        fig.suptitle(f"Individual evaluation episodes: initial solution P = {p_level:g} µM")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                   ncols=len(handles), frameon=False, title="IPPO seed")
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        path = output_dir / f"trade-only-ippo-trade-rates-p-{p_level:g}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_seed_transfers(records: dict[float, dict[str, np.ndarray]], output_dir: Path) -> list[Path]:
    """Write one two-panel, seed-resolved realised-transfer figure per P level."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colours = plt.get_cmap("tab10").colors
    for p_level in sorted(records):
        record = records[p_level]
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True)
        for index, seed in enumerate(record["seeds"]):
            colour = colours[index]
            for axis, field, label in (
                (axes[0], "plant_c_export", "Plant C exported\n(per daily interval)"),
                (axes[1], "fungus_p_export", "Fungal P exported\n(per daily interval)"),
            ):
                axis.plot(
                    record["time_days"], record[field][index], color=colour,
                    marker="o", markersize=2.0, linewidth=1.0, label=f"Seed {seed}",
                )
                axis.set_ylabel(label)
                axis.grid(axis="y", color="#dddddd", linewidth=0.6)
                axis.set_xlim(record["time_days"][0], record["time_days"][-1])
        axes[1].set_xlabel("Time (days; one point per day)")
        fig.suptitle(
            f"Individual realised transfers: initial solution P = {p_level:g} µM"
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                   ncols=len(handles), frameon=False, title="IPPO seed")
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        path = output_dir / f"trade-only-ippo-realised-transfers-p-{p_level:g}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def main_trade_rates() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path, nargs="?", default=Path("outputs/trade-only-ippo"))
    parser.add_argument(
        "output", type=Path, nargs="?",
        default=Path("outputs/trade-only-ippo/figures/trade-only-ippo-trade-rates.png"),
    )
    parser.add_argument(
        "--seed-output-dir", type=Path,
        help="write six per-P figures with individual seed evaluation episodes",
    )
    parser.add_argument(
        "--transfer-output", type=Path,
        help="write the six-P mean/IQR figure for realised plant-C and fungal-P exports",
    )
    parser.add_argument(
        "--transfer-seed-output-dir", type=Path,
        help="write six per-P figures with individual realised-transfer episodes",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = load_trade_rates(args.study_dir)
    plot(records, args.output)
    print(args.output)
    if args.seed_output_dir:
        for path in plot_seed_episodes(records, args.seed_output_dir):
            print(path)
    if args.transfer_output:
        args.transfer_output.parent.mkdir(parents=True, exist_ok=True)
        plot_realised_transfers(records, args.transfer_output)
        print(args.transfer_output)
    if args.transfer_seed_output_dir:
        for path in plot_seed_transfers(records, args.transfer_seed_output_dir):
            print(path)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path, nargs="?", default=Path("outputs/trade-only-ippo"))
    args = parser.parse_args()
    output_dir = args.study_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ippo(load_records(args.study_dir), output_dir / "trade-only-ippo-plant-performance.png")
    trade_records = load_trade_rates(args.study_dir)
    plot_trade_rates(trade_records, output_dir / "trade-only-ippo-trade-rates.png")
    plot_realised_transfers(trade_records, output_dir / "trade-only-ippo-realised-transfers.png")

    fitness_records, baselines = load_fitness(args.study_dir)
    plot_pareto_overlay(fitness_records, baselines, output_dir / "trade-only-ippo-fitness-pareto-overlay.png")
    plot_pareto_fixed_reference(fitness_records, baselines, output_dir / "trade-only-ippo-fitness-pareto-fixed-trade-reference.png")

    rates = load_episode_rates(args.study_dir)
    baseline_rates = load_baseline_rates(args.study_dir)
    for mode in baselines:
        for p_level, values in baselines[mode].items():
            values.update({f"fitness_{metric}": values[metric] for metric in ("plant", "fungus")})
            values.pop("plant", None); values.pop("fungus", None); values.pop("combined", None)
            values.update(baseline_rates[mode][p_level])
    for direction, name in (("fungus_to_plant", "fungal-to-plant"), ("plant_to_fungus", "plant-to-fungus")):
        plot_overlay(rates=rates, records=fitness_records, baselines=baselines, direction=direction,
                     output=output_dir / f"trade-only-ippo-{name}-rate-vs-fitness-overlay.png")

if __name__ == "__main__":
    main()

