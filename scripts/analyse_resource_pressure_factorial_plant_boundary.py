"""Analyse issue #48's observed factorial plant P/C-limitation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mycormarl.resource_pressure_analysis import factorial_plant_boundary_rows


def load_entries(checkpoint_directory: Path) -> list[dict]:
    """Load the condition-level checkpoint records retained by the experiment."""
    entries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(checkpoint_directory.glob("condition-*.json"))
    ]
    if not entries:
        raise FileNotFoundError(f"no condition checkpoints found in {checkpoint_directory}")
    return entries


def _axis_levels(rows: list[dict]) -> tuple[list[float], list[float]]:
    return (
        sorted({float(row["plant_kappa_c"]) for row in rows}),
        sorted({float(row["plant_trade"]) for row in rows}),
    )


def plot_p_response_curves(rows: list[dict], output: Path) -> None:
    """Plot raw observed P-response curves for each plant-κC × trade cell."""
    kappas, trades = _axis_levels(rows)
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True, constrained_layout=True)
    colour_map = plt.get_cmap("viridis", len(trades))
    by_kappa = {kappa: [row for row in rows if row["plant_kappa_c"] == kappa] for kappa in kappas}
    for axis, kappa in zip(axes.ravel(), kappas):
        for colour_index, trade in enumerate(trades):
            row = next(item for item in by_kappa[kappa] if item["plant_trade"] == trade)
            points = row["p_response"]
            axis.plot(
                [point["initial_solution_p_micromolar"] for point in points],
                [point["plant_p_limited_fraction"] for point in points],
                marker="o", markersize=3.5, linewidth=1.1,
                color=colour_map(colour_index), label=f"{trade:.02f}",
            )
        axis.axhline(.5, color="#333333", linestyle="--", linewidth=.8)
        axis.set_title(f"Plant κC multiplier {kappa:g}", fontsize=9)
        axis.set(xlim=(.48, 1.32), ylim=(-.03, 1.03), xlabel="Initial solution P (µM)", ylabel="Plant P-limited fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, title="Plant→fungus trade", ncol=9, loc="lower center", bbox_to_anchor=(.5, -.02))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_boundary_heatmap(rows: list[dict], output: Path) -> None:
    """Plot observed/interpolated thresholds; mark censored cells explicitly."""
    kappas, trades = _axis_levels(rows)
    matrix = np.full((len(trades), len(kappas)), np.nan)
    statuses: dict[tuple[int, int], str] = {}
    for row in rows:
        column = kappas.index(row["plant_kappa_c"])
        line = trades.index(row["plant_trade"])
        if row["threshold_initial_p_micromolar"] is not None:
            matrix[line, column] = row["threshold_initial_p_micromolar"]
        statuses[(line, column)] = row["threshold_status"]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d0d0d0")
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=.5, vmax=1.3)
    for (line, column), status in statuses.items():
        if status == "lower-censored":
            axis.text(column, line, "<0.5", ha="center", va="center", fontsize=7)
        elif status == "upper-censored":
            axis.text(column, line, ">1.3", ha="center", va="center", fontsize=7, color="white")
        elif status not in {"observed-crossing", "observed-level"}:
            axis.text(column, line, "?", ha="center", va="center", fontsize=8)
    axis.set(
        xticks=range(len(kappas)), xticklabels=[f"{value:g}" for value in kappas],
        yticks=range(len(trades)), yticklabels=[f"{value:.02f}" for value in trades],
        xlabel="Plant κC multiplier", ylabel="Plant→fungus trade fraction",
        title="Observed plant P/C transition initial-P boundary (µM)",
    )
    figure.colorbar(image, ax=axis, label="Interpolated initial solution P (µM)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _normalised_biomass_difference(plant: np.ndarray, fungus: np.ndarray) -> np.ndarray:
    """Return min–max-scaled plant biomass minus fungal biomass.

    Biomass is right-skewed in this experiment, so each organism is first
    normalised by its median within the P stratum.  The resulting surfaces are
    then independently min–max scaled to [0, 1], making their difference a
    dimensionless contrast in [-1, 1].
    """
    def scale(matrix: np.ndarray) -> np.ndarray:
        valid = np.isfinite(matrix)
        centre = float(np.median(matrix[valid]))
        relative = matrix / centre if centre > 0 else matrix.copy()
        low, high = np.nanmin(relative), np.nanmax(relative)
        if high == low:
            return np.zeros_like(relative)
        return (relative - low) / (high - low)

    return scale(plant) - scale(fungus)


def plot_biomass_heatmaps_by_initial_p(
    entries: list[dict], output: Path, *, include_difference: bool = False
) -> None:
    """Show biomass and an optional normalised plant-minus-fungus contrast.

    Set ``include_difference=True`` to add the third contrast column.
    """
    p_levels = sorted({float(entry["initial_p_micromolar"]) for entry in entries})
    kappas = sorted({float(entry["factors"]["plant_kappa_c"]) for entry in entries})
    trades = sorted({float(entry["factors"]["plant_trade"]) for entry in entries})
    panel_count = 3 if include_difference else 2
    figure, axes = plt.subplots(
        len(p_levels), panel_count, figsize=(6 * panel_count, 3.2 * len(p_levels)),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    panels = (
        ("plant", "Plant final biomass (g)", "YlGn"),
        ("fungus", "Fungal final biomass (g)", "YlOrBr"),
    )
    for p_row_index, p_level in enumerate(p_levels):
        selected = [entry for entry in entries if float(entry["initial_p_micromolar"]) == p_level]
        matrices: dict[str, np.ndarray] = {}
        for organism, _label, _cmap in panels:
            matrices[organism] = np.full((len(trades), len(kappas)), np.nan)
        for entry in selected:
            factors = entry["factors"]
            trade_row_index = trades.index(float(factors["plant_trade"]))
            column_index = kappas.index(float(factors["plant_kappa_c"]))
            for organism in matrices:
                matrices[organism][trade_row_index, column_index] = float(entry["biomass"][organism])
        panels_for_row = list(panels)
        if include_difference:
            panels_for_row.append(("difference", "Normalised plant − fungus biomass", "RdBu_r"))
        for column_index, (organism, label, cmap) in enumerate(panels_for_row):
            matrix = (
                _normalised_biomass_difference(matrices["plant"], matrices["fungus"])
                if organism == "difference"
                else matrices[organism]
            )
            axis = axes[p_row_index, column_index]
            image_kwargs = {"vmin": -1, "vmax": 1} if organism == "difference" else {}
            image = axis.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, **image_kwargs)
            axis.set_title(f"{label}, initial P = {p_level:g} µM")
            axis.set(
                xticks=range(len(kappas)), xticklabels=[f"{value:g}" for value in kappas],
                yticks=range(len(trades)), yticklabels=[f"{value:.02f}" for value in trades],
                xlabel="Plant κC multiplier", ylabel="Plant→fungus trade fraction",
            )
            figure.colorbar(image, ax=axis, label=label)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--with-biomass-difference", action="store_true",
        help="add the normalised plant-minus-fungus biomass heatmap",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args.checkpoints)
    rows = factorial_plant_boundary_rows(entries)
    (args.output_directory / "resource-pressure-factorial-boundary-summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    plot_p_response_curves(rows, args.output_directory / "resource-pressure-factorial-p-response-curves.png")
    plot_boundary_heatmap(rows, args.output_directory / "resource-pressure-factorial-boundary-heatmap.png")
    plot_biomass_heatmaps_by_initial_p(
        entries,
        args.output_directory / "resource-pressure-factorial-biomass-heatmaps-by-initial-p.png",
        include_difference=args.with_biomass_difference,
    )


if __name__ == "__main__":
    main()
