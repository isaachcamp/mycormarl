"""Generate the fixed-reservoir time-dependent depletion-gradient diagnostic."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mycormarl-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "svg.fonttype": "none",
    }
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from mycormarl.params import EnvConfig
from mycormarl.soil.depletion_gradient_diagnostic import (
    run_native_geometry_closure_comparisons,
    run_time_dependent_depletion_gradient_diagnostic,
)


_DENSITY_COLOURS = ("#0072B2", "#E69F00", "#009E73")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_cumulative_uptake(
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    output_dir: Path,
    reference_time_days: float,
) -> None:
    """Render cumulative blended uptake from diagnostic rows only."""
    radii = sorted({float(row["absorber_radius_cm"]) for row in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(radii), figsize=(7.2, 3.2), sharey=True)
    if len(radii) == 1:
        axes = [axes]
    default_geometries = {(1e-2, 1.0), (5e-4, 2_000.0)}
    for axis, radius, panel_name in zip(
        axes, radii, ("Plant-scale absorber", "Fungus-scale absorber")
    ):
        panel_rows = [row for row in rows if row["absorber_radius_cm"] == radius]
        densities = sorted({float(row["length_density_cm_cm3"]) for row in panel_rows})
        for colour, density in zip(_DENSITY_COLOURS, densities):
            is_default_geometry = (radius, density) in default_geometries
            trajectory = [
                row
                for row in panel_rows
                if row["length_density_cm_cm3"] == density
            ]
            axis.plot(
                [row["time_days"] for row in trajectory],
                [row["cumulative_uptake_micromol"] for row in trajectory],
                color=colour,
                linewidth=1.7,
                alpha=1.0 if is_default_geometry else 0.4,
                label=f"Blended, λ={density:g} cm cm⁻³",
            )
            marker = next(
                (row for row in trajectory if row["is_diffusion_overlap_time"]), None
            )
            if marker is not None:
                axis.scatter(
                    marker["time_days"],
                    marker["cumulative_uptake_micromol"],
                    color=colour,
                    alpha=1.0 if is_default_geometry else 0.4,
                    edgecolor="white",
                    linewidth=0.6,
                    s=24,
                    zorder=4,
                )
        axis.axvline(
            reference_time_days,
            color="#666666",
            linestyle=":",
            linewidth=1.1,
        )
        axis.set_title(panel_name)
        axis.set_xlabel("Experiment time (day)")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    fig.supylabel(
        "Cumulative P uptake by represented cell (µmol P)",
        x=0.02,
        y=0.5,
        fontsize=8,
        va="center",
    )
    densities = sorted({float(row["length_density_cm_cm3"]) for row in rows})
    handles = [
        Line2D(
            [0],
            [0],
            color=colour,
            linewidth=1.7,
            alpha=1.0,
            label=f"λ={density:g} cm cm⁻³",
        )
        for colour, density in zip(_DENSITY_COLOURS, densities)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.25, top=0.88, wspace=0.06)
    fig.savefig(
        output_dir / "depletion_gradient_cumulative_uptake.svg",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        output_dir / "depletion_gradient_cumulative_uptake.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def _plot_native_geometry_closure_comparison(
    rows: list[dict[str, object]], output_dir: Path
) -> None:
    """Render native-geometry closure comparisons from exported trajectories."""
    panels = (
        ("closure_limits", "Sparse vs continuous", {
            "sparse_only": ("#D55E00", "Sparse"),
            "continuous_only": ("#0072B2", "Continuous"),
        }),
        ("blend_time_reference", "Fixed T_ref vs t_sim", {
            "fixed_t_ref": ("#CC79A7", "Fixed T_ref"),
            "simulation_time": ("#009E73", "t_sim"),
        }),
    )
    styles = {
        "plant_default": "-",
        "fungus_default": "--",
        "transition_scale": "-.",
    }
    panel_geometries = {
        "closure_limits": ("plant_default", "fungus_default"),
        "blend_time_reference": tuple(styles),
    }
    reference_time_days = float(rows[0]["reference_time_days"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for axis, (panel, title, treatments) in zip(axes, panels):
        for treatment, (colour, label) in treatments.items():
            for organism in panel_geometries[panel]:
                trajectory = [
                    row
                    for row in rows
                    if row["comparison_panel"] == panel
                    and row["treatment"] == treatment
                    and row["organism_geometry"] == organism
                ]
                axis.plot(
                    [row["time_days"] for row in trajectory],
                    [row["cumulative_uptake_micromol"] for row in trajectory],
                    color=colour,
                    linestyle=styles[organism],
                    linewidth=1.8,
                )
                if panel == "blend_time_reference" and treatment == "simulation_time":
                    marker = next(
                        (
                            row
                            for row in trajectory
                            if row["is_diffusion_overlap_time"]
                        ),
                        None,
                    )
                    if marker is not None:
                        axis.scatter(
                            marker["time_days"],
                            marker["cumulative_uptake_micromol"],
                            color=colour,
                            edgecolor="white",
                            linewidth=0.6,
                            s=24,
                            zorder=4,
                        )
        axis.set_title(title)
        axis.set_xlabel("Experiment time (day)")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.axvline(
            reference_time_days,
            color="#666666",
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
        )
        axis.legend(
            handles=[
                Line2D([0], [0], color=colour, linewidth=1.8, label=label)
                for _treatment, (colour, label) in treatments.items()
            ],
            loc="upper left",
            frameon=False,
        )
    axes[0].set_ylabel("Cumulative P uptake by represented cell (µmol P)")
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.18, top=0.88, wspace=0.06)
    fig.savefig(
        output_dir / "native_geometry_closure_comparison.svg",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        output_dir / "native_geometry_closure_comparison.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def write_diagnostic_artifacts(
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    output_dir: Path,
    reference_time_days: float,
) -> None:
    """Write tables and figures derived exclusively from runner output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "depletion_gradient_time_series.csv", rows)
    _write_csv(output_dir / "depletion_gradient_summary.csv", summaries)
    _plot_cumulative_uptake(rows, summaries, output_dir, reference_time_days)


def write_native_geometry_comparison_artifacts(
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Write rows and the native-geometry comparison figure."""
    _write_csv(output_dir / "native_geometry_closure_comparison.csv", rows)
    _write_csv(
        output_dir / "native_geometry_closure_comparison_summary.csv", summaries
    )
    _plot_native_geometry_closure_comparison(rows, output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-days", type=float, default=30.0)
    parser.add_argument("--sample-count", type=int, default=301)
    args = parser.parse_args(argv)
    if args.duration_days <= 0.0:
        parser.error("--duration-days must be greater than zero")
    if args.sample_count < 2:
        parser.error("--sample-count must be at least two")

    config = EnvConfig()
    times = np.linspace(0.0, args.duration_days, args.sample_count)
    rows, summaries = run_time_dependent_depletion_gradient_diagnostic(
        times_days=times,
        config=config,
    )
    write_diagnostic_artifacts(
        rows, summaries, args.output_dir, config.uptake_reference_time_days
    )
    comparison_rows, comparison_summaries = run_native_geometry_closure_comparisons(
        times_days=times,
        config=config,
    )
    write_native_geometry_comparison_artifacts(
        comparison_rows, comparison_summaries, args.output_dir
    )


if __name__ == "__main__":
    main()
