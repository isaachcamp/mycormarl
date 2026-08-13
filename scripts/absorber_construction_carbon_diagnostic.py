"""Generate isolated absorber construction-carbon and depletion diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mycormarl-matplotlib"),
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
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np

from mycormarl.soil.absorber_diagnostic import run_absorber_geometry_sweep


_MODE_TITLES = {
    "fixed_reservoir": "Fixed reservoir",
    "finite_inventory": "Finite inventory",
}
_MARKER_STYLES = {
    "plant_native": ("o", "Plant-native"),
    "fungus_geometry_plant_economics": (
        "s",
        "Fungal geometry, plant economics",
    ),
    "fungus_equivalent_plant_geometry": ("^", "Fungus-equivalent plant geometry"),
    "fungus_native": ("*", "Fungus-native"),
}


def _surface_matrix(rows, mode: str, metric: str):
    """Return sorted geometry axes and a metric matrix from tabular rows."""
    selected = [
        row
        for row in rows
        if row["record_type"] == "surface" and row["experiment_mode"] == mode
    ]
    radii = np.array(sorted({row["absorber_radius_cm"] for row in selected}))
    densities = np.array(
        sorted({row["length_density_cm_cm3"] for row in selected})
    )
    lookup = {
        (row["length_density_cm_cm3"], row["absorber_radius_cm"]): row[metric]
        for row in selected
    }
    values = np.array(
        [[lookup[(density, radius)] for radius in radii] for density in densities],
        dtype=float,
    )
    return radii, densities, values


def _positive_norm(values: np.ndarray) -> LogNorm:
    """Build a stable logarithmic colour norm for positive scientific metrics."""
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return LogNorm(vmin=1e-30, vmax=1.0)
    vmin = float(positive.min())
    vmax = float(positive.max())
    if vmax <= vmin:
        vmax = vmin * 1.01
    return LogNorm(vmin=vmin, vmax=vmax)


def _log_edges(centres: np.ndarray) -> np.ndarray:
    """Return geometric cell edges around positive logarithmic centres."""
    log_centres = np.log10(centres)
    if len(log_centres) == 1:
        return 10.0 ** np.array([log_centres[0] - 0.25, log_centres[0] + 0.25])
    interior = 0.5 * (log_centres[:-1] + log_centres[1:])
    first = log_centres[0] - (interior[0] - log_centres[0])
    last = log_centres[-1] + (log_centres[-1] - interior[-1])
    return 10.0 ** np.concatenate(([first], interior, [last]))


def _add_reference_markers(
    ax, rows, mode: str, metric: str, *, efficiency: bool
) -> None:
    """Overlay tabular marker rows without recomputing metrics."""
    for row in rows:
        if row["record_type"] != "marker" or row["experiment_mode"] != mode:
            continue
        if efficiency:
            if row["marker_label"] == "fungus_native":
                continue
            if (
                row["marker_label"] == "fungus_equivalent_plant_geometry"
                and row["marker_metric"] != metric
            ):
                continue
        elif row["marker_label"] in {
            "fungus_equivalent_plant_geometry",
            "fungus_geometry_plant_economics",
        }:
            continue
        if row["marker_solve_status"] == "unavailable":
            continue
        marker, _ = _MARKER_STYLES[row["marker_label"]]
        marker_size = {"o": 42, "s": 58, "^": 100, "*": 80}[marker]
        ax.scatter(
            row["absorber_radius_cm"],
            row["length_density_cm_cm3"],
            marker=marker,
            s=marker_size,
            facecolors="white",
            edgecolors="black",
            linewidths=1.0,
            zorder=5 if marker == "*" else 4,
        )


def _format_geometry_axes(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Absorber radius (cm)")
    ax.set_ylabel("Length density (cm cm⁻³)")


def _add_invalid_geometry_boundary(
    ax, radii: np.ndarray, densities: np.ndarray
) -> None:
    """Show the strict cylinder-territory boundary and identify invalid cells."""
    radius_lower = max(radii.min(), 1.0 / np.sqrt(np.pi * densities.max()))
    radius_upper = min(radii.max(), 1.0 / np.sqrt(np.pi * densities.min()))
    boundary_radii = np.geomspace(radius_lower, radius_upper, 200)
    boundary_density = 1.0 / (np.pi * boundary_radii**2)
    ax.plot(
        boundary_radii, boundary_density, color="white", linewidth=2.2, zorder=3
    )
    ax.plot(
        boundary_radii,
        boundary_density,
        color="black",
        linewidth=0.8,
        linestyle="--",
        zorder=3,
    )


def _marker_legend(handles, fig) -> None:
    """Place one shared marker legend outside the upper-right plot."""
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.98, 0.96),
        frameon=False,
    )


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    """Write a vector master and high-resolution raster preview."""
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_pair(rows, output_dir: Path, *, efficiency: bool) -> None:
    """Plot integrated and maximum uptake for both experiment modes."""
    if efficiency:
        metrics = (
            "integrated_uptake_per_construction_carbon_micromol_g_c",
            "maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s",
        )
        column_titles = (
            "Integrated P uptake / Construction C",
            "Maximum P uptake rate / Construction C",
        )
        colour_labels = ("µmol P g C⁻¹", "µmol P g C⁻¹ s⁻¹")
        stem = "construction_carbon_efficiency"
    else:
        metrics = (
            "integrated_uptake_micromol",
            "maximum_instantaneous_uptake_rate_micromol_s",
        )
        column_titles = ("Integrated P uptake", "Maximum P uptake rate")
        colour_labels = ("µmol P", "µmol P s⁻¹")
        stem = "uptake_scale"

    modes = ("fixed_reservoir", "finite_inventory")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), constrained_layout=True)
    for row_index, mode in enumerate(modes):
        for column_index, (metric, title, colour_label) in enumerate(
            zip(metrics, column_titles, colour_labels, strict=True)
        ):
            ax = axes[row_index, column_index]
            radii, densities, values = _surface_matrix(rows, mode, metric)
            norm = _positive_norm(values)
            mesh = ax.pcolormesh(
                _log_edges(radii),
                _log_edges(densities),
                values,
                shading="flat",
                cmap="viridis",
                norm=norm,
                rasterized=True,
            )
            _add_invalid_geometry_boundary(ax, radii, densities)
            _add_reference_markers(
                ax, rows, mode, metric, efficiency=efficiency
            )
            _format_geometry_axes(ax)
            ax.set_title(f"{title}\n{_MODE_TITLES[mode]}")
            fig.colorbar(mesh, ax=ax, label=colour_label, shrink=0.82)

    handles = [
        Line2D(
            [0],
            [0],
            marker=style[0],
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            label=style[1],
        )
        for label, style in _MARKER_STYLES.items()
        if (efficiency and label != "fungus_native")
        or (
            not efficiency
            and label
            not in {
                "fungus_equivalent_plant_geometry",
                "fungus_geometry_plant_economics",
            }
        )
    ]
    _marker_legend(handles, fig)
    _save_figure(fig, output_dir, stem)


def _plot_depletion_timescale(rows, output_dir: Path, reference_time_days: float) -> None:
    """Plot finite-inventory first-depletion time with explicit not-reached cells."""
    radii, densities, times = _surface_matrix(
        rows, "finite_inventory", "t_1_percent_days"
    )
    masked = np.ma.masked_invalid(times)
    cmap = matplotlib.colormaps["cividis"].copy()
    cmap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    mesh = ax.pcolormesh(
        _log_edges(radii),
        _log_edges(densities),
        masked,
        shading="flat",
        cmap=cmap,
        norm=_positive_norm(times),
        rasterized=True,
    )
    _format_geometry_axes(ax)
    _add_invalid_geometry_boundary(ax, radii, densities)
    ax.set_title("Finite inventory — t₁% surface-depletion timescale")
    fig.colorbar(mesh, ax=ax, label="t₁% (day)")

    _add_reference_markers(
        ax, rows, "finite_inventory", "t_1_percent_days", efficiency=False
    )
    _marker_legend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", label="Plant-native"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="white", markeredgecolor="black", label="Fungus-native"),
        ],
        fig,
    )
    _save_figure(fig, output_dir, "depletion_timescale")


def _plot_synthesis(rows, output_dir: Path) -> None:
    """Summarise finite-inventory efficiency, scale, depletion, and capture speed."""
    mode = "finite_inventory"
    efficiency_metric = "integrated_uptake_per_construction_carbon_micromol_g_c"
    uptake_metric = "integrated_uptake_micromol"
    rate_metric = "maximum_instantaneous_uptake_rate_micromol_s"
    time_metric = "t_1_percent_days"
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), constrained_layout=True)

    surface_panels = (
        (
            axes[0, 0],
            efficiency_metric,
            "P acquired per construction C",
            "µmol P g C⁻¹",
            "viridis",
            True,
        ),
        (
            axes[0, 1],
            uptake_metric,
            "P acquired over one day",
            "µmol P",
            "viridis",
            False,
        ),
        (
            axes[1, 0],
            time_metric,
            "Surface-P depletion timescale",
            "t₁% (day)",
            "cividis",
            False,
        ),
    )
    for ax, metric, title, colour_label, cmap_name, efficiency in surface_panels:
        radii, densities, values = _surface_matrix(rows, mode, metric)
        cmap = matplotlib.colormaps[cmap_name].copy()
        cmap.set_bad("#d9d9d9")
        mesh = ax.pcolormesh(
            _log_edges(radii),
            _log_edges(densities),
            np.ma.masked_invalid(values),
            shading="flat",
            cmap=cmap,
            norm=_positive_norm(values),
            rasterized=True,
        )
        _add_invalid_geometry_boundary(ax, radii, densities)
        _add_reference_markers(ax, rows, mode, metric, efficiency=efficiency)
        _format_geometry_axes(ax)
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label=colour_label, shrink=0.82)

    frontier = axes[1, 1]
    surface_rows = [
        row
        for row in rows
        if row["record_type"] == "surface"
        and row["experiment_mode"] == mode
        and row["geometry_valid"]
        and row[efficiency_metric] > 0.0
        and row[rate_metric] > 0.0
    ]
    all_radii = np.array([row["absorber_radius_cm"] for row in surface_rows])
    all_densities = np.array(
        [row["length_density_cm_cm3"] for row in surface_rows]
    )
    unique_radii = np.unique(all_radii)
    unique_densities = np.unique(all_densities)
    sampled_radii = set(unique_radii[::max(len(unique_radii) // 10, 1)])
    sampled_densities = set(
        unique_densities[::max(len(unique_densities) // 10, 1)]
    )
    sampled_rows = [
        row
        for row in surface_rows
        if row["absorber_radius_cm"] in sampled_radii
        and row["length_density_cm_cm3"] in sampled_densities
    ]
    radii = np.array([row["absorber_radius_cm"] for row in sampled_rows])
    densities = np.array([row["length_density_cm_cm3"] for row in sampled_rows])
    efficiency = np.array([row[efficiency_metric] for row in sampled_rows])
    rates = np.array([row[rate_metric] for row in sampled_rows])
    density_log = np.log10(densities)
    density_log_min = np.log10(all_densities.min())
    density_log_span = np.log10(all_densities.max()) - density_log_min
    sizes = 12.0 + 70.0 * (density_log - density_log_min) / density_log_span
    scatter = frontier.scatter(
        efficiency,
        rates,
        c=radii,
        s=sizes,
        cmap="plasma_r",
        norm=LogNorm(vmin=float(radii.min()), vmax=float(radii.max())),
        alpha=0.3,
        edgecolors="none",
        rasterized=True,
    )
    colourbar_mappable = matplotlib.cm.ScalarMappable(
        norm=scatter.norm, cmap=scatter.cmap
    )
    colourbar_mappable.set_array(radii)
    for row in rows:
        if (
            row["record_type"] != "marker"
            or row["experiment_mode"] != mode
            or row["marker_label"] not in {"plant_native", "fungus_native"}
        ):
            continue
        native_density = row["length_density_cm_cm3"]
        native_size = 12.0 + 70.0 * (
            np.log10(native_density) - density_log_min
        ) / density_log_span
        frontier.scatter(
            row[efficiency_metric],
            row[rate_metric],
            c=[row["absorber_radius_cm"]],
            s=native_size,
            cmap="plasma_r",
            norm=LogNorm(vmin=float(radii.min()), vmax=float(radii.max())),
            alpha=1.0,
            edgecolors="none",
            zorder=4,
        )
        label = "P" if row["marker_label"] == "plant_native" else "F"
        frontier.annotate(
            label,
            (row[efficiency_metric], row[rate_metric]),
            xytext=(4, -5),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            zorder=5,
        )
    frontier.set_xscale("log")
    frontier.set_yscale("log")
    frontier.set_xlabel("P acquired / construction C (µmol P g C⁻¹)")
    frontier.set_ylabel("Initial P capture rate (µmol P s⁻¹)")
    frontier.set_title("P-foraging advantage frontier")
    fig.colorbar(
        colourbar_mappable,
        ax=frontier,
        label="Absorber radius (cm)",
        shrink=0.82,
    )
    _marker_legend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", label="Plant-native"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="black", label="Fungal geometry, plant economics"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor="black", label="Fungus-equivalent plant geometry"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="white", markeredgecolor="black", label="Fungus-native"),
        ],
        fig,
    )
    _save_figure(fig, output_dir, "finite_inventory_foraging_synthesis")


def write_diagnostic_artifacts(
    rows: list[dict[str, object]],
    output_dir: Path,
    reference_time_days: float,
) -> None:
    """Write tabular results and plots derived exclusively from those rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "absorber_geometry_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_metric_pair(rows, output_dir, efficiency=True)
    _plot_metric_pair(rows, output_dir, efficiency=False)
    _plot_depletion_timescale(rows, output_dir, reference_time_days)
    _plot_synthesis(rows, output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius-count", type=int, default=40)
    parser.add_argument("--density-count", type=int, default=40)
    parser.add_argument("--dt-days", type=float, default=0.025)
    parser.add_argument("--reference-time-days", type=float, default=1.0)
    args = parser.parse_args(argv)

    rows = run_absorber_geometry_sweep(
        radius_count=args.radius_count,
        density_count=args.density_count,
        dt_days=args.dt_days,
        reference_time_days=args.reference_time_days,
    )
    write_diagnostic_artifacts(rows, args.output_dir, args.reference_time_days)


if __name__ == "__main__":
    main()
