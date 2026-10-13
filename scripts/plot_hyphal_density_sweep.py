"""Plot structural C:P cost against hyphal density."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

P_MASS_UG_PER_UMOL = 30.9738


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--linear-y", action="store_true")
    args = parser.parse_args()

    with args.summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    concentrations = sorted({float(row["initial_solution_p_um"]) for row in rows})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ncols = 3 if len(concentrations) <= 6 else 8
    nrows = math.ceil(len(concentrations) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.3 * nrows),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for index, (axis, concentration) in enumerate(zip(axes, concentrations)):
        selected = [row for row in rows
                    if float(row["initial_solution_p_um"]) == concentration]
        density = np.array([float(row["density_cm_cm3"]) for row in selected])
        uptake = np.array([float(row["total_uptake_micromol"]) for row in selected])
        construction = np.array([float(row["construction_carbon_g"]) for row in selected])
        c_per_p_mass = (
            construction * 1e6 / (uptake * P_MASS_UG_PER_UMOL)
        )
        points = axis.scatter(
            density,
            c_per_p_mass,
            c=uptake,
            cmap="viridis",
            norm=plt.Normalize(vmin=float(uptake.min()), vmax=float(uptake.max())),
            s=24,
            edgecolors="none",
        )
        if not args.linear_y:
            axis.set_yscale("log")
        axis.set_title(f"Initial P = {concentration:g} µM")
        axis.grid(True, which="both", color="0.85", linewidth=0.6)
        if index // ncols == nrows - 1:
            axis.set_xlabel(r"Hyphal density (cm hypha cm$^{-3}$)")
        if index % ncols == 0:
            axis.set_ylabel(r"Structural C:P ($\mu$g C $\mu$g$^{-1}$ P)")
        colorbar = fig.colorbar(points, ax=axis, pad=0.02, shrink=0.86)
        colorbar.set_label(r"Total 30-day P uptake ($\mu$mol P)")
    for axis in axes[len(concentrations):]:
        axis.set_visible(False)
    fig.suptitle("Finite-reservoir hyphal-density sweep", fontsize=15)
    output_name = "hyphal_density_c_to_p_linear_y" if args.linear_y else "hyphal_density_c_to_p"
    output_stem = args.output_dir / output_name
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
