"""Plot a stylised version of the static-control trajectory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLANT_COLOUR = "#2878b5"
P_UPTAKE_COLOUR = "#E69F00"
C_FIXATION_COLOUR = "#D55E00"


def _style(axis: plt.Axes) -> None:
    axis.grid(axis="both", color="#dddddd", linewidth=0.6)
    axis.spines["top"].set_visible(False)


def plot_stylized(output: Path, *, days: float = 60.0) -> None:
    """Write exponential biomass and linear uptake/fixation curves.

    The endpoints are illustrative values chosen to resemble the trajectory
    figure; these curves are not a new model run or a fit to the transient.
    """
    time = np.linspace(0.0, days, 300)
    plant_biomass = 0.10 * np.exp(np.log(20.0) * time / days)

    # Uptake and fixation are linear functions of biomass.  Because biomass
    # itself is exponential in time, these appear as linear response curves
    # in the two biomass panels below.
    plant_p_rate = 0.001 + (0.020 - 0.001) * (
        plant_biomass - plant_biomass[0]
    ) / (plant_biomass[-1] - plant_biomass[0])
    carbon_fixation = 0.004 + (0.080 - 0.004) * (
        plant_biomass - plant_biomass[0]
    ) / (plant_biomass[-1] - plant_biomass[0])

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 12.0), constrained_layout=True)

    plant_axis = axes[0]
    plant_axis.plot(time, plant_biomass, color=PLANT_COLOUR, linewidth=3.0)
    plant_axis.set_xlabel("")
    plant_axis.set_ylabel("")
    plant_axis.tick_params(axis="y", labelcolor="black")
    _style(plant_axis)

    uptake_axis = axes[1]
    uptake_axis.plot(plant_biomass, plant_p_rate, color=P_UPTAKE_COLOUR, linewidth=3.0)
    uptake_axis.set_xlabel("")
    uptake_axis.set_ylabel("")
    _style(uptake_axis)

    carbon_axis = axes[2]
    carbon_axis.plot(plant_biomass, carbon_fixation, color=C_FIXATION_COLOUR, linewidth=3.0)
    carbon_axis.set_xlabel("")
    carbon_axis.set_ylabel("")
    _style(carbon_axis)

    for axis in axes:
        axis.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    plot_stylized(Path("outputs/static-control-1um-fixed-trade-full-growth/stylized-trajectory.png"))
