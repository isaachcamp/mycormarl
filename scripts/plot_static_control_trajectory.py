"""Plot a fixed-trade, full-growth static-control trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mycormarl.environments.base_mycor import FUNGUS, PLANT
from mycormarl.soil.phosphate_units import michaelis_menten_surface_flux
from mycormarl.static_controls import _actions, _environment


PLANT_COLOUR = "#2878b5"
FUNGUS_COLOUR = "#d95f02"


def _trajectory(manifest: dict, *, soil_radius_cm: float | None = None,
                soil_depth_cm: float | None = None,
                days_override: float | None = None) -> dict[str, np.ndarray]:
    """Run one mixed 1 µM condition and return recorded per-step observables."""
    p_level = 1.0
    seed = int(manifest.get("seeds", [0])[0])
    if soil_radius_cm is not None or soil_depth_cm is not None:
        environment = manifest["model"]["environment"]
        if soil_radius_cm is not None:
            environment["soil_radius_cm"] = soil_radius_cm
        if soil_depth_cm is not None:
            environment["soil_depth_cm"] = soil_depth_cm
        environment.pop("initial_solution_p_depth_profile", None)
    if days_override is not None:
        manifest["horizon"]["days"] = days_override
    env = _environment(manifest, "mixed", p_level)
    actions = _actions(manifest["static_policy"])
    _, state = env.reset(jax.random.PRNGKey(seed))

    days = [0.0]
    plant_biomass = [float(state.plant_biomass[0])]
    fungus_biomass = [float(state.fungus_biomass[0])]
    plant_p_uptake_rate = [0.0]
    fungus_p_uptake_rate = [0.0]
    carbon_fixed_rate = [0.0]

    for step in range(env.max_episode_steps):
        _, state, _, _, info = env.step_env(
            jax.random.PRNGKey(seed + step + 1), state, actions
        )
        plant = info[PLANT]
        fungus = info[FUNGUS]
        days.append((step + 1) * env.config.dt)
        plant_biomass.append(float(state.plant_biomass[0]))
        fungus_biomass.append(float(state.fungus_biomass[0]))
        plant_p_uptake_rate.append(float(plant["direct_p_uptake_mg"][0]))
        fungus_p_uptake_rate.append(float(fungus["direct_p_uptake_mg"][0]))
        carbon_fixed_rate.append(float(plant["carbon_fixed"][0]))
        if bool(state.terminal):
            break

    return {name: np.asarray(values) for name, values in {
        "day": days,
        "plant_biomass": plant_biomass,
        "fungus_biomass": fungus_biomass,
        "plant_p_uptake_rate": plant_p_uptake_rate,
        "fungus_p_uptake_rate": fungus_p_uptake_rate,
        "carbon_fixed_rate": carbon_fixed_rate,
    }.items()}


def _style(axis: plt.Axes) -> None:
    axis.grid(axis="both", color="#dddddd", linewidth=0.6)
    axis.spines["top"].set_visible(False)


def plot_trajectory(data: dict[str, np.ndarray], output: Path) -> None:
    """Write five panels in a three-row, two-column layout."""
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.0), constrained_layout=True)

    plant_biomass_axis, fungus_biomass_axis = axes[0]
    plant_biomass_axis.plot(data["day"], data["plant_biomass"], color=PLANT_COLOUR)
    plant_biomass_axis.set_xlabel("Episode time (days)")
    plant_biomass_axis.set_ylabel("Plant biomass (g)")
    plant_biomass_axis.set_title("Plant biomass")
    _style(plant_biomass_axis)
    fungus_biomass_axis.plot(data["day"], data["fungus_biomass"], color=FUNGUS_COLOUR)
    fungus_biomass_axis.set_xlabel("Episode time (days)")
    fungus_biomass_axis.set_ylabel("Fungus biomass (g)")
    fungus_biomass_axis.set_title("Fungus biomass")
    _style(fungus_biomass_axis)

    plant_uptake_axis, fungus_uptake_axis = axes[1]
    plant_uptake_axis.plot(data["plant_biomass"], data["plant_p_uptake_rate"], color=PLANT_COLOUR)
    plant_uptake_axis.set_xlabel("Plant biomass (g)")
    plant_uptake_axis.set_ylabel("P uptake / timestep (mg P)")
    plant_uptake_axis.set_title("Plant P uptake")
    _style(plant_uptake_axis)
    fungus_uptake_axis.plot(data["fungus_biomass"], data["fungus_p_uptake_rate"], color=FUNGUS_COLOUR)
    fungus_uptake_axis.set_xlabel("Fungus biomass (g)")
    fungus_uptake_axis.set_ylabel("P uptake / timestep (mg P)")
    fungus_uptake_axis.set_title("Fungus P uptake")
    _style(fungus_uptake_axis)

    carbon_axis = axes[2, 0]
    carbon_axis.plot(data["plant_biomass"], data["carbon_fixed_rate"], color=PLANT_COLOUR)
    carbon_axis.set_xlabel("Plant biomass (g)")
    carbon_axis.set_ylabel("C fixation / timestep (g C)")
    carbon_axis.set_title("Plant C fixation")
    _style(carbon_axis)
    axes[2, 1].axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_michaelis_menten(output: Path) -> None:
    """Write the plant Michaelis--Menten response curve."""
    from mycormarl.plant.traits import PlantTraits

    concentration_um = np.geomspace(1e-3, 1e3, 500)
    concentration = jnp.asarray(concentration_um * 1e-3)  # µM -> µmol cm^-3
    plant = PlantTraits()
    plant_flux = np.asarray(michaelis_menten_surface_flux(concentration, plant.jmax, plant.km))

    fig, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    axis.plot(concentration_um, plant_flux, color=PLANT_COLOUR, linewidth=3.0)
    axis.set_xscale("log")
    axis.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
    _style(axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--soil-radius-cm", type=float)
    parser.add_argument("--soil-depth-cm", type=float)
    parser.add_argument("--days", type=float)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    data = _trajectory(
        manifest,
        soil_radius_cm=args.soil_radius_cm,
        soil_depth_cm=args.soil_depth_cm,
        days_override=args.days,
    )
    plot_trajectory(data, args.output_directory / "static-control-trajectory.png")
    plot_michaelis_menten(args.output_directory / "michaelis-menten-response.png")
    print(f"wrote figures to {args.output_directory}")


if __name__ == "__main__":
    main()
