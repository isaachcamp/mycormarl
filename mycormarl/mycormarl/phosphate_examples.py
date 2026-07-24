"""Small deterministic examples using the production phosphate soil step."""

from __future__ import annotations

import argparse
import json
import jax
import jax.numpy as jnp

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_units import MICROMOL_P_TO_MG_P
from mycormarl.soil.soil import evolve_soil_p_with_diagnostics


MODES = ("plant-only", "fungus-only", "mixed")


def example_config() -> EnvConfig:
    """Return a cheap four-cell domain using the qualified timestep."""
    return EnvConfig(
        max_steps=1,
        dt=0.025,
        soil_radius_cm=0.2,
        soil_depth_cm=0.2,
        radial_interval_cm=0.1,
        depth_interval_cm=0.1,
        topsoil_depth_cm=0.2,
        initial_solution_p_um=1.0,
    )


def example_species(mode: str) -> SpeciesParams:
    """Return default traits, disabling the consumer absent from ``mode``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    return SpeciesParams(
        plant=PlantTraits(
            initial_biomass=0.0,
            initial_c_pool=0.0,
            initial_p_pool=0.0,
            jmax=0.0 if mode == "fungus-only" else 3.26e-6,
            kappa_c=0.0,
            kappa_p=0.0,
        ),
        fungus=FungusTraits(
            initial_biomass=0.0,
            initial_c_pool=0.0,
            initial_p_pool=0.0,
            jmax=0.0 if mode == "plant-only" else 3.26e-6,
            kappa_c=0.0,
            kappa_p=0.0,
        ),
    )


def _soil_step_arguments(env: BaseMycorMarl) -> tuple:
    """Return the production soil arguments in their canonical order."""
    return (
        env.config.dt,
        env.species,
        env.cell_volumes,
        env.config.theta_water,
        env.config.b_p,
        env.radial_diffusion_conductance,
        env.vertical_diffusion_conductance,
        env.soil_substeps,
        env.config.phosphate_diffusion_coefficient_cm2_s,
        env.config.phosphate_impedance_factor,
        env.config.uptake_reference_time_days,
        env.config.uptake_transition_exponent,
    )


def run_example(mode: str) -> dict[str, float | int | str]:
    """Run one scenario and return uptake, regime, and balance diagnostics.

    The explicit fixed-density fields make the consumer mode easy to inspect.
    The environment geometry, diffusion coefficients, blending, competition,
    unit conversion, and pool credit all use production functions.
    """
    species = example_species(mode)
    env = BaseMycorMarl(example_config(), species)
    _, state = env.reset(jax.random.PRNGKey(0))
    root_density = jnp.ones(env.grid_shape) if mode != "fungus-only" else jnp.zeros(env.grid_shape)
    hyphal_density = (
        jnp.full(env.grid_shape, species.fungus.saturation_density)
        if mode != "plant-only"
        else jnp.zeros(env.grid_shape)
    )
    state = state.replace(
        root_length_density=root_density,
        hyphae_length_density=hyphal_density,
    )
    soil_before = float(jnp.sum(state.soil_labile_p))
    plant_before = float(state.plant_p_pool[0])
    fungus_before = float(state.fungus_p_pool[0])
    next_state, diagnostics = evolve_soil_p_with_diagnostics(
        state, *_soil_step_arguments(env)
    )
    soil_after = float(jnp.sum(next_state.soil_labile_p))
    plant_uptake = float(next_state.plant_p_pool[0]) - plant_before
    fungus_uptake = float(next_state.fungus_p_pool[0]) - fungus_before
    soil_loss_mg = float((soil_before - soil_after) * MICROMOL_P_TO_MG_P)
    final = diagnostics[-1]
    return {
        "mode": mode,
        "grid_cells": int(env.cell_volumes.size),
        "soil_substeps": int(env.soil_substeps),
        "initial_soil_labile_p_micromol": soil_before,
        "final_soil_labile_p_micromol": soil_after,
        "minimum_soil_labile_p_micromol": float(jnp.min(next_state.soil_labile_p)),
        "soil_loss_mg": soil_loss_mg,
        "plant_uptake_mg": plant_uptake,
        "fungus_uptake_mg": fungus_uptake,
        "balance_error_mg": soil_loss_mg - plant_uptake - fungus_uptake,
        "continuous_weight_mean": float(jnp.mean(final.continuous_weight)),
        "capped_cell_fraction": float(jnp.mean(final.capped)),
    }

def main() -> None:
    """Run the selected scenario, or all three scenarios by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    args = parser.parse_args()
    modes = MODES if args.mode == "all" else (args.mode,)
    print(json.dumps([run_example(mode) for mode in modes], indent=2))


if __name__ == "__main__":
    main()
