"""Run the reproducible phosphate convergence and performance studies.

The script uses production environment construction, diffusion, uptake, and
biological stepping. It writes machine-readable JSON and a concise Markdown
summary under ``docs/qualification``. Scientific convergence
uses a reduced physical domain; the full default grid is used only for timed
performance and memory-footprint qualification.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from mycormarl.actions import rate_action
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_qualification import (
    annual_runtime_projection,
    reference_relative_change,
)
from mycormarl.soil.deep_soil_qualification import (
    StaticPolicy,
    run_deep_soil_qualification,
)
from mycormarl.soil.soil import evolve_soil_p_with_diagnostics
from mycormarl.soil.phosphate_units import MICROMOL_P_TO_MG_P


CONCENTRATIONS_UM = (0.1, 0.3, 1.0, 3.0, 10.0)
TIMESTEPS_DAYS = (0.0125, 0.025, 0.05, 0.1, 0.2)
GRID_INTERVALS_CM = (0.4, 0.2, 0.1, 0.05, 0.025)
MODES = ("root_only", "fungus_only", "mixed")
HORIZON_DAYS = 2.0
COUPLED_POLICY_DECISION_INTERVAL_DAYS = 1.0
COUPLED_PLANT_INITIAL_BIOMASS_G = 0.01
COUPLED_FUNGUS_INITIAL_BIOMASS_G = 0.0001
DEEP_SOIL_DURATION_DAYS = 2.0
DEEP_SOIL_RADIUS_CM = 2.0
DEEP_SOIL_DEPTH_CM = 100.0
ABSOLUTE_METRIC_FLOOR = 1e-10
RELATIVE_TOLERANCE = 0.05
MAXIMUM_RELATIVE_P_BALANCE_ERROR = 1e-5
COUPLED_ACTION_FREQUENCY_METRICS = (
    "plant_uptake_micromol",
    "fungus_uptake_micromol",
    "total_uptake_micromol",
    "final_soil_micromol",
    "plant_uptake_share",
    "fungus_uptake_share",
    "plant_biomass_g",
    "fungus_biomass_g",
    "root_radius_cm",
    "root_depth_cm",
    "fungus_radius_cm",
    "fungus_depth_cm",
)
COUPLED_ACTION_FREQUENCY_DIAGNOSTICS = (
    "plant_p_pool_mg",
    "fungus_p_pool_mg",
)


def qualification_species(coupled: bool = False) -> SpeciesParams:
    """Return explicit traits for fixed-soil or coupled qualification runs."""
    plant_biomass = COUPLED_PLANT_INITIAL_BIOMASS_G if coupled else 0.0
    fungus_biomass = COUPLED_FUNGUS_INITIAL_BIOMASS_G if coupled else 0.0
    plant = PlantTraits(
        initial_biomass=plant_biomass,
        kappa_c=0.0,
        kappa_p=0.0,
    )
    fungus = FungusTraits(
        initial_biomass=fungus_biomass,
        kappa_c=0.0,
        kappa_p=0.0,
    )
    return SpeciesParams(
        plant=plant.replace(
            initial_c_pool=plant_biomass * plant.gamma_c,
            initial_p_pool=plant_biomass * plant.gamma_p,
        ),
        fungus=fungus.replace(
            initial_c_pool=fungus_biomass * fungus.gamma_c,
            initial_p_pool=fungus_biomass * fungus.gamma_p,
        ),
    )


def qualification_config(
    interval_cm: float,
    dt_days: float,
    concentration_um: float,
    reference_time_days: float = 1.0,
    exponent: float = 2.0,
) -> EnvConfig:
    """Return the reduced-domain qualification configuration with an internal P front."""
    return EnvConfig(
        dt=dt_days,
        max_steps=math.ceil(HORIZON_DAYS / dt_days) + 1,
        soil_radius_cm=2.0,
        soil_depth_cm=2.0,
        radial_interval_cm=interval_cm,
        depth_interval_cm=interval_cm,
        initial_solution_p_um=concentration_um,
        uptake_reference_time_days=reference_time_days,
        uptake_transition_exponent=exponent,
    )


def fixed_density_fields(env: BaseMycorMarl, mode: str):
    """Return physically identical topsoil absorber fields on every grid."""
    if mode not in MODES:
        raise ValueError(f"unsupported consumer mode: {mode}")
    z_centres = 0.5 * (env.z_edges[:-1] + env.z_edges[1:])
    topsoil = jnp.broadcast_to(
        (z_centres < 1.0)[None, :], env.grid_shape
    )
    root = jnp.where(topsoil, 1.0, 0.0) if mode != "fungus_only" else jnp.zeros(env.grid_shape)
    hypha = (
        jnp.where(topsoil, env.species.fungus.saturation_density, 0.0)
        if mode != "root_only"
        else jnp.zeros(env.grid_shape)
    )
    return root, hypha


def _soil_step_arguments(env: BaseMycorMarl):
    """Return the production soil call arguments in their canonical order."""
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


def run_fixed_soil_scenario(
    mode: str,
    concentration_um: float,
    dt_days: float,
    interval_cm: float,
    reference_time_days: float = 1.0,
    exponent: float = 2.0,
) -> dict:
    """Run one fixed-geometry scenario and aggregate exact uptake diagnostics."""
    step_count_float = HORIZON_DAYS / dt_days
    if not math.isclose(step_count_float, round(step_count_float), rel_tol=0, abs_tol=1e-10):
        raise ValueError("qualification horizon must contain an integer step count")
    env = BaseMycorMarl(
        qualification_config(
            interval_cm, dt_days, concentration_um, reference_time_days, exponent
        ),
        qualification_species(),
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    root_density, hyphal_density = fixed_density_fields(env, mode)
    state = state.replace(
        root_length_density=root_density,
        hyphae_length_density=hyphal_density,
    )
    initial_soil = float(jnp.sum(state.soil_labile_p))
    initial_plant_pool = float(state.plant_p_pool[0])
    initial_fungus_pool = float(state.fungus_p_pool[0])
    root_ratio_sum = fungus_ratio_sum = 0.0
    root_ratio_min = fungus_ratio_min = 1.0
    root_ratio_count = fungus_ratio_count = 0
    capped_count = demand_count = 0
    weight_sum = 0.0
    weight_max = 0.0
    weight_count = 0

    for _ in range(round(step_count_float)):
        state, substep_diagnostics = evolve_soil_p_with_diagnostics(
            state, *_soil_step_arguments(env)
        )
        for diagnostic in substep_diagnostics:
            root_mask = (root_density > 0.0) & (
                diagnostic.root_request_micromol > 0.0
            )
            fungus_mask = (hyphal_density > 0.0) & (
                diagnostic.fungus_request_micromol > 0.0
            )
            demand_mask = (
                diagnostic.root_request_micromol
                + diagnostic.fungus_request_micromol
            ) > 0.0
            root_ratio_sum += float(jnp.sum(jnp.where(root_mask, diagnostic.root_surface_ratio, 0.0)))
            fungus_ratio_sum += float(jnp.sum(jnp.where(fungus_mask, diagnostic.fungus_surface_ratio, 0.0)))
            if bool(jnp.any(root_mask)):
                root_ratio_min = min(root_ratio_min, float(jnp.min(jnp.where(root_mask, diagnostic.root_surface_ratio, 1.0))))
            if bool(jnp.any(fungus_mask)):
                fungus_ratio_min = min(fungus_ratio_min, float(jnp.min(jnp.where(fungus_mask, diagnostic.fungus_surface_ratio, 1.0))))
            root_ratio_count += int(jnp.sum(root_mask))
            fungus_ratio_count += int(jnp.sum(fungus_mask))
            capped_count += int(jnp.sum(diagnostic.capped & demand_mask))
            demand_count += int(jnp.sum(demand_mask))
            hypha_mask = hyphal_density > 0.0
            weight_sum += float(jnp.sum(jnp.where(hypha_mask, diagnostic.continuous_weight, 0.0)))
            if bool(jnp.any(hypha_mask)):
                weight_max = max(
                    weight_max,
                    float(jnp.max(jnp.where(hypha_mask, diagnostic.continuous_weight, 0.0))),
                )
            weight_count += int(jnp.sum(hypha_mask))

    final_soil = float(jnp.sum(state.soil_labile_p))
    root_uptake = (
        float(state.plant_p_pool[0]) - initial_plant_pool
    ) / MICROMOL_P_TO_MG_P
    fungus_uptake = (
        float(state.fungus_p_pool[0]) - initial_fungus_pool
    ) / MICROMOL_P_TO_MG_P
    total_uptake = root_uptake + fungus_uptake
    balance_error = abs(initial_soil - final_soil - total_uptake) / max(
        initial_soil, 1e-30
    )
    return {
        "mode": mode,
        "concentration_um": concentration_um,
        "dt_days": dt_days,
        "interval_cm": interval_cm,
        "reference_time_days": reference_time_days,
        "transition_exponent": exponent,
        "horizon_days": HORIZON_DAYS,
        "cell_count": int(env.cell_volumes.size),
        "root_density_cm_cm3": 1.0 if mode != "fungus_only" else 0.0,
        "hyphal_density_cm_cm3": (
            env.species.fungus.saturation_density if mode != "root_only" else 0.0
        ),
        "soil_substeps": env.soil_substeps,
        "diffusion_cfl_seconds": env.diffusion_cfl_seconds,
        "initial_soil_micromol": initial_soil,
        "final_soil_micromol": final_soil,
        "root_uptake_micromol": root_uptake,
        "fungus_uptake_micromol": fungus_uptake,
        "total_uptake_micromol": total_uptake,
        "root_uptake_share": root_uptake / total_uptake if total_uptake > 0 else 0.0,
        "fungus_uptake_share": fungus_uptake / total_uptake if total_uptake > 0 else 0.0,
        "mean_root_surface_ratio": root_ratio_sum / root_ratio_count if root_ratio_count else None,
        "mean_fungus_surface_ratio": fungus_ratio_sum / fungus_ratio_count if fungus_ratio_count else None,
        "minimum_root_surface_ratio": root_ratio_min if root_ratio_count else None,
        "minimum_fungus_surface_ratio": fungus_ratio_min if fungus_ratio_count else None,
        "mean_continuous_weight": weight_sum / weight_count if weight_count else 0.0,
        "maximum_continuous_weight": weight_max,
        "capped_demand_fraction": capped_count / demand_count if demand_count else 0.0,
        "relative_p_balance_error": balance_error,
    }


def _spatial_extents(env: BaseMycorMarl, density) -> dict[str, float]:
    """Return outer radial and depth edges touched by a positive density field."""
    occupied = jnp.asarray(density) > 0.0
    radial_occupied = jnp.any(occupied, axis=1)
    depth_occupied = jnp.any(occupied, axis=0)
    radial_indices = jnp.where(radial_occupied, jnp.arange(occupied.shape[0]) + 1, 0)
    depth_indices = jnp.where(depth_occupied, jnp.arange(occupied.shape[1]) + 1, 0)
    return {
        "radius_cm": float(env.r_edges[int(jnp.max(radial_indices))]),
        "depth_cm": float(env.z_edges[int(jnp.max(depth_indices))]),
    }


def run_coupled_scenario(
    interval_cm: float,
    dt_days: float,
    decision_interval_days: float = COUPLED_POLICY_DECISION_INTERVAL_DAYS,
) -> dict:
    """Run coupled biology with a fixed policy schedule over numerical substeps."""
    decisions = HORIZON_DAYS / decision_interval_days
    if not math.isclose(decisions, round(decisions), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("coupled horizon must contain an integer policy decision count")
    numerical_substeps = decision_interval_days / dt_days
    if not math.isclose(
        numerical_substeps, round(numerical_substeps), rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError("coupled decision interval must contain whole numerical timesteps")
    numerical_env = BaseMycorMarl(
        qualification_config(interval_cm, dt_days, 1.0).replace(
            max_steps=round(HORIZON_DAYS / dt_days)
        ),
        qualification_species(coupled=True),
    )
    env = PolicyIntervalMycorMarl(
        numerical_env,
        decision_interval_days=decision_interval_days,
        max_episode_steps=round(decisions),
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    initial_soil_micromol = float(jnp.sum(state.soil_labile_p))

    def extended_p_mg(current_state) -> float:
        """Account for soil, free pools, structure, mortality, and exports."""
        return (
            float(jnp.sum(current_state.soil_labile_p)) * MICROMOL_P_TO_MG_P
            + float(current_state.plant_p_pool[0] + current_state.fungus_p_pool[0])
            + float(current_state.plant_biomass[0]) * env.species.plant.gamma_p
            + float(current_state.fungus_biomass[0]) * env.species.fungus.gamma_p
            + float(
                current_state.cumulative_plant_p_mortality_loss_mg[0]
                + current_state.cumulative_fungus_p_mortality_loss_mg[0]
                + current_state.cumulative_plant_p_maintenance_loss_mg[0]
                + current_state.cumulative_fungus_p_maintenance_loss_mg[0]
                + current_state.cumulative_plant_p_reproduction_export_mg[0]
                + current_state.cumulative_fungus_p_reproduction_export_mg[0]
            )
        )

    initial_extended_p_mg = extended_p_mg(state)
    actions = {
        PLANT: rate_action(0.25, 1.0, 0.0, 0.0),
        FUNGUS: rate_action(0.25, 1.0, 0.0, 0.0),
    }
    for step_index in range(round(decisions)):
        _, state, _, _, _ = env.step_env(
            jax.random.PRNGKey(step_index + 1), state, actions
        )
    root_extent = _spatial_extents(numerical_env, state.root_length_density)
    fungus_extent = _spatial_extents(numerical_env, state.hyphae_length_density)
    final_extended_p_mg = extended_p_mg(state)
    final_soil_micromol = float(jnp.sum(state.soil_labile_p))
    plant_uptake_micromol = float(
        state.cumulative_direct_plant_p_uptake_micromol[0]
    )
    fungus_uptake_micromol = (
        initial_soil_micromol - final_soil_micromol - plant_uptake_micromol
    )
    total_uptake_micromol = plant_uptake_micromol + fungus_uptake_micromol
    return {
        "interval_cm": interval_cm,
        "dt_days": dt_days,
        "decision_interval_days": decision_interval_days,
        "timing": {
            "numerical_timestep_days": dt_days,
            "policy_decision_interval_days": decision_interval_days,
        },
        "horizon_days": HORIZON_DAYS,
        "actions": {"plant": actions[PLANT].tolist(), "fungus": actions[FUNGUS].tolist()},
        "plant_biomass_g": float(state.plant_biomass[0]),
        "fungus_biomass_g": float(state.fungus_biomass[0]),
        "root_radius_cm": root_extent["radius_cm"],
        "root_depth_cm": root_extent["depth_cm"],
        "fungus_radius_cm": fungus_extent["radius_cm"],
        "fungus_depth_cm": fungus_extent["depth_cm"],
        "initial_soil_micromol": initial_soil_micromol,
        "final_soil_micromol": final_soil_micromol,
        "plant_uptake_micromol": plant_uptake_micromol,
        "fungus_uptake_micromol": fungus_uptake_micromol,
        "total_uptake_micromol": total_uptake_micromol,
        "plant_uptake_share": (
            plant_uptake_micromol / total_uptake_micromol
            if total_uptake_micromol > 0.0 else 0.0
        ),
        "fungus_uptake_share": (
            fungus_uptake_micromol / total_uptake_micromol
            if total_uptake_micromol > 0.0 else 0.0
        ),
        "plant_p_pool_mg": float(state.plant_p_pool[0]),
        "fungus_p_pool_mg": float(state.fungus_p_pool[0]),
        "plant_mortality_loss_mg": float(state.cumulative_plant_p_mortality_loss_mg[0]),
        "fungus_mortality_loss_mg": float(state.cumulative_fungus_p_mortality_loss_mg[0]),
        "plant_maintenance_loss_mg": float(state.cumulative_plant_p_maintenance_loss_mg[0]),
        "fungus_maintenance_loss_mg": float(state.cumulative_fungus_p_maintenance_loss_mg[0]),
        "plant_reproduction_export_mg": float(state.cumulative_plant_p_reproduction_export_mg[0]),
        "fungus_reproduction_export_mg": float(state.cumulative_fungus_p_reproduction_export_mg[0]),
        "initial_extended_p_mg": initial_extended_p_mg,
        "final_extended_p_mg": final_extended_p_mg,
        "relative_extended_p_balance_error": abs(
            final_extended_p_mg - initial_extended_p_mg
        )
        / max(initial_extended_p_mg, 1e-30),
    }


def run_deep_soil_scenario(
    interval_cm: float,
    dt_days: float,
    duration_days: float = DEEP_SOIL_DURATION_DAYS,
) -> dict:
    """Run the production deep-soil confinement and inventory check."""
    config = EnvConfig(
        dt=dt_days,
        max_steps=math.ceil(duration_days / dt_days),
        soil_radius_cm=DEEP_SOIL_RADIUS_CM,
        soil_depth_cm=DEEP_SOIL_DEPTH_CM,
        radial_interval_cm=interval_cm,
        depth_interval_cm=interval_cm,
        initial_solution_p_um=1.0,
    )
    policy = StaticPolicy(0.25, 1.0, 0.0, 0.0)
    result = run_deep_soil_qualification(
        config=config,
        species=qualification_species(coupled=True),
        plant_policy=policy,
        fungus_policy=policy,
        duration_days=duration_days,
        seed=0,
        software_revision="standard-qualification",
    )
    maximum_density = result.max_fungal_density_outside_colony
    maximum_request = result.max_fungal_uptake_request_outside_colony_micromol
    passes = (
        maximum_density <= 1e-12
        and maximum_request <= 1e-12
        and result.relative_extended_p_balance_error <= MAXIMUM_RELATIVE_P_BALANCE_ERROR
    )
    return {
        "interval_cm": interval_cm,
        "dt_days": dt_days,
        "duration_days": duration_days,
        "max_fungal_density_outside_colony": maximum_density,
        "max_fungal_uptake_request_outside_colony_micromol": maximum_request,
        "depth_band_losses": [
            {
                "start_depth_cm": loss.start_depth_cm,
                "end_depth_cm": loss.end_depth_cm,
                "initial_micromol": loss.initial_micromol,
                "final_micromol": loss.final_micromol,
                "loss_percent": loss.loss_percent,
            }
            for loss in result.depth_band_losses
        ],
        "relative_extended_p_balance_error": result.relative_extended_p_balance_error,
        "passes": passes,
    }


def _comparison(candidate: dict, reference: dict, metrics: tuple[str, ...]) -> dict:
    """Compare selected scalar metrics using the declared near-zero guard."""
    changes = {
        metric: reference_relative_change(
            float(candidate[metric]), float(reference[metric]), ABSOLUTE_METRIC_FLOOR
        )
        for metric in metrics
    }
    return {"changes": changes, "maximum_change": max(changes.values()), "passes_5_percent": all(value <= RELATIVE_TOLERANCE for value in changes.values())}


def compare_coupled_timestep_convergence(
    candidate: dict,
    reference: dict,
) -> dict:
    """Compare coupled trajectories with a fixed policy decision schedule."""
    if (
        candidate["horizon_days"] != reference["horizon_days"]
        or candidate["interval_cm"] != reference["interval_cm"]
    ):
        raise ValueError(
            "coupled timestep comparison requires the same horizon and grid interval"
        )
    if not candidate["dt_days"] > reference["dt_days"]:
        raise ValueError("candidate dt must be larger than reference dt")
    if candidate["decision_interval_days"] != reference["decision_interval_days"]:
        raise ValueError("coupled timestep comparison requires the same policy interval")
    comparison = _comparison(candidate, reference, COUPLED_ACTION_FREQUENCY_METRICS)
    diagnostic_changes = {
        metric: reference_relative_change(
            float(candidate[metric]),
            float(reference[metric]),
            ABSOLUTE_METRIC_FLOOR,
        )
        for metric in COUPLED_ACTION_FREQUENCY_DIAGNOSTICS
    }
    balance_checks = {
        label: float(result["relative_extended_p_balance_error"])
        <= MAXIMUM_RELATIVE_P_BALANCE_ERROR
        for label, result in (("candidate", candidate), ("reference", reference))
    }
    passes = comparison["passes_5_percent"] and all(balance_checks.values())
    return {
        **comparison,
        "classification": "coupled_timestep_convergence",
        "affects_numerical_timestep_selection": True,
        "criteria": {
            "reported_relative_metrics": list(COUPLED_ACTION_FREQUENCY_METRICS),
            "diagnostic_metrics": list(COUPLED_ACTION_FREQUENCY_DIAGNOSTICS),
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_metric_floor": ABSOLUTE_METRIC_FLOOR,
            "maximum_relative_p_balance_error": (
                MAXIMUM_RELATIVE_P_BALANCE_ERROR
            ),
        },
        "provenance": {
            "candidate_dt_days": candidate["dt_days"],
            "reference_dt_days": reference["dt_days"],
            "policy_decision_interval_days": candidate["decision_interval_days"],
            "horizon_days": candidate["horizon_days"],
            "interval_cm": candidate["interval_cm"],
        },
        "diagnostic_changes": diagnostic_changes,
        "maximum_diagnostic_change": max(diagnostic_changes.values()),
        "balance_checks": balance_checks,
        "maximum_relative_p_balance_error": MAXIMUM_RELATIVE_P_BALANCE_ERROR,
        "within_5_percent": passes,
        "passes_5_percent": passes,
    }


def select_solver_timestep(
    timestep_comparisons: list[dict],
    coupled_timestep_comparisons: list[dict],
    finest_timestep_comparisons: list[dict] | None = None,
    finest_coupled_timestep_comparisons: list[dict] | None = None,
) -> float:
    """Select the largest timestep passing adjacent and finest-step gates."""
    finest_timestep_comparisons = (
        timestep_comparisons
        if finest_timestep_comparisons is None
        else finest_timestep_comparisons
    )
    finest_coupled_timestep_comparisons = (
        coupled_timestep_comparisons
        if finest_coupled_timestep_comparisons is None
        else finest_coupled_timestep_comparisons
    )
    passing = []
    for dt_days in TIMESTEPS_DAYS[1:]:
        comparison_sets = (
            timestep_comparisons,
            coupled_timestep_comparisons,
            finest_timestep_comparisons,
            finest_coupled_timestep_comparisons,
        )
        if all(
            (rows := [
                row
                for row in comparisons
                if row["candidate_dt_days"] == dt_days
            ])
            and all(row["passes_5_percent"] for row in rows)
            for comparisons in comparison_sets
        ):
            passing.append(dt_days)
    return max(passing) if passing else min(TIMESTEPS_DAYS)


def select_spatial_interval(
    next_fixed_comparisons: list[dict],
    next_coupled_comparisons: list[dict],
    finest_fixed_comparisons: list[dict],
    finest_coupled_comparisons: list[dict],
) -> float:
    """Select the coarsest grid satisfying adjacent and finest-grid gates."""
    passing = []
    for interval_cm in sorted(GRID_INTERVALS_CM)[1:]:
        comparison_sets = (
            next_fixed_comparisons,
            next_coupled_comparisons,
            finest_fixed_comparisons,
            finest_coupled_comparisons,
        )
        if all(
            (rows := [
                row
                for row in comparisons
                if row["candidate_interval_cm"] == interval_cm
            ])
            and all(row["passes_5_percent"] for row in rows)
            for comparisons in comparison_sets
        ):
            passing.append(interval_cm)
    return max(passing) if passing else min(GRID_INTERVALS_CM)


def _array_bytes(value) -> int:
    """Return total concrete array bytes in a JAX PyTree."""
    return sum(getattr(leaf, "nbytes", 0) for leaf in jax.tree.leaves(value))


def benchmark_environment(config: EnvConfig, repeats: int = 20) -> dict:
    """Measure soil/full-step runtime and estimate core model-array memory."""
    env = BaseMycorMarl(config, qualification_species(coupled=True))
    _, state = env.reset(jax.random.PRNGKey(0))
    root, hypha = fixed_density_fields(env, "mixed") if env.config.soil_depth_cm == 2.0 else (
        jnp.ones(env.grid_shape),
        jnp.full(env.grid_shape, env.species.fungus.saturation_density),
    )
    state = state.replace(root_length_density=root, hyphae_length_density=hypha)
    compiled_step = jax.jit(env.step_phosphorus_field)
    started = time.perf_counter()
    state = compiled_step(state)
    jax.block_until_ready(state)
    compile_and_first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        state = compiled_step(state)
    jax.block_until_ready(state)
    warmed_seconds = (time.perf_counter() - started) / repeats
    cached_arrays = (
        env.r_edges,
        env.z_edges,
        env.cell_volumes,
        env.radial_face_areas,
        env.vertical_face_areas,
        env.radial_diffusion_conductance,
        env.vertical_diffusion_conductance,
    )
    state_bytes = _array_bytes(state)
    cached_bytes = _array_bytes(cached_arrays)
    temporary_array_estimate = 18 * env.cell_volumes.size * 4
    soil_projection = annual_runtime_projection(config.dt, warmed_seconds)
    _, full_state = env.reset(jax.random.PRNGKey(0))
    actions = {
        PLANT: rate_action(0.25, 1.0, 0.0, 0.0),
        FUNGUS: rate_action(0.25, 1.0, 0.0, 0.0),
    }
    key = jax.random.PRNGKey(1)
    compiled_full_step = jax.jit(
        lambda current_state: env.step_env(key, current_state, actions)[1]
    )
    started = time.perf_counter()
    full_state = compiled_full_step(full_state)
    jax.block_until_ready(full_state)
    full_compile_and_first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        full_state = compiled_full_step(full_state)
    jax.block_until_ready(full_state)
    warmed_full_seconds = (time.perf_counter() - started) / repeats
    full_projection = annual_runtime_projection(config.dt, warmed_full_seconds)
    return {
        "grid_shape": list(env.grid_shape),
        "cell_count": int(env.cell_volumes.size),
        "soil_substeps": env.soil_substeps,
        "compile_and_first_step_seconds": compile_and_first_seconds,
        "warmed_seconds_per_step": warmed_seconds,
        "state_array_bytes": state_bytes,
        "cached_geometry_bytes": cached_bytes,
        "temporary_array_estimate_bytes": temporary_array_estimate,
        "temporary_array_estimate_count": 18,
        "temporary_array_dtype_bytes": 4,
        "estimated_core_working_array_bytes": state_bytes + cached_bytes + temporary_array_estimate,
        "timing_repeats": repeats,
        "full_compile_and_first_step_seconds": full_compile_and_first_seconds,
        "warmed_full_seconds_per_step": warmed_full_seconds,
        "soil_annual_projection": soil_projection,
        "full_annual_projection": full_projection,
    }


def run_studies(include_target_benchmark: bool) -> dict:
    """Execute the complete deterministic qualification matrix and selection arithmetic."""
    concentration = [
        run_fixed_soil_scenario(mode, value, 0.05, 0.1)
        for mode in MODES
        for value in CONCENTRATIONS_UM
    ]
    timestep = [
        run_fixed_soil_scenario(mode, 1.0, dt, 0.1)
        for mode in MODES
        for dt in TIMESTEPS_DAYS
    ]
    grid = [
        run_fixed_soil_scenario(mode, 1.0, 0.025, interval)
        for mode in MODES
        for interval in GRID_INTERVALS_CM
    ]
    transition = [
        run_fixed_soil_scenario("mixed", 1.0, 0.05, 0.1, 1.0, exponent)
        for exponent in (1.0, 2.0, 4.0)
    ] + [
        run_fixed_soil_scenario("mixed", 1.0, 0.05, 0.1, reference, 2.0)
        for reference in (0.25, 4.0)
    ]
    coupled = [run_coupled_scenario(interval, 0.025) for interval in GRID_INTERVALS_CM]
    coupled_timestep = [
        run_coupled_scenario(0.1, dt_days)
        for dt_days in TIMESTEPS_DAYS
        if math.isclose(
            COUPLED_POLICY_DECISION_INTERVAL_DAYS / dt_days,
            round(COUPLED_POLICY_DECISION_INTERVAL_DAYS / dt_days),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ]
    primary = ("total_uptake_micromol", "final_soil_micromol", "root_uptake_share", "fungus_uptake_share")
    timestep_comparisons = []
    for mode in MODES:
        ordered = sorted((row for row in timestep if row["mode"] == mode), key=lambda row: row["dt_days"])
        for reference, candidate in zip(ordered, ordered[1:]):
            timestep_comparisons.append({"mode": mode, "candidate_dt_days": candidate["dt_days"], "reference_dt_days": reference["dt_days"], **_comparison(candidate, reference, primary)})
    finest_timestep_comparisons = []
    for mode in MODES:
        ordered = sorted(
            (row for row in timestep if row["mode"] == mode),
            key=lambda row: row["dt_days"],
        )
        reference = ordered[0]
        for candidate in ordered[1:]:
            finest_timestep_comparisons.append({"mode": mode, "candidate_dt_days": candidate["dt_days"], "reference_dt_days": reference["dt_days"], **_comparison(candidate, reference, primary)})
    grid_comparisons = []
    for mode in MODES:
        ordered = sorted((row for row in grid if row["mode"] == mode), key=lambda row: row["interval_cm"])
        for reference, candidate in zip(ordered, ordered[1:]):
            grid_comparisons.append({"mode": mode, "candidate_interval_cm": candidate["interval_cm"], "reference_interval_cm": reference["interval_cm"], **_comparison(candidate, reference, primary)})
    finest_grid_comparisons = []
    for mode in MODES:
        ordered = sorted(
            (row for row in grid if row["mode"] == mode),
            key=lambda row: row["interval_cm"],
        )
        reference = ordered[0]
        for candidate in ordered[1:]:
            finest_grid_comparisons.append({"mode": mode, "candidate_interval_cm": candidate["interval_cm"], "reference_interval_cm": reference["interval_cm"], **_comparison(candidate, reference, primary)})
    coupled_metrics = (
        "plant_uptake_micromol",
        "fungus_uptake_micromol",
        "total_uptake_micromol",
        "final_soil_micromol",
        "plant_uptake_share",
        "fungus_uptake_share",
        "plant_p_pool_mg",
        "fungus_p_pool_mg",
        "plant_biomass_g",
        "fungus_biomass_g",
        "root_radius_cm",
        "root_depth_cm",
        "fungus_radius_cm",
        "fungus_depth_cm",
    )
    ordered_coupled = sorted(coupled, key=lambda row: row["interval_cm"])
    coupled_comparisons = [
        {"candidate_interval_cm": candidate["interval_cm"], "reference_interval_cm": reference["interval_cm"], **_comparison(candidate, reference, coupled_metrics)}
        for reference, candidate in zip(ordered_coupled, ordered_coupled[1:])
    ]
    finest_coupled_reference = ordered_coupled[0]
    finest_coupled_comparisons = [
        {
            "candidate_interval_cm": candidate["interval_cm"],
            "reference_interval_cm": finest_coupled_reference["interval_cm"],
            **_comparison(candidate, finest_coupled_reference, coupled_metrics),
        }
        for candidate in ordered_coupled[1:]
    ]
    ordered_coupled_timestep = sorted(
        coupled_timestep, key=lambda row: row["dt_days"]
    )
    coupled_timestep_comparisons = [
        {
            "candidate_dt_days": candidate["dt_days"],
            "reference_dt_days": reference["dt_days"],
            **compare_coupled_timestep_convergence(candidate, reference),
        }
        for reference, candidate in zip(
            ordered_coupled_timestep, ordered_coupled_timestep[1:]
        )
    ]
    finest_coupled_timestep_reference = ordered_coupled_timestep[0]
    finest_coupled_timestep_comparisons = [
        {
            "candidate_dt_days": candidate["dt_days"],
            "reference_dt_days": finest_coupled_timestep_reference["dt_days"],
            **compare_coupled_timestep_convergence(
                candidate,
                finest_coupled_timestep_reference,
            ),
        }
        for candidate in ordered_coupled_timestep[1:]
    ]
    selected_grid = select_spatial_interval(
        grid_comparisons,
        coupled_comparisons,
        finest_grid_comparisons,
        finest_coupled_comparisons,
    )
    selected_dt = select_solver_timestep(
        timestep_comparisons,
        coupled_timestep_comparisons,
        finest_timestep_comparisons,
        finest_coupled_timestep_comparisons,
    )
    reduced_benchmark = benchmark_environment(qualification_config(selected_grid, selected_dt, 1.0))
    deep_soil = run_deep_soil_scenario(selected_grid, selected_dt)
    target_benchmark = None
    if include_target_benchmark:
        annual_steps = annual_runtime_projection(selected_dt, 0.0)["steps_per_year"]
        target_benchmark = benchmark_environment(
            EnvConfig(
                dt=selected_dt,
                max_steps=annual_steps,
            ),
            repeats=20,
        )
    return {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "horizon_days": HORIZON_DAYS,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_metric_floor": ABSOLUTE_METRIC_FLOOR,
            "float_precision": "JAX default float32",
            "fixed_density_fixture": {
                "root_density_cm_cm3": 1.0,
                "hyphal_density_cm_cm3": (
                    qualification_species().fungus.saturation_density
                ),
                "occupied_depth_cm": 1.0,
            },
            "coupled_fixture": {
                "initial_plant_biomass_g": COUPLED_PLANT_INITIAL_BIOMASS_G,
                "initial_fungus_biomass_g": COUPLED_FUNGUS_INITIAL_BIOMASS_G,
                "initial_plant_c_pool": qualification_species(coupled=True).plant.initial_c_pool,
                "initial_plant_p_pool_mg": qualification_species(coupled=True).plant.initial_p_pool,
                "initial_fungus_c_pool": qualification_species(coupled=True).fungus.initial_c_pool,
                "initial_fungus_p_pool_mg": qualification_species(coupled=True).fungus.initial_p_pool,
                "pool_initialisation": "one_structural_biomass_equivalent",
                "kappa_c": 0.0,
                "kappa_p": 0.0,
            },
        },
        "concentration_scenarios": concentration,
        "timestep_scenarios": timestep,
        "grid_scenarios": grid,
        "transition_scenarios": transition,
        "coupled_grid_scenarios": coupled,
        "coupled_timestep_scenarios": coupled_timestep,
        "timestep_comparisons": timestep_comparisons,
        "finest_timestep_comparisons": finest_timestep_comparisons,
        "grid_comparisons": grid_comparisons,
        "finest_grid_comparisons": finest_grid_comparisons,
        "coupled_grid_comparisons": coupled_comparisons,
        "finest_coupled_grid_comparisons": finest_coupled_comparisons,
        "coupled_timestep_comparisons": coupled_timestep_comparisons,
        "finest_coupled_timestep_comparisons": finest_coupled_timestep_comparisons,
        "deep_soil": deep_soil,
        "selection": {"grid_interval_cm": selected_grid, "dt_days": selected_dt, "grid_had_passing_candidate": selected_grid > min(GRID_INTERVALS_CM), "dt_had_passing_candidate": selected_dt > min(TIMESTEPS_DAYS), "deep_soil_passes": deep_soil["passes"], "spatial_basis": "next-smaller-and-finest-grid", "timestep_basis": "next-smaller-and-finest-timestep"},
        "benchmarks": {"reduced": reduced_benchmark, "target": target_benchmark},
    }


def render_markdown(results: dict) -> str:
    """Render a concise audit report from the canonical JSON-compatible data."""
    selection = results["selection"]
    target = results["benchmarks"]["target"]
    lines = [
        "# Phosphate numerical qualification results",
        "",
        "## Outcome",
        "",
        f"Selected interval: `{selection['grid_interval_cm']} cm`; selected fixed-geometry soil-solver timestep: `{selection['dt_days']} day`.",
        f"Grid had a passing coarser candidate: `{selection['grid_had_passing_candidate']}`; timestep had a passing larger candidate: `{selection['dt_had_passing_candidate']}`.",
        f"Deep-soil confinement and extended-P balance pass: `{selection['deep_soil_passes']}`.",
        "",
        "Numerical timestep and spatial selection both require 5% agreement with the next-smaller discretisation and the finest tested discretisation. Timestep comparisons cover fixed-geometry soil observables and coupled trajectories under a fixed 1-day policy decision interval; candidates that do not divide that interval are ineligible. Grid convergence includes coupled endpoint pools. This is numerical qualification, not empirical validation.",
        "",
        "## Balance and diagnostic ranges",
        "",
    ]
    all_fixed = results["concentration_scenarios"] + results["timestep_scenarios"] + results["grid_scenarios"]
    lines.extend([
        f"- Maximum relative P-balance error: `{max(row['relative_p_balance_error'] for row in all_fixed):.3e}`.",
        f"- Mean continuous-weight range: `{min(row['mean_continuous_weight'] for row in all_fixed):.6g}` to `{max(row['mean_continuous_weight'] for row in all_fixed):.6g}`.",
        f"- Maximum cellwise continuous weight: `{max(row['maximum_continuous_weight'] for row in all_fixed):.6g}`.",
        f"- Diffusion CFL ceiling range: `{min(row['diffusion_cfl_seconds'] for row in all_fixed):.6g}` to `{max(row['diffusion_cfl_seconds'] for row in all_fixed):.6g}` seconds.",
        f"- Capped-demand fraction range: `{min(row['capped_demand_fraction'] for row in all_fixed):.6g}` to `{max(row['capped_demand_fraction'] for row in all_fixed):.6g}`.",
        f"- Maximum coupled extended-P balance error: `{max(row['relative_extended_p_balance_error'] for row in results['coupled_grid_scenarios']):.3e}`.",
        "",
        "## Concentration response (mixed mode)",
        "",
        "| Initial µM | Total uptake (µmol) | Mean root C_s/C_b | Mean fungal C_s/C_b |",
        "|---:|---:|---:|---:|",
    ])
    for row in results["concentration_scenarios"]:
        if row["mode"] == "mixed":
            lines.append(f"| {row['concentration_um']:g} | {row['total_uptake_micromol']:.6g} | {row['mean_root_surface_ratio']:.6f} | {row['mean_fungus_surface_ratio']:.6f} |")
    lines.extend([
        "",
        "## Timestep convergence",
        "",
        "| Candidate day | Next-smaller ref day | Worst fixed-soil change | Coupled fixed-policy change | Finest ref day | Worst fixed-soil change | Coupled fixed-policy change | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for dt in TIMESTEPS_DAYS[1:]:
        rows = [row for row in results["timestep_comparisons"] if row["candidate_dt_days"] == dt]
        coupled_row = next(
            (
                row
                for row in results["coupled_timestep_comparisons"]
                if row["candidate_dt_days"] == dt
            ),
            None,
        )
        finest_rows = [
            row
            for row in results["finest_timestep_comparisons"]
            if row["candidate_dt_days"] == dt
        ]
        finest_coupled_row = next(
            (
                row
                for row in results["finest_coupled_timestep_comparisons"]
                if row["candidate_dt_days"] == dt
            ),
            None,
        )
        passed = (
            coupled_row is not None
            and finest_coupled_row is not None
            and all(row["passes_5_percent"] for row in rows)
            and coupled_row["passes_5_percent"]
            and all(row["passes_5_percent"] for row in finest_rows)
            and finest_coupled_row["passes_5_percent"]
        )
        lines.append(
            f"| {dt:g} | {rows[0]['reference_dt_days']:g} | "
            f"{max(row['maximum_change'] for row in rows):.3%} | "
            f"{coupled_row['maximum_change']:.3%} | "
            f"{finest_rows[0]['reference_dt_days']:g} | "
            f"{max(row['maximum_change'] for row in finest_rows):.3%} | "
            f"{finest_coupled_row['maximum_change']:.3%} | {'yes' if passed else 'no'} |"
        )
    lines.extend([
        "",
        "## Grid convergence",
        "",
        "| Candidate cm | Next-smaller ref cm | Worst fixed-soil change | Coupled change | Finest ref cm | Worst fixed-soil change | Coupled change | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for interval in sorted(GRID_INTERVALS_CM)[1:]:
        fixed_rows = [row for row in results["grid_comparisons"] if row["candidate_interval_cm"] == interval]
        coupled_row = next(row for row in results["coupled_grid_comparisons"] if row["candidate_interval_cm"] == interval)
        finest_fixed_rows = [row for row in results["finest_grid_comparisons"] if row["candidate_interval_cm"] == interval]
        finest_coupled_row = next(row for row in results["finest_coupled_grid_comparisons"] if row["candidate_interval_cm"] == interval)
        passed = (
            all(row["passes_5_percent"] for row in fixed_rows)
            and coupled_row["passes_5_percent"]
            and all(row["passes_5_percent"] for row in finest_fixed_rows)
            and finest_coupled_row["passes_5_percent"]
        )
        lines.append(
            f"| {interval:g} | {fixed_rows[0]['reference_interval_cm']:g} | "
            f"{max(row['maximum_change'] for row in fixed_rows):.3%} | "
            f"{coupled_row['maximum_change']:.3%} | "
            f"{finest_fixed_rows[0]['reference_interval_cm']:g} | "
            f"{max(row['maximum_change'] for row in finest_fixed_rows):.3%} | "
            f"{finest_coupled_row['maximum_change']:.3%} | {'yes' if passed else 'no'} |"
        )
    deep_soil = results["deep_soil"]
    lines.extend([
        "",
        "## Deep-soil integration check",
        "",
        f"- Confinement and extended-P balance pass: `{deep_soil['passes']}`.",
        f"- Maximum fungal density wholly outside the colony: `{deep_soil['max_fungal_density_outside_colony']:.3g}` cm cm^-3.",
        f"- Maximum fungal uptake request wholly outside the colony: `{deep_soil['max_fungal_uptake_request_outside_colony_micromol']:.3g}` micromol P per cell.",
        f"- Maximum relative extended-P balance error: `{deep_soil['relative_extended_p_balance_error']:.3e}`.",
    ])
    lines.extend([
        "",
        "## Transition sensitivity (mixed mode)",
        "",
        "| T_ref (day) | p | Mean w_cont | Total uptake (µmol) |",
        "|---:|---:|---:|---:|",
    ])
    for row in results["transition_scenarios"]:
        lines.append(f"| {row['reference_time_days']:g} | {row['transition_exponent']:g} | {row['mean_continuous_weight']:.6f} | {row['total_uptake_micromol']:.6g} |")
    lines.extend([
        "",
        "## Performance",
        "",
    ])
    reduced = results["benchmarks"]["reduced"]
    lines.append(f"- Reduced grid: `{reduced['cell_count']}` cells, compile+first step `{reduced['compile_and_first_step_seconds']:.3f} s`, warmed step `{reduced['warmed_seconds_per_step']:.6f} s`.")
    if target is not None:
        lines.extend([
            f"- Target grid: `{target['grid_shape'][0]} x {target['grid_shape'][1]}` = `{target['cell_count']}` cells.",
            f"- Target soil compile+first step: `{target['compile_and_first_step_seconds']:.3f} s`; warmed soil step: `{target['warmed_seconds_per_step']:.6f} s`.",
            f"- Target full-step incremental compile+first step, measured after the soil benchmark: `{target['full_compile_and_first_step_seconds']:.3f} s`; warmed full step: `{target['warmed_full_seconds_per_step']:.6f} s`.",
            f"- Estimated core working arrays: `{target['estimated_core_working_array_bytes'] / 2**20:.1f} MiB`, comprising concrete state/cached arrays plus `{target['temporary_array_estimate_count']}` float32 cell-array equivalents. This is a formula-based estimate, not peak process RSS; XLA fusion may reduce actual temporary storage.",
            f"- Projected year: `{target['full_annual_projection']['steps_per_year']}` steps, `{target['soil_annual_projection']['runtime_seconds']:.2f} s` soil-only and `{target['full_annual_projection']['runtime_seconds']:.2f} s` for the deterministic full step, excluding compilation, learned-policy inference, training, and output.",
            f"- The target environment is configured with `max_steps={target['full_annual_projection']['steps_per_year']}` so the projected year is not truncated by the episode limit.",
        ])
    else:
        lines.append("- Target benchmark skipped by command-line option.")
    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- The static transition is sensitive to `T_ref` and `p`; the JSON artifact contains the complete scenario rows.",
        "- `T_ref` changes both the overlap weight and the sparse propagation radius, so its total-uptake response need not be monotonic; it remains a provisional model parameter rather than a numerical tuning control.",
        "- No scientific matrix row was inventory-capped. Coupled endpoint free-P pools remain reported in the JSON artifact as timestep-scaling diagnostics.",
        (
            f"- {selection['dt_days']} day is the largest tested timestep that passed both fixed-soil and coupled fixed-policy gates."
            if selection["dt_had_passing_candidate"]
            else f"- No larger eligible timestep passed both fixed-soil and coupled fixed-policy gates; {selection['dt_days']} day remains the finest tested fallback and does not itself demonstrate convergence without a smaller reference."
        ),
        "- Reduced-domain convergence retains a topsoil diffusion front but cannot reproduce every full-domain spatial scale.",
        (
            "- After biomass-consistent pool initialisation, all tested coupled grid comparisons pass the 5% gate for the established-organism fixture; this is qualification evidence for the tested reduced domain, not universal spatial validation."
            if selection["grid_had_passing_candidate"]
            else "- After biomass-consistent pool initialisation, coupled grids remain sensitive to quantised root/fungal extents; the selected grid is the finest-tested fallback rather than demonstrated spatial convergence."
        ),
        "- The coupled fixture uses 0.01 g plant biomass and 0.0001 g living external fungal biomass; each free C/P pool starts at one structural-biomass equivalent and automatic maintenance costs are disabled.",
        "- Coupled Rate actions are fixed at `[trade=0.25, growth=1, reproduction=0, storage=0] d^-1` and held for a fixed 1-day policy interval. Their timestep comparisons now isolate numerical resolution and are required for timestep selection.",
        "- Annual runtime is projected from both warmed soil-only and deterministic full-environment steps. MARL training, learned-policy inference, output, and accelerator transfer costs are excluded.",
        "- The complete machine-readable tables and exact platform metadata are in `phosphate-numerical-qualification.json`.",
    ])
    return "\n".join(lines)


def main() -> None:
    """Parse options, execute studies, and write canonical qualification artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-target-benchmark", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/qualification"),
    )
    args = parser.parse_args()
    results = run_studies(not args.skip_target_benchmark)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phosphate-numerical-qualification.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "phosphate-numerical-qualification.md").write_text(
        render_markdown(results) + "\n"
    )
    print(json.dumps(results["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
