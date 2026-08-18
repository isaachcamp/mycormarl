"""Deterministic static-policy controls for uniform initial-P treatments."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import jax
import jax.numpy as jnp

from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_units import MICROMOL_P_TO_MG_P
from mycormarl.soil.phosphate_grid import labile_amount_to_solution_concentration


_AGENTS = (PLANT, FUNGUS)
_ACTION_TOLERANCE = 1e-6


def _species(manifest: dict[str, Any]) -> SpeciesParams:
    declarations = manifest.get("model", {}).get("species", {})
    plant = PlantTraits()
    fungus = FungusTraits()
    for name, values, current in ((PLANT, declarations.get(PLANT, {}), plant), (FUNGUS, declarations.get(FUNGUS, {}), fungus)):
        if not isinstance(values, dict):
            raise ValueError(f"species {name} must be an object")
        unknown = set(values) - set(current.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {name} species traits: {sorted(unknown)}")
        try:
            updated = replace(current, **values)
        except TypeError as error:
            raise ValueError(f"invalid {name} species traits") from error
        if name == PLANT:
            plant = updated
        else:
            fungus = updated
    return SpeciesParams(plant=plant, fungus=fungus)


def _environment(manifest: dict[str, Any], mode: str, p_level: float) -> BaseMycorMarl:
    horizon = manifest["horizon"]
    model = manifest["model"]["environment"]
    config = EnvConfig(
        max_steps=round(horizon["days"] / horizon["timestep_days"]),
        dt=horizon["timestep_days"],
        consumer_mode=mode,
        initial_solution_p_um=p_level,
        topsoil_depth_cm=model.get("topsoil_depth_cm", model.get("soil_depth_cm", 1.0)),
        **{key: model[key] for key in (
            "soil_radius_cm", "soil_depth_cm", "radial_interval_cm",
            "depth_interval_cm", "b_p", "theta_water",
        ) if key in model},
    )
    return BaseMycorMarl(config=config, species=_species(manifest))


def _action(value: Any, agent: str) -> jax.Array:
    try:
        action = jnp.asarray(value, dtype=jnp.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid physical action for {agent}") from error
    if action.shape != (4,) or not bool(jnp.all(jnp.isfinite(action))):
        raise ValueError(f"invalid physical action for {agent}")
    if not (0.0 <= float(action[0]) <= 1.0):
        raise ValueError(f"invalid physical action for {agent}")
    if bool(jnp.any(action[1:] < 0.0)) or not math.isclose(float(jnp.sum(action[1:])), 1.0, abs_tol=_ACTION_TOLERANCE):
        raise ValueError(f"invalid physical action for {agent}")
    return action


def _actions(policy: Any) -> dict[str, jax.Array]:
    if not isinstance(policy, dict) or any(agent not in policy for agent in _AGENTS):
        raise ValueError("static_policy must declare plant and fungus actions")
    return {agent: _action(policy[agent], agent) for agent in _AGENTS}


def _condition(manifest: dict[str, Any], mode: str, p_level: float, seed: int) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        actions = _actions(manifest["static_policy"])
        env = _environment(manifest, mode, p_level)
    except ValueError as error:
        return {"mode": mode, "initial_p_micromolar": p_level, "seed": seed, "status": "rejected", "rejection_reasons": [str(error)]}

    _, state = env.reset(jax.random.PRNGKey(seed))
    initial_soil = float(jnp.sum(state.soil_labile_p) * MICROMOL_P_TO_MG_P)
    active_field = state.soil_labile_p > 0.0
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p,
        env.cell_volumes,
        env.config.theta_water,
        env.config.b_p,
    )
    positive_values = concentration[active_field]
    uniform = bool(jnp.all(jnp.isclose(positive_values, positive_values[0])))
    initial_p = float(
        jnp.sum(state.plant_p_pool)
        + jnp.sum(state.fungus_p_pool)
        + jnp.sum(state.plant_biomass) * env.species.plant.gamma_p
        + jnp.sum(state.fungus_biomass) * env.species.fungus.gamma_p
    )
    initial_total = initial_p + initial_soil
    steps = 0
    uptake = {PLANT: 0.0, FUNGUS: 0.0}
    transfers = {"plant_c_out": 0.0, "fungus_p_out": 0.0}
    biological_deaths = {PLANT: 0, FUNGUS: 0}
    while steps < env.max_episode_steps:
        previous = state
        _, state, _, dones, info = env.step_env(jax.random.PRNGKey(seed + steps + 1), state, actions)
        for agent, trade_out, trade_in, trade_out_key, trade_in_key in (
            (PLANT, "trade_out", "trade_in", "plant_c_out", None),
            (FUNGUS, "trade_out", "trade_in", "fungus_p_out", None),
        ):
            diagnostics = info[agent]
            uptake[agent] += float(
                state.__getattribute__(f"{agent}_p_pool")[0]
                - previous.__getattribute__(f"{agent}_p_pool")[0]
                + diagnostics["maint_p_used"][0]
                + diagnostics["growth_p_used"][0]
                + diagnostics["reproduction_p"][0]
                + (diagnostics[trade_out][0] if agent == FUNGUS else 0.0)
                - diagnostics[trade_in][0]
            )
            transfers[trade_out_key] += float(diagnostics[trade_out][0])
        for agent in _AGENTS:
            biological_deaths[agent] += int(
                bool(info["transitions"][agent].operational_at_start)
                and not bool(info["transitions"][agent].operational_at_end)
            )
        steps += 1
        if bool(dones["__all__"]):
            break

    final_soil = float(jnp.sum(state.soil_labile_p) * MICROMOL_P_TO_MG_P)
    final_pools = float(
        jnp.sum(state.plant_p_pool)
        + jnp.sum(state.fungus_p_pool)
        + jnp.sum(jnp.where(state.plant_dead, 0.0, state.plant_biomass))
        * env.species.plant.gamma_p
        + jnp.sum(jnp.where(state.fungus_dead, 0.0, state.fungus_biomass))
        * env.species.fungus.gamma_p
    )
    losses = float(sum(float(jnp.sum(getattr(state, field))) for field in (
        "cumulative_plant_p_mortality_loss_mg", "cumulative_fungus_p_mortality_loss_mg",
        "cumulative_plant_p_maintenance_loss_mg", "cumulative_fungus_p_maintenance_loss_mg",
        "cumulative_plant_p_reproduction_export_mg", "cumulative_fungus_p_reproduction_export_mg",
    )))
    residual = initial_total - final_soil - final_pools - losses
    if not uniform:
        reasons.append("uniform initial P verification failed")
    if float(jnp.min(state.soil_labile_p)) < -_ACTION_TOLERANCE:
        reasons.append("negative soil P pool")
    for agent, dead in ((PLANT, state.plant_dead), (FUNGUS, state.fungus_dead)):
        if bool(jnp.any(dead)) and (mode == "mixed" or agent == PLANT):
            reasons.append(f"biological failure: {agent} died")
    if any(
        bool(jnp.any(getattr(state, field) < 0.0))
        for field in ("plant_c_pool", "plant_p_pool", "fungus_c_pool", "fungus_p_pool")
    ):
        reasons.append("negative organism resource pool")
    if abs(residual) > 1e-5 * max(1.0, initial_total):
        reasons.append("resource-accounting failure: P residual is nonzero")
    if initial_soil > 0.0 and final_soil <= 1e-12:
        reasons.append("pathological depletion: soil inventory exhausted")

    p_loss_or_export_counters = {
        "plant_mortality": float(jnp.sum(state.cumulative_plant_p_mortality_loss_mg)),
        "fungus_mortality": float(jnp.sum(state.cumulative_fungus_p_mortality_loss_mg)),
        "plant_maintenance": float(jnp.sum(state.cumulative_plant_p_maintenance_loss_mg)),
        "fungus_maintenance": float(jnp.sum(state.cumulative_fungus_p_maintenance_loss_mg)),
        "plant_reproduction": float(jnp.sum(state.cumulative_plant_p_reproduction_export_mg)),
        "fungus_reproduction": float(jnp.sum(state.cumulative_fungus_p_reproduction_export_mg)),
    }
    return {
        "mode": mode, "initial_p_micromolar": p_level, "seed": seed,
        "status": "rejected" if reasons else "completed",
        "rejection_reasons": reasons,
        "steps": steps, "uniform_initial_p": uniform,
        "biomass": {PLANT: float(state.plant_biomass[0]), FUNGUS: float(state.fungus_biomass[0])},
        "c_pools": {PLANT: float(state.plant_c_pool[0]), FUNGUS: float(state.fungus_c_pool[0])},
        "p_pools": {PLANT: float(state.plant_p_pool[0]), FUNGUS: float(state.fungus_p_pool[0])},
        "uptake": uptake, "transfers": transfers, "biological_deaths": biological_deaths,
        "soil_inventory_initial": initial_soil, "soil_inventory_final": final_soil,
        "p_loss_or_export": losses,
        "p_loss_or_export_counters": p_loss_or_export_counters,
        "p_accounting_residual": residual,
        "terminated": bool(state.terminal),
    }


def run_static_controls(manifest: dict[str, Any]) -> dict[str, Any]:
    """Execute every declared mode/P/seed condition with one static action."""
    required = ("horizon", "model", "modes", "initial_p_micromolar", "seeds", "static_policy")
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError("static-control manifest missing: " + ", ".join(missing))
    if set(manifest["modes"]) - {"mixed", "plant-only"}:
        raise ValueError("static controls support only mixed and plant-only modes")
    entries = [
        _condition(manifest, mode, p_level, seed) for mode in manifest["modes"] for p_level in manifest["initial_p_micromolar"] for seed in manifest["seeds"]
    ]
    rejected = sum(entry["status"] == "rejected" for entry in entries)
    return {
        "format": "mycormarl-static-controls",
        "format_version": 1,
        "status": "rejected" if rejected else "complete",
        "entries": entries,
        "completion": {
            "completed": len(entries) - rejected,
            "requested": len(entries)
        }
    }
