"""Temporary legacy manifest adapter; new studies use redbyrbd.static_controls.

Numerical execution is owned by mycormarl.static_rollout. Removed at final cutover.
"""

from __future__ import annotations

from dataclasses import replace
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
_POOL_TOLERANCE = 1e-6
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
        initial_solution_p_depth_profile=model.get("initial_solution_p_depth_profile"),
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
        raise ValueError(f"invalid Rate action for {agent}") from error
    if action.shape != (4,) or not bool(jnp.all(jnp.isfinite(action))):
        raise ValueError(f"invalid Rate action for {agent}")
    if bool(jnp.any(action < 0.0)):
        raise ValueError(f"invalid Rate action for {agent}")
    return action


def _actions(policy: Any) -> dict[str, jax.Array]:
    if not isinstance(policy, dict) or any(agent not in policy for agent in _AGENTS):
        raise ValueError("static_policy must declare plant and fungus actions")
    return {agent: _action(policy[agent], agent) for agent in _AGENTS}


from mycormarl.static_rollout import evaluate_static_policy, evaluate_batched_static_policy

def _condition(manifest, mode, p_level, seed):
    try:
        actions = _actions(manifest["static_policy"])
        env = _environment(manifest, mode, p_level)
    except ValueError as error:
        return {"mode": mode, "initial_p_micromolar": p_level, "seed": seed, "status": "rejected", "rejection_reasons": [str(error)]}
    return evaluate_static_policy(env, actions, seed=seed,
        record_limitation_trace=bool(manifest.get("record_limitation_trace", False)),
        record_resource_accounting=bool(manifest.get("record_resource_accounting", False)))

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
