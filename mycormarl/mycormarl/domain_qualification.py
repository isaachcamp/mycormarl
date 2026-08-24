"""Qualification of candidate axisymmetric soil domains."""

from __future__ import annotations

from dataclasses import replace
import json
import hashlib
import time
import tracemalloc
from typing import Any

import jax
import jax.numpy as jnp

from mycormarl.actions import rate_action
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


_DOMAIN_FIELDS = {
    "soil_radius_cm",
    "soil_depth_cm",
    "radial_interval_cm",
    "depth_interval_cm",
}


def _species(manifest: dict[str, Any]) -> SpeciesParams:
    declarations = manifest["model"].get("species", {})
    plant = replace(PlantTraits(), **declarations.get(PLANT, {}))
    fungus = replace(FungusTraits(), **declarations.get(FUNGUS, {}))
    return SpeciesParams(plant=plant, fungus=fungus)


def _action(value: Any, agent: str):
    action = jnp.asarray(value, dtype=jnp.float32)
    if action.shape != (4,) or not bool(jnp.all(jnp.isfinite(action))):
        raise ValueError(f"invalid Rate action for {agent}")
    if bool(jnp.any(action < 0.0)):
        raise ValueError(f"invalid Rate action for {agent}")
    return rate_action(*[float(item) for item in action])


def _actions(manifest: dict[str, Any]) -> dict[str, Any]:
    policy = manifest.get("static_policy")
    if not isinstance(policy, dict):
        raise ValueError("domain qualification requires static_policy")
    return {agent: _action(policy[agent], agent) for agent in (PLANT, FUNGUS)}


def _environment(
        manifest: dict[str, Any],
        candidate: dict[str, Any],
        mode: str,
        p_level: float,
        depth_profile: list[list[float]] | None,
    ) -> BaseMycorMarl:
    model = dict(manifest["model"]["environment"])
    model.update({key: value for key, value in candidate.items() if key in _DOMAIN_FIELDS})
    model.pop("initial_solution_p_depth_profile", None)
    horizon = manifest["horizon"]
    config = EnvConfig(
        max_steps=round(horizon["days"] / horizon["timestep_days"]),
        dt=horizon["timestep_days"],
        consumer_mode=mode,
        initial_solution_p_um=p_level,
        initial_solution_p_depth_profile=(
            tuple((float(depth), float(factor)) for depth, factor in depth_profile)
            if depth_profile is not None else None
        ),
        **model,
    )
    return BaseMycorMarl(config=config, species=_species(manifest))


def _trajectory(
        manifest: dict[str, Any], candidate: dict[str, Any], mode: str,
        p_level: float, seed: int, actions, depth_profile: list[list[float]] | None,
    ):
    env = _environment(manifest, candidate, mode, p_level, depth_profile)
    _, state = env.reset(jax.random.PRNGKey(seed))
    # Cell-averaged geometry can leave float32 residue in an otherwise empty
    # boundary cell. Treat only material density as contact by default; the
    # threshold remains configurable for deliberately stricter studies.
    contact_tolerance = float(manifest.get("domain_qualification", {}).get("boundary_contact_tolerance", 1e-6))
    initial_contact = jnp.any(state.hyphae_length_density[:, -1] > contact_tolerance)
    initial_soil_inventory = float(jnp.sum(state.soil_labile_p))

    def run_trajectory(initial_state):
        def advance(carry, step):
            current_state, previous_contact, first_contact = carry
            _, next_state, _, _, _ = env.step_env(
                jax.random.fold_in(jax.random.PRNGKey(seed), step + 1),
                current_state,
                actions,
            )
            current_contact = jnp.any(
                next_state.hyphae_length_density[:, -1] > contact_tolerance
            )
            first_contact = jnp.where(
                current_contact & ~previous_contact,
                step + 1,
                first_contact,
            )
            return (
                next_state,
                previous_contact | current_contact,
                first_contact,
            ), jnp.sum(next_state.soil_labile_p)

        return jax.lax.scan(
            advance,
            (initial_state, initial_contact, jnp.asarray(-1, dtype=jnp.int32)),
            jnp.arange(env.max_episode_steps),
        )

    (state, fungal_lower_contact, fungal_lower_first_contact_step), soil_trace = jax.jit(
        run_trajectory
    )(state)
    fungal_lower_contact = bool(fungal_lower_contact)
    first_contact_step = int(fungal_lower_first_contact_step)
    fungal_lower_first_contact_step = 0 if bool(initial_contact) else (
        None if first_contact_step < 0 else first_contact_step
    )
    soil_trace = [initial_soil_inventory, *map(float, soil_trace)]
    return {
        "fungal_lower_boundary_contact": fungal_lower_contact,
        "fungal_lower_boundary_first_contact_step": fungal_lower_first_contact_step,
        "initial_p_inventory_micromol": initial_soil_inventory,
        "final_plant_biomass_g": float(state.plant_biomass[0]),
        "cumulative_direct_plant_p_uptake_micromol": float(
            state.cumulative_direct_plant_p_uptake_micromol[0]
        ),
        "final_soil_inventory_micromol": float(jnp.sum(state.soil_labile_p)),
        "soil_inventory_trace_micromol": soil_trace,
        "depletion_fraction": (soil_trace[0] - soil_trace[-1]) / max(soil_trace[0], 1e-30),
    }


def _ordering(records: list[dict[str, Any]]) -> list[tuple[str, tuple[float, ...]]]:
    """Return the rank order of P responses, independent of their magnitudes."""
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_mode.setdefault(record["mode"], []).append(record)
    return [
        (mode, tuple(record["initial_p_micromolar"] for record in sorted(
            mode_records,
            key=lambda record: record["final_plant_biomass_g"],
        )))
        for mode, mode_records in sorted(by_mode.items())
    ]


def _direct_plant_uptake_difference(candidate: dict[str, Any], reference: dict[str, Any]) -> float:
    """Compare exact cumulative root-to-plant P fluxes for matched conditions."""
    candidate_records = {
        (record["mode"], record["initial_p_micromolar"], record["seed"]): record
        for record in candidate["records"]
    }
    reference_records = {
        (record["mode"], record["initial_p_micromolar"], record["seed"]): record
        for record in reference["records"]
    }
    differences = []
    for key, candidate_record in candidate_records.items():
        reference_record = reference_records[key]
        candidate_uptake = candidate_record["cumulative_direct_plant_p_uptake_micromol"]
        reference_uptake = reference_record["cumulative_direct_plant_p_uptake_micromol"]
        differences.append(
            abs(candidate_uptake - reference_uptake) / max(abs(reference_uptake), 1e-30)
        )
    return max(differences, default=0.0)


def run_domain_qualification(manifest: dict[str, Any]) -> dict[str, Any]:
    """Run candidate domains and return a frozen qualification artifact."""
    declaration = manifest.get("domain_qualification")
    if not isinstance(declaration, dict):
        raise ValueError("domain qualification requires a domain_qualification declaration")
    candidates = declaration.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("domain qualification requires at least two candidate domains")
    if any(not isinstance(candidate, dict) or not candidate.get("name") for candidate in candidates):
        raise ValueError("each domain candidate requires a name")
    model_environment = manifest["model"]["environment"]
    depth_profile = declaration.get("depth_profile", model_environment.get("initial_solution_p_depth_profile"))
    if depth_profile is not None and (not isinstance(depth_profile, list) or len(depth_profile) < 2):
        raise ValueError("profile domain qualification requires a depth_profile with at least two knots")
    actions = _actions(manifest)
    if any("soil_depth_cm" not in candidate for candidate in candidates):
        raise ValueError("each domain candidate requires soil_depth_cm")
    depths = [float(candidate["soil_depth_cm"]) for candidate in candidates]
    if any(not jnp.isfinite(depth) or depth <= 0.0 for depth in depths):
        raise ValueError("candidate soil_depth_cm values must be finite and positive")
    if any(later <= earlier for earlier, later in zip(depths, depths[1:])):
        raise ValueError("domain candidates must be ordered by strictly increasing soil_depth_cm")

    outputs = []
    tracemalloc.start()
    try:
        for candidate in candidates:
            started = time.perf_counter()
            records = []
            for mode in manifest["modes"]:
                for p_level in manifest["initial_p_micromolar"]:
                    for seed in manifest["seeds"]:
                        records.append({
                            "mode": mode,
                            "initial_p_micromolar": p_level,
                            "seed": seed,
                            **_trajectory(
                                manifest, candidate, mode, p_level, seed,
                                actions, depth_profile,
                            ),
                        })
            _, peak = tracemalloc.get_traced_memory()
            outputs.append({
                "name": candidate["name"],
                "comparison_identity": hashlib.sha256(
                    json.dumps(
                        {
                            "candidate": candidate,
                            "horizon": manifest["horizon"],
                            "initial_p_micromolar": manifest["initial_p_micromolar"],
                            "modes": manifest["modes"],
                            "seeds": manifest["seeds"],
                            "static_policy": manifest["static_policy"],
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "domain": {key: candidate[key] for key in _DOMAIN_FIELDS if key in candidate},
                "records": records,
                "fungal_lower_boundary_contact": any(
                    record["fungal_lower_boundary_contact"] for record in records
                ),
                "fungal_lower_boundary_first_contact_step": min(
                    (
                        record["fungal_lower_boundary_first_contact_step"]
                        for record in records
                        if record["fungal_lower_boundary_first_contact_step"] is not None
                    ),
                    default=None,
                ),
                "initial_p_inventory_micromol": sum(record["initial_p_inventory_micromol"] for record in records) / len(records),
                "response_order": _ordering(records),
                "runtime_seconds": time.perf_counter() - started,
                "peak_memory_bytes": peak,
            })
    finally:
        tracemalloc.stop()

    passing = []
    uptake_tolerance = float(declaration.get("direct_plant_uptake_relative_tolerance", 0.05))
    if not 0.0 <= uptake_tolerance or not jnp.isfinite(uptake_tolerance):
        raise ValueError("direct_plant_uptake_relative_tolerance must be finite and non-negative")
    for index, output in enumerate(outputs):
        if index == len(outputs) - 1:
            output["direct_plant_uptake_behavior"] = {
                "compared_to": None,
                "maximum_relative_difference": None,
                "relative_tolerance": uptake_tolerance,
                "stable": None,
            }
            output["status"] = "comparison-reference"
            output["rejection_reasons"] = []
            continue
        reasons = []
        largest_domain = outputs[-1]
        if output["fungal_lower_boundary_contact"]:
            reasons.append("fungal lower boundary contact")
        uptake_difference = _direct_plant_uptake_difference(output, largest_domain)
        output["direct_plant_uptake_behavior"] = {
            "compared_to": largest_domain["name"],
            "maximum_relative_difference": uptake_difference,
            "relative_tolerance": uptake_tolerance,
            "stable": uptake_difference <= uptake_tolerance,
        }
        if uptake_difference > uptake_tolerance:
            reasons.append("direct plant P uptake changed at largest depth domain")
        output["status"] = "rejected" if reasons else "eligible"
        output["rejection_reasons"] = reasons
        if not reasons:
            passing.append(output)
    if not passing:
        raise ValueError("no candidate domain passed qualification")
    accepted = passing[0]
    accepted["status"] = "accepted"
    for output in passing[1:]:
        output["status"] = "eligible"
    return {
        "format": "mycormarl-domain-qualification",
        "format_version": 1,
        "status": "complete",
        "accepted_domain": accepted,
        "comparison_reference_domain": outputs[-1]["name"],
        "candidates": outputs,
        "initial_solution_p_profiled": depth_profile is not None,
        "depth_profile": depth_profile,
        "initial_p_scenario": declaration.get(
            "initial_p_scenario",
            "depth-profile" if depth_profile is not None else "uniform",
        ),
        "static_policy": manifest["static_policy"],
    }
