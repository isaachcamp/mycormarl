"""Deterministic static-policy resource-pressure screening."""

from __future__ import annotations

import math
import random
from pathlib import Path
from dataclasses import replace
from typing import Any

import json
import jax.numpy as jnp

from mycormarl.fungus.traits import FungusTraits
from mycormarl.plant.traits import PlantTraits
from mycormarl.static_controls import run_static_controls
from mycormarl.fungus.mycelium import (
    colony_radius_from_length_axisymmetric,
    hyphal_length_from_fungal_biomass,
)
from mycormarl.plant.roots import root_disc_radii_from_biomass
from mycormarl.soil.phosphate_grid import axisymmetric_edges_from_intervals


def logarithmic_levels(lower: float, upper: float, count: int) -> tuple[float, ...]:
    """Return ``count`` inclusive, increasing logarithmic levels."""
    if lower <= 0.0 or upper < lower or count < 2:
        raise ValueError("logarithmic levels require 0 < lower <= upper and count >= 2")
    log_lower = math.log(lower)
    spacing = (math.log(upper) - log_lower) / (count - 1)
    return tuple(math.exp(log_lower + index * spacing) for index in range(count))


_CONTINUOUS_RESOURCE_PRESSURE_RANGES = {
    "plant_kappa_c": (0.01, logarithmic_levels(0.01, 2.0, 6)[-2], "log"),
    "fungus_kappa_c": (0.01, logarithmic_levels(0.01, 2.0, 6)[-2], "log"),
    "fungus_initial_biomass": (1.0, 100.0, "log"),
    "initial_solution_p_micromolar": (0.1, 1.0, "linear"),
    "plant_trade": (0.05, 0.2, "linear"),
    "fungus_trade": (0.5, 0.8, "linear"),
    "fungus_gamma_p": (0.5, 2.0, "linear"),
}


def build_continuous_resource_pressure_design(
    design_seed: int, sample_count: int = 360
) -> list[dict[str, Any]]:
    """Build a reproducible continuous Latin-hypercube resource-pressure design."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = random.Random(design_seed)
    columns: dict[str, list[float]] = {}
    for name, (lower, upper, scale) in _CONTINUOUS_RESOURCE_PRESSURE_RANGES.items():
        values = [(index + rng.random()) / sample_count for index in range(sample_count)]
        rng.shuffle(values)
        if scale == "log":
            log_lower, log_upper = math.log(lower), math.log(upper)
            columns[name] = [math.exp(log_lower + value * (log_upper - log_lower)) for value in values]
        else:
            columns[name] = [lower + value * (upper - lower) for value in values]
    return [
        {
            "id": f"condition-{index + 1:03d}",
            "factors": {
                **{name: columns[name][index] for name in columns},
                "plant_kappa_p": 0.0,
                "fungus_kappa_p": 0.0,
            },
        }
        for index in range(sample_count)
    ]


def _resource_pressure_condition_traits(factors: dict[str, float], timestep_days: float) -> dict[str, Any]:
    plant = PlantTraits()
    fungus = FungusTraits()
    plant_traits = {
        "kappa_c": plant.kappa_c * factors["plant_kappa_c"],
        "kappa_p": plant.kappa_p * factors["plant_kappa_p"],
    }
    fungus_traits = {
        "kappa_c": fungus.kappa_c * factors["fungus_kappa_c"],
        "kappa_p": fungus.kappa_p * factors["fungus_kappa_p"],
        "initial_biomass": fungus.initial_biomass * factors["fungus_initial_biomass"],
    }
    if "fungus_gamma_p" in factors:
        fungus_traits["gamma_p"] = factors["fungus_gamma_p"]
    return {
        "plant": plant_traits,
        "fungus": fungus_traits,
        "derived_initial_pools": {
            "plant": {
                "c": plant_traits["kappa_c"] * plant.initial_biomass * timestep_days,
                "p": plant_traits["kappa_p"] * plant.initial_biomass * timestep_days,
            },
            "fungus": {
                "c": fungus_traits["kappa_c"] * fungus_traits["initial_biomass"] * timestep_days,
                "p": fungus_traits["kappa_p"] * fungus_traits["initial_biomass"] * timestep_days,
            },
        },
    }


def rank_viable_conditions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank completed conditions with non-negative fungal net biomass change."""
    viable = []
    for entry in entries:
        fungal_net_biomass = (
            entry["biomass"]["fungus"] - entry["initial_biomass"]["fungus"]
        )
        if entry["status"] == "completed" and fungal_net_biomass >= 0.0:
            viable.append({**entry, "fungal_net_biomass": fungal_net_biomass})
    viable.sort(key=lambda entry: (-entry["biomass"]["plant"], entry["id"]))
    return [{**entry, "rank": index + 1} for index, entry in enumerate(viable)]


def run_resource_pressure_experiment(
    manifest: dict[str, Any],
    *,
    design_override: list[dict[str, Any]] | None = None,
    existing_entries: list[dict[str, Any]] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Run issue #47's canonical continuous resource-pressure screen.

    ``design_override`` and ``existing_entries`` are used by the writer to
    resume condition-level checkpoints without changing sampled rows.
    """
    required = ("design_seed", "sample_count", "horizon", "model", "static_policy")
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError("resource-pressure experiment manifest missing: " + ", ".join(missing))
    if manifest.get("format") != "mycormarl-resource-pressure-experiment-manifest":
        raise ValueError("resource-pressure experiment requires the canonical manifest format")
    if manifest.get("sampling") != "continuous_lhs":
        raise ValueError("resource-pressure experiment requires continuous_lhs sampling")
    if design_override is not None:
        design = design_override
    else:
        design = build_continuous_resource_pressure_design(
            manifest["design_seed"], manifest["sample_count"]
        )
    if len(design) != manifest["sample_count"]:
        raise ValueError("resource-pressure design length does not match sample_count")
    existing_by_id = {entry["id"]: entry for entry in (existing_entries or [])}
    entries = []
    for index, declaration in enumerate(design, start=1):
        if declaration["id"] in existing_by_id:
            entries.append(existing_by_id[declaration["id"]])
            if progress_callback is not None:
                progress_callback(index, len(design), declaration, entries[-1], True)
            continue
        factors = declaration["factors"]
        traits = _resource_pressure_condition_traits(factors, manifest["horizon"]["timestep_days"])
        policy = {"plant": [factors["plant_trade"], 1.0, 0.0, 0.0], "fungus": [factors["fungus_trade"], 1.0, 0.0, 0.0]}
        control = run_static_controls({
            "horizon": manifest["horizon"],
            "model": {"environment": manifest["model"]["environment"], "species": {"plant": traits["plant"], "fungus": traits["fungus"]}},
            "modes": ["mixed"], "initial_p_micromolar": [factors["initial_solution_p_micromolar"]],
            "seeds": [manifest["design_seed"]], "static_policy": policy,
            "record_limitation_trace": bool(manifest.get("record_limitation_trace", True)),
            "record_resource_accounting": bool(manifest.get("record_resource_accounting", False)),
        })["entries"][0]
        entries.append({**control, **declaration, "traits": traits, "static_policy": policy,
                        "initial_biomass": {"plant": PlantTraits().initial_biomass, "fungus": traits["fungus"]["initial_biomass"]}})
        if progress_callback is not None:
            progress_callback(index, len(design), declaration, entries[-1], False)
    ranked = rank_viable_conditions(entries)
    retained = [entry["id"] for entry in ranked[:manifest.get("retain_count", 8)]]
    rejected = sum(entry["status"] == "rejected" for entry in entries)
    return {
        "format": "mycormarl-resource-pressure-experiment", "format_version": 1,
        "status": "complete" if not rejected else "completed-with-rejections",
        "manifest": manifest,
        "design": {
            "method": "continuous Latin-hypercube",
            "seed": manifest["design_seed"], "sample_count": len(design),
            "factor_levels": _CONTINUOUS_RESOURCE_PRESSURE_RANGES,
            "conditions": design,
        },
        "entries": entries,
        "selection": {"primary_score": "final_living_plant_biomass", "viability_constraint": "fungal_net_biomass >= 0 and control status completed", "ranked_viable": ranked, "retained_ids": retained, "retained_count": len(retained)},
        "completion": {"completed": len(entries) - rejected, "requested": len(entries)},
    }


def write_resource_pressure_experiment(manifest: dict[str, Any], output_path: Path) -> Path:
    """Run the screen with condition-level checkpoints and resumability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / f"{output_path.stem}-checkpoints"
    progress_path = output_path.with_name(f"{output_path.stem}-progress.json")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    existing_bundle: dict[str, Any] | None = None
    if output_path.exists():
        existing_bundle = json.loads(output_path.read_text(encoding="utf-8"))
        if existing_bundle.get("format") != "mycormarl-resource-pressure-experiment":
            raise ValueError("existing resource-pressure output has an incompatible format")

    saved_progress: dict[str, Any] | None = None
    if progress_path.exists():
        saved_progress = json.loads(progress_path.read_text(encoding="utf-8"))

    if saved_progress is not None:
        saved_manifest = saved_progress.get("manifest", {})
        for key in ("design_seed", "sample_count", "horizon", "model", "static_policy", "record_limitation_trace", "record_resource_accounting", "sampling", "write_combined_bundle"):
            if saved_manifest.get(key) != manifest.get(key):
                raise ValueError(f"existing resource-pressure progress differs in {key}")
        design = saved_progress["design"]["conditions"]
    elif existing_bundle is not None:
        saved_manifest = existing_bundle.get("manifest", {})
        for key in ("design_seed", "sample_count", "horizon", "model", "static_policy", "record_limitation_trace", "record_resource_accounting", "sampling", "write_combined_bundle"):
            if saved_manifest.get(key) != manifest.get(key):
                raise ValueError(f"existing resource-pressure output differs in {key}")
        design = existing_bundle["design"]["conditions"]
    else:
        design = build_continuous_resource_pressure_design(
            manifest["design_seed"], manifest["sample_count"]
        )
    if len(design) < manifest["sample_count"]:
        raise ValueError("continuous LHS sample_count changed; use a fresh output")
    elif len(design) > manifest["sample_count"]:
        raise ValueError("sample_count is smaller than the saved resource-pressure design")

    existing_entries = []
    if existing_bundle is not None:
        existing_entries.extend(existing_bundle.get("entries", []))
    for checkpoint in sorted(checkpoint_dir.glob("condition-*.json")):
        entry = json.loads(checkpoint.read_text(encoding="utf-8"))
        if not any(existing.get("id") == entry.get("id") for existing in existing_entries):
            existing_entries.append(entry)
    completed_ids = {entry["id"] for entry in existing_entries}

    def checkpoint(index: int, total: int, declaration: dict[str, Any], entry: dict[str, Any], skipped: bool) -> None:
        completed_ids.add(declaration["id"])
        if not skipped:
            checkpoint_path = checkpoint_dir / f"{declaration['id']}.json"
            checkpoint_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        progress_path.write_text(json.dumps({
            "format": "mycormarl-resource-pressure-progress",
            "format_version": 1,
            "status": "in-progress" if len(completed_ids) < total else "complete",
            "manifest": manifest,
            "design": {"conditions": design},
            "completed_ids": sorted(completed_ids),
            "completed": len(completed_ids),
            "requested": total,
            "last_condition": declaration["id"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"resource-pressure: {len(completed_ids)}/{total} ({declaration['id']})", flush=True)

    bundle = run_resource_pressure_experiment(
        manifest,
        design_override=design,
        existing_entries=existing_entries,
        progress_callback=checkpoint,
    )
    if manifest.get("write_combined_bundle", True):
        encoded = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
        output_path.write_text(encoded, encoding="utf-8")
    progress_path.write_text(json.dumps({
        "format": "mycormarl-resource-pressure-progress",
        "format_version": 1,
        "status": "complete",
        "manifest": manifest,
        "design": {"conditions": design},
        "completed_ids": [entry["id"] for entry in bundle["entries"]],
        "completed": len(bundle["entries"]),
        "requested": len(design),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def retained_condition_diagnostics(bundle: dict[str, Any]) -> dict[str, Any]:
    """Rerun retained conditions with growth, fixation, and geometry diagnostics."""
    manifest = bundle["manifest"]
    days = manifest["horizon"]["days"]
    environment = manifest["model"]["environment"]
    r_edges, z_edges = axisymmetric_edges_from_intervals(
        environment["soil_radius_cm"], environment["soil_depth_cm"],
        environment["radial_interval_cm"], environment["depth_interval_cm"],
    )
    retained = set(bundle["selection"]["retained_ids"])
    diagnostics = []
    for selected in bundle["entries"]:
        if selected["id"] not in retained:
            continue
        traits = selected["traits"]
        control_manifest = {
            "horizon": manifest["horizon"],
            "model": {
                "environment": environment,
                "species": {"plant": traits["plant"], "fungus": traits["fungus"]},
            },
            "modes": ["mixed"],
            "initial_p_micromolar": [1.0],
            "seeds": [manifest["design_seed"]],
            "static_policy": manifest["static_policy"],
        }
        entry = run_static_controls(control_manifest)["entries"][0]
        plant = replace(PlantTraits(), **traits["plant"])
        fungus = replace(FungusTraits(), **traits["fungus"])
        plant_radii = root_disc_radii_from_biomass(
            jnp.array([entry["biomass"]["plant"]]), plant, z_edges
        )
        fungal_length = hyphal_length_from_fungal_biomass(
            jnp.array([entry["biomass"]["fungus"]]), fungus.gamma_c,
            fungus.hyphal_tissue_carbon_density, fungus.hyphal_radius,
        )
        fungal_radius = colony_radius_from_length_axisymmetric(
            fungal_length, fungus.saturation_density
        )
        cumulative_growth = entry["cumulative_growth"]
        cumulative_p = entry["cumulative_direct_p_uptake_mg"]
        diagnostics.append({
            "id": selected["id"],
            "factors": selected["factors"],
            "traits": traits,
            "final_biomass_g": entry["biomass"],
            "net_biomass_change_g": {
                "plant": entry["biomass"]["plant"] - selected["initial_biomass"]["plant"],
                "fungus": entry["biomass"]["fungus"] - selected["initial_biomass"]["fungus"],
            },
            "cumulative_growth_g": cumulative_growth,
            "episode_average_growth_rate_g_per_day": {
                agent: value / days for agent, value in cumulative_growth.items()
            },
            "cumulative_direct_p_uptake_mg": cumulative_p,
            "episode_average_direct_p_uptake_mg_per_day": {
                agent: value / days for agent, value in cumulative_p.items()
            },
            "cumulative_plant_carbon_fixed_g": entry["cumulative_carbon_fixed"],
            "episode_average_plant_carbon_fixation_g_per_day": entry["cumulative_carbon_fixed"] / days,
            "final_geometry_radius_cm": {
                "plant_root_disc_max": float(jnp.max(plant_radii)),
                "plant_root_disc_surface": float(plant_radii[0]),
                "fungus_colony": float(fungal_radius[0]),
            },
            "steps": entry["steps"],
        })
    return {
        "format": "mycormarl-resource-pressure-retained-diagnostics",
        "format_version": 1,
        "source_bundle": "outputs/resource-pressure-static-screen/result-bundle.json",
        "horizon_days": days,
        "entries": diagnostics,
    }


def write_retained_condition_diagnostics(bundle_path: Path, output_path: Path) -> Path:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    diagnostics = retained_condition_diagnostics(bundle)
    encoded = json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != encoded:
            raise ValueError("retained diagnostics output already exists and differs")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded, encoding="utf-8")
    return output_path
