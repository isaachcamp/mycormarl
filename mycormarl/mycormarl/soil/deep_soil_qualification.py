"""Deep-soil phosphorus qualification through production environment steps."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from mycormarl.actions import rate_action
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.mycelium import (
    colony_radius_from_length_axisymmetric,
    hyphal_length_from_fungal_biomass,
)
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.soil.phosphate_grid import (
    labile_amount_to_solution_concentration,
)
from mycormarl.soil.phosphate_qualification import reference_relative_change
from mycormarl.soil.phosphate_units import MICROMOL_P_TO_MG_P
from mycormarl.soil.phosphate_uptake import blended_uptake_transaction
from mycormarl.soil.soil import uptake_geometry_coefficients


@dataclass(frozen=True)
class StaticPolicy:
    """One constant valid Rate action in ``d^-1`` used during qualification."""

    trade: float
    growth: float
    reproduction: float
    storage: float

    def action(self):
        """Construct the policy's action through the public action boundary."""
        return rate_action(
            self.trade,
            self.growth,
            self.reproduction,
            self.storage,
        )


@dataclass(frozen=True)
class DepthBandLoss:
    """Initial-to-final labile-P inventory change in one depth band."""

    start_depth_cm: float
    end_depth_cm: float
    initial_micromol: float
    final_micromol: float
    loss_percent: float | None


@dataclass(frozen=True)
class DeepSoilQualificationResult:
    """Observable fungal-confinement results for one qualification run."""

    executed_steps: int
    max_fungal_density_outside_colony: float
    max_fungal_uptake_request_outside_colony_micromol: float
    depth_band_losses: tuple[DepthBandLoss, ...]
    endpoint_metrics: dict[str, float]
    integrated_p_fluxes_micromol: dict[str, float]
    relative_extended_p_balance_error: float
    daily_soil_labile_p_micromol: np.ndarray
    manifest: dict


_DEPTH_BANDS_CM = (
    (0.0, 10.0),
    (10.0, 25.0),
    (25.0, 50.0),
    (50.0, 75.0),
    (75.0, 100.0),
)
_OUTPUT_INVENTORY = {
    "daily_soil_p_npz": "daily-soil-p.npz",
    "manifest_json": "manifest.json",
}
_RESIDUAL_POOL_METRICS = frozenset(
    {
        "final_plant_c_pool_g",
        "final_plant_p_pool_mg",
        "final_fungus_c_pool_g",
        "final_fungus_p_pool_mg",
    }
)


def _jsonable(value):
    """Convert nested configuration and trait dataclasses to JSON values."""
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    return value


def _policy_record(policy: StaticPolicy) -> dict:
    """Record policy inputs and their normalized public Physical action."""
    return {
        "trade": policy.trade,
        "growth": policy.growth,
        "reproduction": policy.reproduction,
        "storage_rate_per_day": policy.storage,
        "rate_action_per_day": [float(value) for value in policy.action()],
    }


def _depth_band_losses(env, initial_soil, final_soil) -> tuple[DepthBandLoss, ...]:
    """Aggregate canonical cell inventories over the required depth bands."""
    z_centres = 0.5 * (env.z_edges[:-1] + env.z_edges[1:])
    losses = []
    for start_depth, end_depth in _DEPTH_BANDS_CM:
        in_band = (z_centres >= start_depth) & (z_centres < end_depth)
        initial = float(jnp.sum(jnp.where(in_band[None, :], initial_soil, 0.0)))
        final = float(jnp.sum(jnp.where(in_band[None, :], final_soil, 0.0)))
        losses.append(
            DepthBandLoss(
                start_depth_cm=start_depth,
                end_depth_cm=end_depth,
                initial_micromol=initial,
                final_micromol=final,
                loss_percent=(100.0 * (initial - final) / initial if initial else None),
            )
        )
    return tuple(losses)


def _outside_colony_metrics(env: BaseMycorMarl, state):
    """Observe fungal density and calculated request in wholly exterior cells."""
    total_length = hyphal_length_from_fungal_biomass(
        state.fungus_biomass,
        env.species.fungus.gamma_c,
        env.species.fungus.hyphal_tissue_carbon_density,
        env.species.fungus.hyphal_radius,
    )
    colony_radius = colony_radius_from_length_axisymmetric(
        total_length,
        env.species.fungus.saturation_density,
    )
    r_inner = env.r_edges[:-1, None]
    z_inner = env.z_edges[:-1][None, :]
    wholly_outside = (
        r_inner**2 + z_inner**2 >= jnp.squeeze(colony_radius) ** 2
    )

    root_resistance, fungus_resistance, continuous_weight = (
        uptake_geometry_coefficients(
            state,
            env.species,
            env.config.phosphate_diffusion_coefficient_cm2_s,
            env.config.theta_water,
            env.config.phosphate_impedance_factor,
            env.config.b_p,
            env.config.uptake_reference_time_days,
            env.config.uptake_transition_exponent,
        )
    )
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p,
        env.cell_volumes,
        env.config.theta_water,
        env.config.b_p,
    )
    *_, diagnostics = blended_uptake_transaction(
        state.soil_labile_p,
        concentration,
        state.root_length_density,
        state.hyphae_length_density,
        env.cell_volumes,
        env.species,
        env.config.dt,
        root_resistance,
        fungus_resistance,
        continuous_weight,
    )
    density_max = jnp.max(
        jnp.where(wholly_outside, state.hyphae_length_density, 0.0)
    )
    request_max = jnp.max(
        jnp.where(wholly_outside, diagnostics.fungus_request_micromol, 0.0)
    )
    return density_max, request_max


def _extended_p_inventory_mg(state, species: SpeciesParams) -> float:
    """Return accounted soil, organism, loss, and export phosphorus in mg P."""
    return (
        float(jnp.sum(state.soil_labile_p)) * MICROMOL_P_TO_MG_P
        + float(state.plant_p_pool[0] + state.fungus_p_pool[0])
        + float(state.plant_biomass[0]) * species.plant.gamma_p
        + float(state.fungus_biomass[0]) * species.fungus.gamma_p
        + float(
            state.cumulative_plant_p_mortality_loss_mg[0]
            + state.cumulative_fungus_p_mortality_loss_mg[0]
            + state.cumulative_plant_p_maintenance_loss_mg[0]
            + state.cumulative_fungus_p_maintenance_loss_mg[0]
            + state.cumulative_plant_p_reproduction_export_mg[0]
            + state.cumulative_fungus_p_reproduction_export_mg[0]
        )
    )


def run_deep_soil_qualification(
    *,
    config: EnvConfig,
    species: SpeciesParams,
    plant_policy: StaticPolicy,
    fungus_policy: StaticPolicy,
    duration_days: float,
    seed: int,
    software_revision: str,
) -> DeepSoilQualificationResult:
    """Run a static-policy trajectory and check fungal spatial confinement."""
    started_clock = time.perf_counter()
    step_count = duration_days / config.dt
    if (
        not math.isfinite(step_count)
        or step_count <= 0.0
        or not math.isclose(step_count, round(step_count), abs_tol=1e-10)
    ):
        raise ValueError("duration_days must contain an integer number of steps")
    executed_steps = round(step_count)
    steps_per_day_float = 1.0 / config.dt
    if not math.isclose(
        steps_per_day_float,
        round(steps_per_day_float),
        abs_tol=1e-10,
    ):
        raise ValueError("environment dt must divide one day exactly")
    steps_per_day = round(steps_per_day_float)
    env = BaseMycorMarl(
        config=config,
        species=species,
        max_episode_steps=executed_steps,
    )
    root_key = jax.random.PRNGKey(seed)
    _, state = env.reset(root_key)
    initial_soil = state.soil_labile_p
    initial_extended_p_mg = _extended_p_inventory_mg(state, species)
    daily_soil = [np.asarray(state.soil_labile_p)]
    actions = {
        PLANT: plant_policy.action(),
        FUNGUS: fungus_policy.action(),
    }
    step_environment = jax.jit(env.step_env)
    observe_outside_colony = jax.jit(
        lambda current_state: _outside_colony_metrics(env, current_state)
    )
    maximum_density = 0.0
    maximum_request = 0.0
    plant_uptake_mg = 0.0
    fungus_uptake_mg = 0.0

    for step_index in range(executed_steps + 1):
        density, request = observe_outside_colony(state)
        maximum_density = max(maximum_density, float(density))
        maximum_request = max(maximum_request, float(request))
        if step_index < executed_steps:
            plant_p_before = float(state.plant_p_pool[0])
            fungus_p_before = float(state.fungus_p_pool[0])
            _, state, _, _, info = step_environment(
                jax.random.fold_in(root_key, step_index + 1),
                state,
                actions,
            )
            plant_p_cost_mg = float(
                info[PLANT]["growth"][0] * species.plant.gamma_p
                + info[PLANT]["maint_p_used"][0]
                + info[PLANT]["reproduction_p"][0]
            )
            fungus_p_cost_mg = float(
                info[FUNGUS]["growth"][0] * species.fungus.gamma_p
                + info[FUNGUS]["maint_p_used"][0]
                + info[FUNGUS]["reproduction_p"][0]
            )
            plant_uptake_mg += (
                float(state.plant_p_pool[0])
                - plant_p_before
                + plant_p_cost_mg
                - float(info[PLANT]["trade_in"][0])
            )
            fungus_uptake_mg += (
                float(state.fungus_p_pool[0])
                - fungus_p_before
                + fungus_p_cost_mg
                + float(info[FUNGUS]["trade_out"][0])
            )

            if (step_index + 1) % steps_per_day == 0:
                daily_soil.append(np.asarray(state.soil_labile_p))

    depth_band_losses = _depth_band_losses(
        env,
        initial_soil,
        state.soil_labile_p,
    )
    endpoint_metrics = {
        "final_soil_micromol": float(jnp.sum(state.soil_labile_p)),
        "final_plant_biomass_g": float(state.plant_biomass[0]),
        "final_fungus_biomass_g": float(state.fungus_biomass[0]),
        "final_plant_c_pool_g": float(state.plant_c_pool[0]),
        "final_plant_p_pool_mg": float(state.plant_p_pool[0]),
        "final_fungus_c_pool_g": float(state.fungus_c_pool[0]),
        "final_fungus_p_pool_mg": float(state.fungus_p_pool[0]),
    }
    plant_uptake_micromol = plant_uptake_mg / MICROMOL_P_TO_MG_P
    fungus_uptake_micromol = fungus_uptake_mg / MICROMOL_P_TO_MG_P
    integrated_p_fluxes_micromol = {
        "plant_uptake_micromol": plant_uptake_micromol,
        "fungus_uptake_micromol": fungus_uptake_micromol,
        "total_uptake_micromol": (
            plant_uptake_micromol + fungus_uptake_micromol
        ),
    }
    final_extended_p_mg = _extended_p_inventory_mg(state, species)
    relative_extended_p_balance_error = abs(
        final_extended_p_mg - initial_extended_p_mg
    ) / max(initial_extended_p_mg, 1e-30)
    identity_inputs = {
        "schema_version": "2",
        "duration_days": duration_days,
        "environment": _jsonable(config),
        "traits": _jsonable(species),
        "policies": {
            PLANT: _policy_record(plant_policy),
            FUNGUS: _policy_record(fungus_policy),
        },
        "seed": seed,
        "software_revision": software_revision,
    }
    scenario_id = hashlib.sha256(
        json.dumps(identity_inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        **identity_inputs,
        "scenario_id": scenario_id,
        "executed_steps": executed_steps,
        "qualification": {
            "max_fungal_density_outside_colony": maximum_density,
            "max_fungal_uptake_request_outside_colony_micromol": maximum_request,
            "depth_band_losses": [_jsonable(loss) for loss in depth_band_losses],
            "endpoint_metrics": endpoint_metrics,
            "integrated_p_fluxes_micromol": integrated_p_fluxes_micromol,
            "relative_extended_p_balance_error": (
                relative_extended_p_balance_error
            ),
        },
        "output_inventory": _OUTPUT_INVENTORY,
        "runtime_seconds": time.perf_counter() - started_clock,
    }

    return DeepSoilQualificationResult(
        executed_steps=executed_steps,
        max_fungal_density_outside_colony=maximum_density,
        max_fungal_uptake_request_outside_colony_micromol=maximum_request,
        depth_band_losses=depth_band_losses,
        endpoint_metrics=endpoint_metrics,
        integrated_p_fluxes_micromol=integrated_p_fluxes_micromol,
        relative_extended_p_balance_error=relative_extended_p_balance_error,
        daily_soil_labile_p_micromol=np.stack(daily_soil),
        manifest=manifest,
    )


def write_deep_soil_qualification_outputs(
    result: DeepSoilQualificationResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write a qualified trajectory and its complete provenance separately."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        key: output_dir / filename
        for key, filename in result.manifest["output_inventory"].items()
    }
    existing_outputs = [path for path in paths.values() if path.exists()]
    if paths["manifest_json"].exists():
        try:
            existing_manifest = json.loads(
                paths["manifest_json"].read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError("existing output provenance is incompatible") from error
        if existing_manifest.get("scenario_id") != result.manifest["scenario_id"]:
            raise ValueError("existing output provenance is incompatible")
    elif existing_outputs:
        raise ValueError("existing outputs have no compatible manifest")
    np.savez_compressed(
        paths["daily_soil_p_npz"],
        scenario_id=np.asarray(result.manifest["scenario_id"]),
        days=np.arange(len(result.daily_soil_labile_p_micromol)),
        soil_labile_p_micromol=result.daily_soil_labile_p_micromol,
    )
    paths["manifest_json"].write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _artifact_spatial_summary(manifest: dict, soil_snapshots) -> dict:
    """Recompute depth-band and corner-cell losses from stored canonical fields."""
    snapshots = np.asarray(soil_snapshots)
    initial = snapshots[0]
    final = snapshots[-1]
    depth_interval = manifest["environment"]["depth_interval_cm"]
    z_centres = (np.arange(initial.shape[1]) + 0.5) * depth_interval
    band_rows = []
    for start_depth, end_depth in _DEPTH_BANDS_CM:
        in_band = (z_centres >= start_depth) & (z_centres < end_depth)
        initial_inventory = float(initial[:, in_band].sum())
        final_inventory = float(final[:, in_band].sum())
        band_rows.append(
            {
                "start_depth_cm": start_depth,
                "end_depth_cm": end_depth,
                "initial_micromol": initial_inventory,
                "final_micromol": final_inventory,
                "loss_percent": (
                    100.0
                    * (initial_inventory - final_inventory)
                    / initial_inventory
                    if initial_inventory
                    else None
                ),
            }
        )
    corner_initial = float(initial[-1, -1])
    corner_final = float(final[-1, -1])
    return {
        "depth_band_losses": band_rows,
        "outer_bottom_cell": {
            "radial_index": int(initial.shape[0] - 1),
            "depth_index": int(initial.shape[1] - 1),
            "initial_micromol": corner_initial,
            "final_micromol": corner_final,
            "loss_percent": (
                100.0 * (corner_initial - corner_final) / corner_initial
                if corner_initial
                else None
            ),
        },
    }


def compare_deep_soil_qualification(
    corrected: DeepSoilQualificationResult,
    *,
    original_manifest_path: Path,
    original_npz_path: Path,
) -> dict:
    """Compare a corrected run with an existing diagnostic artifact."""
    original_manifest = json.loads(
        Path(original_manifest_path).read_text(encoding="utf-8")
    )
    with np.load(original_npz_path) as stored:
        original_scenario_id = stored["scenario_id"].item()
        original_snapshots = stored["soil_labile_p_micromol"].copy()
    if original_scenario_id != original_manifest["scenario_id"]:
        raise ValueError("original NPZ and manifest provenance are incompatible")

    corrected_manifest = corrected.manifest
    equivalence = {
        key: original_manifest.get(key) == corrected_manifest.get(key)
        for key in ("duration_days", "environment", "policies", "seed")
    }
    intentional_differences = {}
    for key in ("traits", "software_revision"):
        original_value = original_manifest.get(key)
        corrected_value = corrected_manifest.get(key)
        if original_value != corrected_value:
            intentional_differences[key] = {
                "original": original_value,
                "corrected": corrected_value,
            }

    original_summary = _artifact_spatial_summary(
        original_manifest,
        original_snapshots,
    )
    corrected_summary = _artifact_spatial_summary(
        corrected_manifest,
        corrected.daily_soil_labile_p_micromol,
    )
    ticket_reference = 4.683888
    available_bottom_loss = original_summary["depth_band_losses"][-1][
        "loss_percent"
    ]
    return {
        "configuration_equivalence": equivalence,
        "intentional_provenance_differences": intentional_differences,
        "original": original_summary,
        "corrected": corrected_summary,
        "ticket_reference_bottom_25_cm_loss_percent": ticket_reference,
        "available_original_matches_ticket_reference": (
            available_bottom_loss is not None
            and math.isclose(
                available_bottom_loss,
                ticket_reference,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ),
    }


def compare_temporal_convergence(
    candidate: DeepSoilQualificationResult,
    reference: DeepSoilQualificationResult,
    *,
    relative_tolerance: float = 0.05,
    absolute_floor: float = 1e-10,
    maximum_relative_p_balance_error: float = 1e-5,
    maximum_fungal_density_outside_colony: float = 1e-12,
    maximum_fungal_uptake_request_outside_colony_micromol: float = 1e-12,
) -> dict:
    """Compare a timestep candidate with its next-smaller reference run."""
    candidate_environment = dict(candidate.manifest["environment"])
    reference_environment = dict(reference.manifest["environment"])
    candidate_dt = candidate_environment.pop("dt")
    reference_dt = reference_environment.pop("dt")
    comparable_inputs = (
        "schema_version",
        "duration_days",
        "traits",
        "policies",
        "seed",
        "software_revision",
    )
    same_non_timestep_inputs = (
        candidate_environment == reference_environment
        and all(
            candidate.manifest[key] == reference.manifest[key]
            for key in comparable_inputs
        )
    )
    if not same_non_timestep_inputs:
        raise ValueError(
            "temporal convergence requires equivalent provenance; "
            "only environment dt may differ"
        )
    if not candidate_dt > reference_dt:
        raise ValueError("candidate dt must be larger than reference dt")

    candidate_metrics = dict(candidate.endpoint_metrics)
    reference_metrics = dict(reference.endpoint_metrics)
    candidate_metrics.update(candidate.integrated_p_fluxes_micromol)
    reference_metrics.update(reference.integrated_p_fluxes_micromol)
    for candidate_band, reference_band in zip(
        candidate.depth_band_losses,
        reference.depth_band_losses,
        strict=True,
    ):
        label = (
            f"final_soil_{candidate_band.start_depth_cm:g}_"
            f"{candidate_band.end_depth_cm:g}_cm_micromol"
        )
        candidate_metrics[label] = candidate_band.final_micromol
        reference_metrics[label] = reference_band.final_micromol

    comparisons = []
    diagnostic_comparisons = []
    for metric, candidate_value in candidate_metrics.items():
        reference_value = reference_metrics[metric]
        change = reference_relative_change(
            candidate_value,
            reference_value,
            absolute_floor,
        )
        row = (
            {
                "metric": metric,
                "candidate": candidate_value,
                "reference": reference_value,
                "relative_change": change,
                "passes": change <= relative_tolerance,
            }
        )
        if metric in _RESIDUAL_POOL_METRICS:
            diagnostic_comparisons.append(row)
        else:
            comparisons.append(row)
    maximum_change = max(row["relative_change"] for row in comparisons)
    maximum_diagnostic_change = max(
        (row["relative_change"] for row in diagnostic_comparisons),
        default=0.0,
    )
    absolute_checks = [
        {
            "requirement": f"{label}_relative_extended_p_balance_error",
            "value": result.relative_extended_p_balance_error,
            "maximum": maximum_relative_p_balance_error,
            "passes": (
                result.relative_extended_p_balance_error
                <= maximum_relative_p_balance_error
            ),
        }
        for label, result in (
            ("candidate", candidate),
            ("reference", reference),
        )
    ]
    for label, result in (("candidate", candidate), ("reference", reference)):
        for metric, value, maximum in (
            (
                "max_fungal_density_outside_colony",
                result.max_fungal_density_outside_colony,
                maximum_fungal_density_outside_colony,
            ),
            (
                "max_fungal_uptake_request_outside_colony_micromol",
                result.max_fungal_uptake_request_outside_colony_micromol,
                maximum_fungal_uptake_request_outside_colony_micromol,
            ),
        ):
            absolute_checks.append(
                {
                    "requirement": f"{label}_{metric}",
                    "value": value,
                    "maximum": maximum,
                    "passes": value <= maximum,
                }
            )
    passes_temporal_convergence = all(
        row["passes"] for row in comparisons
    ) and all(row["passes"] for row in absolute_checks)
    return {
        "candidate_dt_days": candidate_dt,
        "reference_dt_days": reference_dt,
        "provenance": {
            "candidate_scenario_id": candidate.manifest["scenario_id"],
            "reference_scenario_id": reference.manifest["scenario_id"],
            "software_revision": candidate.manifest["software_revision"],
            "duration_days": candidate.manifest["duration_days"],
            "seed": candidate.manifest["seed"],
            "only_environment_dt_differs": True,
        },
        "criteria": {
            "accepted_relative_metrics": sorted(
                row["metric"] for row in comparisons
            ),
            "diagnostic_metrics": sorted(
                row["metric"] for row in diagnostic_comparisons
            ),
            "relative_tolerance": relative_tolerance,
            "absolute_floor": absolute_floor,
            "maximum_relative_p_balance_error": (
                maximum_relative_p_balance_error
            ),
            "maximum_fungal_density_outside_colony": (
                maximum_fungal_density_outside_colony
            ),
            "maximum_fungal_uptake_request_outside_colony_micromol": (
                maximum_fungal_uptake_request_outside_colony_micromol
            ),
        },
        "relative_tolerance": relative_tolerance,
        "maximum_relative_p_balance_error": maximum_relative_p_balance_error,
        "maximum_fungal_density_outside_colony": (
            maximum_fungal_density_outside_colony
        ),
        "maximum_fungal_uptake_request_outside_colony_micromol": (
            maximum_fungal_uptake_request_outside_colony_micromol
        ),
        "metric_comparisons": comparisons,
        "diagnostic_comparisons": diagnostic_comparisons,
        "absolute_checks": absolute_checks,
        "maximum_relative_change": maximum_change,
        "maximum_diagnostic_relative_change": maximum_diagnostic_change,
        "passes_temporal_convergence": passes_temporal_convergence,
        "passes_5_percent": passes_temporal_convergence,
    }
