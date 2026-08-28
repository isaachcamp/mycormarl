"""Versioned public orchestration seam for MycorMARL studies."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import threading
from typing import Any

import jax
import jax.numpy as jnp
from flax import serialization

from mycormarl.policy_artifacts import (
    ACTOR_INTERFACE_VERSION,
    ENVIRONMENT_STATE_SCHEMA_VERSION,
)
from mycormarl.checkpoint_evaluation import (
    evaluate_checkpoint,
    evaluate_checkpoint_summary,
    save_evaluation_artifact,
    save_evaluation_summary_artifact,
)
from mycormarl.random_streams import (
    RANDOM_STREAM_DERIVATION_VERSION,
    RANDOM_STREAM_NAMES,
    derive_random_streams,
)
from mycormarl.algos.ppo import PPOConfig, make_train
from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.static_controls import run_batched_static_controls, run_static_controls
from mycormarl.domain_qualification import run_domain_qualification
from mycormarl.trade_only import TOTAL_BIOLOGICAL_RATE_PER_DAY, plant_only_actions


STUDY_RESULT_FORMAT = "mycormarl-study-result"
STUDY_RESULT_VERSION = 2
_STUDY_MODES = frozenset({"mixed", "plant-only"})
_STUDY_STAGES = frozenset({"walking-skeleton", "single-condition-training", "comparison-block-training", "phase-1-pilot", "historical-grid-trade-only-pilot", "phase-1-pilot-analysis", "static-controls", "domain-qualification"})
TRAINING_CHECKPOINT_FORMAT = "mycormarl-ppo-checkpoint"
TRAINING_CHECKPOINT_VERSION = 1
_REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "stage",
    "model",
    "horizon",
    "modes",
    "initial_p_micromolar",
    "seeds",
    "training",
    "evaluation",
    "output",
)
_FIGURE_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class StudyResult:
    """Paths to the machine-readable result and its derived human summary."""

    bundle_path: Path
    summary_path: Path


def _provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    repository_root, git_commit, git_dirty = _repository_state()
    if git_dirty:
        raise ValueError(
            "study execution requires a clean Git working tree; "
            "commit or stash tracked and untracked changes first"
        )
    return {
        "actor_interface_version": ACTOR_INTERFACE_VERSION,
        "dependency_lock_sha256": _file_sha256(repository_root / "uv.lock"),
        "environment_state_schema_version": ENVIRONMENT_STATE_SCHEMA_VERSION,
        "execution_kind": "contract-fixture",
        "git_commit": git_commit,
        "git_dirty": False,
        "jax_version": version("jax"),
        "jaxlib_version": version("jaxlib"),
        "manifest_schema_version": manifest["schema_version"],
        "mycormarl_version": version("mycormarl"),
        "python_version": platform.python_version(),
        "result_format_version": STUDY_RESULT_VERSION,
        "numerical_timestep_days": manifest["horizon"]["timestep_days"],
        "policy_decision_interval_days": manifest["horizon"].get(
            "decision_interval_days", manifest["horizon"]["timestep_days"]
        ),
    }


def _git_stdout(*arguments: str, cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "cannot determine the current Git commit for study provenance"
        ) from error
    return completed.stdout.strip()


def _repository_state() -> tuple[Path, str, bool]:
    package_path = Path(__file__).resolve().parent
    repository_root = Path(
        _git_stdout("rev-parse", "--show-toplevel", cwd=package_path)
    )
    git_commit = _git_stdout(
        "rev-parse", "--verify", "HEAD", cwd=repository_root
    )
    git_status = _git_stdout(
        "status", "--porcelain=v1", "--untracked-files=all", cwd=repository_root
    )
    return repository_root, git_commit, bool(git_status)


def _file_sha256(path: Path) -> str:
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"cannot read dependency lockfile for study provenance: {path}"
        ) from error
    return hashlib.sha256(contents).hexdigest()


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _study_identity(manifest: dict[str, Any]) -> str:
    if not isinstance(manifest, dict):
        raise TypeError("canonical study manifest must be a JSON object")
    scientific_manifest = {
        field: value for field, value in manifest.items() if field != "output"
    }
    return _canonical_sha256(scientific_manifest)


def _execution_identity(
    study_identity: str, provenance: dict[str, Any]
) -> str:
    return _canonical_sha256(
        {"provenance": provenance, "study_identity": study_identity}
    )


def _condition_matrix(
    manifest: dict[str, Any],
) -> list[tuple[str, int | float, int]]:
    """Enumerate the immutable condition inventory for a declared study."""
    if manifest["stage"] == "historical-grid-trade-only-pilot":
        control_seed = manifest["seeds"][0]
        return [
            (mode, p_level, seed)
            for mode in manifest["modes"]
            for p_level in manifest["initial_p_micromolar"]
            for seed in (
                manifest["seeds"] if mode == "mixed" else (control_seed,)
            )
        ]
    return list(itertools.product(
        manifest["modes"], manifest["initial_p_micromolar"], manifest["seeds"],
    ))


def _validate_required_declarations(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("study manifest must be a JSON object")
    missing = [field for field in _REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(
            "missing required manifest fields: " + ", ".join(missing)
        )
    if manifest["schema_version"] != 1:
        raise ValueError(
            "incompatible manifest schema_version; use schema_version 1"
        )
    if manifest["stage"] not in _STUDY_STAGES:
        raise ValueError(
            f"unsupported study stage {manifest['stage']!r}; no executor is registered"
        )
    model = manifest["model"]
    if (
        not isinstance(model, dict)
        or not {"environment", "species"}.issubset(model)
        or not isinstance(model["environment"], dict)
        or not isinstance(model["species"], dict)
    ):
        raise ValueError(
            "model must declare environment and species configuration objects"
        )
    modes = manifest["modes"]
    if (
        not isinstance(modes, list)
        or not modes
        or any(mode not in _STUDY_MODES for mode in modes)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError(
            "modes must contain unique 'mixed' and/or 'plant-only' values"
        )
    seeds = manifest["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be unique non-negative integer IDs")
    initial_p = manifest["initial_p_micromolar"]
    if (
        not isinstance(initial_p, list)
        or not initial_p
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
            for value in initial_p
        )
        or len(set(initial_p)) != len(initial_p)
    ):
        raise ValueError(
            "initial_p_micromolar must contain unique finite positive values"
        )
    horizon = manifest["horizon"]
    if not isinstance(horizon, dict) or not {
        "days",
        "timestep_days",
    }.issubset(horizon):
        raise ValueError("horizon must declare days and timestep_days")
    days = horizon["days"]
    timestep_days = horizon["timestep_days"]
    decision_interval_days = horizon.get("decision_interval_days", timestep_days)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in (days, timestep_days, decision_interval_days)
    ):
        raise ValueError("horizon timing values must be finite and positive")
    decisions = days / decision_interval_days
    numerical_substeps = decision_interval_days / timestep_days
    if not math.isclose(decisions, round(decisions), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("horizon days must contain a whole number of policy decisions")
    if not math.isclose(numerical_substeps, round(numerical_substeps), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("decision_interval_days must contain a whole number of timesteps")
    training = manifest["training"]
    comparison_block_stages = {
        "comparison-block-training", "phase-1-pilot",
        "historical-grid-trade-only-pilot",
    }
    training_fields = (
        ("minimum_transition_budget", "maximum_transition_budget", "checkpoint_interval_timesteps")
        if manifest["stage"] in comparison_block_stages
        else ("total_timesteps", "checkpoint_interval_timesteps")
    )
    if not isinstance(training, dict) or not set(training_fields).issubset(training):
        raise ValueError(
            "training must declare total_timesteps and checkpoint_interval_timesteps"
        )
    if any(
        isinstance(training[field], bool)
        or not isinstance(training[field], int)
        or training[field] <= 0
        for field in training_fields
    ):
        raise ValueError("training timestep budgets must be positive integers")
    maximum_budget = training.get(
        "maximum_transition_budget", training.get("total_timesteps")
    )
    if training["checkpoint_interval_timesteps"] > maximum_budget:
        raise ValueError("training checkpoint interval cannot exceed total_timesteps")
    if manifest["stage"] in comparison_block_stages:
        minimum_budget = training["minimum_transition_budget"]
        if minimum_budget > maximum_budget:
            raise ValueError("training minimum transition budget cannot exceed maximum")
        if any(
            budget % training["checkpoint_interval_timesteps"]
            for budget in (minimum_budget, maximum_budget)
        ):
            raise ValueError("comparison-block training budgets must align with checkpoints")
        stopping = training.get("stopping")
        relative_fitness_fields = (
            "fitness_absolute_floor",
            "fitness_relative",
            "action_absolute",
        )
        legacy_fitness_fields = (
            "plant_fitness_absolute",
            "fungus_fitness_absolute",
            "action_absolute",
        )
        tolerances = (
            stopping.get("plateau_tolerances")
            if isinstance(stopping, dict)
            else None
        )
        scale_aware_declared = isinstance(tolerances, dict) and any(
            field in tolerances
            for field in ("fitness_absolute_floor", "fitness_relative")
        )
        required_tolerance_fields = (
            relative_fitness_fields if scale_aware_declared else legacy_fitness_fields
        )
        if (
            not isinstance(stopping, dict)
            or not isinstance(stopping.get("evaluation_window_checkpoints"), int)
            or isinstance(stopping.get("evaluation_window_checkpoints"), bool)
            or stopping["evaluation_window_checkpoints"] < 2
            or not isinstance(tolerances, dict)
            or any(field not in tolerances for field in required_tolerance_fields)
            or any(
                isinstance(tolerances[field], bool)
                or not isinstance(tolerances[field], (int, float))
                or not math.isfinite(tolerances[field])
                or tolerances[field] < 0.0
                for field in required_tolerance_fields
            )
        ):
            raise ValueError(
                "comparison-block training requires a stopping declaration with "
                "an evaluation window and either scale-aware fitness and action "
                "plateau tolerances or the legacy fitness and action tolerances"
            )
    if manifest["stage"] in {"single-condition-training", *comparison_block_stages}:
        if manifest["stage"] == "single-condition-training" and (
            len(manifest["modes"]) != 1 or len(manifest["initial_p_micromolar"]) != 1 or len(manifest["seeds"]) != 1
        ):
            raise ValueError("single-condition-training requires one mode, one initial_p_micromolar value, and one seed")
        for field in ("num_steps", "num_envs", "update_epochs", "num_minibatches"):
            value = training.get(field, 1)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"training {field} must be a positive integer")
        normalization = training.get(
            "critic_target_normalization", "per-agent-running"
        )
        if normalization not in {"per-agent-running", "raw"}:
            raise ValueError(
                "training critic_target_normalization must be "
                "'per-agent-running' or 'raw'"
            )
        update_size = training.get("num_steps", 1) * training.get("num_envs", 1)
        if training["checkpoint_interval_timesteps"] % update_size != 0:
            raise ValueError("training checkpoint interval must contain whole PPO updates")
        if manifest["stage"] in comparison_block_stages and any(
            budget % update_size
            for budget in (
                training["minimum_transition_budget"],
                training["maximum_transition_budget"],
            )
        ):
            raise ValueError("comparison-block training budgets must contain whole PPO updates")
        if manifest["stage"] in comparison_block_stages:
            parallel_workers = manifest["training"].get("parallel_workers", 1)
            if (
                isinstance(parallel_workers, bool)
                or not isinstance(parallel_workers, int)
                or parallel_workers <= 0
            ):
                raise ValueError("training parallel_workers must be a positive integer")
    if manifest["stage"] == "phase-1-pilot":
        fixture = manifest.get("pilot_fixture", False)
        if not isinstance(fixture, bool):
            raise ValueError("Phase 1 pilot_fixture must be a boolean")
        if not fixture and (
            manifest["modes"] != ["mixed", "plant-only"]
            or manifest["initial_p_micromolar"] != [0.1, 0.3, 1.0, 3.0]
            or len(manifest["seeds"]) != 5
            or manifest["horizon"] != {
                "days": 120.0,
                "timestep_days": 0.025,
                "decision_interval_days": 0.25,
            }
        ):
            raise ValueError(
                "Phase 1 pilot requires the fixed 40-run range-finding design"
            )
        artifacts = manifest.get("qualification_artifacts")
        if (
            not isinstance(artifacts, dict)
            or set(artifacts) != {"plant_growth", "static_controls", "domain"}
            or any(not isinstance(path, str) or not path for path in artifacts.values())
        ):
            raise ValueError(
                "Phase 1 pilot requires plant_growth, static_controls, and domain qualification artifacts"
            )
    if manifest["stage"] == "historical-grid-trade-only-pilot":
        fixture = manifest.get("pilot_fixture", False)
        if not isinstance(fixture, bool):
            raise ValueError("historical trade-only pilot_fixture must be a boolean")
        policy = manifest.get("policy")
        fixed_allocation = (
            policy.get("fixed_allocation") if isinstance(policy, dict) else None
        )
        if (
            not isinstance(policy, dict)
            or policy.get("mode") != "trade-only-fixed-allocation"
            or fixed_allocation != {
                "total_biological_rate_per_day": TOTAL_BIOLOGICAL_RATE_PER_DAY,
                "growth_fraction": 0.9,
                "reproduction_fraction": 0.1,
                "storage_rate_per_day": 0.0,
            }
        ):
            raise ValueError(
                "historical trade-only pilot requires the declared fixed-allocation trade-only policy"
            )
        if not fixture and (
            manifest["modes"] != ["mixed", "plant-only"]
            or manifest["initial_p_micromolar"] != [0.3, 0.75, 1.5, 3.0, 5.0, 10.0]
            or len(manifest["seeds"]) != 5
            or manifest["horizon"] != {
                "days": 120.0,
                "timestep_days": 0.025,
                "decision_interval_days": 0.25,
            }
            or manifest["training"] != {
                "minimum_transition_budget": 49152,
                "maximum_transition_budget": 239616,
                "checkpoint_interval_timesteps": 6144,
                "num_steps": 128,
                "num_envs": 16,
                "parallel_workers": 4,
                "update_epochs": 4,
                "num_minibatches": 8,
                "finite_horizon_returns": True,
                "stopping": {
                    "evaluation_window_checkpoints": 3,
                    "plateau_tolerances": {
                        "fitness_absolute_floor": 0.0001,
                        "fitness_relative": 0.2,
                        "action_absolute": 0.01,
                    },
                },
            }
        ):
            raise ValueError(
                "historical trade-only pilot requires its fixed 36-condition protocol"
            )
        artifacts = manifest.get("qualification_artifacts")
        if (
            not isinstance(artifacts, dict)
            or set(artifacts) != {"plant_growth", "static_controls", "domain"}
            or any(not isinstance(path, str) or not path for path in artifacts.values())
        ):
            raise ValueError(
                "historical trade-only pilot requires plant_growth, static_controls, and domain qualification artifacts"
            )
    if manifest["stage"] == "phase-1-pilot-analysis":
        if not isinstance(manifest.get("pilot_result_bundle"), str) or not manifest["pilot_result_bundle"]:
            raise ValueError("Phase 1 pilot analysis requires a pilot_result_bundle path")
        _validate_dense_design(manifest.get("dense_design"))
    evaluation = manifest["evaluation"]
    if not isinstance(evaluation, dict) or not {
        "protocol",
        "episodes",
    }.issubset(evaluation):
        raise ValueError("evaluation must declare protocol and episodes")
    if evaluation["protocol"] != "latent-location":
        raise ValueError("evaluation protocol must be 'latent-location'")
    episodes = evaluation["episodes"]
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("evaluation episodes must be a positive integer")
    output = manifest["output"]
    if not isinstance(output, dict) or not {"directory", "identity"}.issubset(output):
        raise ValueError("output must declare directory and output identity")
    if not isinstance(output["directory"], str) or not output["directory"]:
        raise ValueError("output directory must be a non-empty path string")
    identity = output["identity"]
    if (
        not isinstance(identity, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identity) is None
    ):
        raise ValueError("output identity must be a name, not a path")
    if manifest["stage"] == "domain-qualification":
        declaration = manifest.get("domain_qualification")
        if not isinstance(declaration, dict):
            raise ValueError("domain-qualification requires a domain_qualification declaration")
        if not isinstance(declaration.get("candidates"), list) or len(declaration["candidates"]) < 2:
            raise ValueError("domain-qualification requires at least two candidate domains")


def _validate_dense_design(design: Any) -> None:
    """Require a complete prospective Phase 1 dense-map declaration."""
    if not isinstance(design, dict):
        raise ValueError("Phase 1 pilot analysis requires a dense_design object")
    required = {
        "initial_p_micromolar", "spacing", "retained_pilot_levels", "seeds",
        "target_delta_am_standard_error", "domain_qualification_artifact",
        "training_budget",
    }
    if not required.issubset(design):
        raise ValueError("dense_design is missing a prospective design declaration")
    levels = design["initial_p_micromolar"]
    retained = design["retained_pilot_levels"]
    seeds = design["seeds"]
    if (
        not isinstance(levels, list) or not levels
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in levels)
        or len(set(levels)) != len(levels)
        or design["spacing"] != "logarithmic"
        or not isinstance(retained, list) or not retained or any(value not in levels for value in retained)
        or not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds)
        or not isinstance(design["target_delta_am_standard_error"], (int, float)) or isinstance(design["target_delta_am_standard_error"], bool) or not math.isfinite(design["target_delta_am_standard_error"]) or design["target_delta_am_standard_error"] <= 0
        or not isinstance(design["domain_qualification_artifact"], str) or not design["domain_qualification_artifact"]
        or not isinstance(design["training_budget"], dict)
    ):
        raise ValueError("dense_design is not a valid prospective dense-map declaration")
    budget = design["training_budget"]
    if (
        not isinstance(budget.get("minimum_transition_budget"), int)
        or not isinstance(budget.get("maximum_transition_budget"), int)
        or budget["minimum_transition_budget"] <= 0
        or budget["maximum_transition_budget"] < budget["minimum_transition_budget"]
    ):
        raise ValueError("dense_design training_budget is invalid")


def _accepted_dense_domain_artifact(raw_path: str, manifest_path: Path) -> str:
    """Verify that the prospective map inherits an accepted domain, not a label."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("dense design domain qualification artifact is unreadable") from error
    if not isinstance(artifact, dict):
        raise ValueError("dense design domain qualification artifact is not accepted")
    qualification = artifact.get("qualification", artifact)
    if (
        artifact.get("status") != "complete"
        or not isinstance(qualification, dict)
        or not isinstance(qualification.get("accepted_domain"), dict)
    ):
        raise ValueError("dense design domain qualification artifact is not accepted")
    return str(path)


def _validated_existing_entries(
    bundle: dict[str, Any],
    requested_conditions: list[tuple[str, int | float, int]],
) -> list[dict[str, Any]]:
    entries = bundle.get("entries")
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) for entry in entries
    ):
        raise ValueError("existing result condition inventory is incompatible")
    requested = set(requested_conditions)
    keys = [
        (
            entry.get("mode"),
            entry.get("initial_p_micromolar"),
            entry.get("seed"),
        )
        for entry in entries
    ]
    if (
        len(set(keys)) != len(keys)
        or any(key not in requested for key in keys)
        or any(entry.get("status") not in {"completed", "pending", "failed", "unconverged"} for entry in entries)
    ):
        raise ValueError("existing result condition inventory is incompatible")
    completed = sum(
        entry["status"] in {"completed", "failed", "unconverged"}
        for entry in entries
    )
    expected_completion = {
        "completed": completed,
        "requested": len(requested_conditions),
    }
    if bundle.get("completion") != expected_completion:
        raise ValueError("existing result condition inventory is incompatible")
    status = bundle.get("status")
    if status == "complete":
        if set(keys) != requested or completed != len(requested_conditions):
            raise ValueError("existing result condition inventory is incompatible")
    elif status != "incomplete":
        raise ValueError("existing result condition inventory is incompatible")
    return entries


def _passed_pilot_qualification_artifacts(
    artifact_paths: dict[str, str], manifest_path: Path,
) -> dict[str, str]:
    """Verify the three persisted prerequisites before training a pilot matrix."""
    artifacts: dict[str, dict[str, Any]] = {}
    for name, raw_path in artifact_paths.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Phase 1 pilot {name} qualification artifact is unreadable"
            ) from error
        if not isinstance(artifact, dict):
            raise ValueError(f"Phase 1 pilot {name} qualification artifact is invalid")
        artifacts[name] = artifact

    # Growth-scale evidence remains required provenance, but it is not an
    # all-or-nothing gate for the range-finding pilot: its sensitivity cases
    # inform interpretation rather than selecting whether the matrix exists.
    static_entries = artifacts["static_controls"].get("entries")
    if (
        artifacts["static_controls"].get("status") != "complete"
        or not isinstance(static_entries, list)
        or not static_entries
        or any(entry.get("status") != "completed" for entry in static_entries if isinstance(entry, dict))
        or any(not isinstance(entry, dict) for entry in static_entries)
    ):
        raise ValueError("Phase 1 pilot static_controls qualification did not pass")
    domain = artifacts["domain"]
    qualification = domain.get("qualification", domain)
    if (
        domain.get("status") != "complete"
        or not isinstance(qualification, dict)
        or not isinstance(qualification.get("accepted_domain"), dict)
    ):
        raise ValueError("Phase 1 pilot domain qualification did not pass")
    return artifact_paths


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    average = _mean(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)


def _sample_covariance(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = _mean(left), _mean(right)
    return sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    ) / (len(left) - 1)


def _pilot_entry_outcomes(entry: dict[str, Any], pilot_root: Path) -> dict[str, float]:
    """Read one terminal latent-location evaluation through its saved artifact."""
    artifacts = entry.get("evaluation_artifacts")
    if not isinstance(artifacts, list) or not artifacts or not isinstance(artifacts[-1], str):
        raise ValueError("pilot entry has no terminal evaluation artifact")
    path = Path(artifacts[-1])
    if not path.is_absolute():
        path = pilot_root / path
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("pilot evaluation artifact is unreadable") from error
    if (
        artifact.get("format") != "mycormarl-checkpoint-evaluation"
        or artifact.get("format_version") != 1
        or artifact.get("protocol") != "latent-location"
        or not isinstance(artifact.get("episodes"), list)
        or not artifact["episodes"]
    ):
        raise ValueError("pilot evaluation artifact is incompatible")
    endpoints: dict[str, list[float]] = {
        "fitness": [], "living_biomass": [], "gross_growth": [],
    }
    for episode in artifact["episodes"]:
        try:
            summary = episode["summary"]
            endpoints["fitness"].append(float(summary["cumulative_reproductive_fitness"]["plant"]))
            endpoints["living_biomass"].append(float(summary["final_living_biomass"]["plant"]))
            endpoints["gross_growth"].append(float(summary["cumulative_gross_growth"]["plant"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("pilot evaluation artifact has incomplete plant endpoints") from error
    if any(not math.isfinite(value) for values in endpoints.values() for value in values):
        raise ValueError("pilot evaluation artifact has non-finite plant endpoints")
    return {name: _mean(values) for name, values in endpoints.items()}


def _analyse_pilot_bundle(pilot_path: Path, dense_design: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the immutable paired pilot evidence without testing a hypothesis."""
    try:
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("pilot_result_bundle is unreadable") from error
    manifest = pilot.get("manifest")
    if (
        pilot.get("format") != STUDY_RESULT_FORMAT or pilot.get("format_version") != STUDY_RESULT_VERSION
        or pilot.get("status") != "complete" or not isinstance(manifest, dict)
        or manifest.get("stage") != "phase-1-pilot"
    ):
        raise ValueError("pilot_result_bundle is not a completed Phase 1 pilot")
    levels = manifest.get("initial_p_micromolar")
    seeds = manifest.get("seeds")
    if not isinstance(levels, list) or not isinstance(seeds, list) or not levels or not seeds:
        raise ValueError("pilot_result_bundle has no declared treatment matrix")
    if any(level not in dense_design["initial_p_micromolar"] for level in dense_design["retained_pilot_levels"]):
        raise ValueError("dense design retained pilot level is not present in the pilot")
    if set(dense_design["retained_pilot_levels"]) - set(levels):
        raise ValueError("dense design retains an undeclared pilot level")
    entries = pilot.get("entries")
    if not isinstance(entries, list):
        raise ValueError("pilot_result_bundle has no condition entries")
    expected = {(mode, level, seed) for mode in _STUDY_MODES for level in levels for seed in seeds}
    found: dict[tuple[str, float, int], dict[str, Any]] = {}
    unresolved_conditions = []
    for entry in entries:
        key = (entry.get("mode"), entry.get("initial_p_micromolar"), entry.get("seed")) if isinstance(entry, dict) else None
        if (
            key not in expected
            or key in found
            or entry.get("status") not in {"completed", "failed", "unconverged"}
        ):
            raise ValueError("pilot_result_bundle has incomplete or unqualified paired outcomes")
        found[key] = entry
    if set(found) != expected:
        raise ValueError("pilot_result_bundle has incomplete or unqualified paired outcomes")
    for mode, level, seed in sorted(found):
        status = found[(mode, level, seed)]["status"]
        if status != "completed":
            unresolved_conditions.append({
                "mode": mode,
                "initial_p_micromolar": level,
                "seed": seed,
                "status": status,
            })
    if unresolved_conditions:
        return {
            "pilot_result_bundle": str(pilot_path),
            "levels": [],
            "unresolved_conditions": unresolved_conditions,
            "tendency": {
                "classification": "unresolved-training-or-range-failure",
                "reason": "pilot contains failed or unconverged learned IPPO outcomes",
                "uses_complete_predeclared_grid": True,
                "adjacent_monotonicity_required": False,
                "confirmatory_inference": False,
            },
            "precision": None,
            "eligible_for_dense_execution": False,
        }

    analyses = []
    paired_variances = []
    for level in levels:
        rows = []
        for seed in seeds:
            mixed = _pilot_entry_outcomes(found[("mixed", level, seed)], pilot_path.parent)
            plant_only = _pilot_entry_outcomes(found[("plant-only", level, seed)], pilot_path.parent)
            rows.append({"seed": seed, "mixed": mixed, "plant_only": plant_only})
        def values(mode: str, endpoint: str) -> list[float]:
            return [row[mode][endpoint] for row in rows]
        mixed_fitness, plant_fitness = values("mixed", "fitness"), values("plant_only", "fitness")
        paired_fitness = [left - right for left, right in zip(mixed_fitness, plant_fitness, strict=True)]
        paired_variance = _sample_variance(paired_fitness)
        if paired_variance is not None:
            paired_variances.append(paired_variance)
        mixed_biomass, plant_biomass = values("mixed", "living_biomass"), values("plant_only", "living_biomass")
        mean_plant_biomass = _mean(plant_biomass)
        mgr = None if mean_plant_biomass == 0.0 else 100.0 * (_mean(mixed_biomass) / mean_plant_biomass - 1.0)
        paired_differences = {
            endpoint: [row["mixed"][endpoint] - row["plant_only"][endpoint] for row in rows]
            for endpoint in ("fitness", "living_biomass", "gross_growth")
        }
        analyses.append({
            "initial_p_micromolar": level,
            "seed_outcomes": rows,
            "marginal_outcomes": {
                mode: {endpoint: values(mode, endpoint) for endpoint in ("fitness", "living_biomass", "gross_growth")}
                for mode in ("mixed", "plant_only")
            },
            "delta_am": _mean(paired_fitness),
            "paired_delta_am_variance": paired_variance,
            "mgr_percent": mgr,
            "mgr_missing_reason": "mean plant-only living biomass is zero" if mgr is None else None,
            "paired_differences": paired_differences,
            "paired_scatter": {
                endpoint: [{"seed": row["seed"], "mixed": row["mixed"][endpoint], "plant_only": row["plant_only"][endpoint]} for row in rows]
                for endpoint in ("fitness", "living_biomass", "gross_growth")
            },
            "descriptive_covariance": {
                endpoint: _sample_covariance(values("mixed", endpoint), values("plant_only", endpoint))
                for endpoint in ("fitness", "living_biomass", "gross_growth")
            },
        })
    variance = max(paired_variances, default=None)
    recommended = max(2, math.ceil((variance or 0.0) / dense_design["target_delta_am_standard_error"] ** 2))
    if len(dense_design["seeds"]) < recommended:
        raise ValueError("dense design replication does not meet pilot-variance precision target")
    log_levels = [math.log(float(level["initial_p_micromolar"])) for level in analyses]
    advantages = [float(level["delta_am"]) for level in analyses]
    if len(analyses) < 2:
        tendency = {
            "classification": "unresolved-training-or-range-failure",
            "log_p_slope": None,
        }
    else:
        log_mean = _mean(log_levels)
        advantage_mean = _mean(advantages)
        slope_denominator = sum((value - log_mean) ** 2 for value in log_levels)
        slope = sum(
            (log_level - log_mean) * (advantage - advantage_mean)
            for log_level, advantage in zip(log_levels, advantages, strict=True)
        ) / slope_denominator
        tendency = {
            "classification": "supported-decreasing" if slope < 0.0 else "unsupported-decreasing",
            "log_p_slope": slope,
        }
    tendency.update({
        "uses_complete_predeclared_grid": True,
        "adjacent_monotonicity_required": False,
        "confirmatory_inference": False,
    })
    return {
        "pilot_result_bundle": str(pilot_path),
        "levels": analyses,
        "unresolved_conditions": [],
        "tendency": tendency,
        "precision": {
            "paired_delta_am_variance": variance,
            "target_delta_am_standard_error": dense_design["target_delta_am_standard_error"],
            "recommended_minimum_replication": recommended,
            "frozen_replication": len(dense_design["seeds"]),
        },
        "eligible_for_dense_execution": True,
    }


def _dense_map_manifest(
    analysis_manifest: dict[str, Any],
    pilot_analysis: dict[str, Any],
    domain_artifact: str,
) -> dict[str, Any]:
    """Freeze a standalone successor manifest before dense outcomes exist."""
    design = analysis_manifest["dense_design"]
    output = analysis_manifest["output"]
    training = dict(analysis_manifest["training"])
    training.update(design["training_budget"])
    return {
        "schema_version": analysis_manifest["schema_version"],
        "stage": "phase-1-dense-map",
        "model": analysis_manifest["model"],
        "horizon": analysis_manifest["horizon"],
        "modes": ["mixed", "plant-only"],
        "initial_p_micromolar": design["initial_p_micromolar"],
        "seeds": design["seeds"],
        "training": training,
        "evaluation": analysis_manifest["evaluation"],
        "output": {
            "directory": output["directory"],
            "identity": f"{output['identity']}-dense-map",
        },
        "qualification_artifacts": {"domain": domain_artifact},
        "parent_pilot_analysis": {
            "pilot_result_bundle": pilot_analysis["pilot_result_bundle"],
            "eligible_for_dense_execution": pilot_analysis["eligible_for_dense_execution"],
        },
        "dense_design": design,
    }


def _write_summary(bundle: dict[str, Any], summary_path: Path) -> None:
    summary = (
        f"# MycorMARL study: {bundle['manifest']['output']['identity']}\n\n"
        f"- Stage: {bundle['manifest']['stage']}\n"
        f"- Status: {bundle['status']}\n"
        "- Completed conditions: "
        f"{bundle['completion']['completed']}/"
        f"{bundle['completion']['requested']}\n"
        f"- Git commit: {bundle['provenance']['git_commit']}\n"
        f"- Study identity: {bundle['study_identity']}\n"
        f"- Execution identity: {bundle['execution_identity']}\n"
    )
    qualification = bundle.get("qualification")
    pilot_analysis = bundle.get("pilot_analysis")
    if pilot_analysis is not None:
        summary += (
            "\n## Phase 1 pilot analysis\n\n"
            "- This range-finding pilot is descriptive; it does not support confirmatory inference.\n"
            f"- Tendency status: {pilot_analysis['tendency']['classification']}\n"
            "- Dense-map replication is frozen from pilot paired-Delta_AM variance and the predeclared precision target.\n\n"
            "| Initial P (micromolar) | Delta_AM | MGR (%) |\n"
            "|---:|---:|---:|\n"
        )
        for level in pilot_analysis["levels"]:
            mgr = "missing" if level["mgr_percent"] is None else f"{level['mgr_percent']:.6g}"
            summary += (
                f"| {level['initial_p_micromolar']:.6g} | {level['delta_am']:.6g} | {mgr} |\n"
            )
    if qualification is not None:
        accepted = qualification["accepted_domain"]
        profile = qualification["initial_solution_p_profiled"]
        uptake_tolerance = qualification["candidates"][0][
            "direct_plant_uptake_behavior"
        ]["relative_tolerance"]
        summary += (
            "\n## Domain qualification\n\n"
            f"- Qualification outcome: {qualification['status']}\n"
            f"- Initial-P scenario: {qualification['initial_p_scenario']}\n"
            f"- Profiled initial Pi: {'yes' if profile else 'no'}\n"
            "- Acceptance gates: no fungal lower-boundary contact; maximum "
            "direct-plant-Pi uptake difference to the largest depth domain "
            f"≤ {uptake_tolerance:.0%}.\n"
        )
        if accepted is None:
            summary += "- Accepted depth: none\n\n"
        else:
            summary += (
                f"- Accepted depth: {accepted['domain']['soil_depth_cm']} cm "
                f"({accepted['name']})\n\n"
            )
        summary += (
            "| Candidate | Depth (cm) | Status | Fungal lower-boundary contact | "
            "Largest-depth comparison | Max uptake difference | Runtime (s) |\n"
            "|---|---:|---|---|---|---:|---:|\n"
        )
        for candidate in qualification["candidates"]:
            uptake = candidate["direct_plant_uptake_behavior"]
            difference = uptake["maximum_relative_difference"]
            comparison = uptake["compared_to"] or "—"
            difference_text = "—" if difference is None else f"{difference:.2%}"
            contact = "yes" if candidate["fungal_lower_boundary_contact"] else "no"
            if candidate["fungal_lower_boundary_first_contact_step"] is not None:
                contact += f" (step {candidate['fungal_lower_boundary_first_contact_step']})"
            summary += (
                f"| {candidate['name']} | {candidate['domain']['soil_depth_cm']} | "
                f"{candidate['status']} | {contact} | {comparison} | "
                f"{difference_text} | {candidate['runtime_seconds']:.2f} |\n"
            )
    summary_path.write_text(summary, encoding="utf-8")


def _run_static_controls_study(
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    study_identity: str,
    execution_identity: str,
    output_dir: Path,
) -> StudyResult:
    """Execute deterministic controls and persist them as a study bundle."""
    bundle_path = output_dir / "result-bundle.json"
    summary_path = output_dir / "summary.md"
    if bundle_path.exists():
        try:
            existing = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing result bundle is unreadable") from error
        if any(existing.get(field) != value for field, value in (
            ("format", STUDY_RESULT_FORMAT),
            ("format_version", STUDY_RESULT_VERSION),
            ("study_identity", study_identity),
            ("execution_identity", execution_identity),
            ("provenance", provenance),
        )):
            raise ValueError("existing static-control result provenance is incompatible")
        if existing.get("status") == "complete":
            if not summary_path.exists():
                _write_summary(existing, summary_path)
            return StudyResult(bundle_path, summary_path)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("existing outputs have no compatible execution identity")

    controls = run_static_controls(manifest)
    bundle = {
        "format": STUDY_RESULT_FORMAT,
        "format_version": STUDY_RESULT_VERSION,
        "study_identity": study_identity,
        "execution_identity": execution_identity,
        "manifest": manifest,
        "provenance": provenance,
        "random_streams": {
            "derivation_version": RANDOM_STREAM_DERIVATION_VERSION,
            "stream_names": list(RANDOM_STREAM_NAMES),
        },
        "entries": controls["entries"],
        "completion": controls["completion"],
        "status": controls["status"],
        "control_format": controls["format"],
        "control_format_version": controls["format_version"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(bundle, summary_path)
    return StudyResult(bundle_path, summary_path)


def _training_environment(manifest: dict[str, Any], mode: str, p_level: float) -> PolicyIntervalMycorMarl:
    model_environment = manifest["model"]["environment"]
    horizon = manifest["horizon"]
    decision_interval_days = horizon.get("decision_interval_days", horizon["timestep_days"])
    config = EnvConfig(
        max_steps=round(horizon["days"] / horizon["timestep_days"]),
        dt=horizon["timestep_days"],
        consumer_mode=mode,
        soil_radius_cm=model_environment.get("soil_radius_cm", 1.0),
        soil_depth_cm=model_environment.get("soil_depth_cm", 1.0),
        radial_interval_cm=model_environment.get("radial_interval_cm", 0.1),
        depth_interval_cm=model_environment.get("depth_interval_cm", 0.1),
        initial_solution_p_um=p_level,
        initial_solution_p_depth_profile=model_environment.get(
            "initial_solution_p_depth_profile"
        ),
    )
    return PolicyIntervalMycorMarl(
        BaseMycorMarl(config, SpeciesParams(PlantTraits(), FungusTraits())),
        decision_interval_days=decision_interval_days,
        max_episode_steps=round(horizon["days"] / decision_interval_days),
    )


def _training_config(manifest: dict[str, Any], timesteps: int) -> PPOConfig:
    training = manifest["training"]
    total_timesteps = training.get(
        "maximum_transition_budget", training.get("total_timesteps")
    )
    return PPOConfig(
        TOTAL_TIMESTEPS=total_timesteps,
        RUN_TIMESTEPS=timesteps,
        NUM_STEPS=training.get("num_steps", 1),
        NUM_ENVS=training.get("num_envs", 1),
        UPDATE_EPOCHS=training.get("update_epochs", 1),
        NUM_MINIBATCHES=training.get("num_minibatches", 1),
        DISCOUNT_HALF_LIFE_DAYS=training.get("discount_half_life_days"),
        FINITE_HORIZON_RETURNS=training.get("finite_horizon_returns", True),
        NORMALIZE_CRITIC_TARGETS=(
            training.get("critic_target_normalization", "per-agent-running")
            == "per-agent-running"
        ),
        TRADE_ONLY=(
            manifest.get("policy", {}).get("mode")
            == "trade-only-fixed-allocation"
        ),
    )


def _actor_configuration(config: PPOConfig) -> dict[str, Any]:
    """Persist every actor setting that changes checkpoint execution."""
    return {
        "activation": config.ACTIVATION,
        "trade_only": config.TRADE_ONLY,
        "critic_target_normalization": (
            "per-agent-running" if config.NORMALIZE_CRITIC_TARGETS else "raw"
        ),
    }


def _checkpoint_metadata(
    manifest: dict[str, Any],
    config: PPOConfig,
    *,
    mode: str,
    p_level: float,
    seed: int,
    transitions: int,
) -> dict[str, Any]:
    """Record the complete public context needed to restore a PPO actor."""
    return {
        "mode": mode,
        "initial_p_micromolar": p_level,
        "seed": seed,
        "transitions": transitions,
        "named_random_streams": derive_random_streams(seed).to_dict(),
        "manifest": manifest,
        "policy_context": manifest.get("policy"),
        "actor_configuration": _actor_configuration(config),
        "actor_interface_version": ACTOR_INTERFACE_VERSION,
        "environment_state_schema_version": ENVIRONMENT_STATE_SCHEMA_VERSION,
    }


def _training_diagnostics(chunks: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Summarise PPO optimizer health since the preceding saved checkpoint."""
    fields = (
        "total_loss", "value_loss", "actor_loss", "approx_kl", "latent_entropy",
        "raw_return_mean", "normalized_return_mean", "raw_critic_mean",
        "normalized_critic_mean", "critic_target_scale",
    )
    diagnostics: dict[str, dict[str, float]] = {}
    for agent in ("plant", "fungus"):
        metrics = [chunk[agent] for chunk in chunks]
        diagnostics[agent] = {
            field: float(jnp.mean(jnp.asarray([
                jnp.mean(getattr(metric, field)) for metric in metrics
            ])))
            for field in fields
        }
        diagnostics[agent]["learning_rate"] = float(
            jnp.ravel(metrics[-1].learning_rate)[-1]
        )
    return diagnostics


def _write_training_diagnostic_figure(
    history: list[dict[str, Any]],
    path: Path,
    *,
    metric: str,
    title: str,
    y_label: str,
) -> None:
    """Atomically replace one rolling PPO diagnostic figure."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    updates = [point["update"] for point in history]
    with _FIGURE_WRITE_LOCK:
        figure = Figure(figsize=(7, 4), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for agent, color in (("plant", "#2a9d8f"), ("fungus", "#7b2cbf")):
            values = [point.get(metric, {}).get(agent) for point in history]
            available = [
                (update, value) for update, value in zip(updates, values, strict=True)
                if value is not None
            ]
            if available:
                x_values, y_values = zip(*available, strict=True)
                axis.plot(x_values, y_values, marker="o", linewidth=1.5,
                          markersize=3, label=agent, color=color)
        axis.set(
            title=title,
            xlabel="Training update",
            ylabel=y_label,
        )
        axis.grid(alpha=0.25)
        axis.legend()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        figure.savefig(temporary_path, format="png", dpi=160)
        temporary_path.replace(path)


def _write_training_diagnostic_figures(
    history: list[dict[str, Any]], path: Path,
) -> None:
    """Replace the three complementary learning-curve diagnostics."""
    _write_training_diagnostic_figure(
        history, path / "training-returns.png",
        metric="returns", title="PPO rollout return during training",
        y_label="Mean rollout return",
    )
    _write_training_diagnostic_figure(
        history, path / "training-entropy.png",
        metric="latent_entropy", title="PPO latent policy entropy during training",
        y_label="Mean latent entropy",
    )
    _write_training_diagnostic_figure(
        history, path / "training-kl.png",
        metric="approx_kl", title="PPO approximate KL during training",
        y_label="Mean approximate KL",
    )


def _checkpoint_bytes(metadata: dict[str, Any], runner_state: Any) -> bytes:
    return serialization.msgpack_serialize({
        "format": TRAINING_CHECKPOINT_FORMAT,
        "format_version": TRAINING_CHECKPOINT_VERSION,
        "metadata": metadata,
        "runner_state": serialization.to_state_dict(runner_state),
    })


def _run_historical_trade_only_controls(
    manifest: dict[str, Any],
    conditions: list[tuple[str, float, int]],
) -> dict[tuple[str, float, int], dict[str, Any]]:
    """Evaluate pending plant-only controls in one vmapped rollout.

    Initial P only affects reset for these homogeneous deterministic controls,
    so the static-control runner can safely batch their state transitions.
    """
    if not conditions:
        return {}
    if any(mode != "plant-only" for mode, _, _ in conditions):
        raise ValueError("historical trade-only controls are plant-only")
    seeds = {seed for _, _, seed in conditions}
    if len(seeds) != 1:
        raise ValueError("historical trade-only controls require one control seed")
    seed = next(iter(seeds))
    actions = plant_only_actions()
    controls = run_batched_static_controls({
        **manifest,
        "modes": ["plant-only"],
        "initial_p_micromolar": [p_level for _, p_level, _ in conditions],
        "seeds": [seed],
        "static_policy": {
            agent: action.tolist() for agent, action in actions.items()
        },
    })
    return {
        (entry["mode"], entry["initial_p_micromolar"], entry["seed"]): {
            **entry,
            "execution_kind": "deterministic-static-control-vmapped",
            "policy": manifest["policy"],
        }
        for entry in controls["entries"]
    }


def _checkpoint_stopping_metrics(
    evaluation: Any, mode: str, *, trade_only: bool = False,
) -> dict[str, Any]:
    """Summarise a deterministic checkpoint without forming a treatment contrast."""
    agents = ("plant", "fungus") if mode == "mixed" else ("plant",)
    fitness = {
        agent: sum(
            episode.summary["cumulative_reproductive_fitness"][agent]
            for episode in evaluation.episodes
        ) / len(evaluation.episodes)
        for agent in agents
    }
    actions = {}
    for agent in agents:
        rows = [
            row["actions"][agent]
            for episode in evaluation.episodes
            for row in episode.trace
        ]
        averaged = [
            sum(row[index] for row in rows) / len(rows)
            for index in range(len(rows[0]))
        ]
        actions[agent] = averaged[:1] if trade_only else averaged
    return {"fitness": fitness, "actions": actions}


def _stopping_decision(
    checkpoints: list[dict[str, Any]], training: dict[str, Any], mode: str,
    *, trade_only: bool = False,
) -> dict[str, Any]:
    """Apply one predeclared, treatment-blind stopping rule at a checkpoint."""
    stopping = training["stopping"]
    tolerances = stopping["plateau_tolerances"]
    window_size = stopping["evaluation_window_checkpoints"]
    latest = checkpoints[-1]
    result: dict[str, Any] = {
        "transitions": latest["transitions"],
        "evaluation_window_checkpoints": window_size,
        "checkpoint_transitions": [
            checkpoint["transitions"] for checkpoint in checkpoints[-window_size:]
        ],
        "plateau_tolerances": tolerances,
        "plateau_metrics": {},
    }
    if latest["transitions"] < training["minimum_transition_budget"]:
        result["outcome"] = "minimum-budget-not-reached"
        return result
    window = checkpoints[-window_size:]
    if len(window) < window_size:
        result["outcome"] = "evaluation-window-incomplete"
        return result

    def span(values: list[float]) -> float:
        return max(values) - min(values)

    def fitness_plateau(values: list[float], legacy_key: str) -> dict[str, Any]:
        fitness_span = span(values)
        if "fitness_relative" not in tolerances:
            return {
                "fitness_span": fitness_span,
                "fitness_tolerance": tolerances[legacy_key],
                "stable": fitness_span <= tolerances[legacy_key],
            }
        fitness_scale = max(abs(value) for value in values)
        fitness_tolerance = (
            tolerances["fitness_absolute_floor"]
            + tolerances["fitness_relative"] * fitness_scale
        )
        return {
            "fitness_span": fitness_span,
            "fitness_scale": fitness_scale,
            "fitness_absolute_floor": tolerances["fitness_absolute_floor"],
            "fitness_relative_tolerance": tolerances["fitness_relative"],
            "fitness_tolerance": fitness_tolerance,
            "stable": fitness_span <= fitness_tolerance,
        }

    result["plateau_metrics"]["plant"] = fitness_plateau(
        [checkpoint["metrics"]["fitness"]["plant"] for checkpoint in window],
        "plant_fitness_absolute",
    )
    action_agents = ("plant", "fungus") if mode == "mixed" else ("plant",)
    action_span = max(
        span([checkpoint["metrics"]["actions"][agent][index] for checkpoint in window])
        for agent in action_agents
        for index in range(len(window[0]["metrics"]["actions"][agent]))
    )
    result["plateau_metrics"]["actions"] = {
        "components": ["trade_rate_per_day"] if trade_only else [
            "trade_rate_per_day", "growth_rate_per_day",
            "reproduction_rate_per_day", "storage_rate_per_day",
        ],
        "maximum_component_span": action_span,
        "stable": action_span <= tolerances["action_absolute"],
    }
    if mode == "mixed":
        result["plateau_metrics"]["fungus"] = fitness_plateau(
            [checkpoint["metrics"]["fitness"]["fungus"] for checkpoint in window],
            "fungus_fitness_absolute",
        )
    stable = all(metric["stable"] for metric in result["plateau_metrics"].values())
    if stable:
        result["outcome"] = "plateau-complete"
    elif latest["transitions"] >= training["maximum_transition_budget"]:
        result["outcome"] = "maximum-budget-unconverged"
    else:
        result["outcome"] = "checkpoint-continued"
    return result


def _run_condition_training(
    manifest: dict[str, Any],
    output_dir: Path,
    mode: str,
    p_level: float,
    seed: int,
) -> dict[str, Any]:
    """Train one independent condition; safe to execute in a worker thread."""
    training = manifest["training"]
    update_size = training.get("num_steps", 1) * training.get("num_envs", 1)
    env = _training_environment(manifest, mode, p_level)
    streams = derive_random_streams(seed)
    condition_name = f"{mode}-p{p_level:g}-seed{seed}"
    condition_dir = output_dir / "conditions" / condition_name
    state = None
    transitions = 0
    checkpoints = []
    training_metric_chunks = []
    training_return_history = []
    stopping_decision = None
    while transitions < training["maximum_transition_budget"]:
        config = _training_config(manifest, update_size)
        trained = jax.jit(make_train(env, config, streams, state))(
            jax.random.PRNGKey(seed)
        )
        state = trained["runner_state"]
        training_metric_chunks.append(trained["metrics"])
        transitions += update_size
        training_return_history.append({
            "update": transitions // update_size,
            "returns": {
                agent: float(jnp.mean(trained["metrics"][agent].raw_return_mean))
                for agent in ("plant", "fungus")
            },
            "latent_entropy": {
                agent: float(jnp.mean(trained["metrics"][agent].latent_entropy))
                for agent in ("plant", "fungus")
            },
            "approx_kl": {
                agent: float(jnp.mean(trained["metrics"][agent].approx_kl))
                for agent in ("plant", "fungus")
            },
        })
        if transitions % training["checkpoint_interval_timesteps"]:
            continue
        metadata = _checkpoint_metadata(
            manifest, config, mode=mode, p_level=p_level, seed=seed,
            transitions=transitions,
        )
        checkpoint_path = condition_dir / f"checkpoints/checkpoint-{transitions:08d}.msgpack"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(_checkpoint_bytes(metadata, state))
        evaluation_path = condition_dir / f"evaluations/checkpoint-{transitions:08d}.json"
        evaluation_summary = evaluate_checkpoint_summary(
            checkpoint_path,
            env,
            episodes=manifest["evaluation"]["episodes"],
            protocol=manifest["evaluation"]["protocol"],
            seed=seed,
        )
        if config.TRADE_ONLY:
            evaluation_summary["actions"] = {
                agent: values[:1]
                for agent, values in evaluation_summary["actions"].items()
            }
        training_diagnostics = _training_diagnostics(training_metric_chunks)
        evaluation_summary["training_diagnostics"] = training_diagnostics
        checkpoints.append({
            "transitions": transitions,
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
            "evaluation": str(evaluation_path.relative_to(output_dir)),
            "metrics": evaluation_summary,
            "training_diagnostics": training_diagnostics,
        })
        _write_training_diagnostic_figures(
            training_return_history, condition_dir,
        )
        training_metric_chunks = []
        stopping_decision = _stopping_decision(
            checkpoints, training, mode, trade_only=config.TRADE_ONLY,
        )
        if stopping_decision["outcome"] in {"plateau-complete", "maximum-budget-unconverged"}:
            evaluation = evaluate_checkpoint(
                checkpoint_path,
                env,
                episodes=manifest["evaluation"]["episodes"],
                protocol=manifest["evaluation"]["protocol"],
                seed=seed,
            )
            save_evaluation_artifact(evaluation_path, evaluation, checkpoint=checkpoint_path)
            break
        save_evaluation_summary_artifact(
            evaluation_path, evaluation_summary, checkpoint=checkpoint_path
        )
    assert stopping_decision is not None
    return {
        "mode": mode,
        "initial_p_micromolar": p_level,
        "seed": seed,
        "execution_kind": (
            "trade-only-ippo" if config.TRADE_ONLY else "independent-ppo"
        ),
        "status": "completed" if stopping_decision["outcome"] == "plateau-complete" else "unconverged",
        "transitions": transitions,
        "checkpoint": checkpoints[-1]["checkpoint"],
        "random_streams": streams.to_dict(),
        "evaluation": {"protocol": manifest["evaluation"]["protocol"], "episodes": manifest["evaluation"]["episodes"]},
        "evaluation_artifacts": [checkpoint["evaluation"] for checkpoint in checkpoints],
        "training_return_history": training_return_history,
        "training_return_figure": str(
            (condition_dir / "training-returns.png").relative_to(output_dir)
        ),
        "training_diagnostic_figures": {
            "latent_entropy": str(
                (condition_dir / "training-entropy.png").relative_to(output_dir)
            ),
            "approx_kl": str(
                (condition_dir / "training-kl.png").relative_to(output_dir)
            ),
        },
        "stopping_decision": stopping_decision,
        "stopping_checkpoints": checkpoints,
    }


def _run_comparison_block_training(
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    study_identity: str,
    execution_identity: str,
    output_dir: Path,
    qualification_artifacts: dict[str, str] | None = None,
    parallel_workers: int | None = None,
) -> StudyResult:
    """Train every declared condition under one frozen, blind stopping protocol."""
    bundle_path = output_dir / "result-bundle.json"
    summary_path = output_dir / "summary.md"
    existing_entries: list[dict[str, Any]] = []
    if bundle_path.exists():
        try:
            existing_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing comparison-block result bundle is unreadable") from error
        if any(existing_bundle.get(field) != value for field, value in (
            ("format", STUDY_RESULT_FORMAT),
            ("format_version", STUDY_RESULT_VERSION),
            ("study_identity", study_identity),
            ("execution_identity", execution_identity),
            ("provenance", provenance),
        )):
            raise ValueError("existing comparison-block result provenance is incompatible")
        if existing_bundle.get("qualification_artifacts") != qualification_artifacts:
            raise ValueError("existing comparison-block qualification provenance is incompatible")
        requested_conditions = _condition_matrix(manifest)
        existing_entries = _validated_existing_entries(
            existing_bundle, requested_conditions,
        )
        if existing_bundle.get("status") == "complete":
            if not summary_path.exists():
                _write_summary(existing_bundle, summary_path)
            return StudyResult(bundle_path, summary_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not bundle_path.exists():
            raise ValueError("existing outputs have no compatible execution identity")

    entries_by_condition = {}
    previous_entries = {
        (entry["mode"], entry["initial_p_micromolar"], entry["seed"]): entry
        for entry in existing_entries
    }
    pending_conditions = []
    requested_conditions = _condition_matrix(manifest)
    for mode, p_level, seed in requested_conditions:
        existing = previous_entries.get((mode, p_level, seed))
        if existing is not None and existing["status"] in {"completed", "failed", "unconverged"}:
            entries_by_condition[(mode, p_level, seed)] = existing
            continue
        pending_conditions.append((mode, p_level, seed))
    control_conditions = [
        condition for condition in pending_conditions
        if manifest["stage"] == "historical-grid-trade-only-pilot"
        and condition[0] == "plant-only"
    ]
    entries_by_condition.update(
        _run_historical_trade_only_controls(manifest, control_conditions)
    )
    training_conditions = [
        condition for condition in pending_conditions
        if condition not in control_conditions
    ]
    configured_workers = (
        manifest["training"].get("parallel_workers", 1)
        if parallel_workers is None else parallel_workers
    )
    workers = min(configured_workers, len(training_conditions))
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                condition: executor.submit(
                    _run_condition_training,
                    manifest,
                    output_dir,
                    *condition,
                )
                for condition in training_conditions
            }
            for condition, future in futures.items():
                entries_by_condition[condition] = future.result()
    entries = [entries_by_condition[condition] for condition in requested_conditions]
    bundle = {
        "format": STUDY_RESULT_FORMAT,
        "format_version": STUDY_RESULT_VERSION,
        "study_identity": study_identity,
        "execution_identity": execution_identity,
        "manifest": manifest,
        "provenance": provenance,
        "entries": entries,
        "completion": {"completed": len(entries), "requested": len(entries)},
        "status": "complete",
    }
    if qualification_artifacts is not None:
        bundle["qualification_artifacts"] = qualification_artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(bundle, summary_path)
    return StudyResult(bundle_path, summary_path)


def _run_single_condition_training(
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    study_identity: str,
    execution_identity: str,
    output_dir: Path,
    *,
    stop_after_timesteps: int | None,
) -> StudyResult:
    """Run one condition in checkpoint-sized PPO chunks and resume it."""
    bundle_path = output_dir / "result-bundle.json"
    summary_path = output_dir / "summary.md"
    mode, p_level, seed = manifest["modes"][0], manifest["initial_p_micromolar"][0], manifest["seeds"][0]
    total = manifest["training"]["total_timesteps"]
    streams = derive_random_streams(seed)
    existing = None
    if bundle_path.exists():
        try:
            existing = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing training bundle is unreadable") from error
        if any(existing.get(field) != value for field, value in (("format", TRAINING_CHECKPOINT_FORMAT), ("format_version", TRAINING_CHECKPOINT_VERSION), ("study_identity", study_identity), ("execution_identity", execution_identity), ("provenance", provenance))):
            raise ValueError("existing training checkpoint provenance is incompatible")
        if existing.get("status") == "complete":
            if not summary_path.exists():
                _write_summary(existing, summary_path)
            return StudyResult(bundle_path, summary_path)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("existing outputs have no compatible execution identity")

    env = _training_environment(manifest, mode, p_level)
    update_size = manifest["training"].get("num_steps", 1) * manifest["training"].get("num_envs", 1)
    completed = 0 if existing is None else existing["entries"][0].get("transitions", 0)
    training_return_history = (
        [] if existing is None
        else list(existing["entries"][0].get("training_return_history", []))
    )
    initial_state = None
    if completed:
        checkpoint_path = output_dir / existing["entries"][0]["checkpoint"]
        try:
            payload = serialization.msgpack_restore(checkpoint_path.read_bytes())
        except (OSError, ValueError) as error:
            raise ValueError("training checkpoint is unreadable") from error
        metadata = payload.get("metadata", {})
        if payload.get("format") != TRAINING_CHECKPOINT_FORMAT or payload.get("format_version") != TRAINING_CHECKPOINT_VERSION or metadata.get("mode") != mode or metadata.get("initial_p_micromolar") != p_level or metadata.get("seed") != seed or metadata.get("transitions") != completed:
            raise ValueError("training checkpoint condition is incompatible")
        template_config = _training_config(manifest, update_size)
        template = jax.jit(make_train(env, template_config, streams))(jax.random.PRNGKey(seed))
        initial_state = serialization.from_state_dict(template["runner_state"], payload["runner_state"])

    remaining = total - completed
    requested = remaining if stop_after_timesteps is None else min(remaining, stop_after_timesteps - completed)
    if requested <= 0 or requested % update_size:
        raise ValueError("training stop boundary must advance by whole PPO updates")
    # Execute one checkpoint-sized PPO update at a time.  Keeping the same
    # chunking for fresh and resumed runs makes the deterministic continuation
    # boundary explicit and ensures every declared interval is materialized.
    trained = None
    state = initial_state
    transitions = completed
    output_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(requested // update_size):
        config = _training_config(manifest, update_size)
        trained = jax.jit(make_train(env, config, streams, state))(jax.random.PRNGKey(seed))
        state = trained["runner_state"]
        transitions += update_size
        training_return_history.append({
            "update": transitions // update_size,
            "returns": {
                agent: float(jnp.mean(trained["metrics"][agent].raw_return_mean))
                for agent in ("plant", "fungus")
            },
            "latent_entropy": {
                agent: float(jnp.mean(trained["metrics"][agent].latent_entropy))
                for agent in ("plant", "fungus")
            },
            "approx_kl": {
                agent: float(jnp.mean(trained["metrics"][agent].approx_kl))
                for agent in ("plant", "fungus")
            },
        })
        intermediate_metadata = _checkpoint_metadata(
            manifest, config, mode=mode, p_level=p_level, seed=seed,
            transitions=transitions,
        )
        intermediate_path = output_dir / f"checkpoints/checkpoint-{transitions:08d}.msgpack"
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        intermediate_path.write_bytes(_checkpoint_bytes(intermediate_metadata, state))
        evaluation = evaluate_checkpoint(
            intermediate_path,
            env,
            episodes=manifest["evaluation"]["episodes"],
            protocol=manifest["evaluation"]["protocol"],
            seed=seed,
        )
        save_evaluation_artifact(
            output_dir / f"evaluations/checkpoint-{transitions:08d}.json",
            evaluation,
            checkpoint=intermediate_path,
        )
        _write_training_diagnostic_figures(
            training_return_history, output_dir,
        )
    checkpoint_name = f"checkpoints/checkpoint-{transitions:08d}.msgpack"
    checkpoint_path = output_dir / checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _checkpoint_metadata(
        manifest, config, mode=mode, p_level=p_level, seed=seed,
        transitions=transitions,
    )
    checkpoint_content = _checkpoint_bytes(metadata, trained["runner_state"])
    checkpoint_path.write_bytes(checkpoint_content)
    state_digest = hashlib.sha256(
        serialization.msgpack_serialize(
            serialization.to_state_dict(trained["runner_state"])
        )
    ).hexdigest()
    entry = {
        "mode": mode,
        "initial_p_micromolar": p_level,
        "seed": seed,
        "status": "completed" if transitions >= total else "pending",
        "transitions": transitions,
        "checkpoint": checkpoint_name,
        "random_streams": streams.to_dict(),
        "evaluation": {
            "protocol": manifest["evaluation"]["protocol"],
            "episodes": manifest["evaluation"]["episodes"],
            "state_sha256": state_digest,
        },
        "evaluation_artifacts": [
            f"evaluations/checkpoint-{checkpoint_step:08d}.json"
            for checkpoint_step in range(update_size, transitions + 1, update_size)
        ],
        "training_return_history": training_return_history,
        "training_return_figure": "training-returns.png",
        "training_diagnostic_figures": {
            "latent_entropy": "training-entropy.png",
            "approx_kl": "training-kl.png",
        },
    }
    bundle = {
        "format": TRAINING_CHECKPOINT_FORMAT,
        "format_version": TRAINING_CHECKPOINT_VERSION,
        "study_identity": study_identity,
        "execution_identity": execution_identity,
        "manifest": manifest,
        "provenance": provenance,
        "entries": [entry],
        "completion": {"completed": int(transitions >= total), "requested": 1},
        "status": "complete" if transitions >= total else "incomplete",
    }
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(bundle, summary_path)
    return StudyResult(bundle_path, summary_path)


def run_study(
    manifest_path: str | Path,
    *,
    stop_after_timesteps: int | None = None,
    parallel_workers: int | None = None,
) -> StudyResult:
    """Run a declared study manifest and persist its versioned result bundle.

    ``parallel_workers`` overrides only local scheduling for comparison blocks;
    it never changes the frozen manifest or scientific study identity.
    """
    if parallel_workers is not None and (
        isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, int)
        or parallel_workers <= 0
    ):
        raise ValueError("parallel_workers must be a positive integer when provided")
    manifest_source = Path(manifest_path)
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    _validate_required_declarations(manifest)
    provenance = _provenance(manifest)
    study_identity = _study_identity(manifest)
    execution_identity = _execution_identity(study_identity, provenance)
    requested_conditions = _condition_matrix(manifest)
    output_dir = (
        Path(manifest["output"]["directory"])
        / manifest["output"]["identity"]
    )
    if manifest["stage"] in {
        "comparison-block-training", "phase-1-pilot",
        "historical-grid-trade-only-pilot",
    }:
        if stop_after_timesteps is not None:
            raise ValueError("comparison-block-training does not support selective stop boundaries")
        qualification_artifacts = None
        if manifest["stage"] in {
            "phase-1-pilot", "historical-grid-trade-only-pilot",
        }:
            qualification_artifacts = _passed_pilot_qualification_artifacts(
                manifest["qualification_artifacts"], manifest_source,
            )
        return _run_comparison_block_training(
            manifest,
            provenance,
            study_identity,
            execution_identity,
            output_dir,
            qualification_artifacts,
            parallel_workers,
        )
    if manifest["stage"] == "phase-1-pilot-analysis":
        if stop_after_timesteps is not None:
            raise ValueError("phase-1-pilot-analysis does not support stop_after_timesteps")
        pilot_path = Path(manifest["pilot_result_bundle"])
        if not pilot_path.is_absolute():
            pilot_path = manifest_source.parent / pilot_path
        bundle_path = output_dir / "result-bundle.json"
        summary_path = output_dir / "summary.md"
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("existing outputs have no compatible execution identity")
        domain_artifact = _accepted_dense_domain_artifact(
            manifest["dense_design"]["domain_qualification_artifact"],
            manifest_source,
        )
        pilot_analysis = _analyse_pilot_bundle(pilot_path, manifest["dense_design"])
        dense_manifest = _dense_map_manifest(
            manifest, pilot_analysis, domain_artifact,
        )
        bundle = {
            "format": STUDY_RESULT_FORMAT,
            "format_version": STUDY_RESULT_VERSION,
            "study_identity": study_identity,
            "execution_identity": execution_identity,
            "manifest": manifest,
            "provenance": provenance,
            "pilot_analysis": pilot_analysis,
            "dense_design": manifest["dense_design"],
            "dense_manifest": dense_manifest,
            "completion": {"completed": 1, "requested": 1},
            "status": "complete",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_summary(bundle, summary_path)
        return StudyResult(bundle_path, summary_path)
    if manifest["stage"] == "single-condition-training":
        if stop_after_timesteps is not None and (
            isinstance(stop_after_timesteps, bool)
            or not isinstance(stop_after_timesteps, int)
            or stop_after_timesteps <= 0
        ):
            raise ValueError("stop_after_timesteps must be a positive integer")
        return _run_single_condition_training(
            manifest,
            provenance,
            study_identity,
            execution_identity,
            output_dir,
            stop_after_timesteps=stop_after_timesteps,
        )
    if manifest["stage"] == "static-controls":
        if "static_policy" not in manifest:
            raise ValueError("static-controls stage requires static_policy")
        if stop_after_timesteps is not None:
            raise ValueError("static-controls stage does not support stop_after_timesteps")
        return _run_static_controls_study(
            manifest,
            provenance,
            study_identity,
            execution_identity,
            output_dir,
        )
    if manifest["stage"] == "domain-qualification":
        if stop_after_timesteps is not None:
            raise ValueError("domain-qualification stage does not support stop_after_timesteps")
        bundle_path = output_dir / "result-bundle.json"
        summary_path = output_dir / "summary.md"
        if bundle_path.exists():
            try:
                existing_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                raise ValueError("existing result bundle is unreadable") from error
            if (
                not isinstance(existing_bundle, dict)
                or existing_bundle.get("format") != STUDY_RESULT_FORMAT
                or existing_bundle.get("format_version") != STUDY_RESULT_VERSION
            ):
                raise ValueError("existing result format is incompatible")
            if existing_bundle.get("study_identity") != study_identity:
                raise ValueError("existing study identity is incompatible")
            if existing_bundle.get("execution_identity") != execution_identity:
                raise ValueError("existing execution identity is incompatible")
            if existing_bundle.get("provenance") != provenance:
                raise ValueError("existing result provenance is incompatible")
            if existing_bundle.get("status") != "complete":
                raise ValueError("existing domain qualification is incomplete")
            if not summary_path.exists():
                _write_summary(existing_bundle, summary_path)
            return StudyResult(bundle_path, summary_path)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("existing outputs have no compatible execution identity")
        qualification = run_domain_qualification(manifest)
        bundle = {
            "format": STUDY_RESULT_FORMAT,
            "format_version": STUDY_RESULT_VERSION,
            "study_identity": study_identity,
            "execution_identity": execution_identity,
            "manifest": manifest,
            "provenance": provenance,
            "qualification": qualification,
            "completion": {"completed": 1, "requested": 1},
            "status": "complete",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_summary(bundle, summary_path)
        return StudyResult(bundle_path, summary_path)
    bundle_path = output_dir / "result-bundle.json"
    summary_path = output_dir / "summary.md"
    existing_bundle = None
    if bundle_path.exists():
        try:
            existing_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError("existing result bundle is unreadable") from error
        if (
            not isinstance(existing_bundle, dict)
            or existing_bundle.get("format") != STUDY_RESULT_FORMAT
            or existing_bundle.get("format_version") != STUDY_RESULT_VERSION
        ):
            raise ValueError("existing result format is incompatible")
        if existing_bundle.get("study_identity") != study_identity:
            raise ValueError("existing study identity is incompatible")
        if existing_bundle.get("execution_identity") != execution_identity:
            raise ValueError("existing execution identity is incompatible")
        if existing_bundle.get("provenance") != provenance:
            raise ValueError("existing result provenance is incompatible")
        existing_manifest = existing_bundle.get("manifest")
        try:
            embedded_study_identity = _study_identity(existing_manifest)
            embedded_execution_identity = _execution_identity(
                embedded_study_identity,
                existing_bundle["provenance"],
            )
        except (KeyError, TypeError, ValueError):
            embedded_study_identity = None
            embedded_execution_identity = None
        if (
            embedded_study_identity != existing_bundle["study_identity"]
            or embedded_execution_identity != existing_bundle["execution_identity"]
        ):
            raise ValueError("existing result manifest provenance is incompatible")
        existing_entries = _validated_existing_entries(
            existing_bundle,
            requested_conditions,
        )
        if existing_bundle.get("status") == "complete":
            if not summary_path.exists():
                _write_summary(existing_bundle, summary_path)
            return StudyResult(bundle_path=bundle_path, summary_path=summary_path)
    elif output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise ValueError("existing outputs have no compatible execution identity")
    completed_entries = {}
    if existing_bundle is not None:
        completed_entries = {
            (
                entry.get("mode"),
                entry.get("initial_p_micromolar"),
                entry.get("seed"),
            ): entry
            for entry in existing_entries
            if entry.get("status") == "completed"
        }
    entries = [
        completed_entries.get(
            (mode, initial_p, seed),
            {
                "mode": mode,
                "initial_p_micromolar": initial_p,
                "seed": seed,
                "status": "completed",
                "random_streams": derive_random_streams(seed).to_dict(),
            },
        )
        for mode, initial_p, seed in requested_conditions
    ]
    bundle = {
        "format": STUDY_RESULT_FORMAT,
        "format_version": STUDY_RESULT_VERSION,
        "study_identity": study_identity,
        "execution_identity": execution_identity,
        "manifest": manifest,
        "provenance": provenance,
        "random_streams": {
            "derivation_version": RANDOM_STREAM_DERIVATION_VERSION,
            "stream_names": list(RANDOM_STREAM_NAMES),
        },
        "entries": entries,
        "completion": {"completed": len(entries), "requested": len(entries)},
        "status": "complete",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    saved_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    _write_summary(saved_bundle, summary_path)
    return StudyResult(bundle_path=bundle_path, summary_path=summary_path)
