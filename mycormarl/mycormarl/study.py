"""Versioned public orchestration seam for MycorMARL studies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import platform
import re
import subprocess
from typing import Any

from mycormarl.policy_artifacts import (
    ACTOR_INTERFACE_VERSION,
    ENVIRONMENT_STATE_SCHEMA_VERSION,
)


STUDY_RESULT_FORMAT = "mycormarl-study-result"
STUDY_RESULT_VERSION = 2
_STUDY_MODES = frozenset({"mixed", "plant-only"})
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
    if manifest["stage"] != "walking-skeleton":
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
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in (days, timestep_days)
    ):
        raise ValueError("horizon days and timestep_days must be finite and positive")
    transitions = days / timestep_days
    if not math.isclose(transitions, round(transitions), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("horizon days must contain a whole number of timesteps")
    training = manifest["training"]
    training_fields = ("total_timesteps", "checkpoint_interval_timesteps")
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
    if training["checkpoint_interval_timesteps"] > training["total_timesteps"]:
        raise ValueError("training checkpoint interval cannot exceed total_timesteps")
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
        or any(entry.get("status") not in {"completed", "pending"} for entry in entries)
    ):
        raise ValueError("existing result condition inventory is incompatible")
    completed = sum(entry["status"] == "completed" for entry in entries)
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


def _write_summary(bundle: dict[str, Any], summary_path: Path) -> None:
    summary_path.write_text(
        f"# MycorMARL study: {bundle['manifest']['output']['identity']}\n\n"
        f"- Stage: {bundle['manifest']['stage']}\n"
        f"- Status: {bundle['status']}\n"
        "- Completed conditions: "
        f"{bundle['completion']['completed']}/"
        f"{bundle['completion']['requested']}\n"
        f"- Git commit: {bundle['provenance']['git_commit']}\n"
        f"- Study identity: {bundle['study_identity']}\n"
        f"- Execution identity: {bundle['execution_identity']}\n",
        encoding="utf-8",
    )


def run_study(manifest_path: str | Path) -> StudyResult:
    """Run a declared study manifest and persist its versioned result bundle."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _validate_required_declarations(manifest)
    provenance = _provenance(manifest)
    study_identity = _study_identity(manifest)
    execution_identity = _execution_identity(study_identity, provenance)
    requested_conditions = list(
        itertools.product(
            manifest["modes"],
            manifest["initial_p_micromolar"],
            manifest["seeds"],
        )
    )
    output_dir = (
        Path(manifest["output"]["directory"])
        / manifest["output"]["identity"]
    )
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
