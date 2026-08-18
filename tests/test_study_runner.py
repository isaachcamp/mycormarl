"""End-to-end contract tests for the public study runner."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pytest

import mycormarl.study as study_module
from mycormarl.study import run_study


_TEST_GIT_COMMIT = "a" * 40
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_repository(monkeypatch):
    """Keep public runner tests independent of the developer's worktree state."""
    monkeypatch.setattr(
        study_module,
        "_repository_state",
        lambda: (_REPOSITORY_ROOT, _TEST_GIT_COMMIT, False),
    )


def _manifest(tmp_path, *, identity="fixture"):
    return {
        "schema_version": 1,
        "stage": "walking-skeleton",
        "model": {
            "environment": {"soil_radius_cm": 1.0, "soil_depth_cm": 1.0},
            "species": {"plant": {}, "fungus": {}},
        },
        "horizon": {"days": 0.05, "timestep_days": 0.025},
        "modes": ["mixed", "plant-only"],
        "initial_p_micromolar": [0.3],
        "seeds": [7],
        "training": {
            "total_timesteps": 2,
            "checkpoint_interval_timesteps": 1,
        },
        "evaluation": {"protocol": "latent-location", "episodes": 1},
        "output": {
            "directory": str(tmp_path / "outputs"),
            "identity": identity,
        },
    }


def _write_manifest(tmp_path, manifest):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_manifest_emits_versioned_bundle_and_derived_summary(tmp_path):
    """A declared fixture travels through the production artifact path."""
    manifest = _manifest(tmp_path, identity="tiny-fixture")
    result = run_study(_write_manifest(tmp_path, manifest))

    output_root = tmp_path / "outputs"
    assert result.bundle_path == output_root / "tiny-fixture" / "result-bundle.json"
    assert result.summary_path == output_root / "tiny-fixture" / "summary.md"
    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    assert bundle["format"] == "mycormarl-study-result"
    assert bundle["format_version"] == 2
    assert bundle["manifest"] == manifest
    assert bundle["completion"] == {"completed": 2, "requested": 2}
    assert bundle["status"] == "complete"
    assert bundle["provenance"]["execution_kind"] == "contract-fixture"
    assert {
        (entry["mode"], entry["initial_p_micromolar"], entry["seed"])
        for entry in bundle["entries"]
    } == {("mixed", 0.3, 7), ("plant-only", 0.3, 7)}
    assert all(entry["status"] == "completed" for entry in bundle["entries"])
    assert all(
        entry["random_streams"]["master_seed"] == 7
        for entry in bundle["entries"]
    )
    assert bundle["random_streams"]["derivation_version"] == "named-prng-v1"
    assert result.summary_path.read_text(encoding="utf-8") == (
        "# MycorMARL study: tiny-fixture\n\n"
        "- Stage: walking-skeleton\n"
        "- Status: complete\n"
        "- Completed conditions: 2/2\n"
        f"- Git commit: {_TEST_GIT_COMMIT}\n"
        f"- Study identity: {bundle['study_identity']}\n"
        "- Execution identity: " + bundle["execution_identity"] + "\n"
    )


def test_storage_location_and_name_do_not_redefine_the_scientific_study(tmp_path):
    """Storage choices are excluded from both canonical execution identities."""
    first_manifest = _manifest(tmp_path / "first", identity="first-name")
    second_manifest = _manifest(tmp_path / "second", identity="second-name")

    first = run_study(_write_manifest(tmp_path / "first", first_manifest))
    second = run_study(_write_manifest(tmp_path / "second", second_manifest))

    first_bundle = json.loads(first.bundle_path.read_text(encoding="utf-8"))
    second_bundle = json.loads(second.bundle_path.read_text(encoding="utf-8"))
    assert first_bundle["study_identity"] == second_bundle["study_identity"]
    assert first_bundle["execution_identity"] == second_bundle["execution_identity"]


def test_scientific_manifest_change_redefines_study_and_execution(tmp_path):
    """A treatment change creates a new scientific study and execution."""
    first_manifest = _manifest(tmp_path / "first", identity="first")
    second_manifest = _manifest(tmp_path / "second", identity="second")
    second_manifest["initial_p_micromolar"] = [1.0]

    first = run_study(_write_manifest(tmp_path / "first", first_manifest))
    second = run_study(_write_manifest(tmp_path / "second", second_manifest))

    first_bundle = json.loads(first.bundle_path.read_text(encoding="utf-8"))
    second_bundle = json.loads(second.bundle_path.read_text(encoding="utf-8"))
    assert first_bundle["study_identity"] != second_bundle["study_identity"]
    assert first_bundle["execution_identity"] != second_bundle["execution_identity"]


def test_incomplete_manifest_fails_before_output_is_created(tmp_path):
    """Missing declarations stop the runner before any study work or persistence."""
    manifest = _manifest(tmp_path)
    manifest.pop("evaluation")

    with pytest.raises(ValueError, match="evaluation"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_incompatible_manifest_schema_fails_before_output_is_created(tmp_path):
    """A future or stale declaration cannot enter the current result contract."""
    manifest = _manifest(tmp_path)
    manifest["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_unavailable_git_commit_fails_before_output_is_created(tmp_path, monkeypatch):
    """A result is never emitted with unknown source-code provenance."""

    def git_unavailable():
        raise RuntimeError("cannot determine the current Git commit")

    monkeypatch.setattr(study_module, "_repository_state", git_unavailable)

    with pytest.raises(RuntimeError, match="Git commit"):
        run_study(_write_manifest(tmp_path, _manifest(tmp_path)))

    assert not (tmp_path / "outputs").exists()


def test_dirty_worktree_fails_before_output_is_created(tmp_path, monkeypatch):
    """Formal study results cannot claim a commit while executing modified code."""
    monkeypatch.setattr(
        study_module,
        "_repository_state",
        lambda: (_REPOSITORY_ROOT, _TEST_GIT_COMMIT, True),
    )

    with pytest.raises(ValueError, match="clean Git working tree"):
        run_study(_write_manifest(tmp_path, _manifest(tmp_path)))

    assert not (tmp_path / "outputs").exists()


def test_manifest_rejects_unsupported_consumer_mode_before_execution(tmp_path):
    """The association study accepts only its complete declared mode contrast."""
    manifest = _manifest(tmp_path)
    manifest["modes"] = ["fungus-only"]

    with pytest.raises(ValueError, match="modes"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_incompatible_execution_cannot_overwrite_existing_result_bundle(tmp_path):
    """One output identity cannot join results from different declarations."""
    manifest = _manifest(tmp_path, identity="immutable")
    manifest_path = _write_manifest(tmp_path, manifest)
    original = run_study(manifest_path)
    original_bundle = original.bundle_path.read_bytes()
    original_summary = original.summary_path.read_bytes()
    manifest["initial_p_micromolar"] = [1.0]
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="study identity"):
        run_study(manifest_path)

    assert original.bundle_path.read_bytes() == original_bundle
    assert original.summary_path.read_bytes() == original_summary


def test_execution_cannot_join_outputs_without_an_execution_identity(tmp_path):
    """Orphaned output files cannot be claimed by a new manifest."""
    manifest = _manifest(tmp_path, identity="orphaned")
    output_dir = tmp_path / "outputs" / "orphaned"
    output_dir.mkdir(parents=True)
    orphan = output_dir / "checkpoint.bin"
    orphan.write_bytes(b"existing work")

    with pytest.raises(ValueError, match="no compatible execution identity"):
        run_study(_write_manifest(tmp_path, manifest))

    assert orphan.read_bytes() == b"existing work"
    assert not (output_dir / "result-bundle.json").exists()


def test_compatible_incomplete_execution_preserves_completed_entries(tmp_path):
    """Resume fills missing conditions without replacing valid completed work."""
    manifest = _manifest(tmp_path, identity="resumable")
    manifest_path = _write_manifest(tmp_path, manifest)
    initial = run_study(manifest_path)
    interrupted = json.loads(initial.bundle_path.read_text(encoding="utf-8"))
    completed_entry = interrupted["entries"][0]
    completed_entry["evidence"] = {"sentinel": "must-survive-resume"}
    interrupted["entries"][1]["status"] = "pending"
    interrupted["completion"] = {"completed": 1, "requested": 2}
    interrupted["status"] = "incomplete"
    initial.bundle_path.write_text(
        json.dumps(interrupted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resumed = run_study(manifest_path)

    bundle = json.loads(resumed.bundle_path.read_text(encoding="utf-8"))
    assert bundle["entries"][0] == completed_entry
    assert bundle["entries"][1]["status"] == "completed"
    assert bundle["completion"] == {"completed": 2, "requested": 2}
    assert bundle["status"] == "complete"
    assert "Completed conditions: 2/2" in resumed.summary_path.read_text(
        encoding="utf-8"
    )


def test_manifest_rejects_duplicate_seed_ids_before_execution(tmp_path):
    """Every declared condition has one unambiguous master-seed identity."""
    manifest = _manifest(tmp_path)
    manifest["seeds"] = [7, 7]

    with pytest.raises(ValueError, match="seeds"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_manifest_rejects_duplicate_modes_before_execution(tmp_path):
    """Duplicate mode declarations cannot create duplicate condition identities."""
    manifest = _manifest(tmp_path)
    manifest["modes"] = ["mixed", "mixed"]

    with pytest.raises(ValueError, match="modes"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_manifest_rejects_nonpositive_initial_p_before_execution(tmp_path):
    """A study condition must declare a physically meaningful P treatment."""
    manifest = _manifest(tmp_path)
    manifest["initial_p_micromolar"] = [0.0]

    with pytest.raises(ValueError, match="initial_p_micromolar"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_runner_rejects_a_stage_without_an_executor(tmp_path):
    """Unimplemented scientific stages cannot be reported as completed fixtures."""
    manifest = _manifest(tmp_path)
    manifest["stage"] = "phase-1-pilot"

    with pytest.raises(ValueError, match="stage"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_manifest_rejects_horizon_inconsistent_with_timestep(tmp_path):
    """The declared endpoint must contain a whole number of transitions."""
    manifest = _manifest(tmp_path)
    manifest["horizon"]["days"] = 0.06

    with pytest.raises(ValueError, match="horizon"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_manifest_rejects_missing_training_budget_before_execution(tmp_path):
    """Execution cannot start until its training and checkpoint budgets are fixed."""
    manifest = _manifest(tmp_path)
    manifest["training"].pop("checkpoint_interval_timesteps")

    with pytest.raises(ValueError, match="training"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_manifest_rejects_incomplete_evaluation_settings(tmp_path):
    """Evaluation protocol and replication are fixed before conditions execute."""
    manifest = _manifest(tmp_path)
    manifest["evaluation"].pop("protocol")

    with pytest.raises(ValueError, match="evaluation"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_manifest_rejects_output_identity_that_is_a_path(tmp_path):
    """Output identity cannot redirect persistence outside its declared root."""
    manifest = _manifest(tmp_path)
    manifest["output"]["identity"] = "../escape"

    with pytest.raises(ValueError, match="output identity"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "escape").exists()


def test_manifest_rejects_incomplete_model_configuration(tmp_path):
    """Environment and species configuration are both preserved design inputs."""
    manifest = _manifest(tmp_path)
    manifest["model"].pop("species")

    with pytest.raises(ValueError, match="model"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_compatible_complete_execution_reuses_immutable_artifacts(tmp_path):
    """A completed bundle is returned as-is instead of being rewritten."""
    manifest = _manifest(tmp_path, identity="already-complete")
    manifest_path = _write_manifest(tmp_path, manifest)
    initial = run_study(manifest_path)
    bundle_bytes = initial.bundle_path.read_bytes()
    summary_bytes = initial.summary_path.read_bytes()
    initial.bundle_path.chmod(0o444)
    initial.summary_path.chmod(0o444)
    try:
        repeated = run_study(manifest_path)
    finally:
        initial.bundle_path.chmod(0o644)
        initial.summary_path.chmod(0o644)

    assert repeated == initial
    assert repeated.bundle_path.read_bytes() == bundle_bytes
    assert repeated.summary_path.read_bytes() == summary_bytes


def test_missing_summary_is_derived_without_rewriting_completed_bundle(tmp_path):
    """Human output can be recovered from its immutable machine source."""
    manifest = _manifest(tmp_path, identity="recover-summary")
    manifest_path = _write_manifest(tmp_path, manifest)
    initial = run_study(manifest_path)
    bundle_bytes = initial.bundle_path.read_bytes()
    initial.summary_path.unlink()
    initial.bundle_path.chmod(0o444)
    try:
        recovered = run_study(manifest_path)
    finally:
        initial.bundle_path.chmod(0o644)

    assert recovered.bundle_path.read_bytes() == bundle_bytes
    assert "Completed conditions: 2/2" in recovered.summary_path.read_text(
        encoding="utf-8"
    )


def test_result_bundle_records_versioned_interface_provenance(tmp_path):
    """Persisted results identify the software contracts required to consume them."""
    result = run_study(
        _write_manifest(tmp_path, _manifest(tmp_path, identity="provenance"))
    )

    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    assert bundle["provenance"] == {
        "actor_interface_version": "two-head-latent-v1",
        "dependency_lock_sha256": hashlib.sha256(
            (_REPOSITORY_ROOT / "uv.lock").read_bytes()
        ).hexdigest(),
        "environment_state_schema_version": "state-v2",
        "execution_kind": "contract-fixture",
        "git_commit": _TEST_GIT_COMMIT,
        "git_dirty": False,
        "jax_version": version("jax"),
        "jaxlib_version": version("jaxlib"),
        "manifest_schema_version": 1,
        "mycormarl_version": "0.1.0",
        "python_version": platform.python_version(),
        "result_format_version": 2,
    }


def test_existing_bundle_with_incompatible_provenance_is_rejected(tmp_path):
    """Matching manifest text cannot admit results from a stale interface."""
    manifest = _manifest(tmp_path, identity="stale-interface")
    manifest_path = _write_manifest(tmp_path, manifest)
    result = run_study(manifest_path)
    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    bundle["provenance"]["actor_interface_version"] = "legacy"
    result.bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance"):
        run_study(manifest_path)


def test_version_one_bundle_is_rejected_after_identity_contract_split(tmp_path):
    """Legacy bundles cannot bypass the version-two reproducibility contract."""
    manifest = _manifest(tmp_path, identity="legacy-bundle")
    manifest_path = _write_manifest(tmp_path, manifest)
    result = run_study(manifest_path)
    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    bundle["format_version"] = 1
    result.bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="result format"):
        run_study(manifest_path)


def test_non_object_existing_bundle_is_rejected_as_incompatible(tmp_path):
    """Syntactically valid JSON cannot bypass the result-bundle object contract."""
    manifest = _manifest(tmp_path, identity="non-object-bundle")
    manifest_path = _write_manifest(tmp_path, manifest)
    result = run_study(manifest_path)
    result.bundle_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="result format"):
        run_study(manifest_path)


def test_existing_bundle_without_embedded_manifest_is_rejected(tmp_path):
    """Stored identities must remain derivable from their canonical manifest."""
    manifest = _manifest(tmp_path, identity="missing-embedded-manifest")
    manifest_path = _write_manifest(tmp_path, manifest)
    result = run_study(manifest_path)
    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    bundle.pop("manifest")
    result.bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest provenance"):
        run_study(manifest_path)


def test_existing_bundle_from_another_git_commit_is_rejected(tmp_path, monkeypatch):
    """Results produced by different source commits cannot be joined or reused."""
    manifest = _manifest(tmp_path, identity="stale-commit")
    manifest_path = _write_manifest(tmp_path, manifest)
    result = run_study(manifest_path)
    original_bundle = result.bundle_path.read_bytes()
    original = json.loads(original_bundle)
    monkeypatch.setattr(
        study_module,
        "_repository_state",
        lambda: (_REPOSITORY_ROOT, "b" * 40, False),
    )

    with pytest.raises(ValueError, match="execution identity"):
        run_study(manifest_path)

    assert result.bundle_path.read_bytes() == original_bundle
    assert json.loads(result.bundle_path.read_text())["study_identity"] == original[
        "study_identity"
    ]


def test_existing_bundle_from_another_dependency_lock_is_rejected(
    tmp_path, monkeypatch
):
    """Dependency drift changes execution identity, not scientific study identity."""
    manifest = _manifest(tmp_path, identity="stale-lock")
    manifest_path = _write_manifest(tmp_path, manifest)
    monkeypatch.setattr(study_module, "_file_sha256", lambda path: "1" * 64)
    result = run_study(manifest_path)
    original_bundle = result.bundle_path.read_bytes()
    original = json.loads(original_bundle)
    monkeypatch.setattr(study_module, "_file_sha256", lambda path: "2" * 64)

    with pytest.raises(ValueError, match="execution identity"):
        run_study(manifest_path)

    assert result.bundle_path.read_bytes() == original_bundle
    assert json.loads(result.bundle_path.read_text())["study_identity"] == original[
        "study_identity"
    ]


def test_existing_complete_bundle_with_missing_condition_is_rejected(tmp_path):
    """Completion cannot conceal a missing member of the declared condition matrix."""
    manifest = _manifest(tmp_path, identity="missing-condition")
    manifest_path = _write_manifest(tmp_path, manifest)
    result = run_study(manifest_path)
    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    bundle["entries"].pop()
    result.bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="condition inventory"):
        run_study(manifest_path)
