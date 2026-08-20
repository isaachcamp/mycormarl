"""End-to-end contract tests for the public study runner."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from flax import serialization
import pytest

import mycormarl.domain_qualification as domain_qualification_module
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


def _accepted_domain_artifact(tmp_path):
    path = tmp_path / "accepted-domain.json"
    path.write_text(json.dumps({
        "status": "complete",
        "qualification": {"accepted_domain": {"name": "qualified"}},
    }), encoding="utf-8")
    return path


def _passed_pilot_qualifications(tmp_path):
    """Persist the three passed prerequisite artifacts for a reduced fixture."""
    artifacts = {
        "plant_growth": {"status": "passed"},
        "static_controls": {
            "status": "complete",
            "entries": [{"status": "completed"}],
        },
        "domain": {
            "status": "complete",
            "qualification": {"accepted_domain": {"name": "qualified"}},
        },
    }
    paths = {}
    for name, artifact in artifacts.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        paths[name] = str(path)
    return paths


def _pilot_manifest(tmp_path, *, fixture=True):
    manifest = _manifest(tmp_path, identity="phase-1-fixture")
    manifest["stage"] = "phase-1-pilot"
    manifest["pilot_fixture"] = fixture
    manifest["qualification_artifacts"] = _passed_pilot_qualifications(tmp_path)
    manifest["training"] = {
        "minimum_transition_budget": 1,
        "maximum_transition_budget": 1,
        "checkpoint_interval_timesteps": 1,
        "num_steps": 1,
        "num_envs": 1,
        "update_epochs": 1,
        "num_minibatches": 1,
        "stopping": {
            "evaluation_window_checkpoints": 2,
            "plateau_tolerances": {
                "plant_fitness_absolute": 0.0,
                "fungus_fitness_absolute": 0.0,
                "action_absolute": 0.0,
            },
        },
    }
    return manifest


def test_reduced_phase_1_fixture_uses_the_pilot_matrix_path(tmp_path):
    """A reduced fixture retains the Phase 1 qualification and matrix contract."""
    manifest = _pilot_manifest(tmp_path)

    bundle = json.loads(
        run_study(_write_manifest(tmp_path, manifest)).bundle_path.read_text()
    )

    assert bundle["manifest"]["stage"] == "phase-1-pilot"
    assert bundle["qualification_artifacts"] == manifest["qualification_artifacts"]
    assert {(entry["mode"], entry["initial_p_micromolar"], entry["seed"])
            for entry in bundle["entries"]} == {
        ("mixed", 0.3, 7), ("plant-only", 0.3, 7),
    }


def test_phase_1_checkpoint_records_ppo_diagnostics_for_training_analysis(tmp_path):
    """Every saved pilot checkpoint exposes optimizer and PPO-health summaries."""
    manifest = _pilot_manifest(tmp_path)
    manifest["training"].update({
        "minimum_transition_budget": 2,
        "maximum_transition_budget": 2,
    })

    bundle = json.loads(
        run_study(_write_manifest(tmp_path, manifest)).bundle_path.read_text()
    )
    diagnostics = bundle["entries"][0]["stopping_checkpoints"][0]["training_diagnostics"]

    assert set(diagnostics) == {"plant", "fungus"}
    for agent in diagnostics.values():
        assert {
            "learning_rate", "total_loss", "value_loss", "actor_loss",
            "approx_kl", "latent_entropy",
        } <= agent.keys()


def test_scientific_phase_1_manifest_fixes_the_range_finding_design(tmp_path):
    """The named pilot cannot silently drift from its predeclared 40-run design."""
    manifest = _pilot_manifest(tmp_path, fixture=False)
    manifest["initial_p_micromolar"] = [0.1, 0.3, 1.0, 3.0]
    manifest["seeds"] = [11, 12, 13, 14, 15]
    manifest["horizon"] = {"days": 120.0, "timestep_days": 0.025}
    manifest["modes"] = ["mixed", "plant-only"]
    manifest["initial_p_micromolar"] = [0.1, 0.3, 1.0]

    with pytest.raises(ValueError, match="Phase 1 pilot"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_phase_1_training_environment_uses_the_declared_depth_profile(tmp_path):
    """The qualified soil profile reaches PPO rather than only static controls."""
    manifest = _pilot_manifest(tmp_path)
    manifest["model"]["environment"].update({
        "soil_radius_cm": 1.0,
        "soil_depth_cm": 2.0,
        "radial_interval_cm": 1.0,
        "depth_interval_cm": 1.0,
        "initial_solution_p_depth_profile": [[0.0, 1.0], [2.0, 0.5]],
    })

    environment = study_module._training_environment(manifest, "mixed", 0.3)

    assert environment.config.initial_solution_p_depth_profile == [
        [0.0, 1.0], [2.0, 0.5]
    ]


def test_phase_1_pilot_retains_growth_qualification_as_provenance(tmp_path):
    """Growth qualification is recorded but does not gate range finding."""
    manifest = _pilot_manifest(tmp_path)
    plant_growth_path = Path(manifest["qualification_artifacts"]["plant_growth"])
    plant_growth_path.write_text('{"status": "failed"}', encoding="utf-8")

    bundle = json.loads(
        run_study(_write_manifest(tmp_path, manifest)).bundle_path.read_text()
    )

    assert bundle["qualification_artifacts"]["plant_growth"] == str(plant_growth_path)


def test_phase_1_fixture_resumes_only_the_missing_condition(tmp_path):
    """A compatible interrupted pilot preserves valid completed evidence."""
    manifest = _pilot_manifest(tmp_path)
    manifest_path = _write_manifest(tmp_path, manifest)
    first = run_study(manifest_path)
    interrupted = json.loads(first.bundle_path.read_text(encoding="utf-8"))
    completed_entry = interrupted["entries"][0]
    completed_entry["evidence"] = {"sentinel": "preserve completed pilot run"}
    interrupted["entries"][1] = {
        "mode": "plant-only",
        "initial_p_micromolar": 0.3,
        "seed": 7,
        "status": "pending",
    }
    interrupted["completion"] = {"completed": 1, "requested": 2}
    interrupted["status"] = "incomplete"
    first.bundle_path.write_text(
        json.dumps(interrupted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resumed = run_study(manifest_path)
    bundle = json.loads(resumed.bundle_path.read_text(encoding="utf-8"))

    assert bundle["entries"][0] == completed_entry
    assert bundle["entries"][1]["status"] in {"completed", "unconverged"}
    assert bundle["completion"] == {"completed": 2, "requested": 2}
    assert bundle["status"] == "complete"


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
    manifest["stage"] = "phase-1-dense-map"

    with pytest.raises(ValueError, match="stage"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_phase_1_pilot_analysis_reports_mode_level_endpoints_and_freezes_dense_design(
    tmp_path,
):
    """The runner derives pilot evidence and a prospective dense design together."""
    pilot_root = tmp_path / "pilot"
    pilot_root.mkdir()
    entries = []
    outcomes = {
        1: {"mixed": (10.0, 20.0), "plant-only": (4.0, 10.0)},
        2: {"mixed": (14.0, 4.0), "plant-only": (8.0, 2.0)},
    }
    for seed, modes in outcomes.items():
        for mode, (fitness, biomass) in modes.items():
            artifact = pilot_root / f"{mode}-{seed}.json"
            artifact.write_text(json.dumps({
                "format": "mycormarl-checkpoint-evaluation",
                "format_version": 1,
                "protocol": "latent-location",
                "episodes": [{"summary": {
                    "cumulative_reproductive_fitness": {"plant": fitness},
                    "final_living_biomass": {"plant": biomass},
                    "cumulative_gross_growth": {"plant": biomass + 1.0},
                }}],
            }), encoding="utf-8")
            entries.append({
                "mode": mode,
                "initial_p_micromolar": 0.3,
                "seed": seed,
                "status": "completed",
                "evaluation_artifacts": [artifact.name],
            })
    pilot_bundle = {
        "format": "mycormarl-study-result",
        "format_version": 2,
        "manifest": {
            "stage": "phase-1-pilot",
            "initial_p_micromolar": [0.3],
            "modes": ["mixed", "plant-only"],
            "seeds": [1, 2],
            "pilot_fixture": True,
        },
        "entries": entries,
        "completion": {"completed": 4, "requested": 4},
        "status": "complete",
    }
    pilot_path = pilot_root / "result-bundle.json"
    pilot_path.write_text(json.dumps(pilot_bundle), encoding="utf-8")
    domain_path = _accepted_domain_artifact(tmp_path)

    manifest = _manifest(tmp_path, identity="pilot-analysis")
    manifest["stage"] = "phase-1-pilot-analysis"
    manifest["pilot_result_bundle"] = str(pilot_path)
    manifest["dense_design"] = {
        "initial_p_micromolar": [0.1, 0.3, 1.0],
        "spacing": "logarithmic",
        "retained_pilot_levels": [0.3],
        "seeds": [11, 12, 13, 14],
        "target_delta_am_standard_error": 1.0,
        "domain_qualification_artifact": str(domain_path),
        "training_budget": {"minimum_transition_budget": 100, "maximum_transition_budget": 200},
    }

    result = run_study(_write_manifest(tmp_path, manifest))
    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))

    level = bundle["pilot_analysis"]["levels"][0]
    assert level["delta_am"] == pytest.approx(6.0)
    assert level["mgr_percent"] == pytest.approx(100.0)
    assert level["paired_differences"]["fitness"] == [6.0, 6.0]
    assert level["paired_delta_am_variance"] == pytest.approx(0.0)
    assert level["paired_scatter"]["fitness"] == [
        {"seed": 1, "mixed": 10.0, "plant_only": 4.0},
        {"seed": 2, "mixed": 14.0, "plant_only": 8.0},
    ]
    assert bundle["dense_design"] == manifest["dense_design"]
    assert bundle["pilot_analysis"]["precision"]["recommended_minimum_replication"] == 2
    assert bundle["dense_manifest"]["stage"] == "phase-1-dense-map"
    assert bundle["dense_manifest"]["initial_p_micromolar"] == [0.1, 0.3, 1.0]
    assert bundle["dense_manifest"]["qualification_artifacts"]["domain"] == str(domain_path)
    assert "confirmatory inference" in result.summary_path.read_text(encoding="utf-8")


def test_phase_1_pilot_analysis_marks_mgr_missing_for_zero_mean_plant_only_biomass(
    tmp_path,
):
    """A zero denominator remains explicit rather than becoming an epsilon ratio."""
    pilot_root = tmp_path / "pilot"
    pilot_root.mkdir()
    entries = []
    for seed, mode, biomass in ((1, "mixed", 3.0), (1, "plant-only", 0.0)):
        artifact = pilot_root / f"{mode}-{seed}.json"
        artifact.write_text(json.dumps({
            "format": "mycormarl-checkpoint-evaluation",
            "format_version": 1,
            "protocol": "latent-location",
            "episodes": [{"summary": {
                "cumulative_reproductive_fitness": {"plant": 1.0},
                "final_living_biomass": {"plant": biomass},
                "cumulative_gross_growth": {"plant": 1.0},
            }}],
        }), encoding="utf-8")
        entries.append({
            "mode": mode, "initial_p_micromolar": 0.3, "seed": seed,
            "status": "completed", "evaluation_artifacts": [artifact.name],
        })
    pilot_path = pilot_root / "result-bundle.json"
    pilot_path.write_text(json.dumps({
        "format": "mycormarl-study-result", "format_version": 2,
        "manifest": {"stage": "phase-1-pilot", "initial_p_micromolar": [0.3],
                     "modes": ["mixed", "plant-only"], "seeds": [1], "pilot_fixture": True},
        "entries": entries, "completion": {"completed": 2, "requested": 2}, "status": "complete",
    }), encoding="utf-8")
    domain_path = _accepted_domain_artifact(tmp_path)
    manifest = _manifest(tmp_path, identity="zero-mgr")
    manifest.update({"stage": "phase-1-pilot-analysis", "pilot_result_bundle": str(pilot_path), "dense_design": {
        "initial_p_micromolar": [0.1, 0.3], "spacing": "logarithmic",
        "retained_pilot_levels": [0.3], "seeds": [11, 12],
        "target_delta_am_standard_error": 10.0, "domain_qualification_artifact": str(domain_path),
        "training_budget": {"minimum_transition_budget": 100, "maximum_transition_budget": 200},
    }})

    bundle = json.loads(run_study(_write_manifest(tmp_path, manifest)).bundle_path.read_text())

    level = bundle["pilot_analysis"]["levels"][0]
    assert level["mgr_percent"] is None
    assert level["mgr_missing_reason"] == "mean plant-only living biomass is zero"


def test_phase_1_pilot_analysis_supports_an_overall_noisy_decline_on_the_complete_grid(
    tmp_path,
):
    """One upward noisy step does not discard the predeclared P-response tendency."""
    pilot_root = tmp_path / "pilot"
    pilot_root.mkdir()
    levels = [0.1, 0.3, 1.0, 3.0]
    deltas = [8.0, 5.0, 6.0, 0.0]
    entries = []
    for level, delta in zip(levels, deltas, strict=True):
        for mode, fitness in (("mixed", delta), ("plant-only", 0.0)):
            artifact = pilot_root / f"{mode}-{level}.json"
            artifact.write_text(json.dumps({
                "format": "mycormarl-checkpoint-evaluation", "format_version": 1,
                "protocol": "latent-location", "episodes": [{"summary": {
                    "cumulative_reproductive_fitness": {"plant": fitness},
                    "final_living_biomass": {"plant": 1.0},
                    "cumulative_gross_growth": {"plant": 1.0},
                }}],
            }), encoding="utf-8")
            entries.append({"mode": mode, "initial_p_micromolar": level, "seed": 1,
                            "status": "completed", "evaluation_artifacts": [artifact.name]})
    pilot_path = pilot_root / "result-bundle.json"
    pilot_path.write_text(json.dumps({
        "format": "mycormarl-study-result", "format_version": 2,
        "manifest": {"stage": "phase-1-pilot", "initial_p_micromolar": levels,
                     "modes": ["mixed", "plant-only"], "seeds": [1], "pilot_fixture": True},
        "entries": entries, "completion": {"completed": 8, "requested": 8}, "status": "complete",
    }), encoding="utf-8")
    domain_path = _accepted_domain_artifact(tmp_path)
    manifest = _manifest(tmp_path, identity="noisy-tendency")
    manifest.update({"stage": "phase-1-pilot-analysis", "pilot_result_bundle": str(pilot_path), "dense_design": {
        "initial_p_micromolar": levels, "spacing": "logarithmic",
        "retained_pilot_levels": levels, "seeds": [11, 12],
        "target_delta_am_standard_error": 10.0, "domain_qualification_artifact": str(domain_path),
        "training_budget": {"minimum_transition_budget": 100, "maximum_transition_budget": 200},
    }})

    bundle = json.loads(run_study(_write_manifest(tmp_path, manifest)).bundle_path.read_text())

    tendency = bundle["pilot_analysis"]["tendency"]
    assert tendency["classification"] == "supported-decreasing"
    assert tendency["uses_complete_predeclared_grid"] is True
    assert tendency["adjacent_monotonicity_required"] is False


def test_phase_1_pilot_analysis_reports_unconverged_runs_as_unresolved(tmp_path):
    """A training failure remains visible and cannot become a response tendency."""
    pilot_root = tmp_path / "pilot"
    pilot_root.mkdir()
    entries = []
    for mode in ("mixed", "plant-only"):
        artifact = pilot_root / f"{mode}.json"
        artifact.write_text(json.dumps({
            "format": "mycormarl-checkpoint-evaluation", "format_version": 1,
            "protocol": "latent-location", "episodes": [{"summary": {
                "cumulative_reproductive_fitness": {"plant": 1.0},
                "final_living_biomass": {"plant": 1.0},
                "cumulative_gross_growth": {"plant": 1.0},
            }}],
        }), encoding="utf-8")
        entries.append({"mode": mode, "initial_p_micromolar": 0.3, "seed": 1,
                        "status": "unconverged" if mode == "mixed" else "completed",
                        "evaluation_artifacts": [artifact.name]})
    pilot_path = pilot_root / "result-bundle.json"
    pilot_path.write_text(json.dumps({
        "format": "mycormarl-study-result", "format_version": 2,
        "manifest": {"stage": "phase-1-pilot", "initial_p_micromolar": [0.3],
                     "modes": ["mixed", "plant-only"], "seeds": [1], "pilot_fixture": True},
        "entries": entries, "completion": {"completed": 2, "requested": 2}, "status": "complete",
    }), encoding="utf-8")
    domain_path = _accepted_domain_artifact(tmp_path)
    manifest = _manifest(tmp_path, identity="unresolved-pilot")
    manifest.update({"stage": "phase-1-pilot-analysis", "pilot_result_bundle": str(pilot_path), "dense_design": {
        "initial_p_micromolar": [0.1, 0.3], "spacing": "logarithmic",
        "retained_pilot_levels": [0.3], "seeds": [11, 12],
        "target_delta_am_standard_error": 1.0, "domain_qualification_artifact": str(domain_path),
        "training_budget": {"minimum_transition_budget": 100, "maximum_transition_budget": 200},
    }})

    bundle = json.loads(run_study(_write_manifest(tmp_path, manifest)).bundle_path.read_text())

    assert bundle["pilot_analysis"]["tendency"]["classification"] == "unresolved-training-or-range-failure"
    assert bundle["pilot_analysis"]["unresolved_conditions"] == [
        {"mode": "mixed", "initial_p_micromolar": 0.3, "seed": 1, "status": "unconverged"},
    ]


def test_phase_1_pilot_analysis_rejects_an_unaccepted_dense_domain_artifact(tmp_path):
    """A prospective dense map inherits evidence, not an arbitrary path label."""
    domain_path = tmp_path / "rejected-domain.json"
    domain_path.write_text('{"status": "failed"}', encoding="utf-8")
    manifest = _manifest(tmp_path, identity="unaccepted-domain")
    manifest.update({"stage": "phase-1-pilot-analysis", "pilot_result_bundle": "missing-pilot.json", "dense_design": {
        "initial_p_micromolar": [0.1, 0.3], "spacing": "logarithmic",
        "retained_pilot_levels": [0.3], "seeds": [11, 12],
        "target_delta_am_standard_error": 1.0, "domain_qualification_artifact": str(domain_path),
        "training_budget": {"minimum_transition_budget": 100, "maximum_transition_budget": 200},
    }})

    with pytest.raises(ValueError, match="domain qualification artifact is not accepted"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_domain_qualification_emits_one_frozen_accepted_candidate(tmp_path):
    """Domain qualification records evidence and freezes the smallest safe grid."""
    static_policy = {
        "plant": [0.0, 1.0, 0.0, 0.0],
        "fungus": [0.0, 1.0, 0.0, 0.0],
    }

    domain_manifest = _manifest(tmp_path / "domain", identity="domain")
    domain_manifest["stage"] = "domain-qualification"
    domain_manifest["modes"] = ["plant-only"]
    domain_manifest["static_policy"] = static_policy
    domain_manifest["model"]["species"]["plant"] = {"max_rooting_depth_cm": 0.5, "kfroot": 0.001}
    domain_manifest["domain_qualification"] = {
        "depth_profile": [[0.0, 1.0], [2.0, 1.0]],
        "candidates": [
                {"name": "small", "soil_radius_cm": 10.0, "soil_depth_cm": 1.0},
                {"name": "enlarged", "soil_radius_cm": 20.0, "soil_depth_cm": 2.0},
        ],
    }
    result = run_study(_write_manifest(tmp_path / "domain", domain_manifest))

    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    assert bundle["status"] == "complete"
    assert bundle["qualification"]["accepted_domain"]["name"] == "small"
    assert len(bundle["qualification"]["candidates"]) == 2
    assert all("runtime_seconds" in candidate for candidate in bundle["qualification"]["candidates"])
    assert all("peak_memory_bytes" in candidate for candidate in bundle["qualification"]["candidates"])
    assert bundle["qualification"]["accepted_domain"]["direct_plant_uptake_behavior"]["stable"]
    first_bytes = result.bundle_path.read_bytes()
    repeated = run_study(_write_manifest(tmp_path / "domain", domain_manifest))
    assert repeated.bundle_path.read_bytes() == first_bytes


def test_domain_qualification_records_exact_direct_plant_uptake_comparison(tmp_path):
    """Depth qualification compares the cumulative root-to-plant P flux."""
    static_policy = {
        "plant": [0.0, 1.0, 0.0, 0.0],
        "fungus": [0.0, 1.0, 0.0, 0.0],
    }

    domain_manifest = _manifest(tmp_path / "domain", identity="direct-uptake-domain")
    domain_manifest["stage"] = "domain-qualification"
    domain_manifest["modes"] = ["plant-only"]
    domain_manifest["static_policy"] = static_policy
    domain_manifest["model"]["species"]["plant"] = {"kfroot": 0.001}
    domain_manifest["domain_qualification"] = {
        "direct_plant_uptake_relative_tolerance": 1.0,
        "depth_profile": [[5.0, 1.0], [15.0, 0.345], [30.0, 0.170], [60.0, 0.103], [100.0, 0.069]],
        "candidates": [
            {"name": "small", "soil_radius_cm": 10.0, "soil_depth_cm": 0.5},
            {"name": "reference", "soil_radius_cm": 10.0, "soil_depth_cm": 1.0},
        ],
    }
    bundle = json.loads(run_study(_write_manifest(tmp_path / "domain", domain_manifest)).bundle_path.read_text())

    record = bundle["qualification"]["accepted_domain"]["records"][0]
    comparison = bundle["qualification"]["accepted_domain"]["direct_plant_uptake_behavior"]
    assert record["cumulative_direct_plant_p_uptake_micromol"] >= 0.0
    assert comparison["relative_tolerance"] == pytest.approx(1.0)
    assert comparison["stable"]


def test_domain_qualification_records_uniform_depth_treatment_without_a_profile(tmp_path):
    """A qualification can request uniform P across each candidate's full depth."""
    static_policy = {
        "plant": [0.0, 1.0, 0.0, 0.0],
        "fungus": [0.0, 1.0, 0.0, 0.0],
    }

    domain_manifest = _manifest(tmp_path / "domain", identity="uniform-domain")
    domain_manifest["stage"] = "domain-qualification"
    domain_manifest["static_policy"] = static_policy
    domain_manifest["model"]["species"]["plant"] = {"max_rooting_depth_cm": 0.5, "kfroot": 0.001}
    domain_manifest["domain_qualification"] = {
        "candidates": [
            {"name": "small", "soil_radius_cm": 10.0, "soil_depth_cm": 1.0},
            {"name": "reference", "soil_radius_cm": 20.0, "soil_depth_cm": 2.0},
        ],
    }

    bundle = json.loads(run_study(_write_manifest(tmp_path / "domain", domain_manifest)).bundle_path.read_text())

    assert not bundle["qualification"]["initial_solution_p_profiled"]


def test_domain_qualification_selects_the_smallest_largest_domain_match_without_fungal_contact(
    tmp_path, monkeypatch,
):
    """The selected depth passes both the fungal boundary and largest-depth gates."""
    static_policy = {
        "plant": [0.0, 1.0, 0.0, 0.0],
        "fungus": [0.0, 1.0, 0.0, 0.0],
    }

    uptake = {"5-cm": 80.0, "10-cm": 90.0, "20-cm": 98.0, "30-cm": 100.0}

    def trajectory(manifest, candidate, mode, p_level, seed, actions, depth_profile):
        return {
            "fungal_lower_boundary_contact": candidate["name"] == "5-cm",
            "fungal_lower_boundary_first_contact_step": 1 if candidate["name"] == "5-cm" else None,
            "initial_p_inventory_micromol": 1.0,
            "final_plant_biomass_g": 1.0,
            "cumulative_direct_plant_p_uptake_micromol": uptake[candidate["name"]],
            "final_soil_inventory_micromol": 0.5,
            "soil_inventory_trace_micromol": [1.0, 0.5],
            "depletion_fraction": 0.5,
        }

    monkeypatch.setattr(domain_qualification_module, "_trajectory", trajectory)
    domain_manifest = _manifest(tmp_path / "domain", identity="adjacent-depth-domain")
    domain_manifest["stage"] = "domain-qualification"
    domain_manifest["static_policy"] = static_policy
    domain_manifest["domain_qualification"] = {
        "candidates": [
            {"name": "5-cm", "soil_radius_cm": 1.0, "soil_depth_cm": 5.0},
            {"name": "10-cm", "soil_radius_cm": 1.0, "soil_depth_cm": 10.0},
            {"name": "20-cm", "soil_radius_cm": 1.0, "soil_depth_cm": 20.0},
            {"name": "30-cm", "soil_radius_cm": 1.0, "soil_depth_cm": 30.0},
        ],
    }

    result = run_study(_write_manifest(tmp_path / "domain", domain_manifest))
    qualification = json.loads(result.bundle_path.read_text())["qualification"]

    assert qualification["accepted_domain"]["name"] == "20-cm"
    candidates = {candidate["name"]: candidate for candidate in qualification["candidates"]}
    assert "fungal lower boundary contact" in candidates["5-cm"]["rejection_reasons"]
    assert candidates["10-cm"]["direct_plant_uptake_behavior"]["compared_to"] == "30-cm"
    assert candidates["10-cm"]["status"] == "rejected"
    assert candidates["20-cm"]["direct_plant_uptake_behavior"]["compared_to"] == "30-cm"
    assert candidates["30-cm"]["status"] == "comparison-reference"
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "## Domain qualification" in summary
    assert "20-cm" in summary
    assert "Profiled initial Pi: no" in summary


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


def test_comparison_block_requires_predeclared_blind_stopping_rule(tmp_path):
    """A comparison block cannot begin without its common stopping protocol."""
    manifest = _manifest(tmp_path)
    manifest["stage"] = "comparison-block-training"
    manifest["training"] = {
        "minimum_transition_budget": 1,
        "maximum_transition_budget": 2,
        "checkpoint_interval_timesteps": 1,
    }

    with pytest.raises(ValueError, match="stopping"):
        run_study(_write_manifest(tmp_path, manifest))

    assert not (tmp_path / "outputs").exists()


def test_comparison_block_retains_non_plateau_runs_as_unconverged(tmp_path, monkeypatch):
    """Maximum-budget failures remain auditable rather than becoming endpoints."""
    manifest = _manifest(tmp_path, identity="unconverged-block")
    manifest["stage"] = "comparison-block-training"
    manifest["modes"] = ["plant-only"]
    manifest["training"] = {
        "minimum_transition_budget": 1,
        "maximum_transition_budget": 2,
        "checkpoint_interval_timesteps": 1,
        "num_steps": 1,
        "num_envs": 1,
        "update_epochs": 1,
        "num_minibatches": 1,
        "stopping": {
            "evaluation_window_checkpoints": 2,
            "plateau_tolerances": {
                "plant_fitness_absolute": 0.0,
                "fungus_fitness_absolute": 0.0,
                "action_absolute": 0.0,
            },
        },
    }
    calls = {"plant-only": 0}

    def metrics(checkpoint, _environment, **_kwargs):
        mode = checkpoint.parent.parent.name.split("-p", maxsplit=1)[0]
        calls[mode] += 1
        actions = {"plant": [float(calls[mode])] * 4}
        fitness = {"plant": 1.0}
        return {"fitness": fitness, "actions": actions}

    monkeypatch.setattr(study_module, "evaluate_checkpoint_summary", metrics)

    result = run_study(_write_manifest(tmp_path, manifest))
    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))

    assert bundle["status"] == "complete"
    assert {entry["status"] for entry in bundle["entries"]} == {"unconverged"}
    assert {entry["transitions"] for entry in bundle["entries"]} == {2}
    for entry in bundle["entries"]:
        decision = entry["stopping_decision"]
        assert decision["outcome"] == "maximum-budget-unconverged"
        assert decision["evaluation_window_checkpoints"] == 2
        assert set(decision["plateau_metrics"]) == {"plant", "actions"} | (
            {"fungus"} if entry["mode"] == "mixed" else set()
        )


def test_comparison_block_never_stops_before_minimum_and_records_plateau(tmp_path):
    """A stable checkpoint only completes after the shared minimum budget."""
    manifest = _manifest(tmp_path, identity="plateau-block")
    manifest["stage"] = "comparison-block-training"
    manifest["training"] = {
        "minimum_transition_budget": 2,
        "maximum_transition_budget": 3,
        "checkpoint_interval_timesteps": 1,
        "num_steps": 1,
        "num_envs": 1,
        "update_epochs": 1,
        "num_minibatches": 1,
        "stopping": {
            "evaluation_window_checkpoints": 2,
            "plateau_tolerances": {
                "plant_fitness_absolute": 1e9,
                "fungus_fitness_absolute": 1e9,
                "action_absolute": 1e9,
            },
        },
    }

    bundle = json.loads(
        run_study(_write_manifest(tmp_path, manifest)).bundle_path.read_text(
            encoding="utf-8"
        )
    )

    assert {entry["status"] for entry in bundle["entries"]} == {"completed"}
    assert {entry["transitions"] for entry in bundle["entries"]} == {2}
    assert all(
        entry["stopping_decision"]["outcome"] == "plateau-complete"
        and entry["stopping_decision"]["checkpoint_transitions"] == [1, 2]
        for entry in bundle["entries"]
    )


def test_fitness_plateau_uses_relative_window_scale_with_absolute_floor():
    """Fitness stability is meaningful across both low- and high-fitness agents."""
    training = {
        "minimum_transition_budget": 2,
        "maximum_transition_budget": 2,
        "stopping": {
            "evaluation_window_checkpoints": 3,
            "plateau_tolerances": {
                "fitness_absolute_floor": 1e-4,
                "fitness_relative": 0.2,
                "action_absolute": 0.0,
            },
        },
    }
    checkpoints = [
        {"transitions": transition, "metrics": {
            "fitness": {"plant": fitness},
            "actions": {"plant": [0.0] * 4},
        }}
        for transition, fitness in ((1, 0.010), (2, 0.012), (3, 0.011))
    ]

    decision = study_module._stopping_decision(checkpoints, training, "plant-only")

    plant = decision["plateau_metrics"]["plant"]
    assert plant["fitness_span"] == pytest.approx(0.002)
    assert plant["fitness_scale"] == pytest.approx(0.012)
    assert plant["fitness_tolerance"] == pytest.approx(0.0025)
    assert plant["stable"] is True


def test_scale_aware_fitness_tolerance_requires_both_scale_terms(tmp_path):
    """A partial relative declaration cannot silently fall back to a raw rule."""
    manifest = _manifest(tmp_path)
    manifest["stage"] = "comparison-block-training"
    manifest["training"] = {
        "minimum_transition_budget": 1,
        "maximum_transition_budget": 2,
        "checkpoint_interval_timesteps": 1,
        "stopping": {
            "evaluation_window_checkpoints": 2,
            "plateau_tolerances": {
                "fitness_relative": 0.2,
                "plant_fitness_absolute": 0.0,
                "fungus_fitness_absolute": 0.0,
                "action_absolute": 0.0,
            },
        },
    }

    with pytest.raises(ValueError, match="stopping"):
        run_study(_write_manifest(tmp_path, manifest))


def test_mixed_mode_requires_fungal_fitness_stability(tmp_path, monkeypatch):
    """Fungal improvement keeps a mixed run open even when plant metrics plateau."""
    manifest = _manifest(tmp_path, identity="fungal-gate")
    manifest["stage"] = "comparison-block-training"
    manifest["training"] = {
        "minimum_transition_budget": 1,
        "maximum_transition_budget": 2,
        "checkpoint_interval_timesteps": 1,
        "num_steps": 1,
        "num_envs": 1,
        "update_epochs": 1,
        "num_minibatches": 1,
        "stopping": {
            "evaluation_window_checkpoints": 2,
            "plateau_tolerances": {
                "plant_fitness_absolute": 0.0,
                "fungus_fitness_absolute": 0.1,
                "action_absolute": 0.0,
            },
        },
    }
    calls = {"mixed": 0, "plant-only": 0}

    def metrics(checkpoint, _environment, **_kwargs):
        mode = checkpoint.parent.parent.name.split("-p", maxsplit=1)[0]
        calls[mode] += 1
        result = {"fitness": {"plant": 1.0}, "actions": {"plant": [0.0] * 4}}
        if mode == "mixed":
            result["fitness"]["fungus"] = float(calls[mode] - 1)
            result["actions"]["fungus"] = [0.0] * 4
        return result

    monkeypatch.setattr(study_module, "evaluate_checkpoint_summary", metrics)
    bundle = json.loads(
        run_study(_write_manifest(tmp_path, manifest)).bundle_path.read_text(
            encoding="utf-8"
        )
    )
    entries = {entry["mode"]: entry for entry in bundle["entries"]}

    assert entries["plant-only"]["status"] == "completed"
    assert entries["mixed"]["status"] == "unconverged"
    assert not entries["mixed"]["stopping_decision"]["plateau_metrics"]["fungus"]["stable"]


def test_comparison_block_rejects_selective_stop_boundary(tmp_path):
    """A caller cannot grant one condition extra optimization effort."""
    manifest = _manifest(tmp_path, identity="no-selective-extension")
    manifest["stage"] = "comparison-block-training"
    manifest["training"] = {
        "minimum_transition_budget": 1,
        "maximum_transition_budget": 2,
        "checkpoint_interval_timesteps": 1,
        "stopping": {
            "evaluation_window_checkpoints": 2,
            "plateau_tolerances": {
                "plant_fitness_absolute": 0.0,
                "fungus_fitness_absolute": 0.0,
                "action_absolute": 0.0,
            },
        },
    }

    with pytest.raises(ValueError, match="selective"):
        run_study(_write_manifest(tmp_path, manifest), stop_after_timesteps=1)

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


def test_single_condition_training_can_stop_and_resume_from_checkpoint(tmp_path):
    """A stopped training condition resumes to the uninterrupted endpoint."""
    manifest = _manifest(tmp_path, identity="training-resume")
    manifest["stage"] = "single-condition-training"
    manifest["modes"] = ["plant-only"]
    manifest["training"].update(
        {"num_steps": 1, "num_envs": 1, "update_epochs": 1, "num_minibatches": 1}
    )
    manifest_path = _write_manifest(tmp_path, manifest)

    stopped = run_study(manifest_path, stop_after_timesteps=1)
    stopped_bundle = json.loads(stopped.bundle_path.read_text(encoding="utf-8"))
    assert stopped_bundle["status"] == "incomplete"
    assert stopped_bundle["entries"][0]["status"] == "pending"
    checkpoint = tmp_path / "outputs" / "training-resume" / "checkpoints" / "checkpoint-00000001.msgpack"
    assert checkpoint.exists()
    checkpoint_payload = serialization.msgpack_restore(checkpoint.read_bytes())
    assert checkpoint_payload["format"] == "mycormarl-ppo-checkpoint"
    assert checkpoint_payload["metadata"]["mode"] == "plant-only"
    assert checkpoint_payload["metadata"]["initial_p_micromolar"] == 0.3
    assert checkpoint_payload["metadata"]["seed"] == 7
    assert checkpoint_payload["metadata"]["transitions"] == 1
    assert set(checkpoint_payload["runner_state"]) == {"0", "1", "2", "3"}
    assert set(checkpoint_payload["runner_state"]["0"]) == {"plant", "fungus"}
    assert "opt_state" in checkpoint_payload["runner_state"]["0"]["plant"]
    first_evaluation = (
        tmp_path / "outputs" / "training-resume" / "evaluations"
        / "checkpoint-00000001.json"
    )
    assert first_evaluation.exists()
    assert json.loads(first_evaluation.read_text(encoding="utf-8"))["protocol"] == "latent-location"

    resumed = run_study(manifest_path)
    resumed_bundle = json.loads(resumed.bundle_path.read_text(encoding="utf-8"))
    assert resumed_bundle["status"] == "complete"
    assert resumed_bundle["entries"][0]["status"] == "completed"
    assert resumed_bundle["entries"][0]["transitions"] == 2
    assert (
        tmp_path / "outputs" / "training-resume" / "evaluations"
        / "checkpoint-00000002.json"
    ).exists()

    uninterrupted_manifest = _manifest(tmp_path / "uninterrupted", identity="training-full")
    uninterrupted_manifest["stage"] = "single-condition-training"
    uninterrupted_manifest["modes"] = ["plant-only"]
    uninterrupted_manifest["training"].update(
        {"num_steps": 1, "num_envs": 1, "update_epochs": 1, "num_minibatches": 1}
    )
    full = run_study(
        _write_manifest(tmp_path / "uninterrupted", uninterrupted_manifest)
    )
    full_bundle = json.loads(full.bundle_path.read_text(encoding="utf-8"))
    assert resumed_bundle["entries"][0]["evaluation"] == full_bundle["entries"][0]["evaluation"]
