import jax.numpy as jnp
import pytest
from pathlib import Path
import json

import mycormarl.study as study_module
from mycormarl.static_controls import run_batched_static_controls, run_static_controls
from mycormarl.study import run_study


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _manifest():
    return {
        "horizon": {"days": 0.1, "timestep_days": 0.05},
        "modes": ["mixed", "plant-only"],
        "initial_p_micromolar": [0.3, 1.0],
        "seeds": [7],
        "model": {
            "environment": {
                "soil_radius_cm": 1.0,
                "soil_depth_cm": 1.0,
                "radial_interval_cm": 0.5,
                "depth_interval_cm": 0.5,
            },
            "species": {"plant": {}, "fungus": {}},
        },
        "static_policy": {
            "plant": [0.0, 0.0, 0.0, 1.0],
            "fungus": [0.0, 0.0, 0.0, 1.0],
        },
    }


def test_static_controls_report_uniform_reset_and_finite_inventory_dynamics():
    result = run_static_controls(_manifest())

    assert result["status"] == "complete"
    assert len(result["entries"]) == 4
    assert all(entry["uniform_initial_p"] for entry in result["entries"])
    assert all(entry["soil_inventory_initial"] > 0.0 for entry in result["entries"])
    assert all("soil_inventory_final" in entry for entry in result["entries"])
    assert all("p_accounting_residual" in entry for entry in result["entries"])
    assert all(abs(entry["p_accounting_residual"]) < 1e-5 for entry in result["entries"])
    assert all(
        set(entry["p_loss_or_export_counters"])
        == {
            "plant_mortality",
            "fungus_mortality",
            "plant_maintenance",
            "fungus_maintenance",
            "plant_reproduction",
            "fungus_reproduction",
        }
        for entry in result["entries"]
    )


def test_static_controls_accept_independent_nonnegative_per_day_rates():
    manifest = _manifest()
    manifest["static_policy"] = {
        "plant": [1.25, 2.0, 0.5, 0.25],
        "fungus": [1.25, 2.0, 0.5, 0.25],
    }

    result = run_static_controls(manifest)

    assert result["status"] == "complete"


def test_rate_action_qualification_static_controls_use_declared_trade_and_allocation_rates():
    manifest_path = (
        _REPOSITORY_ROOT
        / "docs/qualification/rate-action-49/manifests/static-controls.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["static_policy"] == {
        "plant": [0.051293, 6.907755, 0.0, 0.0],
        "fungus": [1.203973, 6.907755, 0.0, 0.0],
    }
    assert manifest["seeds"] == [0]
    assert manifest["output"]["identity"] == "phase-1-static-controls-growth-only-13p"
    assert manifest["initial_p_micromolar"] == [
        0.1,
        0.3,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
        1.125,
        1.25,
        1.5,
        3.0,
        5.0,
        10.0,
    ]


def test_static_controls_emit_the_common_study_result_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study_module,
        "_repository_state",
        lambda: (_REPOSITORY_ROOT, "a" * 40, False),
    )
    manifest = _manifest()
    manifest.update(
        {
            "schema_version": 1,
            "stage": "static-controls",
            "training": {"total_timesteps": 1, "checkpoint_interval_timesteps": 1},
            "evaluation": {"protocol": "latent-location", "episodes": 1},
            "output": {"directory": str(tmp_path / "outputs"), "identity": "controls"},
        }
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_study(path)
    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))

    assert bundle["format"] == "mycormarl-study-result"
    assert bundle["manifest"]["stage"] == "static-controls"
    assert bundle["completion"] == {"completed": 4, "requested": 4}
    assert bundle["status"] == "complete"


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param([0.0, 0.0, 0.0, 1.0], id="reserve-only"),
        pytest.param([0.0, 1.0, 0.0, 0.0], id="growth-only"),
        pytest.param([0.0, 0.0, 1.0, 0.0], id="reproduction-only"),
        pytest.param([0.5, 0.0, 0.0, 1.0], id="fixed-trade-and-reserve"),
    ],
)
def test_static_controls_execute_each_declared_allocation_policy(policy):
    """Every valid physical allocation is exercised through the control seam."""
    manifest = _manifest()
    manifest["initial_p_micromolar"] = [1.0]
    manifest["static_policy"] = {"plant": policy, "fungus": policy}

    result = run_static_controls(manifest)

    assert result["status"] == "complete"
    assert {entry["mode"] for entry in result["entries"]} == {"mixed", "plant-only"}
    assert all(entry["status"] == "completed" for entry in result["entries"])
    assert all(entry["uniform_initial_p"] for entry in result["entries"])
    assert all(abs(entry["p_accounting_residual"]) < 1e-5 for entry in result["entries"])


def test_static_controls_report_fixed_trade_as_a_transfer_observable():
    manifest = _manifest()
    manifest["initial_p_micromolar"] = [1.0]
    manifest["static_policy"] = {
        "plant": [0.5, 0.0, 0.0, 1.0],
        "fungus": [0.5, 0.0, 0.0, 1.0],
    }

    result = run_static_controls(manifest)
    mixed = next(entry for entry in result["entries"] if entry["mode"] == "mixed")

    assert mixed["transfers"]["plant_c_out"] > 0.0
    assert mixed["transfers"]["fungus_p_out"] > 0.0


def test_batched_static_controls_match_serial_controls_for_homogeneous_mixed_policy():
    """P-specific resets share one vmapped rollout without changing endpoints."""
    manifest = _manifest()
    manifest["modes"] = ["mixed"]
    manifest["static_policy"] = {
        "plant": [0.05, 1.0, 0.0, 0.0],
        "fungus": [0.75, 1.0, 0.0, 0.0],
    }
    serial = run_static_controls(manifest)
    batched = run_batched_static_controls(manifest)

    assert batched["completion"] == serial["completion"]
    for expected, actual in zip(serial["entries"], batched["entries"], strict=True):
        assert actual["initial_p_micromolar"] == expected["initial_p_micromolar"]
        assert actual["transfers"] == pytest.approx(expected["transfers"])
        assert actual["final_living_biomass"] == pytest.approx(expected["final_living_biomass"])


def test_static_controls_record_gamma_normalized_limitation_trace():
    manifest = _manifest()
    manifest.update({
        "modes": ["mixed"],
        "initial_p_micromolar": [1.0],
        "record_limitation_trace": True,
        "static_policy": {
            "plant": [0.0, 1.0, 0.0, 0.0],
            "fungus": [0.0, 1.0, 0.0, 0.0],
        },
    })

    entry = run_static_controls(manifest)["entries"][0]
    trace = entry["limitation_trace"]

    assert len(trace) == entry["steps"] == 2
    assert {row["agents"]["plant"]["limiting_resource"] for row in trace} <= {
        "none", "carbon", "phosphate", "balanced"
    }
    for row in trace:
        for agent in ("plant", "fungus"):
            values = row["agents"][agent]
            assert values["allocated_c_normalized"] >= 0.0
            assert values["allocated_p_normalized"] >= 0.0
            assert values["used_c_normalized"] >= 0.0
            assert values["used_p_normalized"] >= 0.0
            assert values["acquired_c"] >= 0.0
            assert values["acquired_p"] >= 0.0
            assert values["maintenance_c_used"] >= 0.0
            assert values["maintenance_p_used"] >= 0.0
            assert values["maintenance_fraction_of_prior_acquired_c"] >= 0.0
            assert values["maintenance_fraction_of_prior_acquired_p"] >= 0.0
            assert "signed_pressure" in values
            assert values["signed_pressure"] == pytest.approx(
                values["allocated_c_normalized"] - values["allocated_p_normalized"]
            )
            assert "trade_out_raw" in values
            assert "trade_in_raw" in values


def test_static_controls_reject_non_finite_rate_actions_without_running_a_condition():
    manifest = _manifest()
    manifest["static_policy"]["plant"] = [jnp.nan, 0.0, 0.0, 1.0]

    result = run_static_controls(manifest)

    assert result["status"] == "rejected"
    assert result["entries"][0]["status"] == "rejected"
    assert "invalid Rate action" in result["entries"][0]["rejection_reasons"][0]


def test_static_controls_make_accounting_and_depletion_failures_explicit():
    manifest = _manifest()
    manifest["horizon"]["days"] = 0.5
    manifest["static_policy"]["plant"] = [0.0, 1.0, 0.0, 0.0]
    manifest["static_policy"]["fungus"] = [0.0, 1.0, 0.0, 0.0]
    manifest["model"]["species"] = {
        "plant": {"initial_p_pool": 0.0, "kappa_p": 1.0, "kappa_c": 1.0, "death_fraction": 0.99},
        "fungus": {"initial_p_pool": 0.0, "kappa_p": 1.0, "kappa_c": 1.0, "death_fraction": 0.99},
    }

    result = run_static_controls(manifest)

    assert result["status"] == "rejected"
    assert any(
        "biological failure" in reason
        for entry in result["entries"]
        for reason in entry["rejection_reasons"]
    )
