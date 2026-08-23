import json
import math
from pathlib import Path

import pytest

import mycormarl.resource_pressure_screen as screen
from mycormarl.resource_pressure_screen import (
    build_factorial_resource_pressure_design,
    build_continuous_resource_pressure_design,
    run_resource_pressure_experiment,
)
from mycormarl.resource_pressure_analysis import (
    daily_amf_trace_rows,
    factorial_plant_boundary_rows,
    reconstruct_biomass_trajectory,
)


def test_factorial_design_declares_405_isolated_plant_boundary_conditions():
    design = build_factorial_resource_pressure_design()

    assert len(design) == 405
    assert len({condition["id"] for condition in design}) == 405

    factors = [condition["factors"] for condition in design]
    assert {factor["plant_kappa_c"] for factor in factors} == {
        0.01, 0.0178, 0.0316, 0.0562, 0.1, 0.1778, 0.3162, 0.5623, 1.0,
    }
    assert {factor["initial_solution_p_micromolar"] for factor in factors} == {
        0.5, 0.7, 0.9, 1.1, 1.3,
    }
    assert {factor["plant_trade"] for factor in factors} == {
        0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
    }
    assert {
        tuple(sorted((name, value) for name, value in factor.items() if name not in {
            "plant_kappa_c", "initial_solution_p_micromolar", "plant_trade",
        }))
        for factor in factors
    } == {
        (
            ("fungus_gamma_p", 0.5),
            ("fungus_initial_biomass", 10.0),
            ("fungus_kappa_c", 0.1),
            ("fungus_kappa_p", 0.0),
            ("fungus_trade", 0.75),
            ("plant_kappa_p", 0.0),
        )
    }


def test_factorial_runner_applies_focal_and_fixed_reference_controls(monkeypatch):
    captured = {}

    def fake_static_controls(manifest):
        captured.update(manifest)
        return {"entries": [{
            "status": "completed", "rejection_reasons": [],
            "biomass": {"plant": 1.0, "fungus": 0.2},
            "initial_p_micromolar": manifest["initial_p_micromolar"][0], "steps": 1,
        }]}

    monkeypatch.setattr(screen, "run_static_controls", fake_static_controls)
    declaration = build_factorial_resource_pressure_design()[0]
    result = run_resource_pressure_experiment({
        "format": "mycormarl-resource-pressure-experiment-manifest",
        "format_version": 1, "sampling": "discrete_factorial",
        "design_seed": 4801, "sample_count": 1,
        "horizon": {"days": 0.05, "timestep_days": 0.05},
        "model": {"environment": {"soil_radius_cm": 1.0, "soil_depth_cm": 1.0,
                                  "radial_interval_cm": 0.5, "depth_interval_cm": 0.5}},
        "static_policy": {"plant": [0.02, 1.0, 0.0, 0.0], "fungus": [0.75, 1.0, 0.0, 0.0]},
    }, design_override=[declaration])

    entry = result["entries"][0]
    assert entry["static_policy"] == {"plant": [0.02, 1.0, 0.0, 0.0], "fungus": [0.75, 1.0, 0.0, 0.0]}
    assert captured["initial_p_micromolar"] == [0.5]
    assert captured["model"]["species"]["plant"]["kappa_p"] == 0.0
    assert captured["model"]["species"]["fungus"] == {
        "kappa_c": pytest.approx(0.0015),
        "kappa_p": 0.0,
        "initial_biomass": pytest.approx(0.01),
        "gamma_p": 0.5,
    }


def test_factorial_runner_uses_declared_manifest_levels_and_reference_controls(monkeypatch):
    def fake_static_controls(manifest):
        return {"entries": [{
            "status": "completed", "rejection_reasons": [],
            "biomass": {"plant": 1.0, "fungus": 0.2},
            "initial_p_micromolar": manifest["initial_p_micromolar"][0], "steps": 1,
        }]}

    monkeypatch.setattr(screen, "run_static_controls", fake_static_controls)
    result = run_resource_pressure_experiment({
        "format": "mycormarl-resource-pressure-experiment-manifest",
        "format_version": 1, "sampling": "discrete_factorial",
        "design_seed": 4801, "sample_count": 2,
        "horizon": {"days": 0.05, "timestep_days": 0.05},
        "model": {"environment": {"soil_radius_cm": 1.0, "soil_depth_cm": 1.0,
                                  "radial_interval_cm": 0.5, "depth_interval_cm": 0.5}},
        "static_policy": {"plant": [0.08, 1.0, 0.0, 0.0], "fungus": [0.6, 1.0, 0.0, 0.0]},
        "factor_levels": {
            "plant_kappa_c_multiplier": [0.3],
            "initial_solution_p_micromolar": [0.5, 0.7],
            "plant_trade": [0.08],
        },
        "reference_controls": {
            "fungus_kappa_c_multiplier": 0.25,
            "fungus_gamma_p_mg_per_g_dm": 1.2,
            "fungus_initial_biomass_multiplier": 2.0,
            "fungus_trade": 0.6,
            "plant_kappa_p_multiplier": 0.0,
            "fungus_kappa_p_multiplier": 0.0,
            "plant_initial_biomass_g": 0.01,
        },
    })

    assert [entry["factors"]["initial_solution_p_micromolar"] for entry in result["entries"]] == [0.5, 0.7]
    assert {entry["factors"]["plant_kappa_c"] for entry in result["entries"]} == {0.3}
    assert {entry["factors"]["fungus_gamma_p"] for entry in result["entries"]} == {1.2}
    assert [entry["traits"]["fungus"]["initial_biomass"] for entry in result["entries"]] == [
        pytest.approx(0.002), pytest.approx(0.002),
    ]


def test_factorial_writer_reuses_matching_checkpoints_and_rejects_changed_manifest(tmp_path, monkeypatch):
    calls = []

    def fake_static_controls(manifest):
        calls.append(manifest["initial_p_micromolar"][0])
        return {"entries": [{
            "status": "completed", "rejection_reasons": [],
            "biomass": {"plant": 1.0, "fungus": 0.2},
            "initial_p_micromolar": manifest["initial_p_micromolar"][0], "steps": 1,
        }]}

    monkeypatch.setattr(screen, "run_static_controls", fake_static_controls)
    manifest = {
        "format": "mycormarl-resource-pressure-experiment-manifest",
        "format_version": 1, "sampling": "discrete_factorial",
        "design_seed": 4801, "sample_count": 405, "write_combined_bundle": False,
        "horizon": {"days": 0.05, "timestep_days": 0.05},
        "model": {"environment": {"soil_radius_cm": 1.0, "soil_depth_cm": 1.0,
                                  "radial_interval_cm": 0.5, "depth_interval_cm": 0.5}},
        "static_policy": {"plant": [0.02, 1.0, 0.0, 0.0], "fungus": [0.75, 1.0, 0.0, 0.0]},
        "factor_levels": {
            "plant_kappa_c_multiplier": [0.01, 0.0178, 0.0316, 0.0562, 0.1, 0.1778, 0.3162, 0.5623, 1.0],
            "initial_solution_p_micromolar": [0.5, 0.7, 0.9, 1.1, 1.3],
            "plant_trade": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        },
        "reference_controls": {
            "fungus_kappa_c_multiplier": 0.1,
            "fungus_gamma_p_mg_per_g_dm": 0.5,
            "fungus_initial_biomass_multiplier": 10.0,
            "fungus_trade": 0.75,
            "plant_kappa_p_multiplier": 0.0,
            "fungus_kappa_p_multiplier": 0.0,
            "plant_initial_biomass_g": 0.01,
        },
    }
    output = tmp_path / "result-bundle.json"

    screen.write_resource_pressure_experiment(manifest, output)
    assert len(calls) == 405
    screen.write_resource_pressure_experiment(manifest, output)
    assert len(calls) == 405

    with pytest.raises(ValueError, match="manifest differs"):
        screen.write_resource_pressure_experiment({
            **manifest,
            "reference_controls": {**manifest["reference_controls"], "fungus_gamma_p_mg_per_g_dm": 0.75},
        }, output)


def test_factorial_threshold_analysis_uses_only_adjacent_observed_p_brackets():
    def entry(kappa_c, trade, initial_p, p_limited_steps):
        trace = []
        for step in range(10):
            trace.append({"agents": {"plant": {
                "no_realized_growth": False,
                "limiting_resource": "phosphate" if step < p_limited_steps else "carbon",
            }}})
        return {
            "id": f"{kappa_c}-{trade}-{initial_p}",
            "factors": {
                "plant_kappa_c": kappa_c,
                "plant_trade": trade,
                "initial_solution_p_micromolar": initial_p,
            },
            "biomass": {"plant": 1.2, "fungus": 0.3},
            "limitation_trace": trace,
        }

    levels = (0.5, 0.7, 0.9, 1.1, 1.3)
    entries = [
        *(entry(0.01, 0.02, p, steps) for p, steps in zip(levels, (10, 10, 3, 0, 0))),
        *(entry(0.02, 0.02, p, 10) for p in levels),
        *(entry(0.03, 0.02, p, 0) for p in levels),
        *(entry(0.04, 0.02, p, steps) for p, steps in zip(levels, (10, 4, 8, 2, 0))),
    ]

    rows = factorial_plant_boundary_rows(entries)
    by_kappa = {row["plant_kappa_c"]: row for row in rows}

    assert by_kappa[0.01]["threshold_status"] == "observed-crossing"
    assert by_kappa[0.01]["threshold_initial_p_micromolar"] == pytest.approx(0.8428571428571429)
    assert by_kappa[0.02]["threshold_status"] == "upper-censored"
    assert by_kappa[0.02]["threshold_initial_p_micromolar"] is None
    assert by_kappa[0.03]["threshold_status"] == "lower-censored"
    assert by_kappa[0.03]["threshold_initial_p_micromolar"] is None
    assert by_kappa[0.04]["threshold_status"] == "observed-crossing"
    assert by_kappa[0.04]["threshold_initial_p_micromolar"] == pytest.approx(0.6666666666666666)
    assert by_kappa[0.04]["response_is_monotonic_nonincreasing"] is False


def test_factorial_threshold_analysis_does_not_interpolate_across_no_growth_p_level():
    def entry(initial_p, limiting_resource, no_realized_growth=False):
        return {
            "id": f"condition-{initial_p}",
            "factors": {
                "plant_kappa_c": 0.01,
                "plant_trade": 0.02,
                "initial_solution_p_micromolar": initial_p,
            },
            "biomass": {"plant": 1.0, "fungus": 0.2},
            "limitation_trace": [{"agents": {"plant": {
                "no_realized_growth": no_realized_growth,
                "limiting_resource": limiting_resource,
            }}}],
        }

    row = factorial_plant_boundary_rows([
        entry(0.5, "phosphate"),
        entry(0.7, "carbon", no_realized_growth=True),
        entry(0.9, "carbon"),
    ])[0]

    assert row["threshold_status"] == "unbracketed"
    assert row["threshold_initial_p_micromolar"] is None


def test_factorial_manifest_declares_the_causal_405_condition_follow_up():
    manifest = json.loads(Path(
        "docs/qualification/resource-pressure-factorial-plant-boundary-manifest.json"
    ).read_text(encoding="utf-8"))

    assert manifest["sampling"] == "discrete_factorial"
    assert manifest["sample_count"] == 405
    assert manifest["write_combined_bundle"] is False
    assert manifest["horizon"] == {"days": 80.0, "timestep_days": 0.05}
    assert manifest["model"]["environment"] == {
        "soil_radius_cm": 40.0,
        "soil_depth_cm": 60.0,
        "radial_interval_cm": 0.2,
        "depth_interval_cm": 0.2,
    }
    assert manifest["factor_levels"]["initial_solution_p_micromolar"] == [
        0.5, 0.7, 0.9, 1.1, 1.3,
    ]
    assert manifest["reference_controls"] == {
        "fungus_kappa_c_multiplier": 0.1,
        "fungus_gamma_p_mg_per_g_dm": 0.5,
        "fungus_initial_biomass_multiplier": 10.0,
        "fungus_trade": 0.75,
        "plant_kappa_p_multiplier": 0.0,
        "fungus_kappa_p_multiplier": 0.0,
        "plant_initial_biomass_g": 0.01,
    }


def test_default_continuous_design_is_a_reproducible_360_condition_lhs():
    first = build_continuous_resource_pressure_design(4047)
    second = build_continuous_resource_pressure_design(4047)

    assert first == second
    assert len(first) == 360
    assert len({row["id"] for row in first}) == 360

    ranges = {
        "plant_kappa_c": (0.01, 0.6931448431551466, "log"),
        "fungus_kappa_c": (0.01, 0.6931448431551466, "log"),
        "fungus_initial_biomass": (1.0, 100.0, "log"),
        "initial_solution_p_micromolar": (0.1, 1.0, "linear"),
        "plant_trade": (0.05, 0.2, "linear"),
        "fungus_trade": (0.5, 0.8, "linear"),
        "fungus_gamma_p": (0.5, 2.0, "linear"),
    }
    for name, (lower, upper, scale) in ranges.items():
        values = [row["factors"][name] for row in first]
        assert all(lower <= value <= upper for value in values)
        transformed = [
            (math.log(value) - math.log(lower)) / (math.log(upper) - math.log(lower))
            if scale == "log"
            else (value - lower) / (upper - lower)
            for value in values
        ]
        assert {min(359, int(value * 360)) for value in transformed} == set(range(360))
    assert all(row["factors"]["plant_kappa_p"] == 0.0 for row in first)
    assert all(row["factors"]["fungus_kappa_p"] == 0.0 for row in first)


def test_canonical_manifest_declares_the_360_condition_rerun():
    manifest_path = Path("docs/qualification/resource-pressure-canonical-screen-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["sampling"] == "continuous_lhs"
    assert manifest["sample_count"] == 360
    assert manifest["horizon"] == {"days": 80.0, "timestep_days": 0.05}
    assert manifest["model"]["environment"] == {
        "soil_radius_cm": 40.0,
        "soil_depth_cm": 60.0,
        "radial_interval_cm": 0.2,
        "depth_interval_cm": 0.2,
    }


def test_runner_propagates_continuously_sampled_p_trade_and_gamma():
    result = run_resource_pressure_experiment({
        "format": "mycormarl-resource-pressure-experiment-manifest",
        "format_version": 1,
        "sampling": "continuous_lhs",
        "design_seed": 4047,
        "sample_count": 1,
        "horizon": {"days": 0.05, "timestep_days": 0.05},
        "model": {"environment": {
            "soil_radius_cm": 1.0, "soil_depth_cm": 1.0,
            "radial_interval_cm": 0.5, "depth_interval_cm": 0.5,
        }},
        "static_policy": {"plant": [0.1, 1.0, 0.0, 0.0], "fungus": [0.5, 1.0, 0.0, 0.0]},
        "record_limitation_trace": True,
        "record_resource_accounting": True,
    })

    entry = result["entries"][0]
    factors = entry["factors"]
    assert entry["initial_p_micromolar"] == factors["initial_solution_p_micromolar"]
    assert entry["static_policy"]["plant"][0] == factors["plant_trade"]
    assert entry["static_policy"]["fungus"][0] == factors["fungus_trade"]
    assert entry["traits"]["fungus"]["gamma_p"] == factors["fungus_gamma_p"]


def test_writer_rejects_a_changed_continuous_lhs_sample_count(tmp_path, monkeypatch):
    def fake_static_controls(manifest):
        return {"entries": [{
            "status": "completed", "rejection_reasons": [],
            "biomass": {"plant": 1.0, "fungus": 1.0},
            "initial_p_micromolar": manifest["initial_p_micromolar"][0], "steps": 1,
        }]}

    monkeypatch.setattr(screen, "run_static_controls", fake_static_controls)
    manifest = {
        "format": "mycormarl-resource-pressure-experiment-manifest",
        "format_version": 1, "sampling": "continuous_lhs",
        "design_seed": 4047, "sample_count": 2,
        "horizon": {"days": 0.05, "timestep_days": 0.05},
        "model": {"environment": {"soil_radius_cm": 1.0, "soil_depth_cm": 1.0,
                                  "radial_interval_cm": 0.5, "depth_interval_cm": 0.5}},
        "static_policy": {"plant": [0.1, 1.0, 0.0, 0.0], "fungus": [0.5, 1.0, 0.0, 0.0]},
    }
    output = tmp_path / "result-bundle.json"
    screen.write_resource_pressure_experiment(manifest, output)

    with pytest.raises(ValueError, match="sample_count"):
        screen.write_resource_pressure_experiment({**manifest, "sample_count": 3}, output)


def test_reconstructed_biomass_uses_realised_growth_not_initial_biomass_only():
    entry = {
        "initial_biomass": {"plant": 0.1, "fungus": 0.2},
        "limitation_trace": [
            {"day": 0.5, "agents": {"fungus": {"used_c_normalized": 0.03}, "plant": {"used_c_normalized": 0.02}}},
            {"day": 1.0, "agents": {"fungus": {"used_c_normalized": 0.07}, "plant": {"used_c_normalized": 0.05}}},
        ],
    }

    fungus = reconstruct_biomass_trajectory(entry, "fungus")
    plant = reconstruct_biomass_trajectory(entry, "plant")
    assert [day for day, _ in fungus] == [0.5, 1.0]
    assert [biomass for _, biomass in fungus] == pytest.approx([0.23, 0.3])
    assert [biomass for _, biomass in plant] == pytest.approx([0.12, 0.17])


def test_daily_amf_trace_rows_aggregate_fluxes_into_complete_days():
    entry = {
        "id": "condition-test",
        "factors": {
            "plant_kappa_c": 0.02, "fungus_kappa_c": 0.03,
            "fungus_gamma_p": 1.0, "fungus_initial_biomass": 10.0,
            "initial_solution_p_micromolar": 0.5, "plant_trade": 0.1,
            "fungus_trade": 0.6,
        },
        "initial_biomass": {"plant": 0.1, "fungus": 0.2},
        "limitation_trace": [
            {"day": 0.5, "agents": {"fungus": {"used_c_normalized": 0.03, "acquired_p": 0.4, "trade_out_raw": 0.2}, "plant": {"used_c_normalized": 0.0, "acquired_p": 0.6, "trade_in_raw": 0.2}}},
            {"day": 1.0, "agents": {"fungus": {"used_c_normalized": 0.07, "acquired_p": 0.8, "trade_out_raw": 0.4}, "plant": {"used_c_normalized": 0.0, "acquired_p": 1.4, "trade_in_raw": 0.4}}},
            {"day": 1.5, "agents": {"fungus": {"used_c_normalized": 0.05, "acquired_p": 0.2, "trade_out_raw": 0.1}, "plant": {"used_c_normalized": 0.0, "acquired_p": 0.2, "trade_in_raw": 0.1}}},
            {"day": 2.0, "agents": {"fungus": {"used_c_normalized": 0.05, "acquired_p": 0.2, "trade_out_raw": 0.1}, "plant": {"used_c_normalized": 0.0, "acquired_p": 0.2, "trade_in_raw": 0.1}}},
        ],
    }

    rows = daily_amf_trace_rows([entry], day_width=1.0)

    assert [row["day"] for row in rows] == [1.0, 2.0]
    assert rows[0]["fungus_biomass"] == pytest.approx(0.3)
    assert rows[0]["fungus_growth_rate_g_per_day"] == pytest.approx(0.1)
    assert rows[0]["fungus_p_uptake_mg_per_day"] == pytest.approx(1.2)
    assert rows[0]["fungus_p_transfer_mg_per_day"] == pytest.approx(0.6)
    assert rows[0]["plant_indirect_p_fraction"] == pytest.approx(0.6 / 2.6)
