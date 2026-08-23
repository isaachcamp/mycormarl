import json
import math
from pathlib import Path

import pytest

import mycormarl.resource_pressure_screen as screen
from mycormarl.resource_pressure_screen import (
    build_continuous_resource_pressure_design,
    run_resource_pressure_experiment,
)
from mycormarl.resource_pressure_analysis import (
    daily_amf_trace_rows,
    reconstruct_biomass_trajectory,
)


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
