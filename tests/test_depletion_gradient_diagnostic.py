"""Public contracts for the time-dependent depletion-gradient diagnostic."""

import pytest
import csv
from pathlib import Path
import subprocess
import sys

from mycormarl.soil.depletion_gradient_diagnostic import (
    run_native_geometry_closure_comparisons,
    run_time_dependent_depletion_gradient_diagnostic,
)


def test_zero_time_blended_closure_matches_its_sparse_and_continuous_limits():
    """All fixed geometries begin with no depletion resistance or blending."""
    time_series, summaries = run_time_dependent_depletion_gradient_diagnostic(
        times_days=[0.0, 1.0],
        absorber_radii_cm=[1e-2],
        length_densities_cm_cm3=[1.0],
    )

    assert len(time_series) == 2
    assert len(summaries) == 1
    zero_time_rows = [row for row in time_series if row["time_days"] == 0.0]
    assert len(zero_time_rows) == 1
    row = zero_time_rows[0]
    assert row["closure"] == "blended_time_dependent"
    assert row["effective_radius_cm"] == pytest.approx(1e-2)
    assert row["sparse_resistance"] == pytest.approx(0.0)
    assert row["continuous_weight"] == pytest.approx(0.0)
    assert row["sparse_uptake_rate_micromol_s"] == pytest.approx(
        row["continuous_uptake_rate_micromol_s"]
    )
    assert row["total_uptake_rate_micromol_s"] == pytest.approx(
        row["continuous_uptake_rate_micromol_s"]
    )


def test_blended_request_moves_from_sparse_toward_continuous_with_simulation_time():
    """`t_sim`, not fixed T_ref, controls the diagnostic blend weight."""
    time_series, _ = run_time_dependent_depletion_gradient_diagnostic(
        times_days=[0.0, 1.0, 30.0],
        absorber_radii_cm=[1e-2],
        length_densities_cm_cm3=[100.0],
    )

    weights = [row["continuous_weight"] for row in time_series]
    assert weights[0] == pytest.approx(0.0)
    assert weights[1] == pytest.approx(
        1.0 / (1.0 + time_series[0]["diffusion_overlap_time_days"] ** 2)
    )
    assert weights[-1] > weights[1]
    assert time_series[-1]["total_uptake_rate_micromol_s"] > time_series[-1][
        "sparse_uptake_rate_micromol_s"
    ]


def test_sparse_gradient_travels_outward_then_stops_at_territory_boundary():
    """The diagnostic exposes the density-limited depletion-gradient travel."""
    time_series, _ = run_time_dependent_depletion_gradient_diagnostic(
        times_days=[0.0, 0.25, 1.0, 30.0],
        absorber_radii_cm=[5e-4],
        length_densities_cm_cm3=[2_000.0],
    )
    sparse = time_series

    radii = [row["effective_radius_cm"] for row in sparse]
    resistances = [row["sparse_resistance"] for row in sparse]
    assert radii == sorted(radii)
    assert resistances == sorted(resistances)
    assert radii[-1] == pytest.approx(sparse[-1]["territory_radius_cm"])
    assert radii[-1] == pytest.approx(radii[-2])
    assert resistances[-1] == pytest.approx(resistances[-2])


def test_rows_preserve_fixed_experiment_metadata_at_every_timepoint():
    """Global time changes only the diagnostic depletion-gradient radius."""
    time_series, _ = run_time_dependent_depletion_gradient_diagnostic(
        times_days=[0.0, 1.0, 30.0],
        absorber_radii_cm=[1e-2],
        length_densities_cm_cm3=[100.0],
    )

    sparse = time_series
    assert {row["reference_time_days"] for row in sparse} == {1.0}
    assert {row["represented_length_cm"] for row in sparse} == {100.0}
    assert [row["bulk_concentration_micromol_cm3"] for row in sparse] == pytest.approx(
        [1e-3] * len(sparse)
    )
    assert len({row["plant_jmax_micromol_cm2_s"] for row in sparse}) == 1
    assert len({row["plant_km_micromol_cm3"] for row in sparse}) == 1
    overlap_marker = next(row for row in sparse if row["is_diffusion_overlap_time"])
    assert overlap_marker["time_days"] == pytest.approx(
        overlap_marker["diffusion_overlap_time_days"]
    )


def test_component_continuous_rate_is_density_independent_per_length_and_scales_per_cell():
    """The recorded continuous component separates local kinetics from scale."""
    time_series, _ = run_time_dependent_depletion_gradient_diagnostic(
        times_days=[0.0],
        absorber_radii_cm=[1e-2],
        length_densities_cm_cm3=[1.0, 100.0, 2_000.0],
    )
    rates_per_length = [
        row["continuous_uptake_rate_micromol_s"] / row["represented_length_cm"]
        for row in time_series
    ]
    assert rates_per_length == pytest.approx([rates_per_length[0]] * 3)
    assert time_series[1]["continuous_uptake_rate_micromol_s"] == pytest.approx(
        100.0 * time_series[0]["continuous_uptake_rate_micromol_s"]
    )
    assert time_series[2]["continuous_uptake_rate_micromol_s"] == pytest.approx(
        2_000.0 * time_series[0]["continuous_uptake_rate_micromol_s"]
    )


def test_native_geometry_comparison_exposes_closure_and_blend_time_treatments():
    """The second figure compares native geometries at the agreed closures."""
    rows, summaries = run_native_geometry_closure_comparisons(
        times_days=[0.0, 1.0, 30.0]
    )

    assert len(rows) == 44
    assert len(summaries) == 12
    assert {(row["comparison_panel"], row["treatment"]) for row in rows} == {
        ("closure_limits", "sparse_only"),
        ("closure_limits", "continuous_only"),
        ("blend_time_reference", "fixed_t_ref"),
        ("blend_time_reference", "simulation_time"),
    }
    plant = [row for row in rows if row["organism_geometry"] == "plant_default"]
    fungus = [row for row in rows if row["organism_geometry"] == "fungus_default"]
    assert {row["absorber_radius_cm"] for row in plant} == {1e-2}
    assert {row["length_density_cm_cm3"] for row in plant} == {1.0}
    assert {row["absorber_radius_cm"] for row in fungus} == {5e-4}
    assert {row["length_density_cm_cm3"] for row in fungus} == {2_000.0}
    transition = [
        row for row in rows if row["organism_geometry"] == "transition_scale"
    ]
    assert {row["absorber_radius_cm"] for row in transition} == {5e-4}
    assert [row["diffusion_overlap_time_days"] for row in transition] == pytest.approx(
        [10.0] * len(transition)
    )
    for geometry in ("fungus_default", "transition_scale"):
        markers = [
            row
            for row in rows
            if row["organism_geometry"] == geometry
            and row["is_diffusion_overlap_time"]
        ]
        assert len(markers) == 4
        assert [row["time_days"] for row in markers] == pytest.approx(
            [row["diffusion_overlap_time_days"] for row in markers]
        )
    at_zero = [row for row in rows if row["time_days"] == 0.0]
    for organism in ("plant_default", "fungus_default"):
        by_treatment = {
            row["treatment"]: row
            for row in at_zero
            if row["organism_geometry"] == organism
        }
        assert by_treatment["sparse_only"]["total_uptake_rate_micromol_s"] == pytest.approx(
            by_treatment["continuous_only"]["total_uptake_rate_micromol_s"]
        )
        assert by_treatment["simulation_time"]["total_uptake_rate_micromol_s"] == pytest.approx(
            by_treatment["sparse_only"]["total_uptake_rate_micromol_s"]
        )
    fixed_weights = [
        row["continuous_weight"]
        for row in rows
        if row["organism_geometry"] == "fungus_default"
        and row["treatment"] == "fixed_t_ref"
    ]
    simulation_weights = [
        row["continuous_weight"]
        for row in rows
        if row["organism_geometry"] == "fungus_default"
        and row["treatment"] == "simulation_time"
    ]
    assert fixed_weights == pytest.approx([fixed_weights[0]] * len(fixed_weights))
    assert simulation_weights[0] == pytest.approx(0.0)
    assert simulation_weights[-1] > simulation_weights[1]
    transition_by_treatment = {
        row["treatment"]: row
        for row in transition
        if row["time_days"] == 30.0
    }
    assert (
        transition_by_treatment["simulation_time"]["continuous_weight"]
        > transition_by_treatment["fixed_t_ref"]["continuous_weight"]
    )


def test_cli_writes_machine_readable_rows_and_two_panel_figure(tmp_path):
    """The CLI renders its fixed-reservoir figure solely from runner output."""
    script = Path(__file__).parents[1] / "scripts" / "depletion_gradient_diagnostic.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
            "--sample-count",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    with (tmp_path / "depletion_gradient_time_series.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    with (tmp_path / "depletion_gradient_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summaries = list(csv.DictReader(handle))
    assert len(rows) >= 18
    assert len(summaries) == 6
    assert {"0.0", "15.0", "30.0"} <= {row["time_days"] for row in rows}
    assert (tmp_path / "depletion_gradient_cumulative_uptake.svg").stat().st_size > 1_000
    assert (tmp_path / "depletion_gradient_cumulative_uptake.png").stat().st_size > 10_000
    figure = (tmp_path / "depletion_gradient_cumulative_uptake.svg").read_text()
    assert "Plant-scale absorber" in figure
    assert "Fungus-scale absorber" in figure
    assert "T_ref = 1 day" in figure
    assert "Cumulative P uptake by represented cell" in figure
    assert "t_sim = t_diff" in figure
    assert "stroke-opacity: 0.4" in figure
    assert (tmp_path / "native_geometry_closure_comparison.svg").stat().st_size > 1_000
    assert (tmp_path / "native_geometry_closure_comparison.png").stat().st_size > 10_000
    comparison_figure = (tmp_path / "native_geometry_closure_comparison.svg").read_text()
    assert "Sparse vs continuous" in comparison_figure
    assert "Fixed T_ref vs t_sim" in comparison_figure
    assert "Sparse" in comparison_figure
    assert "Continuous" in comparison_figure
    assert "Sparse-only" not in comparison_figure
    assert "Continuous-only" not in comparison_figure
    assert "Fixed T_ref" in comparison_figure
    assert "t_sim" in comparison_figure
    assert "Plant default geometry (solid)" not in comparison_figure
    assert "Fungus default geometry (dashed)" not in comparison_figure
    assert "Transition-scale geometry (dash-dot)" not in comparison_figure
    with (tmp_path / "native_geometry_closure_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        comparison_rows = list(csv.DictReader(handle))
    assert {row["treatment"] for row in comparison_rows} == {
        "sparse_only",
        "continuous_only",
        "fixed_t_ref",
        "simulation_time",
    }
    assert {row["organism_geometry"] for row in comparison_rows} == {
        "plant_default",
        "fungus_default",
        "transition_scale",
    }
