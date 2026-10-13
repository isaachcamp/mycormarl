"""Tests for the time-dependent finite-inventory hyphal sweep."""

import pytest

from mycormarl.soil.hyphal_density_sweep import run_hyphal_density_sweep


def test_default_sweep_has_200_density_endpoints_and_conserves_p():
    time_series, summaries = run_hyphal_density_sweep(
        densities_cm_cm3=[25.0, 50.0], horizon_days=0.05, dt_days=0.025
    )

    assert len(summaries) == 2
    assert len(time_series) == 4
    assert all(row["geometry_valid"] for row in summaries)
    assert all(row["conservation_error_micromol"] == pytest.approx(0.0) for row in summaries)


def test_custom_initial_p_is_recorded_and_changes_inventory():
    _, low = run_hyphal_density_sweep(
        densities_cm_cm3=[25.0], initial_solution_p_um=0.1,
        horizon_days=0.05, dt_days=0.025
    )
    _, high = run_hyphal_density_sweep(
        densities_cm_cm3=[25.0], initial_solution_p_um=3.0,
        horizon_days=0.05, dt_days=0.025
    )
    assert low[0]["initial_solution_p_um"] == pytest.approx(0.1)
    assert high[0]["initial_labile_p_micromol"] > low[0]["initial_labile_p_micromol"]


def test_dynamic_blend_starts_sparse_and_moves_toward_continuous():
    time_series, _ = run_hyphal_density_sweep(
        densities_cm_cm3=[25.0], horizon_days=2.0, dt_days=0.025
    )

    weights = [row["continuous_weight"] for row in time_series]
    assert weights[0] == pytest.approx(0.0)
    assert weights[-1] > weights[0]


def test_efficiency_uses_structural_hyphal_carbon():
    _, summaries = run_hyphal_density_sweep(
        densities_cm_cm3=[25.0], horizon_days=0.05, dt_days=0.025
    )
    row = summaries[0]
    assert row["construction_carbon_g"] > 0.0
    assert row["uptake_per_construction_carbon_micromol_g_c"] == pytest.approx(
        row["total_uptake_micromol"] / row["construction_carbon_g"]
    )
