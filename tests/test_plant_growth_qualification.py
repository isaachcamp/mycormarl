"""Public contracts for the plant-only growth-scale qualification."""

import pytest

from mycormarl.plant_growth_qualification import run_plant_growth_qualification


def test_high_p_qualification_reports_all_fixed_kleaf_cases():
    result = run_plant_growth_qualification()

    assert result["stage"] == "plant-growth-qualification"
    assert result["policy"] == {
        "consumer_mode": "plant-only",
        "trade": 0.0,
        "growth": 1.0,
        "reproduction": 0.0,
        "storage": 0.0,
    }
    assert tuple(result["cases"]) == (
        "kleaf_0.300",
        "kleaf_0.450",
        "kleaf_0.500",
        "kleaf_0.600",
        "kleaf_0.650",
        "kleaf_0.675",
        "kleaf_0.680",
        "kleaf_0.700",
    )
    assert result["reference"]["forto_endpoint_g_dm"] == pytest.approx(23.26)
    assert result["reference"]["rgr_windows_per_day"] == pytest.approx(
        [0.066, 0.065, 0.060, 0.042]
    )
    assert result["analytical_carbon_only"]["sustained_rgr_per_day"] == pytest.approx(
        (0.68 * 0.05 - 0.007) / 0.402
    )
    assert result["selected_kleaf"] == pytest.approx(0.68)
    assert result["cases"]["kleaf_0.680"]["biomass_g_dm"]["120"] == pytest.approx(
        31.3776, rel=1e-4
    )
    assert result["cases"]["kleaf_0.680"]["biomass_g_dm"]["120"] == pytest.approx(
        result["analytical_carbon_only"]["upper_bound_endpoint_g_dm"]
    )
    endpoints = [case["biomass_g_dm"]["120"] for case in result["cases"].values()]
    assert endpoints == sorted(endpoints)
    assert result["cases"]["kleaf_0.680"]["status"] == "passed"
    assert result["cases"]["kleaf_0.700"]["status"] == "failed"
    assert all("biomass_g_dm" in case for case in result["cases"].values())
