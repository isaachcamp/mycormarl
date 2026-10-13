"""Contracts for the mass-conserving colony-scale video experiment."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("colony_data", Path(__file__).parents[1] / "scripts/colony_uptake_video_data.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
simulate = _module.simulate_colony


def test_equal_area_ring_edges_and_three_dimensional_volumes():
    d = simulate(horizon_days=2)
    r = d["ring_edges_cm"]
    areas = np.pi/2 * np.diff(r*r)
    np.testing.assert_allclose(areas, areas[0], rtol=1e-12)
    np.testing.assert_allclose(d["semiring_area_cm2"], areas, rtol=1e-12)
    volumes = np.pi * np.diff(d["r_edges_cm"]**2)[:, None] * np.diff(d["z_edges_cm"])[None, :]
    np.testing.assert_allclose(d["cell_volume_cm3"], volumes, rtol=1e-5)
    shell_volumes = (d["shell_cell_weights"]*volumes).sum(axis=(1,2))
    np.testing.assert_allclose(shell_volumes, 2*np.pi/3*np.diff(r**3), rtol=1e-10)
    assert np.all(d["shell_cell_weights"] >= 0)
    assert np.all(d["shell_cell_weights"].sum(axis=0) <= 1+1e-12)


def test_closed_soil_and_accumulated_network_uptake_conserve_phosphate():
    d = simulate(horizon_days=4)
    balance = d["soil_amount_micromol"].sum(axis=(1,2), dtype=float) + d["cumulative_uptake_micromol"]
    np.testing.assert_allclose(balance, d["initial_total_micromol"], rtol=1e-5)
    assert np.all(d["soil_amount_micromol"] >= 0)
    assert np.all(d["accepted_uptake_micromol"] >= 0)
    assert d["far_field_half_life_days"] is None
    assert d["dt_days"]*86400 <= d["safe_cfl_seconds"]


def test_partial_geometry_length_and_total_rate_are_consistent():
    d = simulate(horizon_days=4)
    radii = d["colony_radius_cm"]
    length = d["hyphal_density_cm_cm3"] * (2*np.pi/3) * radii**3
    represented = (d["hyphal_density_field_cm_cm3"]*d["cell_volume_cm3"]).sum(axis=(1,2))
    np.testing.assert_allclose(represented, length, rtol=2e-5)
    expected = d["accepted_uptake_micromol"].sum(axis=(1,2), dtype=float)*1000/d["output_interval_days"]
    np.testing.assert_allclose(d["total_uptake_nmol_day"], expected, rtol=1e-12)
    # Radial flux density is volume weighted over depth, not a top-layer sample.
    shell_amount = np.einsum("trz,krz->tk", d["accepted_uptake_micromol"], d["shell_cell_weights"], dtype=float)
    expected = shell_amount/d["ring_volume_cm3"] /d["output_interval_days"] * 30.973761998
    np.testing.assert_allclose(d["radial_flux_ng_mm3_day"], expected, rtol=1e-12)
    reconstructed_total = (d["radial_flux_ng_mm3_day"] * d["ring_volume_cm3"]).sum(axis=1) * 1000/30.973761998
    np.testing.assert_allclose(d["total_uptake_nmol_day"], reconstructed_total, rtol=2e-5)


def test_no_absorption_ahead_of_the_growing_colony():
    d = simulate(horizon_days=4)
    r, z = d["r_edges_cm"][:-1], d["z_edges_cm"][:-1]
    outside = r[None,:,None]**2+z[None,None,:]**2 >= d["colony_radius_cm"][:,None,None]**2
    assert np.all(d["accepted_uptake_micromol"][outside] == 0)
    assert np.all(np.isfinite(d["radial_flux_ng_mm3_day"]))


def test_timestep_refinement_preserves_total_absorption():
    d = simulate(horizon_days=4)
    f = simulate(horizon_days=4, dt_days=0.0025)
    assert f["cumulative_uptake_micromol"][-1] == pytest.approx(d["cumulative_uptake_micromol"][-1], rel=0.02)


@pytest.mark.parametrize("kwargs", [{"dt_days":1}, {"radial_growth_cm_day":0}, {"n_rings":0}, {"horizon_days":50}, {"output_interval_days":0.013}])
def test_invalid_geometry_or_time_is_rejected(kwargs):
    with pytest.raises(ValueError):
        simulate(**kwargs)
