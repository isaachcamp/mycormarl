"""Physics contracts for the isolated sparse-uptake animation."""
import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("video_data", Path(__file__).parents[1] / "scripts/phosphate_uptake_video_data.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
simulate = _module.simulate_uptake_front


def test_no_grid_two_day_far_field_decay_and_uncapped_radius():
    d = simulate()
    assert d["format_version"] == 4
    assert "cell_volume_cm3" not in d
    assert d["absorber_length_cm"] == 0.1
    assert d["far_field_half_life_days"] == 2
    np.testing.assert_allclose(d["bulk_solution_p_um"], d["initial_solution_p_um"] * 2.**(-d["time_days"]/2), rtol=1e-14)
    assert d["bulk_solution_p_um"][200] == pytest.approx(d["initial_solution_p_um"]/2)
    assert np.all(np.diff(d["effective_radius_cm"]) > 0)


def test_identical_curves_shifted_by_contact():
    d = simulate()
    np.testing.assert_array_equal(d["arrival_days"], [0, 6, 12, 18])
    for site, arrival in enumerate(d["arrival_days"]):
        shift = round(arrival / d["dt_days"])
        assert np.all(d["rate_nmol_day"][:shift, site] == 0)
        np.testing.assert_array_equal(d["rate_nmol_day"][shift:, site], d["isolated_rate_nmol_day"][:len(d["time_days"])-shift])


def test_initial_flux_and_length_scaling():
    from mycormarl.fungus.traits import FungusTraits
    from mycormarl.params import EnvConfig
    f, c = FungusTraits(), EnvConfig()
    d = simulate()
    bulk = c.initial_solution_p_um * 1e-3
    expected = f.jmax * bulk/(f.km+bulk) * 2*math.pi*f.hyphal_radius * 0.1*86400*1000
    assert d["sample_rate_nmol_day"][0] == pytest.approx(expected, rel=2e-6)
    np.testing.assert_allclose(d["isolated_rate_nmol_day"], 10*d["sample_rate_nmol_day"], rtol=1e-14)
    longer = simulate(absorber_length_cm=0.2)
    np.testing.assert_allclose(longer["sample_rate_nmol_day"], 2*d["sample_rate_nmol_day"])
    np.testing.assert_allclose(longer["isolated_rate_nmol_day"], d["isolated_rate_nmol_day"])


def test_independent_sparse_formula_and_radial_supply_balance():
    d = simulate()
    p = d["parameters"]
    a, j, km = p["hyphal_radius_cm"], p["jmax_micromol_cm2_s"], p["km_micromol_cm3"]
    cb = d["bulk_solution_p_um"] * 1e-3
    outer = a + np.sqrt(p["apparent_diffusivity_cm2_s"]*d["time_days"]*86400)
    np.testing.assert_allclose(d["effective_radius_cm"], outer, rtol=2e-6)
    k = a*j/p["amount_flux_diffusivity_cm2_s"]*np.log(outer/a)
    b = cb-km-k
    cs = 2*cb*km/(np.sqrt(b*b+4*cb*km)-b)
    np.testing.assert_allclose(d["surface_solution_p_um"], cs*1000, rtol=2e-6)
    supply = p["amount_flux_diffusivity_cm2_s"]*(cb[1:]-cs[1:])/(a*np.log(outer[1:]/a))
    np.testing.assert_allclose(supply, j*cs[1:]/(km+cs[1:]), rtol=2e-6)
    profile = d["radial_solution_p_um"]
    np.testing.assert_allclose(profile[:, 0], d["surface_solution_p_um"], rtol=2e-6)
    np.testing.assert_allclose(profile[:, -1], d["bulk_solution_p_um"])
    assert np.all(np.diff(profile, axis=1) >= -1e-12)
    assert np.all(np.diff(profile, axis=0) <= 1e-12)


def test_monotonic_depletion_and_front_contrast():
    d = simulate()
    q = d["isolated_rate_nmol_day"]
    assert np.all(np.isfinite(q)) and np.all(q > 0)
    assert np.all(np.diff(q) < 0)
    dt = d["dt_days"]
    slopes = [(q[i+1]-q[i-1])/(2*dt) for i in [round(1.99/dt), round(2.01/dt)]]
    assert abs(slopes[1]/slopes[0]-1) < 0.05
    rates = d["rate_nmol_day"][round(18.2/dt)]
    assert rates[3] > rates[2] > rates[1] > rates[0]


def test_sampling_refinement():
    coarse, fine = simulate(), simulate(dt_days=0.005)
    np.testing.assert_allclose(coarse["rate_nmol_day"], fine["rate_nmol_day"][::2], rtol=2e-6)


def test_decay_options_and_fixed_far_field_reference():
    d = simulate()
    fixed = simulate(far_field_half_life_days=None)
    three = simulate(far_field_half_life_days=3)
    six = simulate(far_field_half_life_days=6)
    np.testing.assert_array_equal(fixed["bulk_solution_p_um"], fixed["initial_solution_p_um"])
    assert six["bulk_solution_p_um"][600] == pytest.approx(six["initial_solution_p_um"]/2)
    assert three["isolated_rate_nmol_day"][600] == pytest.approx(0.01597, abs=5e-6)
    assert three["isolated_rate_nmol_day"][1800] == pytest.approx(0.00094, abs=5e-6)
    assert np.all(d["isolated_rate_nmol_day"][1:] < three["isolated_rate_nmol_day"][1:])
    assert np.all(three["isolated_rate_nmol_day"][1:] < six["isolated_rate_nmol_day"][1:])
    assert np.all(six["isolated_rate_nmol_day"][1:] < fixed["isolated_rate_nmol_day"][1:])


@pytest.mark.parametrize("kwargs", [{"dt_days":0}, {"horizon_days":-1}, {"dt_days":100}, {"absorber_length_cm":0}, {"presentation_length_cm":float("nan")}, {"far_field_half_life_days":0}, {"far_field_half_life_days":-3}, {"far_field_half_life_days":float("nan")}, {"far_field_half_life_days":float("inf")}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        simulate(**kwargs)
