"""Validate and plot the isolated sparse-cylinder animation diagnostic."""
from __future__ import annotations
import json
from pathlib import Path
from phosphate_uptake_video_data import simulate_uptake_front
import matplotlib.pyplot as plt
import numpy as np


def validate(output_dir=Path("media/phosphate-uptake")):
    output_dir.mkdir(parents=True, exist_ok=True)
    d = simulate_uptake_front()
    fine = simulate_uptake_front(dt_days=0.005)
    error = float(np.max(np.abs(d["rate_nmol_day"]-fine["rate_nmol_day"][::2])))
    p = d["parameters"]
    outer, cs = d["effective_radius_cm"], d["surface_solution_p_um"] * 1e-3
    bulk = d["bulk_solution_p_um"] * 1e-3
    a, j, km = p["hyphal_radius_cm"], p["jmax_micromol_cm2_s"], p["km_micromol_cm3"]
    supply = p["amount_flux_diffusivity_cm2_s"]*(bulk[1:]-cs[1:])/(a*np.log(outer[1:]/a))
    demand = j*cs[1:]/(km+cs[1:])
    balance_error = float(np.max(abs(supply/demand-1)))
    shift_error = 0.
    for site, arrival in enumerate(d["arrival_days"]):
        shift = round(arrival/d["dt_days"])
        shift_error = max(shift_error, float(np.max(abs(d["rate_nmol_day"][shift:, site]-d["isolated_rate_nmol_day"][:len(outer)-shift]))))
        assert np.all(d["rate_nmol_day"][:shift, site] == 0)
    assert error < 1e-7 and shift_error == 0
    assert balance_error < 2e-6
    assert np.all(np.diff(d["isolated_rate_nmol_day"]) < 0)
    report = {
        "experiment": "isolated sparse cylinder, prescribed 2-day far-field P half-life, local elapsed time",
        "horizon_days": float(d["time_days"][-1]),
        "dt_days": d["dt_days"],
        "absorber_length_cm": d["absorber_length_cm"],
        "presentation_length_cm": d["presentation_length_cm"],
        "presentation_scale": d["presentation_scale"],
        "far_field_half_life_days": d["far_field_half_life_days"],
        "arrival_days": d["arrival_days"].tolist(),
        "initial_rate_nmol_day_for_1cm": float(d["isolated_rate_nmol_day"][0]),
        "final_rate_nmol_day_for_1cm": float(d["isolated_rate_nmol_day"][-1]),
        "maximum_sampling_refinement_absolute_error_nmol_day": error,
        "maximum_time_shift_absolute_error_nmol_day": shift_error,
        "maximum_relative_radial_flux_balance_error": balance_error,
        "parameters": p,
        "interpretation": d["assumptions"],
        "source": d["source"],
        "smoothing_applied": False,
        "mass_balance": "Not mass conserving: prescribed decaying far field; checked local quasi-steady supply-demand balance only.",
    }
    (output_dir / "two-day-decay-validation.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(output_dir / "two-day-decay-data.npz", **{k:v for k,v in d.items() if isinstance(v, np.ndarray)})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for site, letter, colour in zip(range(4), "ABCD", ["#B68B2D", "#367BB0", "#9566B4", "#249C89"]):
        active = d["time_days"] >= d["arrival_days"][site]
        axes[0].plot(d["time_days"][active], d["rate_nmol_day"][active, site], color=colour, label=letter)
    for age in [0, 0.1, 1, 6, 18, 24]:
        i = round(age/d["dt_days"])
        axes[1].plot(d["radial_positions_cm"]*10, d["radial_solution_p_um"][i], label=f"{age:g} d")
    axes[0].set(xlabel="Simulation time (days)", ylabel="Uptake (nmol P/day, equivalent 1 cm)", xlim=(0,24), ylim=(0,0.15))
    axes[1].set(xlabel="Distance from cylinder axis (mm)", ylabel="Solution P (µM)", xscale="log", ylim=(0,1.05))
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Isolated sparse uptake · prescribed 2-day far-field half-life")
    fig.savefig(output_dir / "two-day-decay-rates.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    validate()
