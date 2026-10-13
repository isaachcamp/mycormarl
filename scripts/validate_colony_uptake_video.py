"""Run, validate and plot the representative colony-scale soil experiment."""
from __future__ import annotations
import json
from pathlib import Path

# Set the plotting cache before importing matplotlib or the colony module.
import phosphate_uptake_video_data
from colony_uptake_video_data import simulate_colony
import matplotlib.pyplot as plt
import numpy as np


def validate(out=Path("media/phosphate-uptake/colony")):
    out.mkdir(parents=True,exist_ok=True)
    d = simulate_colony()
    def balance_error(data):
        total = data["soil_amount_micromol"].sum(axis=(1,2),dtype=float)+data["cumulative_uptake_micromol"]
        return float(np.max(abs(total/data["initial_total_micromol"]-1)))
    profiles = d["radial_flux_ng_mm3_day"]
    reconstructed = (profiles*d["ring_volume_cm3"]).sum(axis=1)*1000/30.973761998
    integral_error = float(np.max(abs(reconstructed/d["total_uptake_nmol_day"]-1)))
    report = {
        "experiment":"Production mass-conserving buffered soil, prescribed hemispherical growth",
        "parameters":d["parameters"],
        "horizon_days":float(d["time_days"][-1]),"dt_days":d["dt_days"],
        "output_interval_days":d["output_interval_days"],
        "initial_colony_radius_cm":d["initial_colony_radius_cm"],
        "radial_growth_cm_day":d["radial_growth_cm_day"],
        "safe_cfl_seconds":d["safe_cfl_seconds"],
        "maximum_relative_phosphate_balance_error":balance_error(d),
        "maximum_relative_profile_integral_error":integral_error,
        "initial_total_micromol":d["initial_total_micromol"],
        "total_absorbed_micromol":float(d["cumulative_uptake_micromol"][-1]),
        "interpretation":d["assumptions"],"reference":d["reference"],
        "refinements":{},"snapshot_days":[6,10,14,18,22],
    }
    for name,kwargs in [("half_timestep",{"dt_days":0.0025}),
                        ("finer_grid",{"grid_cells":96,"dt_days":0.0025})]:
        f = simulate_colony(**kwargs)
        change = float(abs(f["cumulative_uptake_micromol"][-1]/d["cumulative_uptake_micromol"][-1]-1))
        late = d["time_days"] >= 6
        numerator = (abs(f["radial_flux_ng_mm3_day"]-profiles)*d["ring_volume_cm3"]).sum(axis=1)
        denominator = (f["radial_flux_ng_mm3_day"]*d["ring_volume_cm3"]).sum(axis=1)
        profile_error = float(np.max(numerator[late]/denominator[late]))
        report["refinements"][name] = {**kwargs,"relative_total_absorption_change":change,
            "maximum_relative_phosphate_balance_error":balance_error(f),
            "maximum_volume_weighted_relative_L1_profile_change_after_day6":profile_error}
        assert change < 0.03 and balance_error(f) < 1e-5 and profile_error < .2
    assert balance_error(d) < 1e-5 and integral_error < 2e-5
    peak_radii = []
    for day in report["snapshot_days"]:
        i = np.searchsorted(d["time_days"],day)
        peak_radii.append(float(d["ring_radius_mm"][np.argmax(profiles[i])]))
    report["profile_peak_radii_mm"] = peak_radii
    assert np.all(np.diff(peak_radii) > 0)
    np.savez_compressed(out/"data.npz",**{k:v for k,v in d.items() if isinstance(v,np.ndarray)})
    (out/"metadata.json").write_text(json.dumps({k:v for k,v in d.items() if not isinstance(v,np.ndarray)},indent=2)+"\n")
    (out/"validation.json").write_text(json.dumps(report,indent=2)+"\n")

    fig,axes = plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    colours = plt.cm.viridis(np.linspace(.25,.95,5))
    for day,colour in zip(report["snapshot_days"],colours):
        i = np.searchsorted(d["time_days"],day)
        axes[0].plot(d["ring_radius_mm"],profiles[i],color=colour,lw=2,label=f"Day {day}")
    axes[0].set(xlabel="Distance from colony origin (mm)",ylabel="P uptake flux (ng P mm⁻³ day⁻¹)",xlim=(0,6),ylim=(0,None))
    axes[0].legend(frameon=False)
    axes[1].plot(d["time_days"],d["total_uptake_nmol_day"],color="#249C89",lw=2)
    axes[1].set(xlabel="Colony simulation time (days)",ylabel="Total network uptake (nmol P/day)",ylim=(0,None))
    soil = d["soil_amount_micromol"].sum(axis=(1,2),dtype=float)
    axes[2].plot(d["time_days"],soil,label="Remaining soil P")
    axes[2].plot(d["time_days"],d["cumulative_uptake_micromol"],label="Accumulated fungal P")
    axes[2].plot(d["time_days"],soil+d["cumulative_uptake_micromol"],"--",label="Conserved total")
    axes[2].set(xlabel="Colony simulation time (days)",ylabel="P amount (µmol)",ylim=(0,None))
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.18)
    fig.suptitle("Representative colony · closed soil · no prescribed P decay")
    fig.savefig(out/"profiles-total-and-balance.png",dpi=180)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    validate()
