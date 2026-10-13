"""Illustrative comparison only; does not modify animation or production data."""
from pathlib import Path

from phosphate_uptake_video_data import simulate_uptake_front
from mycormarl.soil.phosphate_uptake import sparse_surface_concentration, sparse_uptake_resistance
import matplotlib.pyplot as plt
import numpy as np


def main():
    d = simulate_uptake_front(far_field_half_life_days=None)
    age = d["time_days"]
    p = d["parameters"]
    a, j, km = p["hyphal_radius_cm"], p["jmax_micromol_cm2_s"], p["km_micromol_cm3"]
    resistance = sparse_uptake_resistance(0., a, j, p["amount_flux_diffusivity_cm2_s"],
                                          p["apparent_diffusivity_cm2_s"], age)
    curves = {}
    for half_life in [3, 6]:
        far_field_um = d["initial_solution_p_um"] * 2.**(-age / half_life)
        cs = np.asarray(sparse_surface_concentration(far_field_um * 1e-3, km, resistance), dtype=float)
        sample_rate = j * cs/(km+cs) * (2*np.pi*a*d["absorber_length_cm"]) * 86400*1000
        rate = sample_rate * d["presentation_scale"]
        assert np.all(np.diff(rate) < 0) and np.all(rate > 0)
        assert np.isclose(far_field_um[round(half_life/d["dt_days"])], d["initial_solution_p_um"]/2)
        assert np.isclose(rate[0], d["isolated_rate_nmol_day"][0], rtol=2e-6)
        # Independent cylindrical supply-demand check, excluding zero age.
        supply = p["amount_flux_diffusivity_cm2_s"]*(far_field_um[1:]*1e-3-cs[1:])/(a*np.log(d["effective_radius_cm"][1:]/a))
        np.testing.assert_allclose(supply, j*cs[1:]/(km+cs[1:]), rtol=2e-6)
        curves[half_life] = (far_field_um, rate)

    bg, fg, muted = "#071018", "#E6F1F0", "#9CB2BA"
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":11,
                         "figure.facecolor":bg, "axes.facecolor":bg,
                         "text.color":fg, "axes.labelcolor":fg,
                         "xtick.color":muted, "ytick.color":muted,
                         "axes.edgecolor":"#3B515D", "savefig.facecolor":bg}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
        fig.subplots_adjust(left=.075, right=.975, bottom=.24, top=.76, wspace=.28)
        fig.text(.075, .925, "Prescribed far-field decay", fontsize=21, weight="bold")
        fig.text(.075, .855, "3-day vs 6-day concentration half-life · same evolving sparse radial resistance", color=muted, fontsize=12)
        axes[0].plot(age, d["bulk_solution_p_um"], color="#91A1AD", ls="--", lw=1.7, label="Fixed far field")
        axes[1].plot(age, d["isolated_rate_nmol_day"], color="#91A1AD", ls="--", lw=1.7, label="Fixed far field")
        for half_life, colour in [(3,"#EDC77E"),(6,"#7AE6CF")]:
            bulk, rate = curves[half_life]
            axes[0].plot(age, bulk, color=colour, lw=2.5, label=f"{half_life}-day half-life")
            axes[1].plot(age, rate, color=colour, lw=2.5, label=f"{half_life}-day half-life")
        axes[0].set(ylabel="Far-field phosphate (µM)", ylim=(0,1.06))
        axes[1].set(ylabel="Uptake (nmol P/day, equivalent 1 cm)", ylim=(0,.14))
        for ax in axes:
            ax.set(xlim=(0,20), xlabel="Time since local contact (days)", xticks=[0,5,10,15,20])
            ax.grid(color="#3B515D", alpha=.35, lw=.7)
            ax.spines[["top","right"]].set_visible(False)
        axes[0].legend(frameon=True, facecolor=bg, edgecolor=bg, framealpha=1,
                       labelcolor=fg, loc="upper right", fontsize=10)
        fig.text(.075, .11, "Illustrative forcing, not a mass-conserving depletion simulation. Half-life refers to concentration—not uptake.", color=muted, fontsize=10)
        fig.text(.075, .055, "1 mm absorber ×10 for presentation. For A–D, repeat each uptake curve with the existing contact-time offsets.", color=muted, fontsize=10)
        out = Path("media/phosphate-uptake/far-field-decay-comparison.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        plt.close(fig)
    print(out.resolve())
    for half_life, (_, rate) in curves.items():
        print(f"{half_life}-day concentration half-life: uptake at age 6 d = {rate[600]:.5f}; at 18 d = {rate[1800]:.5f} nmol P/day")


if __name__ == "__main__":
    main()
