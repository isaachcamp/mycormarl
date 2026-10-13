"""Isolated sparse-cylinder diagnostic for the illustrative uptake animation.

Prescribed far-field P decay, no grid, no finite reservoir, no overlap/blending. The
production sparse closure is evaluated at elapsed local contact time, with
infinite territory. Reconstructed logarithmic radial profiles are quasi-steady
closure profiles, not a transient PDE solution or the exact Roose 2001 formula.
Production model dynamics are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mycormarl-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "mycormarl-cache"))

import numpy as np
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig
from mycormarl.soil.phosphate_uptake import sparse_uptake_resistance, sparse_surface_concentration


def simulate_uptake_front(*, dt_days=0.01, horizon_days=24.0,
                          absorber_length_cm=0.1, presentation_length_cm=1.0,
                          far_field_half_life_days=2.0):
    """Sample one fixed 1-mm cylinder, then shift its curve to A/B/C/D.

    Rates are instantaneous nmol P/day. ``sample_rate_nmol_day`` is the
    physical 1-mm rate; displayed rates scale by presentation/sample length.
    Sampling interval does not enter the closure, only elapsed age does.
    Far-field concentration halves every two local-age days by default;
    ``None`` retains the earlier fixed-far-field reference. This imposed
    concentration is not calculated by phosphate inventory conservation.
    """
    values = [dt_days, horizon_days, absorber_length_cm, presentation_length_cm]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("times and lengths must be finite and positive")
    if far_field_half_life_days is not None and (
        not math.isfinite(far_field_half_life_days) or far_field_half_life_days <= 0
    ):
        raise ValueError("far_field_half_life_days must be finite and positive, or None")
    count = round(horizon_days / dt_days)
    if count < 1 or not math.isclose(count * dt_days, horizon_days, rel_tol=1e-10):
        raise ValueError("dt_days must divide horizon_days")
    arrivals = np.array([0., 6., 12., 18.])
    if not np.allclose(arrivals / dt_days, np.rint(arrivals / dt_days), rtol=0, atol=1e-8):
        raise ValueError("dt_days must resolve all contact times exactly")
    times = np.arange(count + 1, dtype=float) * dt_days
    c, f = EnvConfig(), FungusTraits()
    bulk_um = np.full_like(times, c.initial_solution_p_um)
    if far_field_half_life_days is not None:
        bulk_um *= 2.**(-times / far_field_half_life_days)
    bulk = bulk_um * 1e-3  # µM -> µmol/cm³
    dflux = c.phosphate_diffusion_coefficient_cm2_s * c.theta_water * c.phosphate_impedance_factor
    dapp = dflux / (c.theta_water + c.b_p)
    radius = f.hyphal_radius
    outer = radius + np.sqrt(dapp * times * 86400)
    # Zero neighbouring length density means infinite territory. Represented
    # length is specified independently below, not inferred from a soil cell.
    resistance = sparse_uptake_resistance(0., radius, f.jmax, dflux, dapp, times)
    surface = np.asarray(sparse_surface_concentration(bulk, f.km, resistance), dtype=float)
    flux = f.jmax * surface / (f.km + surface)
    sample_rate = flux * 2 * np.pi * radius * absorber_length_cm * 86400 * 1000
    isolated_rate = sample_rate * (presentation_length_cm / absorber_length_cm)
    rates = np.zeros((len(times), len(arrivals)))
    for site, arrival in enumerate(arrivals):
        shift = round(arrival / dt_days)
        if shift < len(times):
            rates[shift:, site] = isolated_rate[:len(times)-shift]

    radial_positions = np.geomspace(radius, outer[-1] * 1.1, 192)
    log_width = np.log(outer / radius)
    fraction = np.clip(np.log(radial_positions[None, :] / radius) /
                       np.where(log_width > 0, log_width, 1)[:, None], 0, 1)
    profile = (surface[:, None] + (bulk[:, None] - surface[:, None]) * fraction) * 1000
    # Explicit uniform initial condition avoids float32 roundoff at zero age.
    surface[0] = bulk[0]
    profile[0] = bulk[0] * 1000
    return {
        "format_version": 4,
        "time_days": times,
        "dt_days": dt_days,
        "arrival_days": arrivals,
        "plot_sites": np.arange(4),
        "absorber_length_cm": absorber_length_cm,
        "presentation_length_cm": presentation_length_cm,
        "presentation_scale": presentation_length_cm / absorber_length_cm,
        "far_field_half_life_days": far_field_half_life_days,
        "sample_rate_nmol_day": sample_rate,
        "isolated_rate_nmol_day": isolated_rate,
        "rate_nmol_day": rates,
        "initial_solution_p_um": c.initial_solution_p_um,
        "bulk_solution_p_um": bulk_um,
        "surface_solution_p_um": surface * 1000,
        "effective_radius_cm": outer,
        "radial_positions_cm": radial_positions,
        "radial_solution_p_um": profile,
        "parameters": {
            "hyphal_radius_cm": radius,
            "jmax_micromol_cm2_s": f.jmax,
            "km_micromol_cm3": f.km,
            "theta_water": c.theta_water,
            "buffer_power": c.b_p,
            "phosphate_diffusion_coefficient_cm2_s": c.phosphate_diffusion_coefficient_cm2_s,
            "phosphate_impedance_factor": c.phosphate_impedance_factor,
            "amount_flux_diffusivity_cm2_s": dflux,
            "apparent_diffusivity_cm2_s": dapp,
        },
        "assumptions": "Isolated cylindrical lateral uptake, no end effects; far-field P is prescribed as C0*2^(-local_age/half_life), or fixed when half_life is None; not a mass-conserving depletion simulation; sparse closure only; uncapped a+sqrt(D_app*local_age) depletion radius; quasi-steady logarithmic radial profile, not resolved transient PDE; identical independent segments shifted by contact time; illustrative branching and speed; no production dynamics changed. Greater uptake per unit length at the front does not imply most colony-integrated uptake.",
        "source": "Repository sparse closure; cylindrical depletion framework: Roose, Fowler & Darrah (2001), doi:10.1007/s002850000075. Fungal parameters adapt the root-cylinder framework; this is not the paper's exact asymptotic formula.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("media/phosphate-uptake/data.json"))
    args = parser.parse_args()
    data = simulate_uptake_front()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}, indent=2) + "\n")
    print(f"Saved {args.output}")
