"""Generate the retained analyses for issue #47's canonical screen.

This is the single plotting entry point for the canonical 360-condition screen.
Older exploratory figure generators are intentionally not called here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from mycormarl.resource_pressure_analysis import daily_amf_trace_rows


AMF_TIMING_OUTCOMES = (
    ("fungus_growth_rate_g_per_day", "Fungal realised growth (g day⁻¹)"),
    ("fungus_p_uptake_mg_per_day", "Fungal soil-P uptake (mg day⁻¹)"),
    ("fungus_p_transfer_mg_per_day", "Fungal P transfer to plant (mg day⁻¹)"),
    ("plant_indirect_p_fraction", "Plant indirect-P fraction"),
)

SCREEN_FACTORS = (
    ("plant_kappa_c", "Plant κC", "log"),
    ("fungus_kappa_c", "Fungus κC", "log"),
    ("fungus_gamma_p", "Fungal γP", "linear"),
    ("initial_solution_p_micromolar", "Initial P (µM)", "linear"),
    ("fungus_initial_biomass", "Initial AMF biomass", "log"),
    ("plant_trade", "Plant→fungus trade", "linear"),
    ("fungus_trade", "Fungus→plant trade", "linear"),
)


def load_entries(directory: Path) -> list[dict]:
    """Load canonical-screen condition checkpoints."""
    entries = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("condition-*.json"))]
    if not entries:
        raise FileNotFoundError(f"no condition checkpoints found in {directory}")
    return entries


def p_limited_fraction(entry: dict, agent: str) -> float:
    """Return the fraction of realised-growth steps limited by phosphate."""
    trace = [step["agents"][agent] for step in entry.get("limitation_trace", [])]
    realised = [not item["no_realized_growth"] for item in trace]
    return sum(
        item["limiting_resource"] == "phosphate" and active
        for item, active in zip(trace, realised)
    ) / max(sum(realised), 1)


def plot_gamma_p_limitation(entries: list[dict], output: Path) -> None:
    """Plot plant and fungal P-limitation fractions across P and fungal γP."""
    gamma = np.asarray([entry["factors"]["fungus_gamma_p"] for entry in entries], dtype=float)
    initial_p = np.asarray(
        [entry["factors"]["initial_solution_p_micromolar"] for entry in entries], dtype=float
    )
    plant_limitation = np.asarray([p_limited_fraction(entry, "plant") for entry in entries])
    fungus_limitation = np.asarray([p_limited_fraction(entry, "fungus") for entry in entries])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True, sharex=True, sharey=True)
    for axis, values, label in zip(axes, (plant_limitation, fungus_limitation), ("Plant", "Fungus")):
        scatter = axis.scatter(
            gamma, initial_p, c=values, cmap="magma", vmin=0, vmax=1,
            s=52, alpha=.8, edgecolors="white", linewidths=.3,
        )
        axis.set_title(f"{label} P limitation")
        axis.set_xlabel("Fungal γP")
        axis.set_ylabel("Initial solution P (µM)")
        fig.colorbar(scatter, ax=axis, label="P-limited timestep fraction")
    fig.suptitle("Initial P × fungal γP limitation landscape")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _tertile_edges(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) / 3)], ordered[int(2 * (len(ordered) - 1) / 3)]


def _tertile(value: float, edges: tuple[float, float]) -> int:
    return 0 if value <= edges[0] else 1 if value <= edges[1] else 2


def _plant_transition_rows(entries: list[dict]) -> list[dict]:
    return [
        {
            "id": entry["id"],
            "plant_kappa_c": float(entry["factors"]["plant_kappa_c"]),
            "plant_trade": float(entry["factors"]["plant_trade"]),
            "initial_p": float(entry["factors"]["initial_solution_p_micromolar"]),
            "plant_p_fraction": p_limited_fraction(entry, "plant"),
        }
        for entry in entries
    ]


def _fit_plant_transition(rows: list[dict]) -> tuple[float, str, float]:
    initial_p = np.asarray([row["initial_p"] for row in rows])
    p_fraction = np.asarray([row["plant_p_fraction"] for row in rows])
    slope, intercept = (
        np.polyfit(initial_p, p_fraction, 1)
        if len(rows) >= 2 and np.ptp(initial_p) > 0
        else (0.0, float(np.mean(p_fraction)))
    )
    if slope < 0:
        threshold = float((.5 - intercept) / slope)
        if .1 <= threshold <= 1.0:
            return threshold, "crossing", float(slope)
        if threshold > 1.0 or np.mean(p_fraction[initial_p >= np.quantile(initial_p, .8)]) > .5:
            return 1.0, ">1 (not crossed)", float(slope)
        return .1, "<0.1", float(slope)
    return (
        (1.0 if np.mean(p_fraction) > .5 else .1),
        (">1 (no downward crossing)" if np.mean(p_fraction) > .5 else "<0.1 (no downward crossing)"),
        float(slope),
    )


def plot_plant_transition_sensitivity(entries: list[dict], output: Path, summary_path: Path) -> None:
    """Show descriptive P/C transition trends across κC and trade tertiles."""
    rows = _plant_transition_rows(entries)
    kappa_edges = _tertile_edges([row["plant_kappa_c"] for row in rows])
    trade_edges = _tertile_edges([row["plant_trade"] for row in rows])
    grouped: dict[tuple[int, int], list[dict]] = {(kappa, trade): [] for kappa in range(3) for trade in range(3)}
    for row in rows:
        grouped[(_tertile(row["plant_kappa_c"], kappa_edges), _tertile(row["plant_trade"], trade_edges))].append(row)

    summaries = []
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True, constrained_layout=True)
    x_grid = np.linspace(.1, 1.0, 100)
    for axis, ((kappa_bin, trade_bin), subset) in zip(axes.ravel(), grouped.items()):
        threshold, status, slope = _fit_plant_transition(subset)
        summaries.append({
            "plant_kappa_c_bin": kappa_bin,
            "plant_trade_bin": trade_bin,
            "kappa_c_min": min(row["plant_kappa_c"] for row in subset),
            "kappa_c_max": max(row["plant_kappa_c"] for row in subset),
            "trade_min": min(row["plant_trade"] for row in subset),
            "trade_max": max(row["plant_trade"] for row in subset),
            "n": len(subset),
            "threshold_estimate": threshold,
            "threshold_status": status,
            "slope": slope,
        })
        initial_p = np.asarray([row["initial_p"] for row in subset])
        p_fraction = np.asarray([row["plant_p_fraction"] for row in subset])
        axis.scatter(initial_p, p_fraction, s=22, color="#777777", alpha=.6, edgecolors="white", linewidths=.25)
        if len(subset) >= 2 and np.ptp(initial_p) > 0:
            slope_fit, intercept_fit = np.polyfit(initial_p, p_fraction, 1)
            axis.plot(x_grid, np.clip(intercept_fit + slope_fit * x_grid, 0, 1), color="#4c72b0", linewidth=1.7)
        axis.axhline(.5, color="#333333", linestyle="--", linewidth=.8)
        axis.axvline(1.0, color="#c44e52", linestyle=":", linewidth=.9)
        axis.set_title(f"κC bin {kappa_bin + 1}, trade bin {trade_bin + 1}\nthreshold: {status}", fontsize=9)
        axis.set_xlim(.08, 1.02)
        axis.set_ylim(-.02, 1.02)
        axis.set_xlabel("Initial P (µM)")
        axis.set_ylabel("Plant P-limited fraction")
    fig.suptitle("Plant P/C transition sensitivity to plant κC and plant→fungus trade")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    _write_rows(summaries, summary_path)


def _limitation_fractions(entry: dict, agent: str) -> tuple[float, float]:
    trace = [step["agents"][agent] for step in entry["limitation_trace"]]
    realized = [not item["no_realized_growth"] for item in trace]
    denominator = max(sum(realized), 1)
    p_fraction = sum(item["limiting_resource"] == "phosphate" and active for item, active in zip(trace, realized)) / denominator
    c_fraction = sum(item["limiting_resource"] == "carbon" and active for item, active in zip(trace, realized)) / denominator
    return float(p_fraction), float(c_fraction)


def _regime(p_fraction: float, c_fraction: float) -> str:
    if p_fraction > 0.5:
        return "P-limited"
    if c_fraction > 0.5:
        return "C-limited"
    return "mixed"


def aggregate_regimes(entries: list[dict]) -> list[dict]:
    """Summarise limitation fractions and regimes for every condition."""
    rows = []
    for entry in entries:
        row = {"id": entry["id"]}
        row.update({name: float(entry["factors"][name]) for name, _, _ in SCREEN_FACTORS})
        row["plant_biomass"] = float(entry["biomass"]["plant"])
        row["fungus_biomass"] = float(entry["biomass"]["fungus"])
        for agent in ("plant", "fungus"):
            p_fraction, c_fraction = _limitation_fractions(entry, agent)
            row[f"{agent}_p_limited_fraction"] = p_fraction
            row[f"{agent}_c_limited_fraction"] = c_fraction
            row[f"{agent}_regime"] = _regime(p_fraction, c_fraction)
        rows.append(row)
    return rows


def write_regime_summary(rows: list[dict], output: Path) -> None:
    _write_rows(rows, output)


def plot_joint_regimes(rows: list[dict], output: Path) -> None:
    colors = {"P-limited": "#4c72b0", "C-limited": "#dd8452", "mixed": "#999999"}
    markers = {"P-limited": "o", "C-limited": "s", "mixed": "x"}
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for combination in sorted({(row["plant_regime"], row["fungus_regime"]) for row in rows}):
        subset = [row for row in rows if (row["plant_regime"], row["fungus_regime"]) == combination]
        label = f"plant {combination[0].removesuffix('-limited')}, fungus {combination[1].removesuffix('-limited')}"
        axis.scatter(
            [row["plant_p_limited_fraction"] for row in subset],
            [row["fungus_p_limited_fraction"] for row in subset],
            c=colors[combination[0]], marker=markers[combination[1]], s=48, alpha=.75,
            edgecolors="white", linewidths=.3, label=f"{label} (n={len(subset)})",
        )
    axis.axvline(.5, color="#333333", linestyle="--", linewidth=.9)
    axis.axhline(.5, color="#333333", linestyle="--", linewidth=.9)
    axis.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="Plant P-limited timestep fraction", ylabel="Fungus P-limited timestep fraction", title="Joint plant–fungus limitation regimes")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def factor_rows(entries: list[dict]) -> list[dict[str, float]]:
    rows = []
    for entry in entries:
        row = {name: float(entry["factors"][name]) for name, _, _ in SCREEN_FACTORS}
        row["plant_biomass"] = float(entry["biomass"]["plant"])
        row["fungus_biomass"] = float(entry["biomass"]["fungus"])
        rows.append(row)
    return rows


def plot_initial_p_biomass_response(entries: list[dict], output: Path) -> list[dict]:
    """Plot median plant and fungal biomass in 0.05-µM initial-P bins.

    The canonical Latin-hypercube design has 360 conditions over 0.10–1.00 µM,
    so each of these 18 bins contains 20 conditions.
    """
    edges = np.round(np.arange(.10, 1.0001, .05), 10)
    summaries = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        subset = [
            entry for entry in entries
            if lower <= float(entry["factors"]["initial_solution_p_micromolar"]) < upper
            or (upper == edges[-1] and lower <= float(entry["factors"]["initial_solution_p_micromolar"]) <= upper)
        ]
        plant_values = np.asarray([entry["biomass"]["plant"] for entry in subset], dtype=float)
        fungus_values = np.asarray([entry["biomass"]["fungus"] for entry in subset], dtype=float)
        summaries.append({
            "initial_p_lower_micromolar": float(lower),
            "initial_p_upper_micromolar": float(upper),
            "initial_p_midpoint_micromolar": float((lower + upper) / 2),
            "n_conditions": len(subset),
            "median_plant_biomass_g": float(np.median(plant_values)),
            "plant_biomass_q25_g": float(np.quantile(plant_values, .25)),
            "plant_biomass_q75_g": float(np.quantile(plant_values, .75)),
            "median_fungus_biomass_g": float(np.median(fungus_values)),
            "fungus_biomass_q25_g": float(np.quantile(fungus_values, .25)),
            "fungus_biomass_q75_g": float(np.quantile(fungus_values, .75)),
        })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, constrained_layout=True)
    for axis, key, q25_key, q75_key, label, colour in (
        (axes[0], "median_plant_biomass_g", "plant_biomass_q25_g", "plant_biomass_q75_g", "Final plant biomass (g)", "#4c72b0"),
        (axes[1], "median_fungus_biomass_g", "fungus_biomass_q25_g", "fungus_biomass_q75_g", "Final fungal biomass (g)", "#dd8452"),
    ):
        x_values = np.asarray([row["initial_p_midpoint_micromolar"] for row in summaries])
        y_values = np.asarray([row[key] for row in summaries])
        errors = np.asarray([
            y_values - np.asarray([row[q25_key] for row in summaries]),
            np.asarray([row[q75_key] for row in summaries]) - y_values,
        ])
        axis.errorbar(x_values, y_values, yerr=errors, fmt="o", color=colour, markersize=5.5, capsize=3, linewidth=1.1)
        axis.set_xlabel("Initial solution P (µM)")
        axis.set_ylabel(label)
        axis.set_xlim(.10, 1.00)
        axis.grid(color="#dddddd", linewidth=.5)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summaries


def _binned_eta_squared(rows: list[dict[str, float]], factor: str, outcome: str, bins: int = 8) -> float:
    values = np.asarray([row[factor] for row in rows], dtype=float)
    outcomes = np.asarray([row[outcome] for row in rows], dtype=float)
    total = max(float(np.var(outcomes)), 1e-15)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    groups = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    overall = float(np.mean(outcomes))
    between = sum(float(mask.sum()) * (float(np.mean(outcomes[mask])) - overall) ** 2 for group in np.unique(groups) if (mask := groups == group).any())
    return between / len(outcomes) / total


def rank_factors(rows: list[dict[str, float]], outcome: str) -> list[tuple[str, str, float]]:
    return sorted(((name, label, _binned_eta_squared(rows, name, outcome)) for name, label, _ in SCREEN_FACTORS), key=lambda item: item[2], reverse=True)


def plot_factor_ranking(effects: list[tuple[str, str, float]], title: str, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    labels = [label for _, label, _ in effects][::-1]
    values = [value for _, _, value in effects][::-1]
    bars = axis.barh(labels, values, color="#4169a1")
    axis.set_xlabel("Between-bin variance / total variance (η²)")
    axis.set_xlim(0, max(max(values) * 1.12, .05))
    axis.set_title(title)
    for bar, value in zip(bars, values):
        axis.text(value + axis.get_xlim()[1] * .012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def load_design(progress_path: Path) -> list[dict[str, float]]:
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    return [row["factors"] for row in progress["design"]["conditions"]]


def _scaled(values: np.ndarray, scale: str) -> np.ndarray:
    transformed = np.log10(values) if scale == "log" else values
    lo, hi = float(np.min(transformed)), float(np.max(transformed))
    return (transformed - lo) / (hi - lo) if hi > lo else np.zeros_like(transformed)


def plot_design(factors: list[dict[str, float]], output: Path) -> None:
    """Plot the canonical Latin-hypercube design and its marginal coverage."""
    values = {name: np.asarray([row[name] for row in factors], dtype=float) for name, _, _ in SCREEN_FACTORS}
    scaled = np.column_stack([_scaled(values[name], scale) for name, _, scale in SCREEN_FACTORS])
    colors = values["initial_solution_p_micromolar"]
    norm = plt.Normalize(float(colors.min()), float(colors.max()))
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(3.2, 1.25), wspace=.08)
    axis = fig.add_subplot(grid[0, 0])
    x = np.arange(len(SCREEN_FACTORS))
    for row, colour in zip(scaled, colors):
        axis.plot(x, row, color=plt.cm.viridis(norm(colour)), alpha=.22, linewidth=.8)
    axis.set(xlim=(0, len(SCREEN_FACTORS) - 1), ylim=(0, 1), ylabel="Within-factor percentile (κC and AMF biomass log-scaled)", title="Canonical Latin-hypercube design")
    axis.set_xticks(x, [label for _, label, _ in SCREEN_FACTORS], rotation=28, ha="right")
    axis.set_yticks([0, .5, 1], ["low", "mid", "high"])
    axis.grid(axis="y", color="#dddddd", linewidth=.6)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axis, pad=.02, label="Initial solution P (µM)")
    hist_grid = grid[0, 1].subgridspec(len(SCREEN_FACTORS), 1, hspace=.8)
    for index, (name, label, scale) in enumerate(SCREEN_FACTORS):
        hist_axis = fig.add_subplot(hist_grid[index, 0])
        hist_axis.hist(values[name], bins=12, color="#4c72b0", alpha=.8, edgecolor="white")
        hist_axis.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=8, fontsize=8)
        hist_axis.set_yticks([])
        if scale == "log":
            hist_axis.set_xscale("log")
        if index < len(SCREEN_FACTORS) - 1:
            hist_axis.set_xticks([])
        else:
            hist_axis.set_xlabel("Sampled value")
        hist_axis.spines[["top", "right", "left"]].set_visible(False)
        hist_axis.grid(axis="x", color="#eeeeee", linewidth=.5)
    fig.suptitle("Canonical parameter-space coverage", fontsize=15)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _amf_groups(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Return equal-count initial-AMF tertiles and data-derived labels."""
    values = np.asarray([float(row["fungus_initial_biomass"]) for row in rows])
    lower, upper = np.quantile(values, [1 / 3, 2 / 3])
    groups = np.digitize(values, [lower, upper], right=True)
    labels = [
        f"Low initial AMF (≤{lower:.2g}×)",
        f"Mid initial AMF ({lower:.2g}–{upper:.2g}×)",
        f"High initial AMF (>{upper:.2g}×)",
    ]
    return groups, labels


def _write_rows(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_amf_timing_trajectories(rows: list[dict], output: Path) -> list[dict]:
    """Plot daily functional trajectories by initial-AMF tertile.

    This is descriptive: the LHS balances factors overall but does not create
    matched AMF-inoculum pairs.  The adjusted companion figure reports the
    conditional association.
    """
    groups, labels = _amf_groups(rows)
    days = sorted({float(row["day"]) for row in rows})
    colours = ("#4c72b0", "#55a868", "#c44e52")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    summary: list[dict] = []
    for axis, (outcome, label) in zip(axes.flat, AMF_TIMING_OUTCOMES):
        for group, (group_label, colour) in enumerate(zip(labels, colours)):
            median, q25, q75 = [], [], []
            for day in days:
                values = np.asarray([
                    float(row[outcome]) for row, assigned_group in zip(rows, groups)
                    if assigned_group == group and float(row["day"]) == day
                ])
                median.append(float(np.median(values)))
                q25.append(float(np.quantile(values, .25)))
                q75.append(float(np.quantile(values, .75)))
                summary.append({
                    "outcome": outcome, "day": day, "amf_group": group_label,
                    "n_conditions": len(values), "median": median[-1],
                    "q25": q25[-1], "q75": q75[-1],
                })
            axis.plot(days, median, color=colour, label=group_label, linewidth=1.8)
            axis.fill_between(days, q25, q75, color=colour, alpha=.16, linewidth=0)
        axis.set_ylabel(label)
        axis.grid(color="#dddddd", linewidth=.5)
    axes[1, 0].set_xlabel("Simulation day")
    axes[1, 1].set_xlabel("Simulation day")
    axes[1, 1].set_ylim(-.02, 1.02)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary


def _linear_sse(y: np.ndarray, predictors: np.ndarray) -> float:
    design = np.column_stack((np.ones(len(y)), predictors))
    residuals = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.dot(residuals, residuals))


def _amf_partial_r_squared(rows: list[dict], outcome: str, bootstrap_seed: int = 3047, bootstrap_count: int = 200) -> list[dict]:
    """Estimate daily AMF partial R² after linear adjustment for six factors.

    Flux outcomes are log1p-transformed to avoid a few high-rate conditions
    dominating the fit; the bounded plant indirect-P fraction remains on its
    original 0–1 scale.  Bootstrap intervals resample whole conditions within
    each day and quantify screen-sample uncertainty, not simulator stochasticity.
    """
    rng = np.random.default_rng(bootstrap_seed)
    results = []
    for day in sorted({float(row["day"]) for row in rows}):
        day_rows = [row for row in rows if float(row["day"]) == day]
        y = np.asarray([float(row[outcome]) for row in day_rows])
        if outcome != "plant_indirect_p_fraction":
            y = np.log1p(y)
        base = np.asarray([[
            np.log(float(row["plant_kappa_c"])),
            np.log(float(row["fungus_kappa_c"])),
            float(row["fungus_gamma_p"]),
            float(row["initial_solution_p_micromolar"]),
            float(row["plant_trade"]),
            float(row["fungus_trade"]),
        ] for row in day_rows])
        amf = np.log(np.asarray([float(row["fungus_initial_biomass"]) for row in day_rows]))[:, None]

        def partial_r2(indices: np.ndarray) -> float:
            sse_base = _linear_sse(y[indices], base[indices])
            if sse_base <= 1e-18:
                return 0.0
            sse_full = _linear_sse(y[indices], np.column_stack((base[indices], amf[indices])))
            return max(0.0, (sse_base - sse_full) / sse_base)

        all_indices = np.arange(len(day_rows))
        estimate = partial_r2(all_indices)
        bootstrap = np.asarray([partial_r2(rng.integers(0, len(day_rows), len(day_rows))) for _ in range(bootstrap_count)])
        results.append({
            "outcome": outcome, "day": day, "partial_r_squared": estimate,
            "bootstrap_q05": float(np.quantile(bootstrap, .05)),
            "bootstrap_q95": float(np.quantile(bootstrap, .95)),
            "n_conditions": len(day_rows), "transform": "log1p" if outcome != "plant_indirect_p_fraction" else "identity",
        })
    return results


def plot_amf_adjusted_timing(rows: list[dict], output: Path) -> list[dict]:
    """Plot time-local AMF partial R² conditional on the other sampled factors."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    results: list[dict] = []
    for axis, (outcome, label) in zip(axes, AMF_TIMING_OUTCOMES[:3]):
        estimates = _amf_partial_r_squared(rows, outcome)
        days = [row["day"] for row in estimates]
        values = [row["partial_r_squared"] for row in estimates]
        lower = [row["bootstrap_q05"] for row in estimates]
        upper = [row["bootstrap_q95"] for row in estimates]
        axis.plot(days, values, color="#4c72b0", linewidth=1.6)
        axis.fill_between(days, lower, upper, color="#4c72b0", alpha=.2, linewidth=0)
        axis.set_ylabel(f"AMF partial R²\n{label}")
        axis.set_ylim(bottom=0)
        axis.grid(color="#dddddd", linewidth=.5)
        results.extend(estimates)
    axes[-1].set_xlabel("Simulation day")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return results


def uptake_rows(checkpoint_dir: Path) -> list[dict]:
    """Load plant P-source ratios and limitation outcomes from checkpoints."""
    rows = []
    for path in sorted(checkpoint_dir.glob("condition-*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        trace = [step["agents"]["plant"] for step in entry["limitation_trace"]]
        realized = [not step["no_realized_growth"] for step in trace]
        p_limited = sum(step["limiting_resource"] == "phosphate" and active for step, active in zip(trace, realized)) / max(sum(realized), 1)
        direct = float(entry["cumulative_direct_p_uptake_mg"]["plant"])
        indirect = float(entry["transfers"]["fungus_p_out"])
        total = direct + indirect
        rows.append({
            "id": entry["id"],
            "initial_p": float(entry["factors"]["initial_solution_p_micromolar"]),
            "ratio": indirect / direct if direct > 0 else np.inf,
            "direct_fraction": direct / total if total > 0 else 0.0,
            "indirect_fraction": indirect / total if total > 0 else 0.0,
            "p_limited": p_limited,
            "plant_biomass": float(entry["biomass"]["plant"]),
        })
    return rows


def plot_ordered_uptake_ratios(rows: list[dict], output: Path, order_key: str, title: str) -> None:
    """Plot direct/indirect P-source fractions in four initial-P strata."""
    ordered = sorted(rows, key=lambda row: row["initial_p"])
    strata = np.array_split(ordered, 4)
    rolling_window = 15
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True, constrained_layout=True)
    for ax, subset in zip(axes, reversed(strata)):
        lo = min(row["initial_p"] for row in subset)
        hi = max(row["initial_p"] for row in subset)
        subset = sorted(list(subset), key=lambda row: row[order_key], reverse=True)
        y = np.arange(len(subset))
        direct = np.asarray([row["direct_fraction"] for row in subset])
        indirect = np.asarray([row["indirect_fraction"] for row in subset])
        ax.barh(y, direct, color="#4c72b0", edgecolor="white", linewidth=.25, height=.8, label="Direct soil-P uptake")
        ax.barh(y, indirect, left=direct, color="#dd8452", edgecolor="white", linewidth=.25, height=.8, label="Indirect fungal P transfer")
        half_window = rolling_window // 2
        lower = np.empty(len(subset))
        median = np.empty(len(subset))
        upper = np.empty(len(subset))
        for index in range(len(subset)):
            start = max(0, index - half_window)
            stop = min(len(subset), index + half_window + 1)
            window = indirect[start:stop]
            lower[index], median[index], upper[index] = np.quantile(window, [.25, .5, .75])
        # The stacked-bar boundary is the direct fraction, i.e. one minus the
        # indirect fraction used for the rolling summary above.
        ax.fill_betweenx(
            y, 1 - upper, 1 - lower, color="#202020", alpha=.24,
            linewidth=0, label="Rolling IQR (15 conditions)",
        )
        ax.plot(1 - median, y, color="white", linewidth=2.5, zorder=4)
        ax.plot(1 - median, y, color="#202020", linewidth=1.1, zorder=5, label="Rolling median (15 conditions)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(len(subset) - 0.5, -0.5)
        ax.set_title(f"Initial P: {lo:.3f}–{hi:.3f} µM (n={len(subset)})", loc="left", fontsize=10)
        ax.set_yticks([])
        ax.grid(axis="x", color="#dddddd", linewidth=.5)
    axes[-1].set_xlabel("Fraction of plant P uptake")
    axes[-1].legend(loc="lower right", frameon=True)
    if order_key == "plant_biomass":
        annotation = axes[0].annotate(
            "Increasing final plant biomass",
            xy=(1.04, .18), xytext=(1.04, .82), xycoords="axes fraction",
            ha="center", va="center", rotation=90, fontsize=10, color="#333333",
            arrowprops={"arrowstyle": "-|>", "mutation_scale": 10, "linewidth": 1.0, "color": "#333333"},
            annotation_clip=False,
        )
        annotation.set_in_layout(False)
    fig.suptitle(title, fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plant_limitation_matrices(entries: list[dict]) -> list[tuple[list[dict], np.ndarray, np.ndarray]]:
    """Return biomass-ranked plant pressure and limitation-sign matrices by P stratum.

    Pressure is reconstructed from the allocation fields, rather than from the
    stored signed value, so the figure has the new P-positive convention for
    both historical and future checkpoint traces.
    """
    ordered = sorted(entries, key=lambda entry: float(entry["factors"]["initial_solution_p_micromolar"]))
    matrices = []
    for stratum in np.array_split(ordered, 4):
        ranked = sorted(stratum.tolist(), key=lambda entry: float(entry["biomass"]["plant"]), reverse=True)
        pressure_rows = []
        sign_rows = []
        for entry in ranked:
            trace = [step["agents"]["plant"] for step in entry["limitation_trace"]]
            pressure = np.asarray([
                float(step["allocated_c_normalized"]) - float(step["allocated_p_normalized"])
                for step in trace
            ])
            sign = np.asarray([
                {"carbon": -1.0, "balanced": 0.0, "phosphate": 1.0}.get(step["limiting_resource"], np.nan)
                for step in trace
            ])
            no_growth = np.asarray([bool(step["no_realized_growth"]) for step in trace])
            pressure[no_growth] = np.nan
            sign[no_growth] = np.nan
            pressure_rows.append(pressure)
            sign_rows.append(sign)
        matrices.append((ranked, np.vstack(pressure_rows), np.vstack(sign_rows)))
    return matrices


def _plot_plant_limitation_time_heatmap(
    matrices: list[tuple[list[dict], np.ndarray, np.ndarray]],
    output: Path,
    *,
    mode: str,
) -> None:
    """Plot plant limitation pressure or categorical sign through time."""
    if mode not in {"pressure", "sign"}:
        raise ValueError(f"unsupported limitation heatmap mode: {mode}")
    all_pressure = np.concatenate([pressure[np.isfinite(pressure)] for _, pressure, _ in matrices])
    scale = float(np.quantile(np.abs(all_pressure), .99)) if all_pressure.size else 1.0
    scale = max(scale, 1e-12)
    pressure_cmap = plt.colormaps["coolwarm"].copy()
    pressure_cmap.set_bad("#b8b8b8")
    sign_cmap = ListedColormap(["#3b73b9", "#ffffff", "#c74b50"])
    sign_cmap.set_bad("#b8b8b8")
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True, constrained_layout=True)
    image = None
    for axis, (ranked, pressure, sign) in zip(axes, reversed(matrices)):
        values = pressure if mode == "pressure" else sign
        days = [float(step["day"]) for step in ranked[0]["limitation_trace"]]
        if mode == "pressure":
            image = axis.imshow(
                values, aspect="auto", interpolation="nearest", origin="upper",
                extent=(0.0, days[-1], len(ranked), 0.0), cmap=pressure_cmap,
                norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
            )
        else:
            image = axis.imshow(
                values, aspect="auto", interpolation="nearest", origin="upper",
                extent=(0.0, days[-1], len(ranked), 0.0), cmap=sign_cmap,
                norm=BoundaryNorm([-1.5, -.5, .5, 1.5], sign_cmap.N),
            )
        initial_p = [float(entry["factors"]["initial_solution_p_micromolar"]) for entry in ranked]
        axis.set_title(f"Initial P: {min(initial_p):.3f}–{max(initial_p):.3f} µM (n={len(ranked)})", loc="left", fontsize=10)
        axis.set_yticks([])
        axis.set_ylabel("Plant biomass\nrank: high → low")
    axes[-1].set_xlabel("Simulation day")
    if mode == "pressure":
        fig.suptitle("Plant P-positive limitation pressure through time")
        colorbar = fig.colorbar(image, ax=axes, shrink=.7, pad=.015)
        colorbar.set_label("C-equivalent − P-equivalent allocation (g biomass-equivalent)")
    else:
        fig.suptitle("Plant limitation sign through time")
        colorbar = fig.colorbar(image, ax=axes, shrink=.7, pad=.015, ticks=[-1, 0, 1])
        colorbar.ax.set_yticklabels(["C-limited", "Balanced", "P-limited"])
        colorbar.set_label("Limitation state; grey = no realised growth")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_plant_limitation_time_heatmaps(entries: list[dict], output_dir: Path) -> None:
    """Write the canonical plant pressure-magnitude and limitation-sign figures."""
    matrices = _plant_limitation_matrices(entries)
    _plot_plant_limitation_time_heatmap(
        matrices, output_dir / "resource-pressure-plant-limitation-pressure-through-time.png", mode="pressure"
    )
    _plot_plant_limitation_time_heatmap(
        matrices, output_dir / "resource-pressure-plant-limitation-sign-through-time.png", mode="sign"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--progress", type=Path, help="Optional experiment progress JSON for the design plot.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Retained factor-effect summaries including all sampled factors.
    entries = load_entries(args.checkpoint_dir)
    factor_data = factor_rows(entries)
    rankings = []
    for outcome, label, slug in (("plant_biomass", "plant", "plant-biomass"), ("fungus_biomass", "fungal", "fungal-biomass")):
        effects = rank_factors(factor_data, outcome)
        plot_factor_ranking(
            effects,
            f"Factor-effect ranking for final {label} biomass (all factors)",
            args.output_dir / f"resource-pressure-canonical-factor-effects-{slug}-all-factors.png",
        )
        rankings.extend(
            {"outcome": outcome, "scope": "all-factors", "factor": factor, "label": factor_label, "eta_squared": eta}
            for factor, factor_label, eta in effects
        )
    with (args.output_dir / "resource-pressure-canonical-factor-effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["outcome", "scope", "factor", "label", "eta_squared"])
        writer.writeheader(); writer.writerows(rankings)

    # Retained gamma-P limitation figure and initial-P biomass response.
    plot_gamma_p_limitation(entries, args.output_dir / "resource-pressure-gamma-p-limitation.png")
    initial_p_summary = plot_initial_p_biomass_response(entries, args.output_dir / "resource-pressure-initial-p-biomass-response.png")
    _write_rows(initial_p_summary, args.output_dir / "resource-pressure-initial-p-biomass-response.csv")

    # Retained regime outputs: joint regimes and a tabular summary.  The
    # gamma-P limitation figure above already displays the P-limitation view.
    regime_rows = aggregate_regimes(entries)
    write_regime_summary(regime_rows, args.output_dir / "resource-pressure-regime-summary.csv")
    plot_joint_regimes(regime_rows, args.output_dir / "resource-pressure-joint-regimes.png")

    # Retained plant-threshold sensitivity outputs.
    plot_plant_transition_sensitivity(
        entries,
        args.output_dir / "resource-pressure-plant-threshold-sensitivity.png",
        args.output_dir / "resource-pressure-plant-threshold-sensitivity.csv",
    )

    plot_plant_limitation_time_heatmaps(entries, args.output_dir)

    source_rows = uptake_rows(args.checkpoint_dir)
    plot_ordered_uptake_ratios(source_rows, args.output_dir / "resource-pressure-plant-p-uptake-ratio-ordered-by-biomass.png", "plant_biomass", "Plant indirect:direct P uptake ratio ordered by final plant biomass")

    # AMF initial-biomass timing: functional fluxes rather than absolute fungal
    # biomass, whose starting value differs by construction.
    timing_rows = daily_amf_trace_rows(entries)
    timing_summary = plot_amf_timing_trajectories(
        timing_rows,
        args.output_dir / "resource-pressure-amf-initial-biomass-timing.png",
    )
    _write_rows(timing_summary, args.output_dir / "resource-pressure-amf-initial-biomass-timing.csv")
    adjusted_summary = plot_amf_adjusted_timing(
        timing_rows,
        args.output_dir / "resource-pressure-amf-initial-biomass-adjusted-effect.png",
    )
    _write_rows(adjusted_summary, args.output_dir / "resource-pressure-amf-initial-biomass-adjusted-effect.csv")

    if args.progress is not None:
        plot_design(load_design(args.progress), args.output_dir / "resource-pressure-continuous-design.png")
    print(f"wrote retained continuous gamma-P analyses for {len(result_rows)} conditions to {args.output_dir}")


if __name__ == "__main__":
    main()
