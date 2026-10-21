"""Illustrate the agreed action model; does not import the production environment.

Run with the project Python. All elemental quantities are divided by a common
growth cost (gamma_C or gamma_P), so one unit supports one growth-output unit.
Both illustrative organisms share those costs. This is not a calibration.
"""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mycormarl-allocation-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.integrate import solve_ivp

OUT = Path(__file__).resolve().parent
CASES = [
    dict(id="A", name="Balanced pools, no trade", c=1., p=1., kt=0., kb=1., a=.25, s=1.),
    dict(id="B", name="Carbon surplus, C trade", c=2., p=.5, kt=1., kb=1., a=.25, s=1.),
    dict(id="C", name="Balanced pools, C trade", c=1., p=1., kt=1., kb=1., a=.25, s=1.),
    dict(id="D", name="Complementary partners", c=2., p=.2, kt=1., kb=1., a=.25, s=1., reciprocal=True),
    dict(id="E", name="More reproductive investment", c=1., p=1., kt=1., kb=1., a=.75, s=1.),
    dict(id="F", name="Double reproductive cost", c=1., p=1., kt=1., kb=1., a=.25, s=2.),
]
COLORS = ["#0072B2", "#E69F00", "#CC79A7", "#B8BEC5"]
LABELS = ["Growth use", "Reproductive use", "Trade out", "Retained"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "medium", "savefig.facecolor": "white"})


def run(case, max_step=.0025):
    # Per organism: free C, free P, cumulative growth, reproduction output,
    # C traded out, P traded out, C received, P received.
    initial = np.zeros((2, 8))
    initial[0, :2] = [case["c"], case["p"]]
    if case.get("reciprocal"):
        initial[1, :2] = [.2, 2.]
    ks = np.array([case["kt"], 1. if case.get("reciprocal") else 0.])
    kb = np.array([case["kb"], 1. if case.get("reciprocal") else 0.])
    shares = np.array([case["a"], .25])
    costs = np.array([case["s"], 1.])

    def rhs(t, flat):
        y = flat.reshape(2, 8)
        q = np.minimum(y[:, 0], y[:, 1])
        g = (1 - shares) * kb * q
        r = shares * kb * q / costs
        tc = np.array([ks[0] * y[0, 0], 0.])
        tp = np.array([0., ks[1] * y[1, 1]])
        ic, ip = tc[::-1], tp[::-1]
        dy = np.column_stack((-g-costs*r-tc+ic, -g-costs*r-tp+ip,
                              g, r, tc, tp, ic, ip))
        return dy.ravel()

    solution = solve_ivp(rhs, (0., 1.), initial.ravel(), method="DOP853",
                         rtol=1e-11, atol=1e-13, max_step=max_step,
                         t_eval=np.linspace(0, 1, 401))
    if not solution.success:
        raise RuntimeError(solution.message)
    history = solution.y.T.reshape(-1, 2, 8)
    if history[:, :, :2].min() < -1e-10:
        raise AssertionError("Negative free stock")
    # Verify every organism/resource at every output time, not just totals.
    consumption = history[:, :, 2] + costs * history[:, :, 3]
    balance_c = initial[:, 0] + history[:, :, 6] - history[:, :, 0] - consumption - history[:, :, 4]
    balance_p = initial[:, 1] + history[:, :, 7] - history[:, :, 1] - consumption - history[:, :, 5]
    residual = float(max(np.abs(balance_c).max(), np.abs(balance_p).max()))
    assert residual < 1e-9, residual
    assert np.max(np.abs(history[:, 0, 4] - history[:, 1, 6])) < 1e-10
    assert np.max(np.abs(history[:, 1, 5] - history[:, 0, 7])) < 1e-10
    return solution.t, history, residual


def main():
    results, rows = [], []
    max_balance = max_refinement = 0.
    for case in CASES:
        t, y, residual = run(case)
        _, finer, _ = run(case, .00125)
        error = float(np.max(np.abs(y - finer)))
        assert error < 1e-8, (case["id"], error)
        max_balance, max_refinement = max(max_balance, residual), max(max_refinement, error)
        f = y[-1, 0]
        row = dict(case=case["id"], name=case["name"], c_initial=case["c"],
            p_initial=case["p"], trade_rate=case["kt"], biological_rate=case["kb"],
            reproductive_share=case["a"], reproductive_cost=case["s"],
            growth=f[2], reproductive_output=f[3], reproductive_resource_use=case["s"]*f[3],
            c_trade_out=f[4], p_trade_out=f[5], c_received=f[6], p_received=f[7],
            c_retained=f[0], p_retained=f[1])
        rows.append(row)
        results.append((case, t, y))
    assert abs(rows[0]["growth"] - .75 * (1-np.exp(-1))) < 1e-10
    for field in ("c_retained", "p_retained", "c_trade_out"):
        assert abs(rows[2][field] - rows[4][field]) < 1e-10
    for field in ("c_retained", "p_retained", "growth", "c_trade_out", "reproductive_resource_use"):
        assert abs(rows[2][field] - rows[5][field]) < 1e-10
    assert abs(rows[2]["reproductive_output"] - 2*rows[5]["reproductive_output"]) < 1e-10
    # B remains P-limited throughout the day, so C trade does not reduce biology.
    assert abs(rows[1]["growth"] - .75*.5*(1-np.exp(-1))) < 1e-10
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "trajectories.json").write_text(json.dumps([
        {"parameters": c, "time_days": t.tolist(), "state": y.tolist()}
        for c, t, y in results], indent=2))

    fig, axes = plt.subplots(3, 2, figsize=(13, 11.5), sharex=True)
    for ax, (case, t, y), row in zip(axes.flat, results, rows):
        f = y[-1, 0]
        parts = np.array([[f[2], case["s"]*f[3], f[4], f[0]],
                          [f[2], case["s"]*f[3], f[5], f[1]]])
        left = np.zeros(2)
        for j, color in enumerate(COLORS):
            ax.barh([1, 0], parts[:, j], left=left, height=.38, color=color)
            for i, ypos in enumerate([1, 0]):
                width = parts[i, j]
                if width > .13:
                    ax.text(left[i]+width/2, ypos, f"{width:.2f}", ha="center", va="center", fontsize=10,
                            color="white" if j == 0 else "#20252A")
            left += parts[:, j]
        ax.set_yticks([1, 0], ["C", "P"])
        ax.set_ylim(-.7, 1.8)
        ax.set_xlim(0, 2.15)
        ax.set_xticks([0, .5, 1, 1.5, 2])
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.set_title(f"{case['id']}  {case['name']}\n"
            f"C₀={case['c']:g}, P₀={case['p']:g}  |  kₜ={case['kt']:g}/d, kᵦ=1/d, a={case['a']:.2f}, s={case['s']:g}",
            loc="left", fontsize=11, pad=12)
        ax.text(0, -.6, f"Growth output {f[2]:.3f}   ·   Reproductive output {f[3]:.3f}", fontsize=10)
        if case.get("reciprocal"):
            ax.text(0, 1.55, f"P received +{f[7]:.3f}; bars include incoming resources", fontsize=10)
        ax.set_xlabel("Resource amount / growth resource cost")
    fig.suptitle("Where C and P go over one day", fontsize=19, y=.985)
    fig.text(.5, .948, "Fixed commands · 25% reproductive expenditure except E · no maintenance, photosynthesis or soil uptake",
             ha="center", fontsize=11)
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(COLORS, LABELS)],
               loc="upper center", bbox_to_anchor=(.5, .925), ncol=4, frameon=False)
    fig.subplots_adjust(top=.85, bottom=.06, hspace=.68, wspace=.20, left=.06, right=.98)
    fig.savefig(OUT / "allocation-budgets.png", dpi=180)
    fig.savefig(OUT / "allocation-budgets.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharex=True, sharey=True)
    for ax, index in zip(axes, [1, 2, 3]):
        case, t, y = results[index]
        for field, color, style, label in [(0,"#0072B2","-","Free C"), (1,"#CC79A7","--","Free P"),
                (2,"#009E73","-.","Cumulative growth"), (3,"#E69F00",":","Cumulative reproduction")]:
            ax.plot(t, y[:, 0, field], color=color, linestyle=style, linewidth=2.4, label=label)
        ax.set_title(f"{case['id']}  {case['name']}", fontsize=12, loc="left")
        ax.set_xlabel("Time (days)")
        ax.set_ylim(0, 2.1)
        ax.set_xlim(0, 1)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Normalised resource / output units")
    fig.suptitle("Pools and production evolve together", fontsize=18, y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.91), ncol=4, frameon=False)
    fig.subplots_adjust(top=.73, bottom=.15, left=.07, right=.98, wspace=.18)
    fig.savefig(OUT / "allocation-dynamics.png", dpi=180)
    fig.savefig(OUT / "allocation-dynamics.pdf")
    plt.close(fig)
    checks = dict(max_resource_balance_residual=max_balance,
                  max_refinement_difference=max_refinement, analytic_checks="passed")
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps({"summary": rows, "verification": checks}, indent=2))


if __name__ == "__main__":
    main()
