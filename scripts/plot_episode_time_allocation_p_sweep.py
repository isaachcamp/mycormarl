"""Recreate the plant-only episode time-allocation summary figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


POLICY_ORDER = (
    "static-growth-10-reproduction-90",
    "static-growth-90-reproduction-10",
    "grow-to-reproduce-day-90",
)
POLICY_LABELS = {
    POLICY_ORDER[0]: "static 10/90",
    POLICY_ORDER[1]: "static 90/10",
    POLICY_ORDER[2]: "switch day 90",
}
POLICY_COLOURS = {POLICY_ORDER[0]: "tab:blue", POLICY_ORDER[1]: "tab:orange", POLICY_ORDER[2]: "tab:green"}
P_LEVELS = (0.75, 3.0, 10.0)
P_STYLES = {0.75: ("-", "o"), 3.0: ("--", "s"), 10.0: (":", "^")}
POLICY_MARKERS = {
    POLICY_ORDER[0]: "o",
    POLICY_ORDER[1]: "s",
    POLICY_ORDER[2]: "^",
}


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for condition in payload["conditions"]:
        p = float(condition["initial_solution_p_um"])
        for row in condition["entries"]:
            rows.append({"p": p, **row})
    return rows


def plot(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2), constrained_layout=True)
    labels = ("a.", "b.", "c.")
    for axis, label in zip(axes, labels, strict=True):
        axis.text(-0.12, 1.02, label, transform=axis.transAxes,
                  ha="right", va="bottom", fontsize=10, clip_on=False)

    for policy in POLICY_ORDER:
        subset = [row for row in rows if row["policy"]["identifier"] == policy]
        for p in P_LEVELS:
            endpoint = [row for row in subset if row["p"] == p]
            axes[0].scatter(
                [row["final_living_biomass_g"] for row in endpoint],
                [row["cumulative_reproductive_fitness"] for row in endpoint],
                color=POLICY_COLOURS[policy], marker=P_STYLES[p][1], s=58,
                label=POLICY_LABELS[policy] if p == P_LEVELS[0] else None,
            )
        for row in subset:
            linestyle, marker = P_STYLES[row["p"]]
            axes[1].plot(
                [point["elapsed_days"] for point in row["trajectory"]],
                [point["final_living_biomass_g"] for point in row["trajectory"]],
                color=POLICY_COLOURS[policy], linestyle=linestyle,
                label=POLICY_LABELS[policy] if row["p"] == P_LEVELS[0] else None,
            )
            axes[2].plot(
                [point["elapsed_days"] for point in row["trajectory"]],
                [point["cumulative_reproductive_fitness"] for point in row["trajectory"]],
                color=POLICY_COLOURS[policy], linestyle=linestyle,
                label=POLICY_LABELS[policy] if row["p"] == P_LEVELS[0] else None,
            )

    axes[0].set_xlabel("Final living biomass (g)")
    axes[0].set_ylabel("Cumulative reproductive fitness")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(axis="both", which="both", color="#dddddd", linewidth=.5)
    policy_legend = axes[0].legend(frameon=True, loc="upper left")
    axes[0].add_artist(policy_legend)

    for axis, ylabel in zip(
        axes[1:], ("Living biomass (g)", "Cumulative reproductive fitness"), strict=True
    ):
        axis.set_xlabel("Elapsed days")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(axis="both", which="both", color="#dddddd", linewidth=.5)

    handles = [
        plt.Line2D([], [], color="black", linestyle=P_STYLES[p][0], marker=P_STYLES[p][1], label=fr"$P_0={p:g}$ µM")
        for p in P_LEVELS
    ]
    axes[0].legend(handles=handles, frameon=True, loc="upper left", bbox_to_anchor=(0.28, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(load_rows(args.result_bundle), args.output)


if __name__ == "__main__":
    main()
