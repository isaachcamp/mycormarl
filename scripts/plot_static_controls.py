"""Plot the paired AM biomass ratio in a static-controls result bundle."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def summarise(entries: list[dict]) -> list[dict[str, float]]:
    """Return paired $\\Delta_{AM}$ ratios and per-mode median plant biomass by P."""
    by_treatment: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for entry in entries:
        if entry["status"] != "completed":
            raise ValueError("static-controls bundle contains an incomplete entry")
        by_treatment[(float(entry["initial_p_micromolar"]), entry["mode"])].append(entry)

    phosphorus = sorted({p for p, _ in by_treatment})
    records = []
    for p in phosphorus:
        paired = {
            mode: {entry["seed"]: float(entry["biomass"]["plant"])
                   for entry in by_treatment[(p, mode)]}
            for mode in ("mixed", "plant-only")
        }
        if set(paired["mixed"]) != set(paired["plant-only"]):
            raise ValueError(f"mixed and plant-only seeds do not match at {p:g} µM P")
        if any(value <= 0 for value in paired["plant-only"].values()):
            raise ValueError(f"plant-only biomass must be positive at {p:g} µM P")
        delta_am = np.median([
            paired["mixed"][seed] / paired["plant-only"][seed]
            for seed in paired["mixed"]
        ])
        records.append({
            "p": p,
            "delta_am": float(delta_am),
            "mixed": float(np.median(list(paired["mixed"].values()))),
            "plant_only": float(np.median(list(paired["plant-only"].values()))),
        })
    return records


def plot(records: list[dict[str, float]], output: Path) -> None:
    p = [record["p"] for record in records]
    delta_am = [record["delta_am"] for record in records]
    mixed = [record["mixed"] for record in records]
    plant_only = [record["plant_only"] for record in records]

    fig, (delta_axis, biomass_axis) = plt.subplots(
        2, 1, figsize=(7.2, 7.2), sharex=True, constrained_layout=True,
    )
    delta_axis.axhline(1, color="#555555", linewidth=0.8, zorder=0)
    delta_axis.scatter(p, delta_am, color="#2878b5", s=54, zorder=2)
    delta_axis.plot(p, delta_am, color="#2878b5", linewidth=1.1, zorder=1)
    delta_axis.set_ylabel(r"Paired median $\Delta_{AM}$ (mixed / plant-only)")
    delta_axis.set_title("Static controls: mycorrhizal plant biomass ratio")
    delta_axis.grid(axis="y", color="#dddddd", linewidth=0.6)

    biomass_axis.scatter(p, mixed, color="#2878b5", marker="o", s=54, label="Mixed")
    biomass_axis.plot(p, mixed, color="#2878b5", linewidth=1.1)
    biomass_axis.scatter(p, plant_only, color="#d95f02", marker="s", s=46, label="Plant-only")
    biomass_axis.plot(p, plant_only, color="#d95f02", linewidth=1.1)
    biomass_axis.set_xscale("log")
    biomass_axis.set_xticks(p, [f"{value:g}" for value in p])
    biomass_axis.set_xlabel("Initial solution P (µM)")
    biomass_axis.set_ylabel("Median final plant biomass (g)")
    biomass_axis.set_title("Median final plant biomass")
    biomass_axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    biomass_axis.legend(frameon=False)

    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.result_bundle.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = summarise(bundle["entries"])
    plot(records, args.output)
    print(f"wrote {args.output} for {len(records)} phosphorus concentrations")


if __name__ == "__main__":
    main()
