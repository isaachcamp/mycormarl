"""Plot learned trade rates as approximate per-decision pool fractions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AGENTS = ("plant", "fungus")
COLOURS = {"plant": "#2878b5", "fungus": "#d95f02"}


def load_fraction_series(
    study_directory: Path,
    phosphorus_levels: tuple[float, ...],
    decision_interval_days: float,
    transitions_per_update: int,
) -> dict[float, dict[str, dict[int, dict[int, float]]]]:
    """Load checkpoint action rates and convert rates to per-step fractions."""
    result: dict[float, dict[str, dict[int, dict[int, float]]]] = {}
    for phosphorus in phosphorus_levels:
        prefix = f"mixed-p{phosphorus:g}-seed"
        records = {agent: defaultdict(dict) for agent in AGENTS}
        for condition in sorted(
            path for path in (study_directory / "conditions").glob(f"{prefix}*")
            if path.is_dir()
        ):
            seed = int(condition.name.removeprefix(prefix))
            for path in sorted((condition / "evaluations").glob("checkpoint-*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                actions = payload.get("summary", {}).get("actions")
                if not isinstance(actions, dict):
                    continue
                update = int(path.stem.removeprefix("checkpoint-")) / transitions_per_update
                for agent in AGENTS:
                    values = actions.get(agent)
                    if not isinstance(values, list) or not values:
                        continue
                    rate = float(values[0])
                    fraction = -np.expm1(-rate * decision_interval_days)
                    records[agent][seed][int(round(update))] = float(fraction)
        result[phosphorus] = records
    return result


def plot_fraction_series(
    records: dict[float, dict[str, dict[int, dict[int, float]]]], output: Path
) -> None:
    levels = sorted(records)
    figure, axes = plt.subplots(
        len(levels), 2, figsize=(12, 3.0 * len(levels)), sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, phosphorus in enumerate(levels):
        for column_index, agent in enumerate(AGENTS):
            axis = axes[row_index, column_index]
            values_by_update: dict[int, list[float]] = defaultdict(list)
            for seed, observations in sorted(records[phosphorus][agent].items()):
                updates = sorted(observations)
                values = [observations[update] for update in updates]
                axis.plot(
                    updates, values, color=COLOURS[agent], alpha=.5,
                    linewidth=1.1, label=f"Seed {seed}",
                )
                for update, value in observations.items():
                    values_by_update[update].append(value)
            mean_updates = sorted(
                update for update, values in values_by_update.items() if len(values) >= 2
            )
            axis.plot(
                mean_updates,
                [np.mean(values_by_update[update]) for update in mean_updates],
                color=COLOURS[agent], linewidth=2.4, marker="o", markersize=2.5,
                label="Seed mean",
            )
            axis.set_ylim(0, 1)
            axis.set_ylabel("Approx. pool fraction")
            axis.set_title(f"{agent.capitalize()} — initial P = {phosphorus:g} µM")
            axis.grid(axis="y", color="#dddddd", linewidth=.6)
            axis.legend(frameon=False, ncol=3, fontsize=7, loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("Training update (PPO updates)")
    figure.suptitle(
        "Learned trade as approximate pool fraction per policy decision\n"
        "fraction = 1 − exp(−trade rate × 0.25 d)",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--decision-interval-days", type=float, default=.25)
    parser.add_argument("--transitions-per-update", type=int, default=2048)
    parser.add_argument("--phosphorus", nargs="+", type=float, default=[.3, .75, 1.5, 3.0, 5.0])
    args = parser.parse_args()
    records = load_fraction_series(
        args.study_directory, tuple(args.phosphorus), args.decision_interval_days,
        args.transitions_per_update,
    )
    plot_fraction_series(records, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
