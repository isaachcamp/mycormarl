"""Plot realised checkpoint returns across seeds for one historical-grid P level."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AGENTS = ("plant", "fungus")
COLOURS = {"plant": "#2878b5", "fungus": "#d95f02"}


def load_returns(study_directory: Path, phosphorus: float) -> dict[str, dict[int, dict[int, float]]]:
    """Load realised evaluation returns keyed by agent, seed, and training update."""
    records: dict[str, dict[int, dict[int, float]]] = {
        agent: defaultdict(dict) for agent in AGENTS
    }
    prefix = f"mixed-p{phosphorus:g}-seed"
    conditions = sorted(path for path in (study_directory / "conditions").glob(f"{prefix}*") if path.is_dir())
    if not conditions:
        raise FileNotFoundError(f"no mixed conditions found for P={phosphorus:g} µM")
    for condition in conditions:
        seed = int(condition.name.removeprefix(prefix))
        for path in sorted((condition / "evaluations").glob("checkpoint-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            fitness = payload.get("summary", {}).get("fitness")
            if not isinstance(fitness, dict) or not all(agent in fitness for agent in AGENTS):
                continue
            update = int(path.stem.removeprefix("checkpoint-"))
            for agent in AGENTS:
                records[agent][seed][update] = float(fitness[agent])
    return records


def plot_returns(
    records: dict[str, dict[int, dict[int, float]]], phosphorus: float, output: Path,
    transitions_per_update: int,
) -> None:
    """Plot each seed (alpha=.5) and the mean when at least two seeds were evaluated."""
    figure, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True, constrained_layout=True)
    for axis, agent in zip(axes, AGENTS):
        seed_records = records[agent]
        values_by_update: dict[int, list[float]] = defaultdict(list)
        for seed, observations in sorted(seed_records.items()):
            updates = sorted(observations)
            values = [observations[update] for update in updates]
            axis.plot(
                [update / transitions_per_update for update in updates], values,
                color=COLOURS[agent], alpha=.5, linewidth=1.15, label=f"Seed {seed}",
            )
            for update, value in observations.items():
                values_by_update[update].append(value)

        # Once a seed has stopped, it has no realised return at later updates.
        # Do not present a lone remaining seed as a cross-seed mean.
        mean_updates = sorted(
            update for update, values in values_by_update.items() if len(values) >= 2
        )
        means = [np.mean(values_by_update[update]) for update in mean_updates]
        axis.plot(
            [update / transitions_per_update for update in mean_updates], means,
            color=COLOURS[agent], linewidth=2.5, marker="o", markersize=2.8, label="Seed mean",
        )
        axis.set_ylabel("Realised episode return")
        axis.set_title(agent.capitalize())
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
        axis.legend(frameon=False, ncol=3, fontsize=8, loc="best")

    axes[-1].set_xlabel("Training update (PPO updates)")
    figure.suptitle(f"Realised evaluation returns across seeds (initial P = {phosphorus:g} µM)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phosphorus", type=float, default=.3)
    parser.add_argument("--transitions-per-update", type=int, default=2048)
    args = parser.parse_args()
    records = load_returns(args.study_directory, args.phosphorus)
    plot_returns(records, args.phosphorus, args.output, args.transitions_per_update)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
