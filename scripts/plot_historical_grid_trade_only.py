"""Reproduce the previous trade-only figure layouts for a historical-grid bundle."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_trade_only import (  # noqa: E402
    plot_ippo,
    plot_overlay,
    plot_pareto_fixed_reference,
    plot_pareto_overlay,
    plot_realised_transfers,
    plot_trade_rates,
)

P_LEVELS = (0.3, 0.75, 1.5, 3.0, 5.0, 10.0)
AGENTS = ("plant", "fungus")
FRACTION_INTERVAL_DAYS = 0.25


def rate_to_daily_fraction(rate: float) -> float:
    """Convert a Rate-action hazard to fraction traded per policy interval."""
    return float(-np.expm1(-float(rate) * FRACTION_INTERVAL_DAYS))


def load_data(study_dir: Path) -> tuple[dict[float, list[dict]], dict[str, dict[float, dict]]]:
    bundle = json.loads((study_dir / "result-bundle.json").read_text(encoding="utf-8"))
    records: dict[float, list[dict]] = defaultdict(list)
    for entry in bundle["entries"]:
        if entry["mode"] != "mixed" or not entry.get("evaluation_artifacts"):
            continue
        path = study_dir / entry["evaluation_artifacts"][-1]
        episode = json.loads(path.read_text(encoding="utf-8"))["episodes"][0]
        records[float(entry["initial_p_micromolar"])].append({
            "seed": int(entry["seed"]), "summary": episode["summary"], "trace": episode["trace"],
        })
    if tuple(sorted(records)) != P_LEVELS or any(sorted(r["seed"] for r in rows) != [0, 1, 2, 3, 4] for rows in records.values()):
        raise ValueError("historical-grid bundle must contain five mixed seeds at each P level")

    bilateral_payload = json.loads((study_dir / "fixed-bilateral-trade-control.json").read_text(encoding="utf-8"))
    bilateral = {float(e["initial_p_micromolar"]): e for e in bilateral_payload["entries"]}
    plant_only = {float(e["initial_p_micromolar"]): e for e in bundle["entries"] if e["mode"] == "plant-only"}
    if set(bilateral) != set(P_LEVELS) or set(plant_only) != set(P_LEVELS):
        raise ValueError("both six-level control sets are required")
    return records, {"bilateral": bilateral, "plant-only": plant_only}


def fitness_records(records: dict[float, list[dict]], controls: dict[str, dict[float, dict]]) -> tuple[dict, dict]:
    learned = {}
    for p, rows in records.items():
        learned[p] = {
            agent: np.asarray([r["summary"]["cumulative_reproductive_fitness"][agent] for r in rows])
            for agent in AGENTS
        }
        learned[p]["combined"] = learned[p]["plant"] + learned[p]["fungus"]
        learned[p]["seeds"] = [r["seed"] for r in rows]
    baselines = {}
    for mode, source in (("plant-only", controls["plant-only"]), ("mixed", controls["bilateral"])):
        baselines[mode] = {
            p: {
                "plant": float(e["cumulative_reproductive_fitness"]["plant"]),
                "fungus": float(e["cumulative_reproductive_fitness"].get("fungus", 0.0)),
                "combined": float(e["cumulative_reproductive_fitness"]["plant"])
                + float(e["cumulative_reproductive_fitness"].get("fungus", 0.0)),
            }
            for p, e in source.items()
        }
    return learned, baselines


def endpoint_records(records: dict[float, list[dict]], controls: dict[str, dict[float, dict]]) -> dict:
    result = {}
    for metric, field in (("biomass", "final_living_biomass"), ("fitness", "cumulative_reproductive_fitness")):
        result[metric] = {
            "p": list(P_LEVELS),
            "plant_only": [float(controls["plant-only"][p][field]["plant"]) for p in P_LEVELS],
            "fixed_mixed": [float(controls["bilateral"][p][field]["plant"]) for p in P_LEVELS],
            "learned": [[float(r["summary"][field]["plant"]) for r in records[p]] for p in P_LEVELS],
        }
    return result


def trade_records(records: dict[float, list[dict]]) -> dict:
    result = {}
    for p, rows in records.items():
        time = np.asarray([s["time_days"] for s in rows[0]["trace"]])
        result[p] = {
            "seeds": np.asarray([r["seed"] for r in rows]), "time_days": time,
            "plant": np.asarray([[rate_to_daily_fraction(s["actions"]["plant"][0]) for s in r["trace"]] for r in rows]),
            "fungus": np.asarray([[rate_to_daily_fraction(s["actions"]["fungus"][0]) for s in r["trace"]] for r in rows]),
            "plant_c_export": np.asarray([[s["transfers"]["plant"]["out"] for s in r["trace"]] for r in rows]),
            "fungus_p_export": np.asarray([[s["transfers"]["fungus"]["out"] for s in r["trace"]] for r in rows]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study_dir", type=Path)
    args = parser.parse_args()
    output = args.study_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    records, controls = load_data(args.study_dir)
    fitness, baselines = fitness_records(records, controls)
    plot_ippo(endpoint_records(records, controls), output / "trade-only-ippo-plant-performance.png")
    trade = trade_records(records)
    plot_trade_rates(trade, output / "trade-only-ippo-trade-rates.png")
    plot_realised_transfers(trade, output / "trade-only-ippo-realised-transfers.png")
    plot_pareto_overlay(fitness, baselines, output / "trade-only-ippo-fitness-pareto-overlay.png")
    plot_pareto_fixed_reference(fitness, baselines, output / "trade-only-ippo-fitness-pareto-fixed-trade-reference.png")

    rate_baselines = {mode: {} for mode in baselines}
    for mode, source in (("plant-only", controls["plant-only"]), ("mixed", controls["bilateral"])):
        for p, entry in source.items():
            rate_baselines[mode][p] = {
                "fitness_plant": baselines[mode][p]["plant"],
                "fitness_fungus": baselines[mode][p]["fungus"],
                "plant_to_fungus": rate_to_daily_fraction(entry["commanded_rate_actions"]["plant"][0]),
                "fungus_to_plant": rate_to_daily_fraction(entry["commanded_rate_actions"]["fungus"][0]),
            }
    learned_rates = {
        p: {
            "plant_to_fungus": np.asarray([np.mean([rate_to_daily_fraction(s["actions"]["plant"][0]) for s in r["trace"]]) for r in rows]),
            "fungus_to_plant": np.asarray([np.mean([rate_to_daily_fraction(s["actions"]["fungus"][0]) for s in r["trace"]]) for r in rows]),
        }
        for p, rows in records.items()
    }
    for direction, name in (("fungus_to_plant", "fungal-to-plant"), ("plant_to_fungus", "plant-to-fungus")):
        plot_overlay(fitness, rate_baselines, learned_rates, direction,
                     output / f"trade-only-ippo-{name}-rate-vs-fitness-overlay.png")
    print(f"wrote previous-layout figures to {output}")


if __name__ == "__main__":
    main()
