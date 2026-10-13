"""Run and write the time-dependent finite-inventory hyphal sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mycormarl.soil.hyphal_density_sweep import run_hyphal_density_sweep


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-density", type=int, default=25)
    parser.add_argument("--max-density", type=int, default=3000)
    parser.add_argument("--density-step", type=int, default=25)
    parser.add_argument("--p-concentrations", nargs="+", type=float,
                        default=[0.1, 0.3, 0.6, 1.0, 1.5, 3.0])
    args = parser.parse_args()
    time_series = []
    summaries = []
    densities = range(args.min_density, args.max_density + 1, args.density_step)
    for concentration in args.p_concentrations:
        concentration_time_series, concentration_summaries = run_hyphal_density_sweep(
            densities_cm_cm3=densities,
            initial_solution_p_um=concentration,
        )
        time_series.extend(concentration_time_series)
        summaries.extend(concentration_summaries)
    _write_csv(args.output_dir / "hyphal_density_sweep_time_series.csv", time_series)
    _write_csv(args.output_dir / "hyphal_density_sweep_summary.csv", summaries)


if __name__ == "__main__":
    main()
