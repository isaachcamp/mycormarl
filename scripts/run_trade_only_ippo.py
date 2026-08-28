"""Launch the reduced-control mixed IPPO study across initial P conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

from mycormarl.trade_only_study import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p",
        dest="initial_p_micromolar",
        type=float,
        nargs="+",
        metavar="MICROMOLAR",
        help="one or more initial solution-P concentrations in micromolar",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    levels = (
        (0.75, 3.0, 10.0)
        if args.initial_p_micromolar is None
        else tuple(args.initial_p_micromolar)
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("outputs/trade-only-ippo")
        if args.initial_p_micromolar is not None:
            cohort = "-".join(f"{level:g}" for level in levels)
            output_dir = Path("outputs") / f"trade-only-ippo-p-{cohort}"
    run_study(output_dir, initial_p_micromolar=levels, workers=args.workers)


if __name__ == "__main__":
    main()
