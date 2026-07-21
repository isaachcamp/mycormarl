"""Print the reproducible phosphate uptake examples as JSON."""

from __future__ import annotations

import argparse
import json

from mycormarl.phosphate_examples import MODES, run_example


def main() -> None:
    """Run the selected scenario, or all three scenarios by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    args = parser.parse_args()
    modes = MODES if args.mode == "all" else (args.mode,)
    print(json.dumps([run_example(mode) for mode in modes], indent=2))


if __name__ == "__main__":
    main()
