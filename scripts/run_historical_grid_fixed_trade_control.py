"""Run the bilateral fixed-trade controls for the historical-grid study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mycormarl.static_controls import run_batched_static_controls
from mycormarl.trade_only import fixed_allocation_rate_action, pool_fraction_to_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["modes"] = ["mixed"]
    manifest["seeds"] = [0]
    manifest["static_policy"] = {
        "plant": fixed_allocation_rate_action(pool_fraction_to_rate(0.05)).tolist(),
        "fungus": fixed_allocation_rate_action(pool_fraction_to_rate(0.75)).tolist(),
    }
    result = run_batched_static_controls(manifest)
    payload = {
        "format": "mycormarl-historical-grid-fixed-trade-control",
        "format_version": 1,
        "protocol": {
            "plant_to_fungus_fraction_per_day": 0.05,
            "fungus_to_plant_fraction_per_day": 0.75,
            "fixed_allocation": manifest["policy"]["fixed_allocation"],
        },
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({result['completion']})")


if __name__ == "__main__":
    main()
