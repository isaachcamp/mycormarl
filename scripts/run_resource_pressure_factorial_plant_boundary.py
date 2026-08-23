"""Run issue #48's factorial plant P/C-limitation boundary experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mycormarl.resource_pressure_screen import write_resource_pressure_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest", type=Path,
        help="Discrete factorial manifest (normally the #48 qualification manifest).",
    )
    parser.add_argument("output", type=Path, help="Combined-bundle path; disabled by the manifest.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("sampling") != "discrete_factorial":
        raise ValueError("issue #48 runner requires a discrete_factorial manifest")
    print(write_resource_pressure_experiment(manifest, args.output))


if __name__ == "__main__":
    main()
