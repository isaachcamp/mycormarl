"""Run issue #47's canonical resource-pressure screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mycormarl.resource_pressure_screen import write_resource_pressure_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(write_resource_pressure_experiment(manifest, args.output))


if __name__ == "__main__":
    main()
