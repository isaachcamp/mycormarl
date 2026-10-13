"""Re-evaluate completed historical-grid checkpoints with current trace fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mycormarl.checkpoint_evaluation import evaluate_checkpoint, save_evaluation_artifact
from mycormarl.study import _training_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    study_dir = args.study_dir
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bundle_path = study_dir / "result-bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    count = 0
    for entry in bundle["entries"]:
        if entry["mode"] != "mixed" or not entry.get("evaluation_artifacts"):
            continue
        checkpoint = study_dir / entry["checkpoint"]
        evaluation_path = study_dir / entry["evaluation_artifacts"][-1]
        environment = _training_environment(
            manifest, entry["mode"], float(entry["initial_p_micromolar"])
        )
        evaluation = evaluate_checkpoint(
            checkpoint,
            environment,
            episodes=manifest["evaluation"]["episodes"],
            protocol=manifest["evaluation"]["protocol"],
            seed=int(entry["seed"]),
        )
        save_evaluation_artifact(evaluation_path, evaluation, checkpoint=checkpoint)
        count += 1
    print(f"refreshed {count} final mixed evaluations")


if __name__ == "__main__":
    main()
