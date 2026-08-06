"""Requalify deep-soil phosphorus against an existing diagnostic artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.deep_soil_qualification import (
    StaticPolicy,
    compare_deep_soil_qualification,
    compare_temporal_convergence,
    run_deep_soil_qualification,
    write_deep_soil_qualification_outputs,
)


def _software_revision() -> str:
    """Return HEAD plus a reproducible fingerprint for dirty model sources."""
    source_paths = ("mycormarl", "scripts", "tests", "pyproject.toml", "uv.lock")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0:
        return "unknown"
    head = revision.stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *source_paths],
        check=False,
        capture_output=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *source_paths],
        check=False,
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0 or untracked.returncode != 0:
        return f"{head}-dirty-unknown"
    untracked_paths = sorted(path for path in untracked.stdout.splitlines() if path)
    if not diff.stdout and not untracked_paths:
        return head
    fingerprint = hashlib.sha256(diff.stdout)
    for relative_path in untracked_paths:
        fingerprint.update(relative_path.encode())
        fingerprint.update(b"\0")
        fingerprint.update(Path(relative_path).read_bytes())
        fingerprint.update(b"\0")
    return f"{head}-dirty-{fingerprint.hexdigest()[:12]}"


def _static_policy(record: dict) -> StaticPolicy:
    """Reconstruct the requested static policy from manifest provenance."""
    return StaticPolicy(
        trade=record["trade"],
        growth=record["growth"],
        reproduction=record["reproduction"],
        reserve=record["reserve"],
    )


def _render_markdown(comparison: dict, corrected_manifest: dict) -> str:
    """Render the artifact-derived spatial comparison as a concise report."""
    original = comparison["original"]
    corrected = comparison["corrected"]
    qualification = corrected_manifest["qualification"]
    lines = [
        "# Deep-soil phosphorus qualification",
        "",
        "## Configuration and provenance",
        "",
        "The environment, policies, duration, and seed were reconstructed from the original manifest. Current `main` trait defaults were used intentionally.",
        "",
        f"- Configuration equivalence: `{json.dumps(comparison['configuration_equivalence'], sort_keys=True)}`.",
        f"- Original software revision: `{comparison['intentional_provenance_differences'].get('software_revision', {}).get('original', corrected_manifest['software_revision'])}`.",
        f"- Corrected software revision: `{corrected_manifest['software_revision']}`.",
        "",
        "## Fungal confinement",
        "",
        f"- Maximum fungal density wholly outside the biomass-derived colony: `{qualification['max_fungal_density_outside_colony']:.9g}` cm cm^-3.",
        f"- Maximum fungal uptake request wholly outside the colony: `{qualification['max_fungal_uptake_request_outside_colony_micromol']:.9g}` micromol P per cell.",
        "",
        "## Initial-to-final labile-P inventory loss",
        "",
        "| Depth (cm) | Original loss | Corrected loss |",
        "|---:|---:|---:|",
    ]
    for old, new in zip(
        original["depth_band_losses"],
        corrected["depth_band_losses"],
        strict=True,
    ):
        lines.append(
            f"| {old['start_depth_cm']:g}-{old['end_depth_cm']:g} | "
            f"{old['loss_percent']:.6f}% | {new['loss_percent']:.6f}% |"
        )
    lines.extend(
        [
            "",
            "## Original-artifact discrepancy",
            "",
            f"The available original NPZ yields `{original['depth_band_losses'][-1]['loss_percent']:.6f}%` loss in the 75-100 cm band. Issue #18 cites `4.683888%`; match: `{comparison['available_original_matches_ticket_reference']}`. The cited 1% value is treated only as a diagnostic reference, not a domain constant.",
            "",
            f"The original outermost/bottommost cell loss is `{original['outer_bottom_cell']['loss_percent']:.6f}%`; the corrected value is `{corrected['outer_bottom_cell']['loss_percent']:.6f}%`.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--temporal-reference-dt", type=float)
    parser.add_argument("--software-revision")
    return parser.parse_args()


def main() -> None:
    """Run one corrected scenario without overwriting the original artifact."""
    args = _parse_args()
    original_manifest_path = args.original_dir / "manifest.json"
    original_npz_path = args.original_dir / "daily-soil-p.npz"
    original_manifest = json.loads(original_manifest_path.read_text())
    environment = dict(original_manifest["environment"])
    if args.dt is not None:
        environment["dt"] = args.dt
    config = EnvConfig(**environment)
    policies = original_manifest["policies"]
    result = run_deep_soil_qualification(
        config=config,
        species=SpeciesParams(plant=PlantTraits(), fungus=FungusTraits()),
        plant_policy=_static_policy(policies["plant"]),
        fungus_policy=_static_policy(policies["fungus"]),
        duration_days=original_manifest["duration_days"],
        seed=original_manifest["seed"],
        software_revision=args.software_revision or _software_revision(),
    )
    temporal_comparison = None
    if args.temporal_reference_dt is not None:
        if args.temporal_reference_dt >= config.dt:
            raise ValueError(
                "temporal reference dt must be smaller than the candidate dt"
            )
        reference_environment = {
            **environment,
            "dt": args.temporal_reference_dt,
        }
        reference = run_deep_soil_qualification(
            config=EnvConfig(**reference_environment),
            species=SpeciesParams(plant=PlantTraits(), fungus=FungusTraits()),
            plant_policy=_static_policy(policies["plant"]),
            fungus_policy=_static_policy(policies["fungus"]),
            duration_days=original_manifest["duration_days"],
            seed=original_manifest["seed"],
            software_revision=args.software_revision or _software_revision(),
        )
        temporal_comparison = compare_temporal_convergence(result, reference)
    comparison = compare_deep_soil_qualification(
        result,
        original_manifest_path=original_manifest_path,
        original_npz_path=original_npz_path,
    )
    write_deep_soil_qualification_outputs(result, args.output_dir)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "comparison.md").write_text(
        _render_markdown(comparison, result.manifest) + "\n"
    )
    if temporal_comparison is not None:
        (args.output_dir / "temporal-convergence.json").write_text(
            json.dumps(temporal_comparison, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "scenario_id": result.manifest["scenario_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
