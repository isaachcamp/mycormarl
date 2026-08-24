"""Run and persist the static plant growth-scale qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from mycormarl.plant_growth_qualification import run_plant_growth_qualification


def _summary(result: dict) -> str:
    cases = result["cases"]
    lines = [
        "# Plant growth-scale qualification",
        "",
        f"- Status: `{result['status']}`",
        f"- Selected reference kleaf: `{result['selected_kleaf']:.2f}`",
        "- Mode: `plant-only` high-P vegetative control",
        "- Rate policy (d⁻¹): trade=0, growth=1, reproduction=0, storage=0",
        "",
        "## Candidate comparison",
        "",
        "| kleaf | day-120 biomass (g DM) | mean 40–120 d RGR (d⁻¹) | limitation | result |",
        "|---:|---:|---:|---|---|",
    ]
    for name, case in cases.items():
        checkpoints = case["checkpoints"]
        rgr = [checkpoints[str(day)]["windowed_rgr_per_day"] for day in (60, 80, 100, 120)]
        mean_rgr = sum(rgr) / len(rgr)
        lines.append(
            f"| {case['kleaf']:.3g} | {case['biomass_g_dm']['120']:.6g} | "
            f"{mean_rgr:.6g} | {case['realized_limitation']} | {case['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Forto reference: 23.26 g DM at day 120 and approximately "
            "0.066, 0.065, 0.060, and 0.042 d⁻¹ windowed RGR.",
            "The candidate with the smallest endpoint and RGR discrepancy is "
            "not automatically accepted if it remains materially outside the "
            "reference envelope.",
            "",
        ]
    )
    return "\n".join(lines)


def _figure(result: dict, path: Path) -> None:
    windows = ("40–60", "60–80", "80–100", "100–120")
    observed = result["reference"]["rgr_windows_per_day"]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(windows, observed, marker="o", linewidth=2.5, color="black", label="Forto observed")
    colors = {0.30: "#4472c4", 0.45: "#70ad47", 0.50: "#ed7d31", 0.60: "#c00000"}
    for case in result["cases"].values():
        values = [
            case["checkpoints"][str(day)]["windowed_rgr_per_day"]
            for day in (60, 80, 100, 120)
        ]
        label = f"static kleaf={case['kleaf']:.3g}"
        if case["cap_contact"]:
            label += " (cap contact)"
        axis.plot(
            windows,
            values,
            marker=".",
            linewidth=1.8,
            color=colors.get(case["kleaf"], None),
            label=label,
        )
    axis.set_ylabel("Windowed relative growth rate (d⁻¹)")
    axis.set_xlabel("20-day growth window (DAS)")
    axis.set_title("Forto RGR pattern versus static leaf allocation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--kleaf",
        type=float,
        nargs="+",
        default=(0.30, 0.45, 0.50, 0.60, 0.65, 0.675, 0.68, 0.70),
        help="candidate fixed kleaf values; defaults to the reference sensitivities",
    )
    args = parser.parse_args()
    result = run_plant_growth_qualification(kleaf_values=args.kleaf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plant-growth-qualification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "plant-growth-qualification.md").write_text(
        _summary(result), encoding="utf-8"
    )
    _figure(result, args.output_dir / "plant-growth-rgr-comparison.png")
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
