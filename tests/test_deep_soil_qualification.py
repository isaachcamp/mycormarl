"""Public contracts for the deep-soil phosphorus qualification."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import pytest

from mycormarl.fungus.mycelium import fungal_biomass_for_colony_radius
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


def test_qualification_confines_fungal_density_and_request_to_current_colony():
    """Every sampled production transition keeps fungal uptake in its colony."""
    fungus = FungusTraits(
        initial_biomass=0.0,
        initial_c_pool=0.0,
        initial_p_pool=0.0,
        kappa_c=0.0,
        kappa_p=0.0,
        death_fraction=0.0,
        hyphal_radius=0.1,
        hyphal_tissue_carbon_density=1.0,
        saturation_density=1.0,
        jmax=1.0,
        km=1e-3,
    )
    fungus = fungus.replace(
        initial_biomass=fungal_biomass_for_colony_radius(0.5, fungus)
    )
    result = run_deep_soil_qualification(
        config=EnvConfig(
            consumer_mode="fungus-only",
            dt=1.0,
            soil_radius_cm=2.0,
            soil_depth_cm=2.0,
            radial_interval_cm=1.0,
            depth_interval_cm=1.0,
            topsoil_depth_cm=2.0,
            initial_solution_p_um=1.0,
            phosphate_diffusion_coefficient_cm2_s=1e-30,
            b_p=0.0,
        ),
        species=SpeciesParams(plant=PlantTraits(), fungus=fungus),
        plant_policy=StaticPolicy(0.0, 0.0, 0.0, 1.0),
        fungus_policy=StaticPolicy(0.0, 0.0, 0.0, 1.0),
        duration_days=1,
        seed=0,
        software_revision="test-revision",
    )

    assert result.executed_steps == 1
    assert result.max_fungal_density_outside_colony == pytest.approx(0.0)
    assert (
        result.max_fungal_uptake_request_outside_colony_micromol
        == pytest.approx(0.0)
    )


def test_qualification_reports_inventory_loss_for_required_depth_bands():
    """The public result reports independently known zero-loss depth bands."""
    species = SpeciesParams(
        plant=PlantTraits(
            initial_biomass=0.0,
            initial_c_pool=0.0,
            initial_p_pool=0.0,
            jmax=0.0,
        ),
        fungus=FungusTraits(
            initial_biomass=0.0,
            initial_c_pool=0.0,
            initial_p_pool=0.0,
            jmax=0.0,
        ),
    )
    result = run_deep_soil_qualification(
        config=EnvConfig(
            dt=1.0,
            soil_radius_cm=1.0,
            soil_depth_cm=100.0,
            radial_interval_cm=1.0,
            depth_interval_cm=5.0,
            topsoil_depth_cm=100.0,
            initial_solution_p_um=1.0,
            phosphate_diffusion_coefficient_cm2_s=1e-30,
        ),
        species=species,
        plant_policy=StaticPolicy(0.0, 0.0, 0.0, 1.0),
        fungus_policy=StaticPolicy(0.0, 0.0, 0.0, 1.0),
        duration_days=1,
        seed=0,
        software_revision="test-revision",
    )

    assert [
        (band.start_depth_cm, band.end_depth_cm)
        for band in result.depth_band_losses
    ] == [
        (0.0, 10.0),
        (10.0, 25.0),
        (25.0, 50.0),
        (50.0, 75.0),
        (75.0, 100.0),
    ]
    assert all(band.loss_percent == pytest.approx(0.0) for band in result.depth_band_losses)


def test_writer_emits_separate_soil_snapshots_and_complete_provenance(tmp_path):
    """Fresh artifacts retain the complete inputs needed to reproduce a run."""
    species = SpeciesParams(
        plant=PlantTraits(initial_biomass=0.0, jmax=0.0),
        fungus=FungusTraits(initial_biomass=0.0, jmax=0.0),
    )
    result = run_deep_soil_qualification(
        config=EnvConfig(
            dt=1.0,
            soil_radius_cm=1.0,
            soil_depth_cm=100.0,
            radial_interval_cm=1.0,
            depth_interval_cm=5.0,
            topsoil_depth_cm=100.0,
            initial_solution_p_um=1.0,
            phosphate_diffusion_coefficient_cm2_s=1e-30,
        ),
        species=species,
        plant_policy=StaticPolicy(0.0, 0.0, 0.0, 1.0),
        fungus_policy=StaticPolicy(0.0, 0.0, 0.0, 1.0),
        duration_days=1,
        seed=7,
        software_revision="test-revision",
    )

    paths = write_deep_soil_qualification_outputs(result, tmp_path)

    assert {path.name for path in paths.values()} == {
        "daily-soil-p.npz",
        "manifest.json",
    }
    manifest = json.loads(paths["manifest_json"].read_text())
    assert manifest["software_revision"] == "test-revision"
    assert manifest["seed"] == 7
    assert manifest["duration_days"] == 1
    assert manifest["environment"]["dt"] == pytest.approx(1.0)
    assert manifest["traits"]["fungus"]["saturation_density"] == pytest.approx(2000.0)
    assert manifest["policies"]["fungus"]["physical_action"] == pytest.approx(
        [0.0, 0.0, 0.0, 1.0]
    )
    with np.load(paths["daily_soil_p_npz"]) as snapshots:
        assert snapshots["scenario_id"].item() == manifest["scenario_id"]
        assert snapshots["days"].tolist() == [0, 1]
        assert snapshots["soil_labile_p_micromol"].shape == (2, 1, 20)


def test_writer_refuses_to_overwrite_incompatible_artifacts(tmp_path):
    """A different scenario cannot replace an existing qualified artifact."""
    config = EnvConfig(
        dt=1.0,
        soil_radius_cm=1.0,
        soil_depth_cm=100.0,
        radial_interval_cm=1.0,
        depth_interval_cm=5.0,
        topsoil_depth_cm=100.0,
        initial_solution_p_um=1.0,
        phosphate_diffusion_coefficient_cm2_s=1e-30,
    )
    species = SpeciesParams(
        plant=PlantTraits(initial_biomass=0.0, jmax=0.0),
        fungus=FungusTraits(initial_biomass=0.0, jmax=0.0),
    )
    policy = StaticPolicy(0.0, 0.0, 0.0, 1.0)
    original = run_deep_soil_qualification(
        config=config,
        species=species,
        plant_policy=policy,
        fungus_policy=policy,
        duration_days=1,
        seed=1,
        software_revision="revision-a",
    )
    incompatible = run_deep_soil_qualification(
        config=config,
        species=species,
        plant_policy=policy,
        fungus_policy=policy,
        duration_days=1,
        seed=2,
        software_revision="revision-b",
    )
    paths = write_deep_soil_qualification_outputs(original, tmp_path)
    original_manifest = paths["manifest_json"].read_bytes()
    original_npz = paths["daily_soil_p_npz"].read_bytes()

    with pytest.raises(ValueError, match="incompatible"):
        write_deep_soil_qualification_outputs(incompatible, tmp_path)

    assert paths["manifest_json"].read_bytes() == original_manifest
    assert paths["daily_soil_p_npz"].read_bytes() == original_npz


def test_comparison_checks_equivalence_and_recomputes_original_loss(tmp_path):
    """Comparison evidence comes from artifacts and identifies provenance drift."""
    config = EnvConfig(
        dt=1.0,
        soil_radius_cm=1.0,
        soil_depth_cm=100.0,
        radial_interval_cm=1.0,
        depth_interval_cm=5.0,
        topsoil_depth_cm=100.0,
        initial_solution_p_um=1.0,
        phosphate_diffusion_coefficient_cm2_s=1e-30,
    )
    species = SpeciesParams(
        plant=PlantTraits(initial_biomass=0.0, jmax=0.0),
        fungus=FungusTraits(initial_biomass=0.0, jmax=0.0),
    )
    policy = StaticPolicy(0.0, 0.0, 0.0, 1.0)
    original = run_deep_soil_qualification(
        config=config,
        species=species,
        plant_policy=policy,
        fungus_policy=policy,
        duration_days=1,
        seed=0,
        software_revision="original-revision",
    )
    original_paths = write_deep_soil_qualification_outputs(
        original,
        tmp_path / "original",
    )
    corrected = run_deep_soil_qualification(
        config=config,
        species=species,
        plant_policy=policy,
        fungus_policy=policy,
        duration_days=1,
        seed=0,
        software_revision="corrected-revision",
    )

    comparison = compare_deep_soil_qualification(
        corrected,
        original_manifest_path=original_paths["manifest_json"],
        original_npz_path=original_paths["daily_soil_p_npz"],
    )

    assert comparison["configuration_equivalence"] == {
        "duration_days": True,
        "environment": True,
        "policies": True,
        "seed": True,
    }
    assert comparison["intentional_provenance_differences"] == {
        "software_revision": {
            "original": "original-revision",
            "corrected": "corrected-revision",
        }
    }
    assert comparison["original"]["depth_band_losses"][-1][
        "loss_percent"
    ] == pytest.approx(0.0)
    assert comparison["original"]["outer_bottom_cell"][
        "loss_percent"
    ] == pytest.approx(0.0)
    assert comparison["ticket_reference_bottom_25_cm_loss_percent"] == pytest.approx(
        4.683888
    )
    assert comparison["available_original_matches_ticket_reference"] is False


def test_temporal_gate_accepts_matching_endpoint_metrics():
    """A finer run passes when all qualified endpoint metrics agree within 5%."""
    species = SpeciesParams(
        plant=PlantTraits(initial_biomass=0.0, jmax=0.0),
        fungus=FungusTraits(initial_biomass=0.0, jmax=0.0),
    )
    policy = StaticPolicy(0.0, 0.0, 0.0, 1.0)

    def run(dt):
        return run_deep_soil_qualification(
            config=EnvConfig(
                dt=dt,
                soil_radius_cm=1.0,
                soil_depth_cm=100.0,
                radial_interval_cm=1.0,
                depth_interval_cm=5.0,
                topsoil_depth_cm=100.0,
                initial_solution_p_um=1.0,
                phosphate_diffusion_coefficient_cm2_s=1e-30,
            ),
            species=species,
            plant_policy=policy,
            fungus_policy=policy,
            duration_days=1,
            seed=0,
            software_revision="test-revision",
        )

    comparison = compare_temporal_convergence(run(1.0), run(0.5))

    assert comparison["candidate_dt_days"] == pytest.approx(1.0)
    assert comparison["reference_dt_days"] == pytest.approx(0.5)
    assert comparison["passes_5_percent"] is True
    assert comparison["maximum_relative_change"] == pytest.approx(0.0)


def test_cli_reruns_diagnosed_configuration_with_current_trait_defaults(tmp_path):
    """One command writes a separate corrected artifact and comparison report."""
    config = EnvConfig(
        dt=1.0,
        soil_radius_cm=1.0,
        soil_depth_cm=100.0,
        radial_interval_cm=1.0,
        depth_interval_cm=5.0,
        topsoil_depth_cm=100.0,
        initial_solution_p_um=1.0,
        phosphate_diffusion_coefficient_cm2_s=1e-30,
    )
    original = run_deep_soil_qualification(
        config=config,
        species=SpeciesParams(
            plant=PlantTraits(initial_biomass=0.0, jmax=0.0),
            fungus=FungusTraits(initial_biomass=0.0, jmax=0.0),
        ),
        plant_policy=StaticPolicy(0.0, 0.0, 0.0, 1.0),
        fungus_policy=StaticPolicy(0.0, 0.0, 0.0, 1.0),
        duration_days=1,
        seed=0,
        software_revision="original-revision",
    )
    original_dir = tmp_path / "original"
    write_deep_soil_qualification_outputs(original, original_dir)
    output_dir = tmp_path / "corrected"
    script = Path(__file__).parents[1] / "scripts" / "deep_soil_qualification.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--original-dir",
            str(original_dir),
            "--output-dir",
            str(output_dir),
            "--software-revision",
            "corrected-revision",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout.splitlines()[-1])["output_dir"] == str(
        output_dir
    )
    assert {path.name for path in output_dir.iterdir()} == {
        "comparison.json",
        "comparison.md",
        "daily-soil-p.npz",
        "manifest.json",
    }
    manifest = json.loads((output_dir / "manifest.json").read_text())
    comparison = json.loads((output_dir / "comparison.json").read_text())
    assert manifest["traits"]["plant"]["initial_biomass"] == pytest.approx(0.001)
    assert manifest["traits"]["fungus"]["initial_biomass"] == pytest.approx(
        7.97e-7
    )
    assert comparison["configuration_equivalence"]["environment"] is True
    assert "traits" in comparison["intentional_provenance_differences"]
