"""Public contracts for the deep-soil phosphorus qualification."""

from dataclasses import replace
import json

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


def test_qualification_records_fixed_horizon_integrated_p_uptake_and_balance():
    """Integrated uptake closes the independently observed soil-P inventory loss."""
    species = SpeciesParams(
        plant=PlantTraits(
            initial_biomass=1e-5,
            initial_c_pool=0.0,
            initial_p_pool=0.0,
            kappa_c=0.0,
            kappa_p=0.0,
            death_fraction=0.0,
        ),
        fungus=FungusTraits(
            initial_biomass=1e-5,
            initial_c_pool=0.0,
            initial_p_pool=0.0,
            kappa_c=0.0,
            kappa_p=0.0,
            death_fraction=0.0,
        ),
    )
    policy = StaticPolicy(0.0, 0.0, 0.0, 1.0)

    result = run_deep_soil_qualification(
        config=EnvConfig(
            dt=0.5,
            soil_radius_cm=1.0,
            soil_depth_cm=2.0,
            radial_interval_cm=1.0,
            depth_interval_cm=1.0,
            initial_solution_p_um=1.0,
            phosphate_diffusion_coefficient_cm2_s=1e-30,
        ),
        species=species,
        plant_policy=policy,
        fungus_policy=policy,
        duration_days=1.0,
        seed=7,
        software_revision="test-revision",
    )

    fluxes = result.integrated_p_fluxes_micromol
    observed_soil_loss = float(
        result.daily_soil_labile_p_micromol[0].sum()
        - result.daily_soil_labile_p_micromol[-1].sum()
    )
    assert fluxes["plant_uptake_micromol"] > 0.0
    assert fluxes["fungus_uptake_micromol"] > 0.0
    assert fluxes["total_uptake_micromol"] == pytest.approx(
        fluxes["plant_uptake_micromol"]
        + fluxes["fungus_uptake_micromol"],
        rel=1e-6,
    )
    assert fluxes["total_uptake_micromol"] == pytest.approx(
        observed_soil_loss,
        rel=2e-5,
        abs=1e-7,
    )
    assert result.relative_extended_p_balance_error <= 1e-5
    assert result.manifest["qualification"]["integrated_p_fluxes_micromol"] == fluxes
    assert result.manifest["qualification"][
        "relative_extended_p_balance_error"
    ] == pytest.approx(result.relative_extended_p_balance_error)


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
    assert manifest["policies"]["fungus"]["rate_action_per_day"] == pytest.approx(
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


def test_temporal_gate_rejects_different_fixed_horizons():
    """Only timestep may differ between candidate and reference provenance."""
    species = SpeciesParams(
        plant=PlantTraits(initial_biomass=0.0, jmax=0.0),
        fungus=FungusTraits(initial_biomass=0.0, jmax=0.0),
    )
    policy = StaticPolicy(0.0, 0.0, 0.0, 1.0)

    def run(dt, duration_days):
        return run_deep_soil_qualification(
            config=EnvConfig(
                dt=dt,
                soil_radius_cm=1.0,
                soil_depth_cm=100.0,
                radial_interval_cm=1.0,
                depth_interval_cm=5.0,
                initial_solution_p_um=1.0,
                phosphate_diffusion_coefficient_cm2_s=1e-30,
            ),
            species=species,
            plant_policy=policy,
            fungus_policy=policy,
            duration_days=duration_days,
            seed=0,
            software_revision="test-revision",
        )

    with pytest.raises(
        ValueError,
        match="only environment dt may differ",
    ):
        compare_temporal_convergence(run(1.0, 1.0), run(0.5, 2.0))


def test_temporal_gate_reports_timestep_scaled_pools_without_failing():
    """Residual pools remain diagnostic when accepted quantities agree."""
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

    candidate = run(1.0)
    reference = run(0.5)
    candidate = replace(
        candidate,
        endpoint_metrics={
            **candidate.endpoint_metrics,
            "final_plant_c_pool_g": 2.0,
            "final_plant_p_pool_mg": 4.0,
            "final_fungus_c_pool_g": 6.0,
            "final_fungus_p_pool_mg": 8.0,
        },
    )
    reference = replace(
        reference,
        endpoint_metrics={
            **reference.endpoint_metrics,
            "final_plant_c_pool_g": 1.0,
            "final_plant_p_pool_mg": 2.0,
            "final_fungus_c_pool_g": 3.0,
            "final_fungus_p_pool_mg": 4.0,
        },
    )

    comparison = compare_temporal_convergence(candidate, reference)

    assert comparison["passes_5_percent"] is True
    assert {row["metric"] for row in comparison["diagnostic_comparisons"]} == {
        "final_plant_c_pool_g",
        "final_plant_p_pool_mg",
        "final_fungus_c_pool_g",
        "final_fungus_p_pool_mg",
    }
    assert comparison["maximum_diagnostic_relative_change"] == pytest.approx(1.0)
    assert comparison["criteria"]["diagnostic_metrics"] == sorted(
        {
            "final_plant_c_pool_g",
            "final_plant_p_pool_mg",
            "final_fungus_c_pool_g",
            "final_fungus_p_pool_mg",
        }
    )
    assert set(comparison["criteria"]["accepted_relative_metrics"]).issuperset(
        {
            "plant_uptake_micromol",
            "fungus_uptake_micromol",
            "total_uptake_micromol",
            "final_plant_biomass_g",
            "final_fungus_biomass_g",
        }
    )


def test_temporal_gate_fails_when_integrated_p_uptake_does_not_converge():
    """Fixed-horizon P uptake is an accepted temporal-convergence quantity."""
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

    candidate = replace(
        run(1.0),
        integrated_p_fluxes_micromol={
            "plant_uptake_micromol": 0.6,
            "fungus_uptake_micromol": 0.6,
            "total_uptake_micromol": 1.2,
        },
    )
    reference = replace(
        run(0.5),
        integrated_p_fluxes_micromol={
            "plant_uptake_micromol": 0.5,
            "fungus_uptake_micromol": 0.5,
            "total_uptake_micromol": 1.0,
        },
    )

    comparison = compare_temporal_convergence(candidate, reference)

    assert comparison["passes_5_percent"] is False
    assert {
        row["metric"] for row in comparison["metric_comparisons"]
    }.issuperset(
        {
            "plant_uptake_micromol",
            "fungus_uptake_micromol",
            "total_uptake_micromol",
        }
    )


def test_temporal_gate_requires_each_run_to_close_extended_p_balance():
    """A converged trajectory still fails if either P ledger is not conservative."""
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

    candidate = replace(
        run(1.0),
        relative_extended_p_balance_error=2e-5,
    )
    reference = run(0.5)

    comparison = compare_temporal_convergence(candidate, reference)

    assert comparison["passes_temporal_convergence"] is False
    assert next(
        row
        for row in comparison["absolute_checks"]
        if row["requirement"] == "candidate_relative_extended_p_balance_error"
    ) == {
        "requirement": "candidate_relative_extended_p_balance_error",
        "value": pytest.approx(2e-5),
        "maximum": pytest.approx(1e-5),
        "passes": False,
    }


def test_temporal_gate_requires_each_run_to_preserve_fungal_confinement():
    """Numerical agreement cannot excuse uptake geometry outside the colony."""
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

    candidate = replace(
        run(1.0),
        max_fungal_density_outside_colony=2e-12,
    )
    comparison = compare_temporal_convergence(candidate, run(0.5))

    assert comparison["passes_temporal_convergence"] is False
    density_check = next(
        row
        for row in comparison["absolute_checks"]
        if row["requirement"]
        == "candidate_max_fungal_density_outside_colony"
    )
    assert density_check["maximum"] == pytest.approx(1e-12)
    assert density_check["passes"] is False
