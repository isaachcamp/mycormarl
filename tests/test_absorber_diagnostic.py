"""Public contracts for the absorber construction-carbon diagnostic."""

import csv
import math
from pathlib import Path
import subprocess
import sys

import pytest

from mycormarl.params import EnvConfig
from mycormarl.fungus.mycelium import fungal_biomass_from_hyphal_length
from mycormarl.fungus.traits import FungusTraits
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.absorber_diagnostic import (
    plant_construction_carbon_g,
    root_tissue_carbon_density_g_cm3,
    run_absorber_geometry_sweep,
)


def test_root_tissue_carbon_density_is_inferred_from_supplied_traits():
    """A worked cylindrical-root fixture has density 2 g C cm^-3."""
    traits = PlantTraits(
        kroot=0.01,
        root_radius=0.1,
        specific_root_length=20.0 / math.pi,
        gamma_c=0.4,
    )

    assert root_tissue_carbon_density_g_cm3(traits) == pytest.approx(2.0)


def test_plant_construction_carbon_scales_with_length_and_radius_squared():
    """Candidate cylinders are priced using the inferred root-tissue density."""
    traits = PlantTraits(
        kroot=0.01,
        root_radius=0.1,
        specific_root_length=20.0 / math.pi,
        gamma_c=0.4,
    )

    baseline = plant_construction_carbon_g(10.0, 0.1, traits)

    assert baseline == pytest.approx(0.2 * math.pi)
    assert plant_construction_carbon_g(20.0, 0.1, traits) == pytest.approx(
        2.0 * baseline
    )
    assert plant_construction_carbon_g(10.0, 0.2, traits) == pytest.approx(
        4.0 * baseline
    )


@pytest.mark.parametrize("length_cm", [-1.0, float("nan"), float("inf")])
def test_plant_construction_carbon_rejects_invalid_lengths(length_cm):
    """Construction economics are undefined for invalid absorber lengths."""
    with pytest.raises(ValueError, match="absorber_length_cm"):
        plant_construction_carbon_g(length_cm, 0.01, PlantTraits())


def test_construction_normalisation_excludes_whole_organism_accounting_traits():
    """Only root tissue geometry and structural carbon define construction cost."""
    traits = PlantTraits(
        kroot=0.5,
        specific_root_length=20.0,
        gamma_c=0.4,
        kleaf=0.99,
        amass=99.0,
        kappa_c=88.0,
        initial_c_pool=77.0,
    )
    row = run_absorber_geometry_sweep(
        absorber_radii_cm=[0.01],
        length_densities_cm_cm3=[2.0],
        modes=("fixed_reservoir",),
        dt_days=1.0,
        reference_time_days=1.0,
        plant_traits=traits,
        include_markers=False,
    )[0]

    expected_density = traits.gamma_c / (
        traits.specific_root_length * math.pi * traits.root_radius**2
    )
    expected_cost = 2.0 * math.pi * 0.01**2 * expected_density
    assert row["root_tissue_carbon_density_g_cm3"] == pytest.approx(
        expected_density
    )
    assert row["construction_carbon_g"] == pytest.approx(expected_cost)


def test_initial_instantaneous_rate_is_per_second_and_timestep_independent():
    """Maximum rate is the uncapped initial closure rate, not accepted uptake/dt."""
    traits = PlantTraits(kroot=0.5, specific_root_length=20.0, gamma_c=0.4)

    rows = [
        run_absorber_geometry_sweep(
            absorber_radii_cm=[0.01],
            length_densities_cm_cm3=[2.0],
            modes=("fixed_reservoir",),
            dt_days=dt_days,
            reference_time_days=1.0,
            plant_traits=traits,
            include_markers=False,
        )[0]
        for dt_days in (0.25, 0.4)
    ]

    row = rows[0]
    assert row["record_type"] == "surface"
    assert row["integrated_uptake_micromol"] > 0.0
    assert row["integrated_uptake_micromol"] == pytest.approx(
        row["maximum_instantaneous_uptake_rate_micromol_s"] * 86_400.0
    )
    assert rows[1]["maximum_instantaneous_uptake_rate_micromol_s"] == pytest.approx(
        row["maximum_instantaneous_uptake_rate_micromol_s"]
    )
    assert row["final_bulk_concentration_micromol_cm3"] == pytest.approx(
        row["initial_bulk_concentration_micromol_cm3"]
    )
    assert row["maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s"] == pytest.approx(
        row["maximum_instantaneous_uptake_rate_micromol_s"]
        / row["construction_carbon_g"]
    )
    assert row["dt_days"] == pytest.approx(0.25)
    assert row["reference_time_days"] == pytest.approx(1.0)
    assert row["amount_flux_diffusivity_cm2_s"] == pytest.approx(9.24e-7)
    assert row["apparent_diffusivity_cm2_s"] < row[
        "amount_flux_diffusivity_cm2_s"
    ]


def test_finite_inventory_event_time_is_not_bounded_by_reference_horizon():
    """Slow valid geometry reaches one-percent surface P after the one-day run."""
    rows = [
        run_absorber_geometry_sweep(
            absorber_radii_cm=[1e-4],
            length_densities_cm_cm3=[0.1],
            modes=("finite_inventory",),
            dt_days=dt_days,
            reference_time_days=1.0,
            include_markers=False,
        )[0]
        for dt_days in (0.25, 1.0)
    ]

    assert rows[0]["conservation_error_micromol"] == pytest.approx(0.0, abs=1e-7)
    assert rows[0]["t_1_percent_reached"] is True
    assert rows[0]["t_1_percent_days"] > 1.0
    assert rows[1]["t_1_percent_days"] == pytest.approx(
        rows[0]["t_1_percent_days"], rel=1e-10
    )


def test_depletion_event_time_matches_reference_quadrature_and_uptake_limit():
    """Canonical slow-cell values agree with an independent high-accuracy run."""
    rows = run_absorber_geometry_sweep(
        absorber_radii_cm=[1e-4],
        length_densities_cm_cm3=[0.1, 10.0],
        modes=("finite_inventory",),
        dt_days=1.0,
        reference_time_days=1.0,
        include_markers=False,
    )

    assert rows[0]["t_1_percent_days"] == pytest.approx(
        483_045.3082646465, rel=2e-8
    )
    assert rows[1]["t_1_percent_days"] == pytest.approx(
        4_830.294816628497, rel=2e-8
    )
    assert rows[1]["t_1_percent_days"] < rows[0]["t_1_percent_days"]


def test_zero_density_has_zero_cost_and_uptake_without_depletion():
    """An absorber-free cell remains a valid limiting-case qualification."""
    row = run_absorber_geometry_sweep(
        absorber_radii_cm=[0.01],
        length_densities_cm_cm3=[0.0],
        modes=("finite_inventory",),
        dt_days=1.0,
        reference_time_days=1.0,
        include_markers=False,
    )[0]

    assert row["construction_carbon_g"] == pytest.approx(0.0)
    assert row["integrated_uptake_micromol"] == pytest.approx(0.0)
    assert row["integrated_uptake_per_construction_carbon_micromol_g_c"] == 0.0
    assert row["t_1_percent_reached"] is False
    assert row["t_1_percent_days"] is None


def test_sweep_retains_but_blanks_touching_and_overlapping_geometries():
    """Only cylinders strictly inside their assigned territories are valid."""
    radius = 0.01
    touching_density = 1.0 / (math.pi * radius**2)
    rows = run_absorber_geometry_sweep(
        absorber_radii_cm=[radius],
        length_densities_cm_cm3=[0.99 * touching_density, touching_density, 1.01 * touching_density],
        modes=("finite_inventory",),
        dt_days=1.0,
        reference_time_days=1.0,
        include_markers=False,
    )

    below, touching, overlapping = rows
    assert below["geometry_valid"] is True
    assert below["territory_radius_cm"] > radius
    assert below["integrated_uptake_micromol"] is not None
    for invalid in (touching, overlapping):
        assert invalid["geometry_valid"] is False
        assert invalid["territory_radius_cm"] <= radius + 1e-14
        for field in (
            "construction_carbon_g",
            "integrated_uptake_micromol",
            "sparse_resistance_micromol_cm3",
            "initial_surface_concentration_micromol_cm3",
            "t_1_percent_days",
        ):
            assert invalid[field] is None


@pytest.mark.parametrize("radius_cm", [0.0, -0.01, float("nan")])
def test_runner_rejects_invalid_absorber_radius(radius_cm):
    """The closure harness rejects radii that cannot define a cylinder."""
    with pytest.raises(ValueError, match="absorber_radius"):
        run_absorber_geometry_sweep(
            absorber_radii_cm=[radius_cm],
            length_densities_cm_cm3=[1.0],
            modes=("fixed_reservoir",),
            include_markers=False,
        )


def test_runner_rejects_invalid_buffer_capacity_before_experiment():
    """A finite cell cannot be constructed with non-positive labile capacity."""
    with pytest.raises(ValueError, match="b_p"):
        run_absorber_geometry_sweep(
            absorber_radii_cm=[0.01],
            length_densities_cm_cm3=[1.0],
            modes=("finite_inventory",),
            config=EnvConfig(b_p=-0.3),
            include_markers=False,
        )


def test_sweep_solves_panel_specific_fungus_equivalent_plant_geometries():
    """Efficiency markers equal fungal targets without snapping to sweep cells."""
    rows = run_absorber_geometry_sweep(
        absorber_radii_cm=[1e-4, 3e-2],
        length_densities_cm_cm3=[0.1, 1e4],
        modes=("fixed_reservoir", "finite_inventory"),
        dt_days=1.0,
        reference_time_days=1.0,
        include_markers=True,
    )

    surfaces = [row for row in rows if row["record_type"] == "surface"]
    markers = [row for row in rows if row["record_type"] == "marker"]
    assert len(surfaces) == 8
    assert len(markers) == 10
    assert {row["marker_label"] for row in markers} == {
        "plant_native",
        "fungus_geometry_plant_economics",
        "fungus_equivalent_plant_geometry",
        "fungus_native",
    }
    assert all(row.keys() == rows[0].keys() for row in rows)
    assert {
        "experiment_mode",
        "economics_mode",
        "uptake_traits",
        "absorber_radius_cm",
        "length_density_cm_cm3",
        "construction_carbon_g",
        "integrated_uptake_micromol",
        "maximum_instantaneous_uptake_rate_micromol_s",
        "sparse_resistance_micromol_cm3",
        "initial_surface_concentration_micromol_cm3",
        "final_surface_concentration_micromol_cm3",
        "t_1_percent_days",
        "t_1_percent_reached",
    } <= rows[0].keys()

    equivalents = [
        row for row in markers
        if row["marker_label"] == "fungus_equivalent_plant_geometry"
    ]
    assert len(equivalents) == 4
    assert all(row["marker_solve_status"] == "solved" for row in equivalents)
    assert all(row["geometry_valid"] is True for row in equivalents)
    for row in equivalents:
        assert row[row["marker_metric"]] == pytest.approx(
            row["target_metric_value"], rel=2e-6
        )
        assert row["absorber_radius_cm"] not in (1e-4, 3e-2)
    finite_equivalents = {
        row["marker_metric"]: row["absorber_radius_cm"]
        for row in equivalents
        if row["experiment_mode"] == "finite_inventory"
    }
    assert finite_equivalents[
        "integrated_uptake_per_construction_carbon_micromol_g_c"
    ] != pytest.approx(
        finite_equivalents[
            "maximum_instantaneous_uptake_rate_per_construction_carbon_micromol_g_c_s"
        ]
    )

    fungus = FungusTraits()
    fungus_native = next(
        row for row in markers
        if row["experiment_mode"] == "fixed_reservoir"
        and row["marker_label"] == "fungus_native"
    )
    expected_fungal_biomass = fungal_biomass_from_hyphal_length(
        2_000.0,
        fungus.gamma_c,
        fungus.hyphal_tissue_carbon_density,
        fungus.hyphal_radius,
    )
    assert fungus_native["uptake_traits"] == "fungus"
    assert fungus_native["economics_mode"] == "fungus"
    assert fungus_native["construction_carbon_g"] == pytest.approx(
        expected_fungal_biomass * fungus.gamma_c
    )

    fungal_geometry_plant_economics = [
        row for row in markers
        if row["marker_label"] == "fungus_geometry_plant_economics"
    ]
    assert len(fungal_geometry_plant_economics) == 2
    assert all(row["economics_mode"] == "plant" for row in fungal_geometry_plant_economics)
    assert all(row["uptake_traits"] == "plant" for row in fungal_geometry_plant_economics)
    assert all(
        row["absorber_radius_cm"] == pytest.approx(fungus.hyphal_radius)
        and row["length_density_cm_cm3"] == pytest.approx(fungus.saturation_density)
        for row in fungal_geometry_plant_economics
    )


def test_sweep_records_unavailable_equivalent_target_without_invalid_search():
    """An unbracketed fungal P-per-C target is explicit and has no coordinates."""
    rows = run_absorber_geometry_sweep(
        absorber_radii_cm=[1e-4, 3e-2],
        length_densities_cm_cm3=[0.1, 1e4],
        modes=("fixed_reservoir",),
        dt_days=1.0,
        reference_time_days=1.0,
        fungus_traits=FungusTraits(hyphal_tissue_carbon_density=1e12),
        include_markers=True,
    )

    unavailable = [
        row for row in rows
        if row["marker_label"] == "fungus_equivalent_plant_geometry"
    ]
    assert len(unavailable) == 2
    assert all(row["marker_solve_status"] == "unavailable" for row in unavailable)
    assert all(row["absorber_radius_cm"] is None for row in unavailable)


def test_equivalent_solver_rejects_fungal_density_outside_plotted_domain():
    """Equivalent geometry cannot be placed beyond the displayed lambda range."""
    rows = run_absorber_geometry_sweep(
        absorber_radii_cm=[1e-4, 3e-2],
        length_densities_cm_cm3=[0.1, 100.0],
        modes=("fixed_reservoir",),
        dt_days=1.0,
        reference_time_days=1.0,
        fungus_traits=FungusTraits(saturation_density=2_000.0),
        include_markers=True,
    )

    equivalents = [
        row for row in rows
        if row["marker_label"] == "fungus_equivalent_plant_geometry"
    ]
    assert all(row["marker_solve_status"] == "unavailable" for row in equivalents)


def test_default_sweep_uses_configurable_logarithmic_canonical_grids():
    """Qualification overrides retain the specified endpoints and log spacing."""
    rows = run_absorber_geometry_sweep(
        modes=("fixed_reservoir",),
        dt_days=1.0,
        reference_time_days=1.0,
        radius_count=3,
        density_count=4,
        include_markers=False,
    )

    radii = sorted({row["absorber_radius_cm"] for row in rows})
    densities = sorted({row["length_density_cm_cm3"] for row in rows})
    assert len(rows) == 12
    assert radii[0] == pytest.approx(1e-4)
    assert radii[-1] == pytest.approx(3e-2)
    assert radii[1] / radii[0] == pytest.approx(radii[2] / radii[1])
    assert densities[0] == pytest.approx(1e-1)
    assert densities[-1] == pytest.approx(1e4)
    assert densities[1] / densities[0] == pytest.approx(
        densities[2] / densities[1]
    )


def test_sweep_exposes_density_territory_and_radius_effects_with_finite_values():
    """Geometry changes both sparse resistance and absolute uptake scale."""
    rows = run_absorber_geometry_sweep(
        absorber_radii_cm=[1e-4, 3e-2],
        length_densities_cm_cm3=[1e-1, 1e4],
        modes=("fixed_reservoir",),
        dt_days=1.0,
        reference_time_days=1.0,
        include_markers=False,
    )
    by_geometry = {
        (row["absorber_radius_cm"], row["length_density_cm_cm3"]): row
        for row in rows
    }
    valid_rows = [row for row in rows if row["geometry_valid"]]
    invalid_rows = [row for row in rows if not row["geometry_valid"]]
    assert invalid_rows
    for row in valid_rows:
        assert math.isfinite(row["sparse_resistance_micromol_cm3"])
        assert math.isfinite(row["integrated_uptake_micromol"])
    for radius in (1e-4,):
        assert by_geometry[(radius, 1e4)]["sparse_resistance_micromol_cm3"] < (
            by_geometry[(radius, 1e-1)]["sparse_resistance_micromol_cm3"]
        )
        assert by_geometry[(radius, 1e4)]["integrated_uptake_micromol"] > (
            by_geometry[(radius, 1e-1)]["integrated_uptake_micromol"]
        )
    assert by_geometry[(1e-4, 1e-1)]["sparse_resistance_micromol_cm3"] != (
        by_geometry[(3e-2, 1e-1)]["sparse_resistance_micromol_cm3"]
    )


def test_cli_writes_tabular_data_and_publication_figure_formats(tmp_path):
    """The command emits every row plus legible vector and raster artifacts."""
    script = Path(__file__).parents[1] / "scripts" / (
        "absorber_construction_carbon_diagnostic.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
            "--radius-count",
            "2",
            "--density-count",
            "2",
            "--dt-days",
            "1",
            "--reference-time-days",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    csv_path = tmp_path / "absorber_geometry_sweep.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    assert {row["marker_label"] for row in rows if row["record_type"] == "marker"} == {
        "plant_native",
        "fungus_geometry_plant_economics",
        "fungus_equivalent_plant_geometry",
        "fungus_native",
    }
    assert any(row["geometry_valid"] == "False" for row in rows)
    assert all(
        row["integrated_uptake_micromol"] == ""
        for row in rows
        if row["record_type"] == "surface" and row["geometry_valid"] == "False"
    )

    for stem in ("construction_carbon_efficiency", "uptake_scale", "depletion_timescale"):
        svg = tmp_path / f"{stem}.svg"
        png = tmp_path / f"{stem}.png"
        assert svg.stat().st_size > 1_000
        assert png.stat().st_size > 10_000

    efficiency_svg = (tmp_path / "construction_carbon_efficiency.svg").read_text()
    assert "Fixed reservoir" in efficiency_svg
    assert "Finite inventory" in efficiency_svg
    assert "Construction C" in efficiency_svg
    assert "µmol P g C⁻¹ s⁻¹" in efficiency_svg
    assert "r=" not in efficiency_svg
    assert "λ=" not in efficiency_svg
    assert "touching or overlapping" not in efficiency_svg
    assert efficiency_svg.count("Plant-native") == 1
    assert efficiency_svg.count("Fungal geometry, plant economics") == 1
    assert efficiency_svg.count("Fungus-equivalent plant geometry") == 1
    timescale_svg = (tmp_path / "depletion_timescale.svg").read_text()
    assert "t₁%" in timescale_svg
    assert "Plant-native" in timescale_svg
    assert "Fungus-native" in timescale_svg
    assert "t₁% (day)" in timescale_svg
