"""Executable contracts for phosphate diagnostics and qualification arithmetic.

The tests ensure that qualification reports observe the production uptake
transaction rather than reimplementing a subtly different scientific model.
They also protect deterministic comparison and annual-cost calculations used
to select a grid and timestep.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_grid import labile_amount_to_solution_concentration
from mycormarl.soil.phosphate_qualification import (
    annual_runtime_projection,
    reference_relative_change,
)
from mycormarl.soil.phosphate_uptake import blended_uptake_transaction
from mycormarl.soil.soil import uptake_geometry_coefficients
from mycormarl.soil.soil import evolve_soil_p_with_diagnostics
from mycormarl.soil.phosphate_units import MICROMOL_P_TO_MG_P


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "phosphate_qualification.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "phosphate_qualification_script", _SCRIPT_PATH
)
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
assert _SCRIPT_SPEC.loader is not None
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)
run_fixed_soil_scenario = _SCRIPT_MODULE.run_fixed_soil_scenario
run_coupled_scenario = _SCRIPT_MODULE.run_coupled_scenario


def _diagnostic_environment():
    """Return a one-cell mixed fixture where diffusion is identically zero."""
    config = EnvConfig(
        dt=0.05,
        soil_radius_cm=1.0,
        soil_depth_cm=1.0,
        radial_interval_cm=1.0,
        depth_interval_cm=1.0,
        topsoil_depth_cm=1.0,
        initial_solution_p_um=1.0,
    )
    species = SpeciesParams(
        plant=PlantTraits(initial_biomass=0.0, initial_p_pool=0.0),
        fungus=FungusTraits(initial_biomass=0.0, initial_p_pool=0.0),
    )
    return BaseMycorMarl(config, species)


def test_diagnostic_transaction_exactly_matches_production_soil_step():
    """Requests, cap, soil loss, and pool credits share one implementation."""
    env = _diagnostic_environment()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        root_length_density=jnp.array([[2.0]]),
        hyphae_length_density=jnp.array([[168.75]]),
    )
    root_resistance, fungus_resistance, weight = uptake_geometry_coefficients(
        state,
        env.species,
        env.config.phosphate_diffusion_coefficient_cm2_s,
        env.config.theta_water,
        env.config.phosphate_impedance_factor,
        env.config.b_p,
        env.config.uptake_reference_time_days,
        env.config.uptake_transition_exponent,
    )
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p,
        env.cell_volumes,
        env.config.theta_water,
        env.config.b_p,
    )

    remaining, accepted_root, accepted_fungus, diagnostics = (
        blended_uptake_transaction(
            state.soil_labile_p,
            concentration,
            state.root_length_density,
            state.hyphae_length_density,
            env.cell_volumes,
            env.species,
            env.config.dt,
            root_resistance,
            fungus_resistance,
            weight,
        )
    )
    stepped = env.step_phosphorus_field(state)

    assert jnp.allclose(stepped.soil_labile_p, remaining, rtol=1e-6)
    assert stepped.plant_p_pool[0] == pytest.approx(
        jnp.sum(accepted_root) * MICROMOL_P_TO_MG_P, rel=1e-6
    )
    assert stepped.fungus_p_pool[0] == pytest.approx(
        jnp.sum(accepted_fungus) * MICROMOL_P_TO_MG_P, rel=1e-6
    )
    assert jnp.all((diagnostics.root_surface_ratio >= 0.0))
    assert jnp.all((diagnostics.root_surface_ratio <= 1.0))
    assert jnp.all((diagnostics.fungus_surface_ratio >= 0.0))
    assert jnp.all((diagnostics.fungus_surface_ratio <= 1.0))
    assert jnp.allclose(
        state.soil_labile_p - remaining,
        accepted_root + accepted_fungus,
        rtol=1e-6,
    )


def test_diagnostic_transaction_reports_inventory_capping():
    """The capped mask is true exactly where final blended demand exceeds P."""
    env = _diagnostic_environment()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        soil_labile_p=jnp.array([[1e-12]]),
        root_length_density=jnp.array([[1e6]]),
        hyphae_length_density=jnp.array([[1e6]]),
    )
    coefficients = uptake_geometry_coefficients(
        state,
        env.species,
        env.config.phosphate_diffusion_coefficient_cm2_s,
        env.config.theta_water,
        env.config.phosphate_impedance_factor,
        env.config.b_p,
        env.config.uptake_reference_time_days,
        env.config.uptake_transition_exponent,
    )
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p,
        env.cell_volumes,
        env.config.theta_water,
        env.config.b_p,
    )

    remaining, accepted_root, accepted_fungus, diagnostics = (
        blended_uptake_transaction(
            state.soil_labile_p,
            concentration,
            state.root_length_density,
            state.hyphae_length_density,
            env.cell_volumes,
            env.species,
            env.config.dt,
            *coefficients,
        )
    )

    assert diagnostics.capped[0, 0]
    assert remaining[0, 0] == pytest.approx(0.0, abs=1e-18)
    assert accepted_root[0, 0] + accepted_fungus[0, 0] == pytest.approx(
        state.soil_labile_p[0, 0], rel=1e-6
    )


def test_diagnostic_evolution_matches_production_and_reports_each_substep():
    """Offline qualification retains observables without changing evolution."""
    env = _diagnostic_environment()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        root_length_density=jnp.array([[2.0]]),
        hyphae_length_density=jnp.array([[168.75]]),
    )

    diagnosed, diagnostics = evolve_soil_p_with_diagnostics(
        state,
        env.config.dt,
        env.species,
        env.cell_volumes,
        env.config.theta_water,
        env.config.b_p,
        env.radial_diffusion_conductance,
        env.vertical_diffusion_conductance,
        env.soil_substeps,
        env.config.phosphate_diffusion_coefficient_cm2_s,
        env.config.phosphate_impedance_factor,
        env.config.uptake_reference_time_days,
        env.config.uptake_transition_exponent,
    )
    production = env.step_phosphorus_field(state)

    assert len(diagnostics) == env.soil_substeps
    assert jnp.allclose(diagnosed.soil_labile_p, production.soil_labile_p)
    assert jnp.allclose(diagnosed.plant_p_pool, production.plant_p_pool)
    assert jnp.allclose(diagnosed.fungus_p_pool, production.fungus_p_pool)


@pytest.mark.parametrize(
    ("candidate", "reference", "expected"),
    [(100.0, 105.0, 5.0 / 105.0), (0.0, 0.0, 0.0), (0.0, 1e-12, 0.0)],
)
def test_reference_relative_change_uses_reference_and_absolute_floor(
    candidate, reference, expected
):
    """Near-zero metrics do not create meaningless relative failures."""
    assert reference_relative_change(candidate, reference, 1e-9) == pytest.approx(
        expected
    )


def test_annual_projection_converts_days_to_steps_and_runtime():
    """A projected year reports exact step count and warmed runtime cost."""
    projection = annual_runtime_projection(dt_days=0.05, seconds_per_step=0.2)

    assert projection["steps_per_year"] == 7300
    assert projection["runtime_seconds"] == pytest.approx(1460.0)
    assert projection["runtime_hours"] == pytest.approx(1460.0 / 3600.0)


def test_environment_uses_configured_episode_length_by_default():
    """An annual production config is not silently truncated at 256 steps."""
    env = BaseMycorMarl(
        EnvConfig(
            max_steps=14600,
            soil_radius_cm=1.0,
            soil_depth_cm=1.0,
            radial_interval_cm=1.0,
            depth_interval_cm=1.0,
            topsoil_depth_cm=1.0,
        ),
        SpeciesParams(
            plant=PlantTraits(initial_biomass=0.0),
            fungus=FungusTraits(initial_biomass=0.0),
        ),
    )

    assert env.max_episode_steps == 14600


def test_fixed_qualification_runner_is_deterministic_and_conservative():
    """The executable study fixture gives repeatable metrics and closes P."""
    first = run_fixed_soil_scenario("mixed", 1.0, 0.4, 0.1)
    second = run_fixed_soil_scenario("mixed", 1.0, 0.4, 0.1)

    assert first == second
    assert first["relative_p_balance_error"] <= 1e-5
    assert first["root_uptake_micromol"] > 0.0
    assert first["fungus_uptake_micromol"] > 0.0
    assert 0.0 <= first["minimum_root_surface_ratio"] <= 1.0
    assert 0.0 <= first["minimum_fungus_surface_ratio"] <= 1.0
    assert first["mean_continuous_weight"] <= first["maximum_continuous_weight"] <= 1.0
    assert first["diffusion_cfl_seconds"] > 0.0


def test_qualification_uptake_increases_across_concentration_range():
    """The scenario matrix preserves the expected nonlinear monotonic response."""
    low = run_fixed_soil_scenario("mixed", 0.1, 0.4, 0.1)
    high = run_fixed_soil_scenario("mixed", 10.0, 0.4, 0.1)

    assert high["total_uptake_micromol"] > low["total_uptake_micromol"]


def test_coupled_qualification_closes_extended_p_balance():
    """Soil, free pools, structure, mortality, and exports remain conservative."""
    result = run_coupled_scenario(interval_cm=0.1, dt_days=0.4)

    assert result["relative_extended_p_balance_error"] <= 1e-5
    assert result["plant_uptake_micromol"] > 0.0
    assert result["fungus_uptake_micromol"] > 0.0
    assert result["total_uptake_micromol"] == pytest.approx(
        result["plant_uptake_micromol"] + result["fungus_uptake_micromol"],
        rel=1e-6,
    )
    assert result["plant_uptake_share"] + result["fungus_uptake_share"] == pytest.approx(
        1.0, rel=1e-6
    )
    assert result["initial_soil_micromol"] - result["final_soil_micromol"] == pytest.approx(
        result["total_uptake_micromol"], rel=1e-5
    )
    assert result["final_soil_micromol"] >= 0.0
