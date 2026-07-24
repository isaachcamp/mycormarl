"""Environment integration and accounting tests for continuous P uptake."""

import jax
import jax.numpy as jnp
import pytest

from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_uptake import (
    allocate_competing_uptake,
    blend_uptake_requests,
    continuous_uptake_request,
    sparse_uptake_request,
    sparse_uptake_resistance,
)
from mycormarl.soil.phosphate_units import MICROMOL_P_TO_MG_P
from mycormarl.soil.soil import evolve_soil_p, uptake_geometry_coefficients


def _p3_config(initial_solution_p_um=1.0):
    """Return a one-cell domain with unbuffered uptake arithmetic."""
    return EnvConfig(
        dt=0.05,
        soil_radius_cm=1.0,
        soil_depth_cm=1.0,
        radial_interval_cm=1.0,
        depth_interval_cm=1.0,
        topsoil_depth_cm=1.0,
        initial_solution_p_um=initial_solution_p_um,
        theta_water=0.3,
        b_p=0.0,
    )


def _p3_species(**overrides):
    """Return small consumers with controllable uptake and loss traits."""
    plant_overrides = overrides.get("plant", {})
    fungus_overrides = overrides.get("fungus", {})
    return SpeciesParams(
        plant=PlantTraits(**{
            "initial_biomass": 1e-4,
            "initial_c_pool": 1.0,
            "initial_p_pool": 0.0,
            "kappa_c": 0.0,
            "kappa_p": 0.0,
            "root_length_density": 100.0,
            **plant_overrides,
        }),
        fungus=FungusTraits(**{
            "initial_biomass": 1e-8,
            "initial_c_pool": 1.0,
            "initial_p_pool": 0.0,
            "kappa_c": 0.0,
            "kappa_p": 0.0,
            **fungus_overrides,
        }),
    )


def test_soil_step_credits_plant_only_uptake_in_milligrams():
    """Checks the soil-to-plant unit boundary with no fungal request."""
    species = _p3_species(fungus={"jmax": 0.0})
    env = BaseMycorMarl(_p3_config(), species)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        root_length_density=jnp.array([[2.0]]),
        hyphae_length_density=jnp.array([[0.0]]),
    )
    concentration = state.soil_labile_p / (env.cell_volumes * 0.3)
    d_flux = (
        env.config.phosphate_diffusion_coefficient_cm2_s
        * env.config.theta_water
        * env.config.phosphate_impedance_factor
    )
    d_app = d_flux / (env.config.theta_water + env.config.b_p)
    resistance = sparse_uptake_resistance(
        state.root_length_density,
        species.plant.root_radius,
        species.plant.jmax,
        d_flux,
        d_app,
        env.config.uptake_reference_time_days,
    )
    expected_request = sparse_uptake_request(
        concentration,
        state.root_length_density,
        env.cell_volumes,
        species.plant.root_radius,
        species.plant.jmax,
        species.plant.km,
        env.config.dt,
        resistance,
    )

    next_state = env.step_phosphorus_field(state)
    accepted = jnp.minimum(state.soil_labile_p, expected_request)

    assert jnp.sum(state.soil_labile_p - next_state.soil_labile_p) == pytest.approx(
        jnp.sum(accepted), rel=1e-6
    )
    assert next_state.plant_p_pool[0] == pytest.approx(
        jnp.sum(accepted) * MICROMOL_P_TO_MG_P, rel=1e-6
    )
    assert next_state.fungus_p_pool[0] == pytest.approx(0.0)


def test_soil_step_supports_fungus_only_uptake():
    """Checks that a zero-root cell still credits fungal uptake correctly."""
    species = _p3_species(plant={"jmax": 0.0})
    env = BaseMycorMarl(_p3_config(), species)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        root_length_density=jnp.array([[0.0]]),
        hyphae_length_density=jnp.array([[20.0]]),
    )

    next_state = env.step_phosphorus_field(state)

    soil_loss_mg = (
        jnp.sum(state.soil_labile_p - next_state.soil_labile_p)
        * MICROMOL_P_TO_MG_P
    )
    assert next_state.fungus_p_pool[0] == pytest.approx(soil_loss_mg, rel=1e-6)
    assert next_state.plant_p_pool[0] == pytest.approx(0.0)


def test_mixed_soil_step_caps_shared_inventory_and_closes_transaction_balance():
    """Verifies mixed over-demand removes exactly the P credited to consumers."""
    species = _p3_species(
        plant={"jmax": 1.0},
        fungus={"jmax": 1.0},
    )
    env = BaseMycorMarl(_p3_config(initial_solution_p_um=1e-6), species)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        root_length_density=jnp.array([[1000.0]]),
        hyphae_length_density=jnp.array([[1000.0]]),
    )

    next_state = env.step_phosphorus_field(state)

    soil_loss_mg = (
        jnp.sum(state.soil_labile_p - next_state.soil_labile_p)
        * MICROMOL_P_TO_MG_P
    )
    pool_gain_mg = (
        next_state.plant_p_pool[0]
        + next_state.fungus_p_pool[0]
        - state.plant_p_pool[0]
        - state.fungus_p_pool[0]
    )
    assert jnp.all(next_state.soil_labile_p >= 0.0)
    assert jnp.sum(next_state.soil_labile_p) == pytest.approx(0.0, abs=1e-12)
    assert pool_gain_mg == pytest.approx(soil_loss_mg, rel=1e-5)


def test_mixed_cell_matches_independent_shared_weight_and_post_blend_cap():
    """Reconstructs blended requests to verify one fungal-derived regime decision."""
    species = _p3_species()
    env = BaseMycorMarl(_p3_config(), species)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        root_length_density=jnp.array([[2.0]]),
        hyphae_length_density=jnp.array([[168.75]]),
    )
    root_resistance, fungus_resistance, weight = uptake_geometry_coefficients(
        state,
        species,
        env.config.phosphate_diffusion_coefficient_cm2_s,
        env.config.theta_water,
        env.config.phosphate_impedance_factor,
        env.config.b_p,
        env.config.uptake_reference_time_days,
        env.config.uptake_transition_exponent,
    )
    concentration = state.soil_labile_p / (
        env.cell_volumes
        * (env.config.theta_water + env.config.b_p)
    )
    root_sparse = sparse_uptake_request(
        concentration,
        state.root_length_density,
        env.cell_volumes,
        species.plant.root_radius,
        species.plant.jmax,
        species.plant.km,
        env.config.dt,
        root_resistance,
    )
    fungus_sparse = sparse_uptake_request(
        concentration,
        state.hyphae_length_density,
        env.cell_volumes,
        species.fungus.hyphal_radius,
        species.fungus.jmax,
        species.fungus.km,
        env.config.dt,
        fungus_resistance,
    )
    root_continuous = continuous_uptake_request(
        concentration,
        state.root_length_density,
        env.cell_volumes,
        species.plant.root_radius,
        species.plant.jmax,
        species.plant.km,
        env.config.dt,
    )
    fungus_continuous = continuous_uptake_request(
        concentration,
        state.hyphae_length_density,
        env.cell_volumes,
        species.fungus.hyphal_radius,
        species.fungus.jmax,
        species.fungus.km,
        env.config.dt,
    )
    root_request = blend_uptake_requests(root_sparse, root_continuous, weight)
    fungus_request = blend_uptake_requests(
        fungus_sparse, fungus_continuous, weight
    )
    expected_soil, expected_root, expected_fungus = allocate_competing_uptake(
        state.soil_labile_p, root_request, fungus_request
    )

    next_state = env.step_phosphorus_field(state)

    assert 0.0 < weight[0, 0] < 1.0
    assert jnp.allclose(next_state.soil_labile_p, expected_soil, rtol=1e-6)
    assert next_state.plant_p_pool[0] - state.plant_p_pool[0] == pytest.approx(
        jnp.sum(expected_root) * MICROMOL_P_TO_MG_P, rel=1e-6
    )
    assert next_state.fungus_p_pool[0] - state.fungus_p_pool[0] == pytest.approx(
        jnp.sum(expected_fungus) * MICROMOL_P_TO_MG_P, rel=1e-6
    )


def test_uptake_credit_cannot_fund_growth_in_the_same_step():
    """Preserves allocation-before-uptake while exposing post-growth geometry."""
    species = _p3_species(fungus={"jmax": 0.0})
    env = BaseMycorMarl(_p3_config(), species)
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        PLANT: jnp.array([0.0, 1.0, 0.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    _, next_state, _, _, infos = env.step_env(
        jax.random.PRNGKey(1), state, actions
    )

    assert infos[PLANT]["growth"][0] == pytest.approx(0.0)
    assert next_state.plant_biomass[0] == pytest.approx(state.plant_biomass[0])
    assert next_state.plant_p_pool[0] > 0.0


def test_end_to_end_uptake_conserves_soil_plus_free_pool_p():
    """Closes the full step balance when no biological P allocation is made."""
    species = _p3_species()
    env = BaseMycorMarl(_p3_config(), species)
    _, state = env.reset(jax.random.PRNGKey(0))
    before_mg = (
        jnp.sum(state.soil_labile_p) * MICROMOL_P_TO_MG_P
        + state.plant_p_pool[0]
        + state.fungus_p_pool[0]
    )
    actions = {
        PLANT: jnp.array([0.0, 0.0, 1.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    _, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)
    after_mg = (
        jnp.sum(next_state.soil_labile_p) * MICROMOL_P_TO_MG_P
        + next_state.plant_p_pool[0]
        + next_state.fungus_p_pool[0]
    )

    assert after_mg == pytest.approx(before_mg, rel=1e-5)


def test_soil_uptake_kernel_is_jittable():
    """Checks the complete amount-to-concentration-to-credit path under JIT."""
    species = _p3_species()
    env = BaseMycorMarl(_p3_config(), species)
    _, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(evolve_soil_p)

    next_state = step(
        state,
        env.config.dt,
        species,
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

    assert jnp.all(jnp.isfinite(next_state.soil_labile_p))
    assert jnp.all(next_state.soil_labile_p >= 0.0)


def test_structural_p_removed_with_biomass_is_recorded_as_mortality_loss():
    """Closes structural-P accounting when maintenance deficit removes biomass."""
    species = _p3_species(
        plant={
            "initial_c_pool": 0.0,
            "jmax": 0.0,
            "kappa_c": 100.0,
        },
        fungus={
            "initial_c_pool": 0.0,
            "jmax": 0.0,
            "kappa_c": 100.0,
        },
    )
    env = BaseMycorMarl(_p3_config(initial_solution_p_um=0.0), species)
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        PLANT: jnp.array([0.0, 0.0, 1.0, 0.0]),
        FUNGUS: jnp.array([0.0, 0.0, 1.0, 0.0]),
    }

    _, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)

    expected_plant_loss = state.plant_biomass[0] * species.plant.gamma_p
    expected_fungus_loss = state.fungus_biomass[0] * species.fungus.gamma_p
    assert next_state.plant_biomass[0] == pytest.approx(0.0)
    assert next_state.fungus_biomass[0] == pytest.approx(0.0)
    assert next_state.cumulative_plant_p_mortality_loss_mg[0] == pytest.approx(
        expected_plant_loss, rel=1e-6
    )
    assert next_state.cumulative_fungus_p_mortality_loss_mg[0] == pytest.approx(
        expected_fungus_loss, rel=1e-6
    )
