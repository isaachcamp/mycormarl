"""Fixed IRM allocation through the public environment and geometry seams."""

import math

import jax
import jax.numpy as jnp
import pytest

from mycormarl.actions import rate_action
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.mycelium import (
    axisymmetric_density_from_biomass,
    fungal_biomass_for_colony_radius,
)
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


def _environment(fraction=0.2, *, mode="mixed", **fungus_overrides):
    return BaseMycorMarl(
        EnvConfig(
            dt=1.0,
            soil_radius_cm=2.0,
            soil_depth_cm=2.0,
            radial_interval_cm=0.5,
            depth_interval_cm=0.5,
            initial_solution_p_um=0.0,
            consumer_mode=mode,
        ),
        SpeciesParams(
            plant=PlantTraits(kappa_c=0.0, kleaf=0.0),
            fungus=FungusTraits(**{
                "irm_fraction": fraction,
                "initial_biomass": 1.0,
                "gamma_c": 1.0,
                "gamma_p": 2.0,
                "hyphal_radius": 1 / math.sqrt(math.pi),
                "hyphal_tissue_carbon_density": 1.0,
                "saturation_density": 3 / (2 * math.pi),
                "kappa_c": 0.2,
                **fungus_overrides,
            }),
        ),
    )


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.1, math.nan, math.inf, -math.inf])
def test_environment_rejects_invalid_irm_fraction(fraction):
    with pytest.raises(ValueError, match="irm_fraction"):
        _environment(fraction)


def test_host_free_fungus_requires_zero_irm_allocation():
    with pytest.raises(ValueError, match="irm_fraction.*fungus-only"):
        _environment(0.2, mode="fungus-only")
    _environment(0.0, mode="fungus-only")


def test_reset_allocates_external_geometry_without_changing_total_investment():
    """One gram total contains 0.8 g ERM, hence 0.8 cm in this fixture."""
    env = _environment()
    legacy = _environment(0.0)
    obs, state = env.reset(jax.random.PRNGKey(0))
    legacy_obs, legacy_state = legacy.reset(jax.random.PRNGKey(0))

    assert float(state.fungus_biomass[0]) == pytest.approx(1.0)
    assert float(jnp.sum(state.hyphae_length_density * env.cell_volumes)) == pytest.approx(0.8)
    assert float(jnp.sum(legacy_state.hyphae_length_density * legacy.cell_volumes)) == pytest.approx(1.0)
    assert float(state.fungus_c_pool[0]) == pytest.approx(0.2)
    assert jnp.array_equal(state.fungus_c_pool, legacy_state.fungus_c_pool)
    assert jnp.array_equal(state.fungus_p_pool, legacy_state.fungus_p_pool)
    assert jnp.array_equal(obs[FUNGUS], legacy_obs[FUNGUS])


def test_radius_inverse_includes_irm_without_rescaling_biomass_observation():
    """A unit-radius hemisphere needs 1 g ERM, or 8 g total at f=0.875."""
    env = _environment(0.875)
    traits = env.species.fungus
    biomass = fungal_biomass_for_colony_radius(1.0, traits)
    assert float(biomass) == pytest.approx(8.0)
    density = axisymmetric_density_from_biomass(biomass, traits, env.r_edges, env.z_edges)
    assert float(jnp.sum(density * env.cell_volumes)) == pytest.approx(1.0)
    assert float(density[0, 0]) == pytest.approx(3 / (2 * math.pi))
    assert jnp.all(density[2:, :] == 0.0)
    assert jnp.all(density[:, 2:] == 0.0)

    legacy = _environment(0.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    assert jnp.array_equal(env.get_obs(state)[FUNGUS], legacy.get_obs(state)[FUNGUS])


def test_partial_mortality_removes_irm_and_erm_proportionally_in_compiled_batches():
    """0.2/0.1 g total losses leave 0.64/0.72 cm ERM, exporting 0.4/0.2 mg P."""
    env = _environment(initial_c_pool=0.0, initial_p_pool=0.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    batch = jax.tree.map(lambda value: jnp.stack([value, value]), state)
    batch = batch.replace(fungus_c_pool=jnp.array([[0.0], [0.1]]))
    actions = {agent: rate_action(0.0, 0.0, 0.0) for agent in (PLANT, FUNGUS)}

    _, result, _, _, infos = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, None)))(
        jax.random.split(jax.random.PRNGKey(1), 2), batch, actions
    )

    assert jnp.allclose(result.fungus_biomass[:, 0], jnp.array([0.8, 0.9]))
    assert jnp.allclose(
        jnp.sum(result.hyphae_length_density * env.cell_volumes, axis=(1, 2)),
        jnp.array([0.64, 0.72]),
    )
    assert jnp.allclose(
        result.cumulative_fungus_p_mortality_loss_mg[:, 0], jnp.array([0.4, 0.2])
    )
    assert jnp.allclose(
        2 * result.fungus_biomass + result.cumulative_fungus_p_mortality_loss_mg, 2.0
    )
    assert jnp.allclose(infos[FUNGUS]["maint_c"], 0.2)
    assert jnp.allclose(infos[FUNGUS]["maint_c_used"][:, 0], jnp.array([0.0, 0.1]))
    assert jnp.all(result.fungus_c_pool == 0.0)
    assert jnp.all(result.fungus_p_pool == 0.0)
    assert not jnp.any(result.fungus_dead)


def test_growth_pays_for_total_tissue_but_only_erm_adds_soil_length():
    """Half of one paired resource unit grows 0.5 g total and 0.4 cm ERM."""
    env = _environment(kappa_c=0.0, initial_c_pool=1.0, initial_p_pool=2.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        PLANT: rate_action(0.0, 0.0, 0.0),
        FUNGUS: rate_action(0.0, math.log(2), 0.0),
    }
    _, result, _, _, infos = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert float(result.fungus_biomass[0]) == pytest.approx(1.5, rel=1e-5)
    assert float(jnp.sum(result.hyphae_length_density * env.cell_volumes)) == pytest.approx(1.2, rel=1e-5)
    assert float(result.fungus_c_pool[0]) == pytest.approx(0.5, rel=1e-5)
    assert float(result.fungus_p_pool[0]) == pytest.approx(1.0, rel=1e-5)
    assert float(infos[FUNGUS]["growth_c_used"][0]) == pytest.approx(0.5, rel=1e-5)
    assert float(infos[FUNGUS]["growth_p_used"][0]) == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize("mode", ["mixed", "plant-only"])
def test_terminal_or_absent_fungus_has_no_geometry_but_keeps_accounted_residual(mode):
    env = _environment(mode=mode, kappa_c=0.97, initial_c_pool=0.0, initial_p_pool=0.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {agent: rate_action(0.0, 0.0, 0.0) for agent in (PLANT, FUNGUS)}
    _, result, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert jnp.all(result.hyphae_length_density == 0.0)
    if mode == "mixed":
        assert bool(result.fungus_dead[0])
        assert float(result.fungus_biomass[0]) == pytest.approx(0.03, abs=1e-7)
        assert float(result.cumulative_fungus_p_mortality_loss_mg[0]) == pytest.approx(1.94)
    else:
        assert float(result.fungus_biomass[0]) == 0.0
        assert float(result.cumulative_fungus_p_mortality_loss_mg[0]) == 0.0


def test_clipping_external_geometry_does_not_remove_irm_or_external_mass_costs():
    original = _environment(initial_c_pool=0.0, initial_p_pool=0.0)
    env = BaseMycorMarl(
        original.config.replace(soil_radius_cm=0.5, soil_depth_cm=0.5),
        original.species,
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {agent: rate_action(0.0, 0.0, 0.0) for agent in (PLANT, FUNGUS)}
    _, result, _, _, infos = env.step_env(jax.random.PRNGKey(1), state, actions)

    # The whole cylindrical cell is inside the surviving ERM hemisphere:
    # (pi / 8 cm³) * (3 / (2*pi) cm/cm³) = 3/16 cm, far below total ERM length.
    assert float(jnp.sum(result.hyphae_length_density * env.cell_volumes)) == pytest.approx(0.1875)
    assert float(result.fungus_biomass[0]) == pytest.approx(0.8)
    assert float(infos[FUNGUS]["maint_c"][0]) == pytest.approx(0.2)
    assert float(result.cumulative_fungus_p_mortality_loss_mg[0]) == pytest.approx(0.4)
