import jax
import pytest

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


def test_reset_derives_active_pools_from_one_maintenance_timestep():
    """Unspecified pools fund exactly one configured maintenance payment."""
    env = BaseMycorMarl(
        config=EnvConfig(
            dt=0.125,
            soil_radius_cm=1.0,
            soil_depth_cm=1.0,
            radial_interval_cm=0.5,
            depth_interval_cm=0.5,
        ),
        species=SpeciesParams(
            plant=PlantTraits(
                initial_biomass=0.02,
                kappa_c=0.4,
                kappa_p=0.1,
            ),
            fungus=FungusTraits(
                initial_biomass=0.01,
                kappa_c=0.2,
                kappa_p=0.3,
            ),
        ),
    )

    _, state = env.reset(jax.random.PRNGKey(0))

    assert state.plant_c_pool[0] == pytest.approx(0.001)
    assert state.plant_p_pool[0] == pytest.approx(0.00025)
    assert state.fungus_c_pool[0] == pytest.approx(0.00025)
    assert state.fungus_p_pool[0] == pytest.approx(0.000375)


def test_reset_preserves_explicit_pool_overrides_and_zero_maintenance():
    """Explicit pools override derivation and zero maintenance derives zero."""
    env = BaseMycorMarl(
        config=EnvConfig(
            dt=0.125,
            soil_radius_cm=1.0,
            soil_depth_cm=1.0,
            radial_interval_cm=0.5,
            depth_interval_cm=0.5,
        ),
        species=SpeciesParams(
            plant=PlantTraits(
                initial_biomass=0.02,
                initial_c_pool=0.7,
                initial_p_pool=0.8,
                kappa_c=0.0,
                kappa_p=0.0,
            ),
            fungus=FungusTraits(
                initial_biomass=0.01,
                kappa_c=0.0,
                kappa_p=0.0,
            ),
        ),
    )

    _, state = env.reset(jax.random.PRNGKey(0))

    assert state.plant_c_pool[0] == pytest.approx(0.7)
    assert state.plant_p_pool[0] == pytest.approx(0.8)
    assert state.fungus_c_pool[0] == pytest.approx(0.0)
    assert state.fungus_p_pool[0] == pytest.approx(0.0)


def test_reset_keeps_an_inactive_partner_poolless_even_with_overrides():
    """An absent partner remains dormant rather than receiving an override."""
    env = BaseMycorMarl(
        config=EnvConfig(
            consumer_mode="plant-only",
            soil_radius_cm=1.0,
            soil_depth_cm=1.0,
            radial_interval_cm=0.5,
            depth_interval_cm=0.5,
        ),
        species=SpeciesParams(
            plant=PlantTraits(),
            fungus=FungusTraits(initial_c_pool=0.7, initial_p_pool=0.8),
        ),
    )

    _, state = env.reset(jax.random.PRNGKey(0))

    assert state.fungus_c_pool[0] == pytest.approx(0.0)
    assert state.fungus_p_pool[0] == pytest.approx(0.0)
