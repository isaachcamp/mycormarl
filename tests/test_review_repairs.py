"""Regression tests for defects found in the post-implementation audit."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from mycormarl.actions import physical_action
from mycormarl.algos.ppo import make_train
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


ROOT = Path(__file__).resolve().parents[1]


def _small_config(mode="mixed"):
    return EnvConfig(
        max_steps=2,
        dt=0.05,
        soil_radius_cm=0.2,
        soil_depth_cm=0.2,
        radial_interval_cm=0.1,
        depth_interval_cm=0.1,
        topsoil_depth_cm=0.2,
        consumer_mode=mode,
    )


def _species():
    return SpeciesParams(
        plant=PlantTraits(
            initial_biomass=1e-4,
            initial_c_pool=0.01,
            initial_p_pool=0.01,
            kappa_c=0.0,
            kappa_p=0.0,
        ),
        fungus=FungusTraits(
            initial_biomass=1e-8,
            initial_c_pool=0.01,
            initial_p_pool=0.01,
            kappa_c=0.0,
            kappa_p=0.0,
        ),
    )


def test_documented_example_script_and_main_entry_point_run():
    """Both commands advertised by the handoff must execute successfully."""
    for entry_point in (ROOT / "scripts/phosphate_examples.py", ROOT / "main.py"):
        completed = subprocess.run(
            [sys.executable, str(entry_point), "--mode", "plant-only"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert '"mode": "plant-only"' in completed.stdout


@pytest.mark.parametrize(
    ("mode", "inactive", "active"),
    (("plant-only", FUNGUS, PLANT), ("fungus-only", PLANT, FUNGUS)),
)
def test_independent_consumer_mode_keeps_absent_partner_dormant(mode, inactive, active):
    """Several complete transitions cannot activate or trade with the absent partner."""
    env = BaseMycorMarl(_small_config(mode), _species())
    _, state = env.reset(jax.random.PRNGKey(0))
    aggressive = {
        PLANT: physical_action(1.0, 1.0, 1.0, 1.0),
        FUNGUS: physical_action(1.0, 1.0, 1.0, 1.0),
    }

    for step in range(2):
        _, state, rewards, _, _ = env.step_env(
            jax.random.PRNGKey(step + 1), state, aggressive
        )
        assert rewards[inactive] == pytest.approx(0.0)

    assert getattr(state, f"{inactive}_biomass")[0] == pytest.approx(0.0)
    assert getattr(state, f"{inactive}_c_pool")[0] == pytest.approx(0.0)
    assert getattr(state, f"{inactive}_p_pool")[0] == pytest.approx(0.0)
    density = state.hyphae_length_density if inactive == FUNGUS else state.root_length_density
    assert jnp.all(density == 0.0)
    assert getattr(state, f"{active}_biomass")[0] > 0.0


def test_absorbing_death_removes_real_root_geometry_and_uptake():
    """A dead active plant retains biomass but has no production soil footprint."""
    env = BaseMycorMarl(_small_config(), _species())
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(plant_dead=jnp.array(True))
    biomass_before = state.plant_biomass.copy()
    c_before = state.plant_c_pool.copy()
    p_before = state.plant_p_pool.copy()
    actions = {
        PLANT: physical_action(1.0, 1.0, 1.0, 1.0),
        FUNGUS: physical_action(0.0, 0.0, 0.0, 1.0),
    }

    _, next_state, rewards, dones, infos = env.step_env(
        jax.random.PRNGKey(1), state, actions
    )

    assert dones[PLANT]
    assert rewards[PLANT] == pytest.approx(0.0)
    assert infos[PLANT]["growth"][0] == pytest.approx(0.0)
    assert next_state.plant_biomass == pytest.approx(biomass_before)
    assert next_state.plant_c_pool == pytest.approx(c_before)
    assert next_state.plant_p_pool == pytest.approx(p_before)
    assert jnp.all(next_state.root_length_density == 0.0)


def test_ppo_stack_accepts_current_agent_identifiers():
    """A minimal PPO update uses the environment's plant/fungus keys."""
    env = BaseMycorMarl(_small_config(), _species())
    cfg = SimpleNamespace(
        TOTAL_TIMESTEPS=2,
        NUM_STEPS=2,
        NUM_ENVS=1,
        NUM_ACTORS=2,
        NUM_MINIBATCHES=1,
        UPDATE_EPOCHS=1,
        LR=2.5e-4,
        ACTIVATION="tanh",
        GAMMA=0.995,
        GAE_LAMBDA=0.95,
        CLIP_EPS=0.2,
        VF_COEF=0.5,
        ENT_COEF=0.01,
    )

    output = make_train(env, cfg)(jax.random.PRNGKey(0))

    assert set(output["runner_state"][0]) == {PLANT, FUNGUS}
    assert output["trajectories"][0].reward.shape[:2] == (1, 2)


@pytest.mark.parametrize(
    ("trait_type", "field", "value"),
    (
        (PlantTraits, "initial_c_pool", -1.0),
        (PlantTraits, "initial_p_pool", float("nan")),
        (PlantTraits, "kappa_c", -0.1),
        (PlantTraits, "kappa_p", float("inf")),
        (PlantTraits, "death_fraction", 1.1),
        (PlantTraits, "biomass_cap", 0.0),
        (PlantTraits, "kleaf", -0.1),
        (PlantTraits, "amass", float("nan")),
        (FungusTraits, "initial_c_pool", -1.0),
        (FungusTraits, "initial_p_pool", float("inf")),
        (FungusTraits, "kappa_c", -0.1),
        (FungusTraits, "kappa_p", float("nan")),
        (FungusTraits, "death_fraction", -0.1),
    ),
)
def test_invalid_state_forming_traits_fail_at_environment_construction(
    trait_type, field, value
):
    """Invalid pools and rates cannot enter the canonical environment state."""
    species = _species()
    invalid = trait_type(**{field: value})
    species = species.replace(
        plant=invalid if trait_type is PlantTraits else species.plant,
        fungus=invalid if trait_type is FungusTraits else species.fungus,
    )

    with pytest.raises(ValueError, match=field):
        BaseMycorMarl(_small_config(), species)


def test_unknown_consumer_mode_is_rejected():
    """Independent-consumer configuration accepts only explicit supported modes."""
    with pytest.raises(ValueError, match="consumer_mode"):
        BaseMycorMarl(_small_config("unknown"), _species())
