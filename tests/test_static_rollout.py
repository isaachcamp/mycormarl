from mycormarl.params import SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.fungus.traits import FungusTraits

def test_public_generic_static_seam():
    from mycormarl.static_rollout import evaluate_static_policy
    from mycormarl.environments.base_mycor import BaseMycorMarl
    from mycormarl.params import EnvConfig
    import jax.numpy as jnp
    env = BaseMycorMarl(EnvConfig(max_steps=2, dt=.05, soil_radius_cm=1., soil_depth_cm=1., radial_interval_cm=.5, depth_interval_cm=.5), SpeciesParams(PlantTraits(), FungusTraits()))
    result = evaluate_static_policy(env, {a:jnp.asarray([0., .1, .2, .3]) for a in env.agents}, seed=7)
    assert result['steps'] == 2
    assert abs(result['p_accounting_residual']) < 1e-5


def test_static_rollout_rejects_invalid_physical_actions():
    from mycormarl.static_rollout import evaluate_static_policy
    from mycormarl.environments.base_mycor import BaseMycorMarl
    from mycormarl.params import EnvConfig
    import pytest
    env = BaseMycorMarl(EnvConfig(max_steps=1, dt=.05, soil_radius_cm=1., soil_depth_cm=1.), SpeciesParams(PlantTraits(), FungusTraits()))
    with pytest.raises(ValueError, match="invalid Rate action"):
        evaluate_static_policy(env, {"plant": [-1.,0.,0.,0.], "fungus": [0.,0.,0.,0.]}, seed=7)


def test_batched_static_rollout_rejects_mixed_dynamics():
    from mycormarl.static_rollout import evaluate_batched_static_policy
    from mycormarl.environments.base_mycor import BaseMycorMarl
    from mycormarl.params import EnvConfig
    import pytest
    envs = [BaseMycorMarl(EnvConfig(max_steps=1, dt=dt, soil_radius_cm=1., soil_depth_cm=1.), SpeciesParams(PlantTraits(), FungusTraits())) for dt in (.025,.05)]
    with pytest.raises(ValueError, match="homogeneous"):
        evaluate_batched_static_policy(envs, {"plant": [0.,0.,0.,0.], "fungus": [0.,0.,0.,0.]}, seed=7)
