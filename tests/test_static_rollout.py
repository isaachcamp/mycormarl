from mycormarl.params import SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.fungus.traits import FungusTraits
import pytest

def test_public_generic_static_seam():
    from mycormarl.static_rollout import evaluate_static_policy
    from mycormarl.environments.base_mycor import BaseMycorMarl
    from mycormarl.params import EnvConfig
    import jax.numpy as jnp
    env = BaseMycorMarl(EnvConfig(max_steps=2, dt=.05, soil_radius_cm=1., soil_depth_cm=1., radial_interval_cm=.5, depth_interval_cm=.5), SpeciesParams(PlantTraits(), FungusTraits()))
    result = evaluate_static_policy(env, {a:jnp.asarray([0., .1, .2]) for a in env.agents}, seed=7)
    assert result['steps'] == 2
    assert abs(result['p_accounting_residual']) < 1e-5


def test_static_rollout_rejects_invalid_physical_actions():
    from mycormarl.static_rollout import evaluate_static_policy
    from mycormarl.environments.base_mycor import BaseMycorMarl
    from mycormarl.params import EnvConfig
    import pytest
    env = BaseMycorMarl(EnvConfig(max_steps=1, dt=.05, soil_radius_cm=1., soil_depth_cm=1.), SpeciesParams(PlantTraits(), FungusTraits()))
    with pytest.raises(ValueError, match="invalid Rate action"):
        evaluate_static_policy(env, {"plant": [-1.,0.,0.], "fungus": [0.,0.,0.]}, seed=7)


def test_batched_static_rollout_rejects_mixed_dynamics():
    from mycormarl.static_rollout import evaluate_batched_static_policy
    from mycormarl.environments.base_mycor import BaseMycorMarl
    from mycormarl.params import EnvConfig
    import pytest
    envs = [BaseMycorMarl(EnvConfig(max_steps=1, dt=dt, soil_radius_cm=1., soil_depth_cm=1.), SpeciesParams(PlantTraits(), FungusTraits())) for dt in (.025,.05)]
    with pytest.raises(ValueError, match="homogeneous"):
        evaluate_batched_static_policy(envs, {"plant": [0.,0.,0.], "fungus": [0.,0.,0.]}, seed=7)


def test_static_rollout_handles_zero_soil_inventory():
    from mycormarl.static_rollout import evaluate_static_policy
    from mycormarl.environments.base_mycor import BaseMycorMarl
    from mycormarl.params import EnvConfig
    env = BaseMycorMarl(EnvConfig(max_steps=1, dt=.05, soil_radius_cm=.2, soil_depth_cm=.2, initial_solution_p_um=0.), SpeciesParams(PlantTraits(), FungusTraits()))
    result = evaluate_static_policy(env, {a:[0.,0.,0.] for a in env.agents}, seed=7)
    assert result["soil_inventory_initial"] == 0.
    assert result["uniform_initial_p"]
    assert abs(result["p_accounting_residual"]) < 1e-5


def test_raw_static_rollout_rejects_and_stops_a_failed_allocation_transition():
    from mycormarl.static_rollout import evaluate_static_policy
    from test_stoichiometric_actions import pool_environment
    env = pool_environment(dt=.05,allocation_max_substeps=1)
    result = evaluate_static_policy(env,{'plant':[0.,1.,0.],'fungus':[0.,0.,0.]},seed=7)
    assert result['status'] == 'rejected'
    assert any('allocation' in reason for reason in result['rejection_reasons'])
    assert result['steps'] == 1


def test_batched_static_rollout_rejects_and_stops_failed_allocation_transitions():
    from mycormarl.static_rollout import evaluate_batched_static_policy
    from test_stoichiometric_actions import pool_environment
    envs = [pool_environment(dt=.05,allocation_max_substeps=1) for _ in range(2)]
    results = evaluate_batched_static_policy(envs,{'plant':[0.,1.,0.],'fungus':[0.,0.,0.]},seed=7)
    for result in results:
        assert result['status'] == 'rejected'
        assert any('allocation' in reason for reason in result['rejection_reasons'])
        assert result['steps'] == 1


def test_static_inventory_accounts_for_retained_dead_structure_in_single_and_batched_runs():
    from mycormarl.static_rollout import evaluate_static_policy,evaluate_batched_static_policy
    from test_stoichiometric_actions import pool_environment
    import pytest
    env = pool_environment(dt=.05,max_steps=1,
        plant=dict(initial_c_pool=0.,kappa_c=10.,death_fraction=.9))
    actions = {'plant':[0.,0.,0.],'fungus':[0.,0.,0.]}
    for result in (evaluate_static_policy(env,actions,seed=7),
                   evaluate_batched_static_policy([env],actions,seed=7)[0]):
        assert result['biomass']['plant'] == pytest.approx(.5)
        assert result['final_living_biomass']['plant'] == 0.
        assert result['p_loss_or_export_counters']['plant_mortality'] == pytest.approx(.5)
        assert result['p_accounting_residual'] == pytest.approx(0.,abs=1e-6)


@pytest.mark.parametrize('mode',['mixed','plant-only','fungus-only'])
def test_static_runs_do_not_classify_an_absent_counterpart_as_a_biological_failure(mode):
    from mycormarl.static_rollout import evaluate_static_policy,evaluate_batched_static_policy
    from test_stoichiometric_actions import pool_environment
    env = pool_environment(dt=.05,max_steps=1,consumer_mode=mode)
    actions = {'plant':[0.,0.,0.],'fungus':[0.,0.,0.]}
    for result in (evaluate_static_policy(env,actions,seed=7),
                   evaluate_batched_static_policy([env],actions,seed=7)[0]):
        assert result['status'] == 'completed',result['rejection_reasons']


def test_limitation_trace_serializes_carbon_and_phosphorus_acquisition_in_the_correct_columns():
    from mycormarl.static_rollout import evaluate_static_policy
    from test_stoichiometric_actions import pool_environment
    env = pool_environment(dt=.05,max_steps=1,fungus=dict(initial_p_pool=1.))
    result = evaluate_static_policy(env,{'plant':[1.,0.,0.],'fungus':[1.,0.,0.]},
                                    seed=7,record_limitation_trace=True)
    agents = result['limitation_trace'][0]['agents']
    assert agents['plant']['acquired_c'] == 0.
    assert agents['plant']['acquired_p'] == pytest.approx(.0487705755,abs=2e-6)
    assert agents['fungus']['acquired_c'] == pytest.approx(.0975411510,abs=2e-6)
    assert agents['fungus']['acquired_p'] == 0.


def test_first_trace_row_has_no_prior_acquisition_maintenance_fraction():
    from mycormarl.static_rollout import evaluate_static_policy
    from test_stoichiometric_actions import pool_environment
    env = pool_environment(dt=.05,max_steps=1,plant=dict(kappa_c=.1,kappa_p=.1))
    result = evaluate_static_policy(env,{'plant':[0.,0.,0.],'fungus':[0.,0.,0.]},
                                    seed=7,record_limitation_trace=True)
    plant = result['limitation_trace'][0]['agents']['plant']
    assert plant['maintenance_c_used'] == pytest.approx(.005)
    assert plant['maintenance_fraction_of_prior_acquired_c'] == 0.
    assert plant['maintenance_fraction_of_prior_acquired_p'] == 0.


def test_static_runs_stop_and_reject_contact_with_the_numerical_biomass_guard():
    from mycormarl.static_rollout import evaluate_static_policy,evaluate_batched_static_policy
    from test_stoichiometric_actions import pool_environment
    env = pool_environment(dt=1.,max_steps=2,
        plant=dict(initial_biomass=.99,biomass_cap=1.,initial_c_pool=.04,kappa_c=.02))
    actions = {'plant':[0.,10.,0.],'fungus':[0.,0.,0.]}
    for result in (evaluate_static_policy(env,actions,seed=7),
                   evaluate_batched_static_policy([env],actions,seed=7)[0]):
        assert result['status'] == 'rejected'
        assert any('guard' in reason for reason in result['rejection_reasons'])
        assert result['steps'] == 1


def test_batched_guard_stop_does_not_count_death_from_discarded_later_dynamics():
    from mycormarl.static_rollout import evaluate_batched_static_policy
    from test_stoichiometric_actions import pool_environment
    env = pool_environment(dt=1.,max_steps=2,
        plant=dict(initial_biomass=.99,biomass_cap=1.,initial_c_pool=.04,kappa_c=.02,death_fraction=.999))
    result = evaluate_batched_static_policy([env],{'plant':[0.,10.,0.],'fungus':[0.,0.,0.]},seed=7)[0]
    assert result['steps'] == 1
    assert result['biological_deaths']['plant'] == 0
