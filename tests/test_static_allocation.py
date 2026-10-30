"""Declared static experiment strategies through the public rollout seam."""

import pytest

from mycormarl import static_rollout
from mycormarl.environments.base_mycor import BaseMycorMarl, PLANT, FUNGUS
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.fungus.traits import FungusTraits


def environment(*, reproduction_cost_scale=1., **config):
    settings = dict(dt=.05, max_steps=1, soil_radius_cm=.2, soil_depth_cm=.2,
                    radial_interval_cm=.1, depth_interval_cm=.1, initial_solution_p_um=0.)
    settings.update(config)
    return BaseMycorMarl(EnvConfig(**settings), SpeciesParams(
        PlantTraits(initial_biomass=1., initial_c_pool=1., initial_p_pool=1.,
                    gamma_c=1., gamma_p=1., kappa_c=0., kappa_p=0., kleaf=0., jmax=0.,
                    reproduction_cost_scale=reproduction_cost_scale),
        FungusTraits(initial_biomass=1., initial_c_pool=0., initial_p_pool=0.,
                     gamma_c=1., gamma_p=1., kappa_c=0., kappa_p=0., jmax=0.),
    ))


def strategy(env, **overrides):
    settings = dict(interval_days=.05, trade_shares={PLANT:.05,FUNGUS:0.},
                    reproductive_shares={PLANT:0.,FUNGUS:0.}, seed=7)
    settings.update(overrides)
    return static_rollout.evaluate_static_allocation(env, **settings)


def test_static_experiment_derives_interval_dependent_intensity_and_records_executed_strategy():
    result = strategy(environment())
    metadata = result['static_allocation']
    assert metadata['total_rate_per_day'] == pytest.approx(138.1551055796, rel=1e-10)
    assert metadata['target_residual_fraction'] == .001
    assert metadata['interval_days'] == metadata['policy_interval_days'] == .05
    assert result['commanded_rate_actions'][PLANT] == pytest.approx([6.907755279,131.247350301,0.], abs=2e-5)
    assert metadata['strategies'][PLANT] == dict(trade_share=.05, reproductive_share=0., reproduction_cost_scale=1.)
    assert result['cumulative_growth'][PLANT] == pytest.approx(.94905, abs=3e-6)
    assert result['transfers']['plant_c_out'] == pytest.approx(.04995, abs=3e-6)
    assert result['c_pools'][PLANT] == pytest.approx(.001, abs=3e-6)


def test_declared_static_interval_must_match_the_actual_environment_interval():
    with pytest.raises(ValueError, match='interval'):
        strategy(environment(), interval_days=.1)


@pytest.mark.parametrize('field', ['trade_shares','reproductive_shares'])
@pytest.mark.parametrize('value', [-.01,1.01,float('nan'),float('inf')])
def test_static_experiment_rejects_invalid_shares_instead_of_sanitising_them(field,value):
    with pytest.raises(ValueError, match='shares'):
        strategy(environment(), **{field:{PLANT:value,FUNGUS:0.}})


@pytest.mark.parametrize('field', ['trade_shares','reproductive_shares'])
def test_static_experiment_requires_an_explicit_strategy_for_both_agents(field):
    with pytest.raises(ValueError, match='shares'):
        strategy(environment(), **{field:{PLANT:0.}})


@pytest.mark.parametrize('epsilon', [0.,-1.,1.,1.01,float('nan'),float('inf')])
def test_static_experiment_requires_a_finite_residual_fraction_strictly_between_zero_and_one(epsilon):
    with pytest.raises(ValueError, match='target_residual_fraction'):
        strategy(environment(), target_residual_fraction=epsilon)


def test_static_experiment_rejects_reusing_a_daily_total_at_a_different_interval():
    with pytest.raises(ValueError, match='total_rate_per_day'):
        strategy(environment(), total_rate_per_day=6.907755279)


def test_static_experiment_records_reproductive_exports_separately_from_output():
    result = strategy(environment(reproduction_cost_scale=2.),
                      reproductive_shares={PLANT:.25,FUNGUS:0.})
    assert result['cumulative_reproductive_fitness'][PLANT] == pytest.approx(.11863125, abs=3e-6)
    assert result['resource_accounting'][PLANT]['c']['reproduction'] == pytest.approx(.2372625, abs=3e-6)
    assert result['resource_accounting'][PLANT]['p']['reproduction'] == pytest.approx(.2372625, abs=3e-6)
    assert result['static_allocation']['strategies'][PLANT]['reproduction_cost_scale'] == 2.
    assert abs(result['p_accounting_residual']) < 4e-6


def test_static_experiment_rejects_an_insufficient_allocation_budget_before_execution():
    with pytest.raises(ValueError, match='allocation.*budget'):
        strategy(environment(allocation_max_substeps=1))


def test_static_experiment_rejects_intensity_that_cannot_be_represented_as_a_rate():
    with pytest.raises(ValueError, match='total.*rate'):
        strategy(environment(dt=1e-40), interval_days=1e-40)


def test_internal_refinement_keeps_the_declared_strategy_and_records_solver_controls():
    ordinary = strategy(environment())
    refined = strategy(environment(allocation_rate_step=.01))
    assert ordinary['commanded_rate_actions'] == refined['commanded_rate_actions']
    assert ordinary['static_allocation']['total_rate_per_day'] == refined['static_allocation']['total_rate_per_day']
    assert refined['cumulative_growth'] == pytest.approx(ordinary['cumulative_growth'], abs=3e-6)
    assert refined['static_allocation']['allocation_solver'] == dict(
        rate_step=.01, max_substeps=65536)


@pytest.mark.parametrize('interval,expected_k', [(.025,276.3102111593),(.05,138.1551055796),(.1,69.0775527898)])
def test_static_screen_interval_changes_intensity_but_preserves_nominal_one_step_turnover(interval,expected_k):
    result = strategy(environment(dt=interval), interval_days=interval)
    assert result['static_allocation']['total_rate_per_day'] == pytest.approx(expected_k, rel=1e-10)
    assert sum(result['commanded_rate_actions'][PLANT]) == pytest.approx(expected_k, rel=1e-7)
    assert result['cumulative_growth'][PLANT] == pytest.approx(.94905, abs=3e-6)
    assert result['c_pools'][PLANT] == pytest.approx(.001, abs=3e-6)


@pytest.mark.parametrize('p,a,growth,reproduction,transfer', [
    (0.,0.,.999,0.,0.), (0.,1.,0.,.999,0.), (1.,1.,0.,0.,.999)])
def test_exact_static_share_endpoints(p,a,growth,reproduction,transfer):
    result = strategy(environment(), trade_shares={PLANT:p,FUNGUS:0.},
                      reproductive_shares={PLANT:a,FUNGUS:0.})
    assert result['cumulative_growth'][PLANT] == pytest.approx(growth, abs=3e-6)
    assert result['cumulative_reproductive_fitness'][PLANT] == pytest.approx(reproduction, abs=3e-6)
    assert result['transfers']['plant_c_out'] == pytest.approx(transfer, abs=3e-6)
    assert result['status'] == 'completed'


@pytest.mark.parametrize('interval', [0.,-1.,float('nan'),float('inf')])
def test_static_experiment_rejects_invalid_intervals(interval):
    with pytest.raises(ValueError, match='interval'):
        strategy(environment(), interval_days=interval)


@pytest.mark.parametrize('total', [0.,-1.,float('nan'),float('inf')])
def test_static_experiment_rejects_invalid_supplied_total(total):
    with pytest.raises(ValueError, match='total_rate_per_day'):
        strategy(environment(), total_rate_per_day=total)


def test_matching_manual_total_is_recorded_with_validation_tolerance_and_json_roundtrips():
    import json
    result = strategy(environment(), total_rate_per_day=138.1551055796)
    restored = json.loads(json.dumps(result, allow_nan=False))
    metadata = restored['static_allocation']
    assert metadata['supplied_total_rate_per_day'] == 138.1551055796
    assert metadata['total_rate_validation_tolerance'] == dict(relative=1e-7,absolute_per_day=1e-10)


def test_static_screen_rejects_a_different_policy_interval():
    from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
    wrapped = PolicyIntervalMycorMarl(environment(), decision_interval_days=.1,max_episode_steps=1)
    with pytest.raises(ValueError, match='interval'):
        strategy(wrapped)


def test_small_intervals_must_match_relatively_rather_than_with_a_fixed_absolute_allowance():
    with pytest.raises(ValueError, match='interval'):
        strategy(environment(dt=2e-13), interval_days=1e-13)


def test_raw_fixed_rate_reference_does_not_require_static_normalisation():
    result = static_rollout.evaluate_static_policy(environment(),
        {PLANT:[0.,1.,0.],FUNGUS:[0.,0.,0.]},seed=7)
    assert result['cumulative_growth'][PLANT] == pytest.approx(.0487705755, abs=2e-6)
    assert 'static_allocation' not in result


def test_raw_reference_rejects_legacy_four_rate_actions():
    with pytest.raises(ValueError, match='invalid Rate action'):
        static_rollout.evaluate_static_policy(environment(),
            {PLANT:[0.,1.,0.,0.],FUNGUS:[0.,0.,0.,0.]},seed=7)
