"""Agreed three-rate biological behavior through public environment steps."""

import jax
import jax.numpy as jnp
import pytest

from mycormarl.actions import rate_action
from mycormarl.environments.base_mycor import BaseMycorMarl, PLANT, FUNGUS
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.fungus.traits import FungusTraits


def pool_environment(*, dt=1.0, max_steps=10, plant=None, fungus=None, **config):
    plant_traits = dict(initial_biomass=1., initial_c_pool=2., initial_p_pool=1.,
                        gamma_c=1., gamma_p=1., kappa_c=0., kappa_p=0.,
                        kleaf=0., jmax=0.)
    fungus_traits = dict(initial_biomass=1., initial_c_pool=0., initial_p_pool=0.,
                         gamma_c=1., gamma_p=1., kappa_c=0., kappa_p=0., jmax=0.)
    plant_traits.update(plant or {})
    fungus_traits.update(fungus or {})
    return BaseMycorMarl(
        EnvConfig(dt=dt, max_steps=max_steps, soil_radius_cm=.2, soil_depth_cm=.2,
                  radial_interval_cm=.1, depth_interval_cm=.1,
                  initial_solution_p_um=0., **config),
        SpeciesParams(PlantTraits(**plant_traits), FungusTraits(**fungus_traits)),
    )


def advance(env, plant_action, fungus_action=(0., 0., 0.)):
    _, initial = env.reset(jax.random.PRNGKey(0))
    result = env.step_env(jax.random.PRNGKey(1), initial,
                          {PLANT: jnp.array(plant_action), FUNGUS: jnp.array(fungus_action)})
    return initial, result


def test_reproduction_exports_only_stoichiometric_resources_and_preserves_surplus():
    env = pool_environment()
    initial, (_, final, rewards, _, info) = advance(env, (0., 0., 1.))

    # Analytic isolated first-order depletion of one P-limited output unit.
    assert float(rewards[PLANT]) == pytest.approx(.6321205588, rel=2e-6)
    assert float(final.plant_c_pool[0]) == pytest.approx(1.3678794412, rel=2e-6)
    assert float(final.plant_p_pool[0]) == pytest.approx(.3678794412, rel=2e-6)
    assert jnp.array_equal(final.plant_biomass, initial.plant_biomass)
    assert float(info[PLANT]['reproduction_c'][0]) == pytest.approx(.6321205588, rel=2e-6)
    assert float(info[PLANT]['reproduction_p'][0]) == pytest.approx(.6321205588, rel=2e-6)


@pytest.mark.parametrize('agent', [PLANT, FUNGUS])
@pytest.mark.parametrize('scale,expected_output', [(.5, 1.2642411176), (2., .3160602794)])
def test_reproductive_cost_scale_changes_output_without_changing_elemental_expenditure(agent, scale, expected_output):
    traits = dict(initial_c_pool=4., initial_p_pool=3., gamma_c=2., gamma_p=3.,
                  reproduction_cost_scale=scale)
    env = pool_environment(**{agent: traits})
    actions = {PLANT: (0., 0., 0.), FUNGUS: (0., 0., 0.)}
    actions[agent] = (0., 0., 1.)
    initial, (_, final, rewards, _, info) = advance(env, actions[PLANT], actions[FUNGUS])

    assert float(rewards[agent]) == pytest.approx(expected_output, rel=2e-6)
    assert float(info[agent]['reproduction_c'][0]) == pytest.approx(1.2642411176, rel=2e-6)
    assert float(info[agent]['reproduction_p'][0]) == pytest.approx(1.8963616764, rel=2e-6)
    assert jnp.array_equal(getattr(final, agent+'_biomass'), getattr(initial, agent+'_biomass'))


@pytest.mark.parametrize('agent', [PLANT, FUNGUS])
@pytest.mark.parametrize('scale', [0., -1., float('nan'), float('inf')])
def test_environment_rejects_invalid_reproductive_costs(agent, scale):
    with pytest.raises(ValueError, match='reproduction_cost_scale'):
        pool_environment(**{agent: dict(reproduction_cost_scale=scale)})


def test_public_action_constructor_and_environment_expose_three_rates():
    action = rate_action(.2, .7, .1)
    env = pool_environment()
    assert action.shape == env.action_spaces[PLANT].shape == (3,)
    _, (_, _, _, _, info) = advance(env, action)
    assert jnp.array_equal(info['transitions'][PLANT].requested_action, action)


def test_equal_trade_and_growth_hazards_compete_without_trade_priority():
    env = pool_environment(plant=dict(initial_c_pool=1., initial_p_pool=1.))
    _, (_, final, _, _, info) = advance(env, (1., 1., 0.))

    # Two equal first-order sinks consume an initially C-limited pool together.
    assert float(info[PLANT]['trade_out'][0]) == pytest.approx(.4323323584, rel=3e-6)
    assert float(info[PLANT]['growth'][0]) == pytest.approx(.4323323584, rel=3e-6)
    assert float(final.plant_c_pool[0]) == pytest.approx(.1353352832, rel=3e-6)
    assert float(final.plant_p_pool[0]) == pytest.approx(.5676676416, rel=3e-6)
    assert float(final.fungus_c_pool[0]) == pytest.approx(.4323323584, rel=3e-6)


@pytest.mark.parametrize('c,p,offered_c,offered_p,used', [
    (2.,.5,.8490904191,.2370452096,.2370452096),
    (1.,1.,.3242492688,.5371246344,.3242492688),
])
def test_growth_offers_preserve_elemental_limitation_information(c,p,offered_c,offered_p,used):
    env = pool_environment(plant=dict(initial_c_pool=c,initial_p_pool=p))
    _, (_, _, _, _, info) = advance(env,(1.,.75,.25))
    plant = info[PLANT]
    assert float(plant['growth_c_allocated'][0]) == pytest.approx(offered_c, abs=3e-6)
    assert float(plant['growth_p_allocated'][0]) == pytest.approx(offered_p, abs=3e-6)
    assert float(plant['growth_c_used'][0]) == pytest.approx(used, abs=3e-6)
    assert float(plant['growth_p_used'][0]) == pytest.approx(used, abs=3e-6)


@pytest.mark.parametrize('case,c,p,share,scale,trade,expected', [
    ('A', 1., 1., .25, 1., 0., (.4740904191, .1580301397, .3678794412, .3678794412, 0.)),
    ('B', 2., .5, .25, 1., 1., (.2370452096, .0790150699, .5518191618, .1839397206, 1.1321205588)),
    ('C', 1., 1., .25, 1., 1., (.3242492688, .1080830896, .1353352832, .5676676416, .4323323584)),
    ('D', 2., .2, .25, 1., 1., (.4133337997, .1377779332, .3818521706, .7159243635, 1.0670360965)),
    ('E', 1., 1., .75, 1., 1., (.1080830896, .3242492688, .1353352832, .5676676416, .4323323584)),
    ('F', 1., 1., .25, 2., 1., (.3242492688, .0540415448, .1353352832, .5676676416, .4323323584)),
])
def test_six_agreed_case_studies_match_independent_continuous_time_reference(case,c,p,share,scale,trade,expected):
    # Values from the independent DOP853 illustrative calculation, not the
    # production integration algorithm. D crosses a limiting-resource boundary.
    fungal = dict(initial_c_pool=.2, initial_p_pool=2.) if case == 'D' else {}
    env = pool_environment(plant=dict(initial_c_pool=c, initial_p_pool=p,
                                     reproduction_cost_scale=scale), fungus=fungal)
    fungal_action = (1., .75, .25) if case == 'D' else (0., 0., 0.)
    initial, (_, final, rewards, _, info) = advance(env, (trade,1-share,share), fungal_action)
    observed = (info[PLANT]['growth'][0], rewards[PLANT], final.plant_c_pool[0],
                final.plant_p_pool[0], info[PLANT]['trade_out'][0])
    assert tuple(map(float, observed)) == pytest.approx(expected, abs=2e-5, rel=2e-5)
    for agent, counterpart in ((PLANT, FUNGUS), (FUNGUS, PLANT)):
        t = getattr(env.species, agent)
        for resource, gamma in (('c',t.gamma_c), ('p',t.gamma_p)):
            before = getattr(initial, f'{agent}_{resource}_pool')[0]
            after = getattr(final, f'{agent}_{resource}_pool')[0]
            incoming = info[agent]['trade_in'][0] if (agent,resource) in ((PLANT,'p'),(FUNGUS,'c')) else 0.
            outgoing = info[agent]['trade_out'][0] if (agent,resource) in ((PLANT,'c'),(FUNGUS,'p')) else 0.
            use = info[agent][f'growth_{resource}_used'][0]+info[agent][f'reproduction_{resource}'][0]
            assert float(after+use+outgoing-incoming) == pytest.approx(float(before), abs=4e-6)
        assert float(info[agent]['trade_in'][0]) == pytest.approx(float(info[counterpart]['trade_out'][0]), abs=1e-7)


def test_environment_rejects_legacy_four_component_actions():
    env = pool_environment()
    with pytest.raises(ValueError, match=r'shape.*3'):
        advance(env, (0.,1.,0.,1.), (0.,0.,0.,0.))


def test_exhausted_allocation_budget_is_reported_instead_of_accepting_an_underresolved_result():
    env = pool_environment(dt=.05, allocation_max_substeps=2)
    _, (_, final, rewards, _, info) = advance(env, (6.907755, 131.24735, 0.))
    assert not bool(info[PLANT]['allocation_converged'])
    assert not jnp.all(jnp.isfinite(final.plant_c_pool))
    assert not jnp.isfinite(rewards[PLANT])


@pytest.mark.parametrize('field,value', [
    ('allocation_rate_step', 0.), ('allocation_rate_step', 1.01),
    ('allocation_rate_step', float('nan')), ('allocation_rate_step', float('inf')),
    ('allocation_max_substeps', 0), ('allocation_max_substeps', -1),
    ('allocation_max_substeps', 1.5), ('allocation_max_substeps', True),
])
def test_environment_validates_allocation_solver_controls(field, value):
    with pytest.raises(ValueError, match=field):
        pool_environment(**{field: value})


def test_fast_static_intensities_consume_the_expected_matched_pool_without_overdrawing():
    env = pool_environment(dt=.05, plant=dict(initial_c_pool=1., initial_p_pool=1.))
    _, (_, final, _, _, info) = advance(env, (6.90775528, 131.2473503, 0.))
    assert bool(info[PLANT]['allocation_converged'])
    assert float(info[PLANT]['growth'][0]) == pytest.approx(.94905, abs=3e-6)
    assert float(info[PLANT]['trade_out'][0]) == pytest.approx(.04995, abs=3e-6)
    assert float(final.plant_c_pool[0]) == pytest.approx(.001, abs=3e-6)
    assert float(final.plant_p_pool[0]) == pytest.approx(.05095, abs=3e-6)
    assert jnp.all(final.plant_c_pool >= 0.)


@pytest.mark.parametrize('c,p', [(0.,1.), (1.,0.), (0.,0.)])
def test_missing_essential_resource_prevents_growth_and_reproduction(c,p):
    env = pool_environment(plant=dict(initial_c_pool=c, initial_p_pool=p))
    initial, (_, final, rewards, _, info) = advance(env, (0.,1.,1.))
    assert float(info[PLANT]['growth'][0]) == 0.
    assert float(rewards[PLANT]) == 0.
    assert jnp.array_equal(final.plant_c_pool, initial.plant_c_pool)
    assert jnp.array_equal(final.plant_p_pool, initial.plant_p_pool)


def test_zero_actions_retain_all_free_resources():
    env = pool_environment()
    initial, (_, final, rewards, _, _) = advance(env, (0.,0.,0.))
    for agent in (PLANT,FUNGUS):
        for element in ('c','p'):
            assert jnp.array_equal(getattr(initial, f'{agent}_{element}_pool'),
                                   getattr(final, f'{agent}_{element}_pool'))
        assert float(rewards[agent]) == 0.


@pytest.mark.parametrize('mode,active,inactive', [('plant-only',PLANT,FUNGUS), ('fungus-only',FUNGUS,PLANT)])
def test_single_consumer_executes_biology_without_trading_with_absent_partner(mode,active,inactive):
    env = pool_environment(consumer_mode=mode, plant=dict(initial_c_pool=1.),
                           fungus=dict(initial_c_pool=1., initial_p_pool=1.))
    _, (_, final, rewards, _, info) = advance(env, (10.,.4,.6), (10.,.4,.6))
    assert float(rewards[active]) == pytest.approx(.3792723353, abs=2e-6)
    assert float(rewards[inactive]) == 0.
    assert float(info[active]['trade_out'][0]) == 0.
    assert not bool(info['transitions'][active].trade_executed)
    assert jnp.all(getattr(final,f'{inactive}_c_pool') == 0.)


def test_growth_guard_preserves_the_cost_of_blocked_growth_in_free_pools():
    env = pool_environment(plant=dict(biomass_cap=1.1))
    _, (_, final, _, _, info) = advance(env, (0.,10.,0.))
    assert float(final.plant_biomass[0]) == pytest.approx(1.1, abs=2e-7)
    assert float(info[PLANT]['growth'][0]) == pytest.approx(.1, abs=2e-7)
    assert float(final.plant_c_pool[0]) == pytest.approx(1.9, abs=2e-6)
    assert float(final.plant_p_pool[0]) == pytest.approx(.9, abs=2e-6)


def test_transfers_conserve_physical_carbon_with_unequal_species_costs():
    env = pool_environment(plant=dict(initial_c_pool=4., initial_p_pool=3., gamma_c=2., gamma_p=3.),
                           fungus=dict(gamma_c=.5, gamma_p=1.3))
    _, (_, final, _, _, info) = advance(env, (1.,1.,0.))
    assert float(info[PLANT]['growth'][0]) == pytest.approx(.6321205588, abs=3e-6)
    assert float(info[PLANT]['trade_out'][0]) == pytest.approx(2., abs=3e-6)
    assert float(final.fungus_c_pool[0]) == pytest.approx(2., abs=3e-6)


def test_coupled_allocation_refines_toward_the_same_limiting_switch_reference():
    def result(step):
        env = pool_environment(plant=dict(initial_c_pool=2., initial_p_pool=.2),
                               fungus=dict(initial_c_pool=.2, initial_p_pool=2.),
                               allocation_rate_step=step)
        _, (_, state, _, _, _) = advance(env, (1.,.75,.25), (1.,.75,.25))
        return float(state.plant_c_pool[0])
    reference = .3818521706
    assert abs(result(.005)-reference) < abs(result(.1)-reference)
    assert result(.005) == pytest.approx(reference, abs=2e-6)


def test_three_rate_execution_is_jittable_and_vectorises_at_public_step():
    env = pool_environment(plant=dict(initial_c_pool=1.))
    _, initial = env.reset(jax.random.PRNGKey(0))
    def step(action):
        return env.step_env(jax.random.PRNGKey(1), initial,
                            {PLANT:action,FUNGUS:jnp.zeros(3)})
    _, states, rewards, _, info = jax.jit(jax.vmap(step))(jnp.array([[0.,0.,1.], [0.,1.,0.]]))
    assert info['transitions'][PLANT].requested_action.shape == (2,3)
    assert rewards[PLANT] == pytest.approx([.6321205588,0.], abs=2e-6)
    assert jnp.all(jnp.isfinite(states.plant_c_pool))
    assert jnp.all(states.plant_c_pool >= 0.)
