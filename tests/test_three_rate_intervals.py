"""Integrated three-rate facts through public held-interval steps."""

import jax
import jax.numpy as jnp
import pytest

from mycormarl.environments.base_mycor import PLANT, FUNGUS
from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
from test_stoichiometric_actions import pool_environment


def step(env, actions):
    _, state = env.reset(jax.random.PRNGKey(0))
    return env.step_env(jax.random.PRNGKey(1), state, actions)


def test_held_reward_diagnostic_sums_realised_output_at_nondefault_cost_scale():
    numerical = pool_environment(dt=.05,plant=dict(initial_c_pool=1.,reproduction_cost_scale=2.))
    held = PolicyIntervalMycorMarl(numerical,decision_interval_days=.1,max_episode_steps=5)
    _, _, rewards, _, info = step(held,{PLANT:jnp.array([0.,0.,1.]),FUNGUS:jnp.zeros(3)})
    assert float(rewards[PLANT]) == pytest.approx(.0475812910,abs=2e-6)
    assert float(info[PLANT]['reproductive_fitness_from_components'][0]) == pytest.approx(.0475812910,abs=2e-6)
    assert float(info[PLANT]['reproduction_c'][0]) == pytest.approx(.0951625820,abs=2e-6)


def test_interval_preserves_execution_before_death_in_a_later_numerical_step():
    numerical = pool_environment(dt=.05,plant=dict(initial_c_pool=.6,kappa_c=10.,death_fraction=.9))
    held = PolicyIntervalMycorMarl(numerical,decision_interval_days=.1,max_episode_steps=5)
    actions = {PLANT:jnp.array([.3,.1,.2]),FUNGUS:jnp.zeros(3)}
    _, final, rewards, _, info = step(held,actions)
    transition = info['transitions'][PLANT]
    assert bool(final.plant_dead[0])
    assert transition.operational_at_start
    assert not transition.operational_at_end
    assert float(rewards[PLANT]) > 0.
    assert transition.allocation_executed
    assert transition.trade_executed
    assert transition.realised_action == pytest.approx(actions[PLANT])


def test_interval_integrated_flows_match_the_identical_numerical_partition():
    numerical = pool_environment(dt=.05,
        plant=dict(kappa_c=.1,kappa_p=.05,reproduction_cost_scale=2.),
        fungus=dict(initial_c_pool=.2,initial_p_pool=2.,kappa_c=.02,kappa_p=.01,reproduction_cost_scale=.5))
    held = PolicyIntervalMycorMarl(numerical,decision_interval_days=.1,max_episode_steps=5)
    actions = {PLANT:jnp.array([.3,.1,.2]),FUNGUS:jnp.array([.4,.2,.1])}
    _, final, rewards, _, info = step(held,actions)
    _, manual = numerical.reset(jax.random.PRNGKey(0))
    rows, totals = [], {PLANT:0.,FUNGUS:0.}
    for i in range(2):
        _, manual, reward, _, row = numerical.step_env(jax.random.PRNGKey(i+1),manual,actions)
        rows.append(row)
        totals = {a:totals[a]+reward[a] for a in totals}
    for a in (PLANT,FUNGUS):
        assert rewards[a] == pytest.approx(totals[a],abs=2e-6)
        for field in ('growth_c_allocated','growth_p_allocated','growth_c_used','growth_p_used',
                      'maint_c','maint_p','maint_c_used','maint_p_used','c_deficit','p_deficit',
                      'maintenance_p_required_mg','growth','reproduction_c','reproduction_p',
                      'trade_in','trade_out','direct_p_uptake_mg','reproductive_fitness_from_components',
                      'allocation_substeps'):
            assert info[a][field] == pytest.approx(rows[0][a][field]+rows[1][a][field],abs=2e-6), field
    for actual,expected in zip(jax.tree.leaves(final),jax.tree.leaves(manual),strict=True):
        assert actual == pytest.approx(expected,abs=2e-6)


@pytest.mark.parametrize('horizon',[1,3,4])
def test_interval_freezes_at_the_numerical_horizon_including_a_short_final_interval(horizon):
    numerical = pool_environment(dt=.05,max_steps=horizon,plant=dict(initial_c_pool=1.))
    held = PolicyIntervalMycorMarl(numerical,decision_interval_days=.1,max_episode_steps=(horizon+1)//2)
    actions = {PLANT:jnp.array([.2,.3,.4]),FUNGUS:jnp.zeros(3)}
    _, grouped = held.reset(jax.random.PRNGKey(0))
    _, manual = numerical.reset(jax.random.PRNGKey(0))
    grouped_reward,manual_reward = 0.,0.
    for i in range((horizon+1)//2):
        observations,grouped,rewards,dones,info = held.step_env(jax.random.PRNGKey(i+1),grouped,actions)
        grouped_reward += rewards[PLANT]
    for i in range(horizon):
        _,manual,rewards,_,_ = numerical.step_env(jax.random.PRNGKey(i+1),manual,actions)
        manual_reward += rewards[PLANT]
    assert int(grouped.step) == horizon
    assert dones['__all__']
    assert info['transitions'][PLANT].truncated
    assert grouped_reward == pytest.approx(manual_reward,abs=2e-6)
    assert observations[PLANT] == pytest.approx(numerical.get_obs(manual)[PLANT],abs=2e-6)
    for actual,expected in zip(jax.tree.leaves(grouped),jax.tree.leaves(manual),strict=True):
        assert actual == pytest.approx(expected,abs=2e-6)
    _,frozen,rewards,_,info = held.step_env(jax.random.PRNGKey(100),grouped,actions)
    assert rewards[PLANT] == 0.
    assert not info['transitions'][PLANT].allocation_executed
    assert not info['transitions'][PLANT].trade_executed
    assert info['transitions'][PLANT].realised_action == pytest.approx(jnp.zeros(3))
    for actual,expected in zip(jax.tree.leaves(frozen),jax.tree.leaves(grouped),strict=True):
        assert jnp.array_equal(actual,expected)


def test_short_interval_preserves_a_failed_allocation_qualification_flag():
    numerical = pool_environment(dt=.05,max_steps=1,allocation_max_substeps=1)
    held = PolicyIntervalMycorMarl(numerical,decision_interval_days=.1,max_episode_steps=1)
    _, _, _, _, info = step(held,{PLANT:jnp.array([0.,1.,0.]),FUNGUS:jnp.zeros(3)})
    assert not info[PLANT]['allocation_converged']


def test_interval_preserves_a_trade_cancellation_event_before_a_dormant_substep():
    numerical = pool_environment(dt=.05,plant=dict(initial_c_pool=.6,kappa_c=10.,death_fraction=.9),
                                 fungus=dict(initial_p_pool=1.))
    held = PolicyIntervalMycorMarl(numerical,decision_interval_days=.15,max_episode_steps=3)
    _, _, _, _, info = step(held,{PLANT:jnp.array([.3,.1,.2]),FUNGUS:jnp.array([.4,0.,0.])})
    assert bool(info[PLANT]['trade_cancelled'][0])


@pytest.mark.parametrize('mode',['mixed','plant-only','fungus-only'])
def test_jitted_vmapped_intervals_preserve_fixed_agent_mappings_and_trade_availability(mode):
    numerical = pool_environment(dt=.05,consumer_mode=mode)
    held = PolicyIntervalMycorMarl(numerical,decision_interval_days=.1,max_episode_steps=5)
    keys = jax.random.split(jax.random.PRNGKey(1),2)
    _,states = jax.vmap(held.reset)(keys)
    actions = {PLANT:jnp.array([.3,.1,.2]),FUNGUS:jnp.array([.4,.2,.1])}
    _,_,_,_,info = jax.jit(jax.vmap(held.step_env,in_axes=(0,0,None)))(keys,states,actions)
    assert set(info['transitions']) == {PLANT,FUNGUS}
    for agent in (PLANT,FUNGUS):
        transition = info['transitions'][agent]
        operational = mode == 'mixed' or mode == agent+'-only'
        assert transition.requested_action.shape == (2,3)
        assert jnp.all(transition.operational_at_end == operational)
        assert jnp.all(transition.allocation_executed == operational)
        assert jnp.all(transition.trade_executed == (mode == 'mixed'))
        expected = actions[agent] if operational else jnp.zeros(3)
        if mode != 'mixed':
            expected = expected.at[0].set(0.)
        assert jnp.allclose(transition.realised_action,jnp.broadcast_to(expected,(2,3)))


def test_public_ppo_preserves_valid_early_execution_when_death_occurs_later_in_the_interval():
    from mycormarl.algos.ppo import PPOConfig,make_train
    numerical = pool_environment(dt=.05,plant=dict(initial_c_pool=.6,kappa_c=10.,death_fraction=.9))
    held = PolicyIntervalMycorMarl(numerical,decision_interval_days=.1,max_episode_steps=5)
    config = PPOConfig(TOTAL_TIMESTEPS=1,NUM_STEPS=1,NUM_ENVS=1,NUM_MINIBATCHES=1,
                       UPDATE_EPOCHS=1,LR=0.,DISCOUNT_HALF_LIFE_DAYS=1.)
    output = jax.jit(make_train(held,config))(jax.random.PRNGKey(1))
    plant = output['trajectories'][0]
    assert jnp.all(plant.terminated)
    assert jnp.all(plant.biological_rate_actor_valid)
    assert jnp.all(plant.trade_actor_valid)
    assert jnp.all(plant.reward > 0.)
    assert plant.rate_action.shape[-1] == 3
    for value in jax.tree.leaves(output['metrics']):
        assert jnp.all(jnp.isfinite(value))
