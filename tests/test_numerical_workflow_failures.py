"""Public workflows reject unqualified numerical transitions."""

import jax
import pytest

from mycormarl.algos.ppo import ActorCritic,PPOConfig,make_train
from mycormarl.checkpoint_evaluation import evaluate_policy_parameters,evaluate_policy_summary
from test_stoichiometric_actions import pool_environment


def policy(env):
    actor = ActorCritic(initial_trade=1.)
    observations,_ = env.reset(jax.random.PRNGKey(0))
    parameters = {agent:actor.init(jax.random.PRNGKey(i+1),observations[agent])
                  for i,agent in enumerate(env.agents)}
    return actor,parameters


@pytest.mark.parametrize('protocol',['latent-location','sampled-policy'])
def test_policy_evaluation_rejects_allocation_solver_failure(protocol):
    env = pool_environment(dt=1.,max_steps=1,allocation_max_substeps=1)
    actor,parameters = policy(env)
    with pytest.raises(ValueError,match='allocation'):
        evaluate_policy_parameters(env,parameters,episodes=1,actor=actor,protocol=protocol)


@pytest.mark.parametrize('protocol',['latent-location','sampled-policy'])
def test_summary_evaluation_rejects_allocation_solver_failure(protocol):
    env = pool_environment(dt=1.,max_steps=1,allocation_max_substeps=1)
    actor,parameters = policy(env)
    with pytest.raises(ValueError,match='allocation'):
        evaluate_policy_summary(env,parameters,episodes=1,actor=actor,protocol=protocol)


@pytest.mark.parametrize('evaluate',[evaluate_policy_parameters,evaluate_policy_summary])
def test_policy_evaluation_rejects_numerical_guard_contact(evaluate):
    env = pool_environment(dt=1.,max_steps=1,plant=dict(biomass_cap=1.001))
    actor,parameters = policy(env)
    with pytest.raises(ValueError,match='guard'):
        evaluate(env,parameters,episodes=1,actor=actor)


def test_jitted_ppo_rejects_a_chunk_containing_allocation_solver_failure():
    env = pool_environment(dt=1.,max_steps=1,allocation_max_substeps=1)
    config = PPOConfig(TOTAL_TIMESTEPS=1,NUM_STEPS=1,NUM_ENVS=1,
                       NUM_MINIBATCHES=1,UPDATE_EPOCHS=1,FINITE_HORIZON_RETURNS=True)
    with pytest.raises(RuntimeError,match='allocation'):
        output = jax.jit(make_train(env,config))(jax.random.PRNGKey(1))
        jax.block_until_ready(output)


def test_jitted_ppo_rejects_a_chunk_containing_numerical_guard_contact():
    env = pool_environment(dt=1.,max_steps=1,plant=dict(biomass_cap=1.001))
    config = PPOConfig(TOTAL_TIMESTEPS=1,NUM_STEPS=1,NUM_ENVS=1,
                       NUM_MINIBATCHES=1,UPDATE_EPOCHS=1,FINITE_HORIZON_RETURNS=True)
    with pytest.raises(RuntimeError,match='guard'):
        output = jax.jit(make_train(env,config))(jax.random.PRNGKey(1))
        jax.block_until_ready(output)


@pytest.mark.parametrize('failure',['allocation','guard'])
def test_numerical_qualification_rejects_failed_transitions(failure):
    from mycormarl.soil.deep_soil_qualification import StaticPolicy,run_deep_soil_qualification
    settings = dict(allocation_max_substeps=1) if failure == 'allocation' else dict(plant=dict(biomass_cap=1.001))
    env = pool_environment(dt=1.,max_steps=1,**settings)
    with pytest.raises(ValueError,match=failure):
        run_deep_soil_qualification(config=env.config,species=env.species,
            plant_policy=StaticPolicy(0.,1.,0.),fungus_policy=StaticPolicy(0.,0.,0.),
            duration_days=1.,seed=7,software_revision='failure-test')


def test_public_benchmark_rejects_failed_allocation_instead_of_timing_it_as_a_valid_step():
    from scripts.phosphate_qualification import benchmark_environment
    env = pool_environment(dt=1.,max_steps=10,allocation_max_substeps=1)
    with pytest.raises(ValueError,match='allocation'):
        benchmark_environment(env.config,repeats=1)
