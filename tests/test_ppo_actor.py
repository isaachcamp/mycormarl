"""Public contracts for the compact two-head IPPO actor."""

import jax.numpy as jnp
import jax

from mycormarl.algos.ppo import (
    ActorCritic,
    PPOConfig,
    latent_to_rate_action,
    make_train,
    normal_log_probability,
)
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


def _small_environment():
    """Build a compact mixed-consumer environment for actor integration tests."""
    return BaseMycorMarl(
        EnvConfig(
            max_steps=2,
            dt=0.05,
            soil_radius_cm=0.2,
            soil_depth_cm=0.2,
            radial_interval_cm=0.1,
            depth_interval_cm=0.1,
        ),
        SpeciesParams(
            plant=PlantTraits(kappa_c=0.0, kappa_p=0.0),
            fungus=FungusTraits(kappa_c=0.0, kappa_p=0.0),
        ),
    )


def test_zero_latents_map_to_independent_nonnegative_per_day_rates():
    action = latent_to_rate_action(
        trade_latent=jnp.array(0.0),
        biological_rate_latent=jnp.zeros(3, dtype=jnp.float32),
    )

    assert action.dtype == jnp.float32
    assert jnp.allclose(action, jnp.full(4, jnp.log(2.0)))


def test_transformed_rate_actions_reach_the_environment_unchanged():
    """The environment observes and executes each transformed policy action exactly."""
    environment = _small_environment()
    _, state = environment.reset(jax.random.PRNGKey(1))
    actions = {
        PLANT: latent_to_rate_action(0.25, jnp.array([0.5, -0.75, 0.0])),
        FUNGUS: latent_to_rate_action(-0.5, jnp.array([-0.25, 1.0, 0.0])),
    }

    _, _, _, _, info = environment.step_env(
        jax.random.PRNGKey(2), state, actions
    )

    for agent in (PLANT, FUNGUS):
        transition = info["transitions"][agent]
        assert jnp.allclose(transition.requested_action, actions[agent])
        assert jnp.allclose(transition.realised_action, actions[agent])


def test_actor_initialises_two_gaussian_heads_and_a_local_critic():
    """Local observations expose the agreed initial actor and critic contract."""
    actor_critic = ActorCritic(activation="tanh")
    observations = jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 15.0
    parameters = actor_critic.init(jax.random.PRNGKey(0), observations)

    policy, values = actor_critic.apply(parameters, observations)

    assert policy.trade_loc.shape == (3,)
    assert policy.trade_log_std.shape == (3,)
    assert policy.biological_rate_loc.shape == (3, 3)
    assert policy.biological_rate_log_std.shape == (3, 3)
    assert values.shape == (3,)
    assert jnp.allclose(jax.nn.softplus(policy.trade_loc), 0.1)
    assert jnp.allclose(policy.biological_rate_loc, 0.0)
    assert jnp.allclose(policy.trade_log_std, 0.0)
    assert jnp.allclose(policy.biological_rate_log_std, 0.0)


def test_actor_trade_head_can_be_preconditioned_at_requested_per_day_rate():
    """The zero-feature trade latent maps exactly to the configured rate."""
    observations = jnp.zeros((2, 5), dtype=jnp.float32)
    for initial_trade in (0.05, 0.75):
        actor_critic = ActorCritic(initial_trade=initial_trade)
        parameters = actor_critic.init(jax.random.PRNGKey(11), observations)
        policy, _ = actor_critic.apply(parameters, observations)
        assert jnp.allclose(jax.nn.softplus(policy.trade_loc), initial_trade)


def test_standard_normal_log_probability_matches_known_density():
    """The explicit latent likelihood uses the standard Gaussian density."""
    log_probability = normal_log_probability(
        sample=jnp.array([0.0, 1.0]),
        location=jnp.array([0.0, 0.0]),
        log_std=jnp.array([0.0, 0.0]),
    )

    assert jnp.allclose(log_probability, jnp.array([-0.9189385, -1.4189385]))


def test_jitted_vectorised_rollout_retains_actions_and_factor_likelihoods():
    """Each independent rollout keeps the sampled and executed policy decision."""
    environment = _small_environment()
    config = PPOConfig(
        TOTAL_TIMESTEPS=4,
        NUM_STEPS=2,
        NUM_ENVS=2,
        NUM_MINIBATCHES=1,
        UPDATE_EPOCHS=1,
        LR=0.0,
        DISCOUNT_HALF_LIFE_DAYS=30.0,
    )

    output = jax.jit(make_train(environment, config))(jax.random.PRNGKey(4))
    train_states = output["runner_state"][0]

    for agent, trajectory in zip(
        (PLANT, FUNGUS), output["trajectories"], strict=True
    ):
        assert trajectory.latent_trade_action.shape == (1, 2, 2)
        assert trajectory.latent_biological_rate_action.shape == (1, 2, 2, 3)
        assert trajectory.rate_action.shape == (1, 2, 2, 4)
        assert trajectory.critic_valid.shape == (1, 2, 2)
        assert trajectory.biological_rate_actor_valid.shape == (1, 2, 2)
        assert trajectory.trade_actor_valid.shape == (1, 2, 2)
        assert trajectory.bootstrap_observation.shape == (1, 2, 2, 5)
        for values in (
            trajectory.latent_trade_action,
            trajectory.latent_biological_rate_action,
            trajectory.rate_action,
            trajectory.trade_log_probability,
            trajectory.biological_rate_log_probability,
            trajectory.value,
        ):
            assert jnp.all(jnp.isfinite(values))
        assert jnp.all(trajectory.rate_action[..., 0] >= 0.0)
        assert jnp.all(trajectory.rate_action[..., 1:] >= 0.0)
        assert jnp.allclose(
            trajectory.rate_action,
            latent_to_rate_action(
                trajectory.latent_trade_action,
                trajectory.latent_biological_rate_action,
            ),
        )

        policy, _ = ActorCritic(activation=config.ACTIVATION).apply(
            train_states[agent].params, trajectory.obs
        )
        recomputed_trade = normal_log_probability(
            trajectory.latent_trade_action,
            policy.trade_loc,
            policy.trade_log_std,
        )
        recomputed_biological_rate = jnp.sum(
            normal_log_probability(
                trajectory.latent_biological_rate_action,
                policy.biological_rate_loc,
                policy.biological_rate_log_std,
            ),
            axis=-1,
        )
        assert jnp.allclose(
            recomputed_trade, trajectory.trade_log_probability, atol=1e-6
        )
        assert jnp.allclose(
            recomputed_biological_rate,
            trajectory.biological_rate_log_probability,
            atol=1e-6,
        )

        advantages = output["advantages"][agent]
        targets = output["targets"][agent]
        returns = output["returns"][agent]
        metrics = output["metrics"][agent]
        assert advantages.shape == (1, 2, 2)
        assert targets.shape == (1, 2, 2)
        assert returns.shape == (1, 2, 2)
        assert jnp.all(jnp.isfinite(trajectory.reward))
        assert jnp.all(jnp.isfinite(advantages))
        assert jnp.all(jnp.isfinite(targets))
        assert jnp.all(jnp.isfinite(returns))
        assert jnp.array_equal(returns, targets)
        assert jnp.all(jnp.isfinite(metrics.total_loss))
        assert jnp.all(jnp.isfinite(metrics.value_loss))
        assert jnp.all(jnp.isfinite(metrics.actor_loss))

    plant_leaves = jax.tree.leaves(train_states[PLANT].params)
    fungus_leaves = jax.tree.leaves(train_states[FUNGUS].params)
    assert any(
        not jnp.array_equal(plant, fungus)
        for plant, fungus in zip(plant_leaves, fungus_leaves, strict=True)
    )
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for state in train_states.values()
        for leaf in jax.tree.leaves(state.params)
    )


def test_resumed_training_anneals_learning_rate_over_the_global_budget():
    """A resumed PPO chunk continues, rather than restarts, LR annealing."""
    environment = _small_environment()
    config = PPOConfig(
        TOTAL_TIMESTEPS=4,
        RUN_TIMESTEPS=2,
        NUM_STEPS=2,
        NUM_ENVS=1,
        NUM_MINIBATCHES=1,
        UPDATE_EPOCHS=1,
        LR=1.0,
        DISCOUNT_HALF_LIFE_DAYS=30.0,
    )

    first = jax.jit(make_train(environment, config))(jax.random.PRNGKey(9))
    second = jax.jit(make_train(
        environment, config, initial_runner_state=first["runner_state"],
    ))(jax.random.PRNGKey(9))

    for output, expected_rate in ((first, 0.5), (second, 0.0)):
        for agent in (PLANT, FUNGUS):
            assert jnp.isclose(
                output["metrics"][agent].learning_rate[-1], expected_rate
            )
