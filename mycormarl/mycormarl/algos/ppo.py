
from dataclasses import dataclass
import math
from typing import Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
import optax

from mycormarl.environments.base_mycor import FUNGUS, PLANT
from mycormarl.random_streams import RandomStreamContract
from mycormarl.transition import Transition


class PolicyParameters(NamedTuple):
    """Parameters of the factorised Gaussian latent policy."""

    trade_loc: jax.Array
    trade_log_std: jax.Array
    allocation_loc: jax.Array
    allocation_log_std: jax.Array


class PPOStepFields(NamedTuple):
    """Learning controls derived from one environment transition."""

    critic_valid: jax.Array
    allocation_actor_valid: jax.Array
    trade_actor_valid: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    bootstrap_valid: jax.Array
    gae_trace_continues: jax.Array
    bootstrap_observation: jax.Array


def transition_to_ppo_fields(transition: Transition) -> PPOStepFields:
    """Convert algorithm-independent lifecycle facts into PPO controls."""
    critic_valid = transition.operational_at_start
    terminated = transition.operational_at_start & ~transition.operational_at_end
    return PPOStepFields(
        critic_valid=critic_valid,
        allocation_actor_valid=transition.allocation_executed,
        trade_actor_valid=transition.trade_executed,
        terminated=terminated,
        truncated=transition.truncated,
        bootstrap_valid=transition.operational_at_end,
        gae_trace_continues=(
            critic_valid & ~terminated & ~transition.truncated
        ),
        bootstrap_observation=transition.final_observation,
    )


def discount_from_half_life(
    dt_days: float,
    half_life_days: float | None,
) -> float:
    """Convert a physical reward half-life to one environment-step discount."""
    dt_days = float(dt_days)
    if not math.isfinite(dt_days) or dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    if half_life_days is None:
        return 1.0
    half_life_days = float(half_life_days)
    if half_life_days == math.inf:
        return 1.0
    if not math.isfinite(half_life_days) or half_life_days <= 0.0:
        raise ValueError("half_life_days must be positive, infinite, or None")
    return math.exp(-math.log(2.0) * dt_days / half_life_days)


def calculate_gae(
    *,
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_values: jax.Array,
    critic_valid: jax.Array,
    bootstrap_valid: jax.Array,
    gae_trace_continues: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Calculate masked GAE from explicit bootstrap and trace controls."""

    def _step(next_advantage, step):
        reward, value, bootstrap_value, valid, bootstrap, trace = step
        delta = reward + gamma * bootstrap * bootstrap_value - value
        advantage = valid * (
            delta + gamma * gae_lambda * trace * next_advantage
        )
        return advantage, advantage

    initial_advantage = jnp.zeros_like(values[-1])
    _, advantages = jax.lax.scan(
        _step,
        initial_advantage,
        (
            rewards,
            values,
            bootstrap_values,
            critic_valid,
            bootstrap_valid,
            gae_trace_continues,
        ),
        reverse=True,
    )
    return advantages, advantages + values


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Return the mean over valid samples, or zero when none are valid."""
    mask = jnp.asarray(mask, dtype=bool)
    weights = mask.astype(values.dtype)
    count = jnp.sum(weights)
    return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(count, 1.0)


def masked_normalize(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Normalise valid values and return zero for every invalid sample."""
    mask = jnp.asarray(mask, dtype=bool)
    mean = masked_mean(values, mask)
    variance = masked_mean(jnp.square(values - mean), mask)
    normalized = (values - mean) / (jnp.sqrt(variance) + 1e-8)
    return jnp.where(mask, normalized, 0.0)


def normal_log_probability(
    sample: jax.Array,
    location: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    """Return elementwise log density under a diagonal Gaussian latent policy."""
    standardised = (sample - location) * jnp.exp(-log_std)
    return -0.5 * jnp.square(standardised) - log_std - 0.5 * jnp.log(2.0 * jnp.pi)


def latent_to_physical_action(
    trade_latent: jax.Array,
    allocation_latent: jax.Array,
) -> jax.Array:
    """Map three policy latents to ``[trade, growth, reproduction, reserve]``."""
    trade_latent = jnp.asarray(trade_latent, dtype=jnp.float32)
    allocation_latent = jnp.asarray(allocation_latent, dtype=jnp.float32)
    first = allocation_latent[..., 0]
    second = allocation_latent[..., 1]
    allocation_logits = jnp.stack(
        (
            first / jnp.sqrt(2.0) + second / jnp.sqrt(6.0),
            -first / jnp.sqrt(2.0) + second / jnp.sqrt(6.0),
            -2.0 * second / jnp.sqrt(6.0),
        ),
        axis=-1,
    )
    allocation = jax.nn.softmax(allocation_logits, axis=-1)
    trade = jax.nn.sigmoid(trade_latent)[..., None]
    return jnp.concatenate((trade, allocation), axis=-1)


@dataclass(frozen=True)
class PPOConfig:
    """Typed configuration for the two-policy PPO training loop."""

    TOTAL_TIMESTEPS: int = 5_000_000
    NUM_STEPS: int = 128
    NUM_ENVS: int = 16
    NUM_ACTORS: int = 2
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 4
    DISCOUNT_HALF_LIFE_DAYS: float | None = None
    GAE_LAMBDA: float = 0.95
    VF_COEF: float = 0.5
    CLIP_EPS: float = 0.2
    ACTIVATION: str = "tanh"
    LR: float = 2.5e-4


class ActorCritic(nn.Module):
    """Shared network architecture used by each independent actor--critic."""

    activation: str = "relu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[PolicyParameters, jnp.ndarray]:
        """Forward pass of the actor-critic model."""
        activation = getattr(jax.nn, self.activation, jax.nn.relu)

        policy_features = nn.Dense(64, name="policy_encoder_0")(obs)
        policy_features = activation(policy_features)
        policy_features = nn.Dense(64, name="policy_encoder_1")(policy_features)
        policy_features = activation(policy_features)
        trade_loc = nn.Dense(
            1,
            kernel_init=constant(0.0),
            bias_init=constant(jnp.log(0.1 / 0.9)),
            name="trade_head",
        )(policy_features)[..., 0]
        allocation_loc = nn.Dense(
            2,
            kernel_init=constant(0.0),
            bias_init=constant(0.0),
            name="allocation_head",
        )(policy_features)
        trade_log_std = self.param("trade_log_std", constant(0.0), (1,))
        allocation_log_std = self.param(
            "allocation_log_std", constant(0.0), (2,)
        )
        policy = PolicyParameters(
            trade_loc=trade_loc,
            trade_log_std=jnp.broadcast_to(trade_log_std[0], trade_loc.shape),
            allocation_loc=allocation_loc,
            allocation_log_std=jnp.broadcast_to(
                allocation_log_std, allocation_loc.shape
            ),
        )

        critic = nn.Dense(64, name="critic_0")(obs)
        critic = activation(critic)
        critic = nn.Dense(64, name="critic_1")(critic)
        critic = activation(critic)
        critic = nn.Dense(1, name="critic_value")(critic)

        return policy, jnp.squeeze(critic, axis=-1)


class Trajectory(NamedTuple):
    """PPO rollout fields for one policy, distinct from environment transitions."""

    latent_trade_action: jnp.ndarray
    latent_allocation_action: jnp.ndarray
    physical_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    trade_log_probability: jnp.ndarray
    allocation_log_probability: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    critic_valid: jnp.ndarray
    allocation_actor_valid: jnp.ndarray
    trade_actor_valid: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    bootstrap_valid: jnp.ndarray
    gae_trace_continues: jnp.ndarray
    bootstrap_observation: jnp.ndarray


class PPOUpdateMetrics(NamedTuple):
    """Per-species PPO losses and rollout-validity diagnostics."""

    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    critic_valid_count: jnp.ndarray
    allocation_actor_valid_count: jnp.ndarray
    trade_actor_valid_count: jnp.ndarray
    critic_valid_fraction: jnp.ndarray
    allocation_actor_valid_fraction: jnp.ndarray
    trade_actor_valid_fraction: jnp.ndarray


def batchify(
    x: Dict[str, jax.Array],
    agent_list: List[str],
    num_envs: int,
    num_actors: int,
) -> jax.Array:
    """Stack per-agent observations as ``(actors, environments, features)``."""
    # I've adapted this as it was collapsing the envs dimension – I don't know how it
    # worked for their code...
    x_inter = jnp.stack([x[a] for a in agent_list])
    return x_inter.reshape((num_actors, num_envs, -1))

def unbatchify(
    x: jax.Array,
    agent_list: List[str],
    num_envs: int,
    num_actors: int,
) -> Dict[str, jax.Array]:
    """Convert an actor-major batch back to the environment's agent mapping."""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(
    env,
    config,
    random_streams: RandomStreamContract | None = None,
    initial_runner_state=None,
):
    """Build a JIT-compatible independent-PPO trainer for the fixed agent API.

    Each policy consumes its own bounded observation and learns only from the
    validity, bootstrap, and trace controls derived from typed Transitions.
    The returned training output contains both policy states, trajectories,
    losses, advantages, targets, and returns for smoke validation.
    """

    gamma = discount_from_half_life(
        env.config.dt, config.DISCOUNT_HALF_LIFE_DAYS
    )
    if gamma == 1.0:
        configured_consumers = []
        if env.config.consumer_mode in ("mixed", "plant-only"):
            configured_consumers.append((PLANT, env.species.plant))
        if env.config.consumer_mode in ("mixed", "fungus-only"):
            configured_consumers.append((FUNGUS, env.species.fungus))
        for agent, traits in configured_consumers:
            if (
                traits.initial_biomass <= 0.0
                or traits.kappa_p <= 0.0
                or traits.death_fraction <= 0.0
            ):
                raise ValueError(
                    "undiscounted PPO requires a guaranteed finite lifetime "
                    f"for each configured consumer; {agent} does not satisfy it"
                )

    if config.NUM_ACTORS != len(env.agents):
        raise ValueError("NUM_ACTORS must match the environment agent count")
    if config.NUM_STEPS % config.NUM_MINIBATCHES != 0:
        raise ValueError("NUM_STEPS must be divisible by NUM_MINIBATCHES")
    if config.TOTAL_TIMESTEPS < config.NUM_STEPS * config.NUM_ENVS:
        raise ValueError("TOTAL_TIMESTEPS must contain at least one PPO update")
    for agent in env.agents:
        if env.observation_spaces[agent].shape != (5,):
            raise ValueError("each independent actor-critic requires five observations")

    NUM_UPDATES = (
        config.TOTAL_TIMESTEPS // config.NUM_STEPS // config.NUM_ENVS
    )
    MINIBATCH_SIZE = (
        # config.NUM_ACTORS * # Two separate networks, so do not multiply by NUM_ACTORS.
        config.NUM_STEPS // config.NUM_MINIBATCHES
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config.NUM_MINIBATCHES * config.UPDATE_EPOCHS)) / NUM_UPDATES
        return config.LR * frac

    def train(rng):
        """
        Main training function for PPO. 
        --- Steps ---
        1. Initialize the actor-critic network and training state.
        2. Initialize the environment.
        3. Scan over the number of updates:
            a. For each update, perform a full step to update the environment and network.
            b. Calculate advantages using Generalized Advantage Estimation (GAE).
        4. Return the final training state and metrics.
        
        Plant and fungal policies consume their own vectorised observations.
        """
        # Initialize independent plant and fungus networks.  The optional
        # runner state is used by the study checkpoint seam to continue the
        # same optimizer, environment, and named-RNG state.
        plant_policy = ActorCritic(activation=config.ACTIVATION)
        fungus_policy = ActorCritic(activation=config.ACTIVATION)

        if initial_runner_state is None:
            if random_streams is None:
                rng, plant_rng, fungus_rng = jax.random.split(rng, 3)
                action_rng = rng
                environment_rng = rng
                minibatch_rng = rng
            else:
                plant_rng = random_streams.key("plant_initialization")
                fungus_rng = random_streams.key("fungal_initialization")
                action_rng = random_streams.key("policy_action_sampling")
                environment_rng = random_streams.key("environment_variation")
                minibatch_rng = random_streams.key("minibatch_ordering")
            init_x = jnp.zeros((1, env.observation_spaces[PLANT].shape[0]))
            plant_tx = optax.adam(learning_rate=config.LR)
            fungus_tx = optax.adam(learning_rate=config.LR)
            plant_train_state = TrainState.create(
                apply_fn=plant_policy.apply,
                params=plant_policy.init(plant_rng, init_x),
                tx=plant_tx,
            )
            fungus_train_state = TrainState.create(
                apply_fn=fungus_policy.apply,
                params=fungus_policy.init(fungus_rng, init_x),
                tx=fungus_tx
            )
            train_state = {PLANT: plant_train_state, FUNGUS: fungus_train_state}
            environment_rng, _rng = jax.random.split(environment_rng)
            reset_rng = jax.random.split(_rng, config.NUM_ENVS)
            obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            runner_rngs = (action_rng, environment_rng, minibatch_rng)
        else:
            train_state, env_state, obs, runner_rngs = initial_runner_state
            action_rng, environment_rng, minibatch_rng = runner_rngs

        def _update_step(runner_state, x):
            """
            Full update step for environment and network.
            Collects trajectories using lax.scan, calculates GAE advantages.
            """
            def _env_step(runner_state, x):
                """
                Execute a single step in the environment.
                
                1. Sample actions from the actor network.
                2. Step the environment with the sampled actions.
                3. Collect Transition for the trajectory.
                """
                train_state, env_state, last_obs, rngs = runner_state
                action_rng, environment_rng, minibatch_rng = rngs
                action_rng, plant_act_rng, fungus_act_rng = jax.random.split(
                    action_rng, 3
                )
                plant_trade_rng, plant_allocation_rng = jax.random.split(
                    plant_act_rng
                )
                fungus_trade_rng, fungus_allocation_rng = jax.random.split(
                    fungus_act_rng
                )

                # Batch observations for the independent plant and fungal policies.
                obs_batch = batchify(
                    last_obs, env.agents, config.NUM_ENVS, config.NUM_ACTORS
                )
                plant_obs_batch, fungus_obs_batch = obs_batch[0], obs_batch[1]

                # Keep each factor's sampling and likelihood explicit in the rollout.
                plant_policy_parameters, plant_value = plant_policy.apply(
                    train_state[PLANT].params, plant_obs_batch
                )
                plant_latent_trade = plant_policy_parameters.trade_loc + jnp.exp(
                    plant_policy_parameters.trade_log_std
                ) * jax.random.normal(
                    plant_trade_rng, plant_policy_parameters.trade_loc.shape
                )
                plant_latent_allocation = (
                    plant_policy_parameters.allocation_loc
                    + jnp.exp(plant_policy_parameters.allocation_log_std)
                    * jax.random.normal(
                        plant_allocation_rng,
                        plant_policy_parameters.allocation_loc.shape,
                    )
                )
                plant_trade_log_probability = normal_log_probability(
                    plant_latent_trade,
                    plant_policy_parameters.trade_loc,
                    plant_policy_parameters.trade_log_std,
                )
                plant_allocation_log_probability = jnp.sum(
                    normal_log_probability(
                        plant_latent_allocation,
                        plant_policy_parameters.allocation_loc,
                        plant_policy_parameters.allocation_log_std,
                    ),
                    axis=-1,
                )
                plant_physical_action = latent_to_physical_action(
                    plant_latent_trade, plant_latent_allocation
                )

                fungus_policy_parameters, fungus_value = fungus_policy.apply(
                    train_state[FUNGUS].params, fungus_obs_batch
                )
                fungus_latent_trade = fungus_policy_parameters.trade_loc + jnp.exp(
                    fungus_policy_parameters.trade_log_std
                ) * jax.random.normal(
                    fungus_trade_rng, fungus_policy_parameters.trade_loc.shape
                )
                fungus_latent_allocation = (
                    fungus_policy_parameters.allocation_loc
                    + jnp.exp(fungus_policy_parameters.allocation_log_std)
                    * jax.random.normal(
                        fungus_allocation_rng,
                        fungus_policy_parameters.allocation_loc.shape,
                    )
                )
                fungus_trade_log_probability = normal_log_probability(
                    fungus_latent_trade,
                    fungus_policy_parameters.trade_loc,
                    fungus_policy_parameters.trade_log_std,
                )
                fungus_allocation_log_probability = jnp.sum(
                    normal_log_probability(
                        fungus_latent_allocation,
                        fungus_policy_parameters.allocation_loc,
                        fungus_policy_parameters.allocation_log_std,
                    ),
                    axis=-1,
                )
                fungus_physical_action = latent_to_physical_action(
                    fungus_latent_trade, fungus_latent_allocation
                )

                # Unbatchify the actions to match the environment's expected input format
                env_act = unbatchify(
                    jnp.stack([plant_physical_action, fungus_physical_action]),
                    env.agents, config.NUM_ENVS, config.NUM_ACTORS
                )

                environment_rng, _rng = jax.random.split(environment_rng)
                rng_step = jax.random.split(_rng, config.NUM_ENVS)
                obs, env_state, reward, _, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                plant_fields = transition_to_ppo_fields(
                    info["transitions"][PLANT]
                )
                fungus_fields = transition_to_ppo_fields(
                    info["transitions"][FUNGUS]
                )

                # Collect Trajectory object
                plant_trajectory = Trajectory(
                    latent_trade_action=plant_latent_trade,
                    latent_allocation_action=plant_latent_allocation,
                    physical_action=plant_physical_action,
                    value=jnp.array(plant_value),
                    reward=reward[PLANT].reshape((config.NUM_ENVS,)),
                    trade_log_probability=plant_trade_log_probability,
                    allocation_log_probability=plant_allocation_log_probability,
                    obs=plant_obs_batch,
                    info=info[PLANT],
                    critic_valid=plant_fields.critic_valid,
                    allocation_actor_valid=plant_fields.allocation_actor_valid,
                    trade_actor_valid=plant_fields.trade_actor_valid,
                    terminated=plant_fields.terminated,
                    truncated=plant_fields.truncated,
                    bootstrap_valid=plant_fields.bootstrap_valid,
                    gae_trace_continues=plant_fields.gae_trace_continues,
                    bootstrap_observation=plant_fields.bootstrap_observation,
                )
                fungus_trajectory = Trajectory(
                    latent_trade_action=fungus_latent_trade,
                    latent_allocation_action=fungus_latent_allocation,
                    physical_action=fungus_physical_action,
                    value=jnp.array(fungus_value),
                    reward=reward[FUNGUS].reshape((config.NUM_ENVS,)),
                    trade_log_probability=fungus_trade_log_probability,
                    allocation_log_probability=fungus_allocation_log_probability,
                    obs=fungus_obs_batch,
                    info=info[FUNGUS],
                    critic_valid=fungus_fields.critic_valid,
                    allocation_actor_valid=fungus_fields.allocation_actor_valid,
                    trade_actor_valid=fungus_fields.trade_actor_valid,
                    terminated=fungus_fields.terminated,
                    truncated=fungus_fields.truncated,
                    bootstrap_valid=fungus_fields.bootstrap_valid,
                    gae_trace_continues=fungus_fields.gae_trace_continues,
                    bootstrap_observation=fungus_fields.bootstrap_observation,
                )

                runner_state = (
                    train_state,
                    env_state,
                    obs,
                    (action_rng, environment_rng, minibatch_rng),
                )

                return runner_state, (plant_trajectory, fungus_trajectory)

            # Scan over the number of steps to collect trajectories for parallel envs, per update.
            runner_state, (plant_traj, fungus_traj) = jax.lax.scan(
                _env_step, runner_state, None, config.NUM_STEPS
            )

            train_state, env_state, last_obs, rngs = runner_state
            _, plant_bootstrap_values = plant_policy.apply(
                train_state[PLANT].params, plant_traj.bootstrap_observation
            )
            _, fungus_bootstrap_values = fungus_policy.apply(
                train_state[FUNGUS].params, fungus_traj.bootstrap_observation
            )
            plant_advantages, plant_targets = calculate_gae(
                rewards=plant_traj.reward,
                values=plant_traj.value,
                bootstrap_values=plant_bootstrap_values,
                critic_valid=plant_traj.critic_valid,
                bootstrap_valid=plant_traj.bootstrap_valid,
                gae_trace_continues=plant_traj.gae_trace_continues,
                gamma=gamma,
                gae_lambda=config.GAE_LAMBDA,
            )
            fungus_advantages, fungus_targets = calculate_gae(
                rewards=fungus_traj.reward,
                values=fungus_traj.value,
                bootstrap_values=fungus_bootstrap_values,
                critic_valid=fungus_traj.critic_valid,
                bootstrap_valid=fungus_traj.bootstrap_valid,
                gae_trace_continues=fungus_traj.gae_trace_continues,
                gamma=gamma,
                gae_lambda=config.GAE_LAMBDA,
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(agent_train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        """
                        Calculate the loss for the PPO update. Same implementation as in original
                        Schulman et al. (2017) PPO paper (section 5, eq.(9)).
                        
                        Loss = -L_actor + L_value
                        """
                        # RERUN NETWORK
                        policy_parameters, value = agent_train_state.apply_fn(
                            params, traj_batch.obs
                        )
                        trade_log_probability = normal_log_probability(
                            traj_batch.latent_trade_action,
                            policy_parameters.trade_loc,
                            policy_parameters.trade_log_std,
                        )
                        allocation_log_probability = jnp.sum(
                            normal_log_probability(
                                traj_batch.latent_allocation_action,
                                policy_parameters.allocation_loc,
                                policy_parameters.allocation_log_std,
                            ),
                            axis=-1,
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * masked_mean(
                            jnp.maximum(value_losses, value_losses_clipped),
                            traj_batch.critic_valid,
                        )

                        # CALCULATE ACTOR LOSS
                        log_probability = allocation_log_probability + jnp.where(
                            traj_batch.trade_actor_valid,
                            trade_log_probability,
                            0.0,
                        )
                        old_log_probability = (
                            traj_batch.allocation_log_probability
                            + jnp.where(
                                traj_batch.trade_actor_valid,
                                traj_batch.trade_log_probability,
                                0.0,
                            )
                        )
                        ratio = jnp.exp(log_probability - old_log_probability)
                        gae = masked_normalize(
                            gae, traj_batch.allocation_actor_valid
                        )
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.CLIP_EPS,
                                1.0 + config.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = masked_mean(
                            -jnp.minimum(loss_actor1, loss_actor2),
                            traj_batch.allocation_actor_valid,
                        )
                        total_loss = loss_actor + config.VF_COEF * value_loss
                        return total_loss, (value_loss, loss_actor)

                    def _apply_update(state):
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        (total_loss, losses), grads = grad_fn(
                            state.params, traj_batch, advantages, targets
                        )
                        return state.apply_gradients(grads=grads), (
                            total_loss,
                            *losses,
                        )

                    def _skip_update(state):
                        zero = jnp.asarray(0.0)
                        return state, (zero, zero, zero)

                    return jax.lax.cond(
                        jnp.any(traj_batch.allocation_actor_valid),
                        _apply_update,
                        _skip_update,
                        agent_train_state,
                    )

                agent_train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Shuffle batch and create minibatches of shape 
                # (NUM_MINIBATCHES, MINIBATCH_SIZE, NUM_ENVS).
                batch_size = MINIBATCH_SIZE * config.NUM_MINIBATCHES
                # assert (
                #     batch_size == config.NUM_STEPS * config.NUM_ACTORS
                # ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)

                # return (train_state, traj_batch, advantages, targets, rng), batch

                # Reshape the batch to have the first dimension as batch_size.
                # batch = jax.tree_util.tree_map(
                #     lambda x: x.reshape((batch_size,) + x.shape[2:]), batch # hard-coded x.shape indices, assuming multiple agents in batch.
                # )

                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[1:]), batch # hard-coded x.shape indices, assuming one agent in batch.
                )

                # Shuffle the batch.
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Create minibatches.
                # Reshape the batch to have the first dimension as NUM_MINIBATCHES.
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                agent_train_state, total_loss = jax.lax.scan(
                    _update_minibatch, agent_train_state, minibatches
                )
                update_state = (agent_train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Update the plant policy network.
            update_plant_state = (
                train_state[PLANT],
                plant_traj,
                plant_advantages,
                plant_targets,
                rngs[2],
            )
            update_plant_state, plant_loss_info = jax.lax.scan(
                _update_epoch, update_plant_state, None, config.UPDATE_EPOCHS
            )

            # Update the fungus policy network.
            update_fungus_state = (
                train_state['fungus'], fungus_traj,
                fungus_advantages, fungus_targets,
                update_plant_state[-1]
            )
            update_fungus_state, fungus_loss_info = jax.lax.scan(
                _update_epoch, update_fungus_state, None, config.UPDATE_EPOCHS
            )

            train_state = {
                PLANT: update_plant_state[0],
                FUNGUS: update_fungus_state[0],
            }
            rngs = (rngs[0], rngs[1], update_fungus_state[-1])

            def _metrics(trajectory, loss_info):
                total_loss, value_loss, actor_loss = loss_info
                sample_count = trajectory.critic_valid.size
                critic_count = jnp.sum(trajectory.critic_valid)
                allocation_count = jnp.sum(
                    trajectory.allocation_actor_valid
                )
                trade_count = jnp.sum(trajectory.trade_actor_valid)
                return PPOUpdateMetrics(
                    total_loss=total_loss,
                    value_loss=value_loss,
                    actor_loss=actor_loss,
                    critic_valid_count=critic_count,
                    allocation_actor_valid_count=allocation_count,
                    trade_actor_valid_count=trade_count,
                    critic_valid_fraction=critic_count / sample_count,
                    allocation_actor_valid_fraction=(
                        allocation_count / sample_count
                    ),
                    trade_actor_valid_fraction=trade_count / sample_count,
                )

            plant_metrics = _metrics(plant_traj, plant_loss_info)
            fungus_metrics = _metrics(fungus_traj, fungus_loss_info)

            runner_state = (train_state, env_state, last_obs, rngs)
            return runner_state, (
                (plant_traj, fungus_traj),
                (plant_metrics, fungus_metrics),
                (plant_advantages, fungus_advantages),
                (plant_targets, fungus_targets),
            )

        # Scan over update steps. Each stochastic subsystem retains its own key.
        if initial_runner_state is None:
            runner_rngs = (action_rng, environment_rng, minibatch_rng)
        runner_state = (train_state, env_state, obs, runner_rngs)
        runner_state, (
            (plant_traj, fungus_traj),
            (plant_metrics, fungus_metrics),
            (plant_advantages, fungus_advantages),
            (plant_targets, fungus_targets),
        ) = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )

        return {
            "runner_state": runner_state,
            "trajectories": (plant_traj, fungus_traj),
            "metrics": {PLANT: plant_metrics, FUNGUS: fungus_metrics},
            "advantages": {
                PLANT: plant_advantages,
                FUNGUS: fungus_advantages,
            },
            "targets": {PLANT: plant_targets, FUNGUS: fungus_targets},
            # GAE lambda-returns are the critic regression targets.
            "returns": {PLANT: plant_targets, FUNGUS: fungus_targets},
        }

    return train
