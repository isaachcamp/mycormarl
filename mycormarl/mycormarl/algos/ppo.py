
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
import optax

from mycormarl.environments.base_mycor import FUNGUS, PLANT


class PolicyParameters(NamedTuple):
    """Parameters of the factorised Gaussian latent policy."""

    trade_loc: jax.Array
    trade_log_std: jax.Array
    allocation_loc: jax.Array
    allocation_log_std: jax.Array


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
    GAMMA: float = 0.995
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

    done: jnp.ndarray
    latent_trade_action: jnp.ndarray
    latent_allocation_action: jnp.ndarray
    physical_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    trade_log_probability: jnp.ndarray
    allocation_log_probability: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    terminal: jnp.ndarray


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

def make_train(env, config):
    """Factory function to create training function for PPO."""

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
        # Initialize independent plant and fungus networks
        plant_policy = ActorCritic(activation=config.ACTIVATION)
        fungus_policy = ActorCritic(activation=config.ACTIVATION)

        rng, plant_rng, fungus_rng = jax.random.split(rng, 3)
        init_x = jnp.zeros((1, env.observation_spaces[PLANT].shape[0]))

        plant_tx = optax.adam(learning_rate=config.LR)
        fungus_tx = optax.adam(learning_rate=config.LR)

        # Initialize training states
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

        # Initialize parallel environments
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.NUM_ENVS)
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

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
                train_state, env_state, last_obs, rng = runner_state
                rng, plant_act_rng, fungus_act_rng = jax.random.split(rng, 3)
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

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.NUM_ENVS)
                obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                # Collect Trajectory object
                plant_trajectory = Trajectory(
                    done=done[PLANT].squeeze(),
                    latent_trade_action=plant_latent_trade,
                    latent_allocation_action=plant_latent_allocation,
                    physical_action=plant_physical_action,
                    value=jnp.array(plant_value),
                    reward=reward[PLANT].squeeze(),
                    trade_log_probability=plant_trade_log_probability,
                    allocation_log_probability=plant_allocation_log_probability,
                    obs=plant_obs_batch,
                    info=info[PLANT],
                    terminal=done["__all__"].squeeze(),
                )
                fungus_trajectory = Trajectory(
                    done=done[FUNGUS].squeeze(),
                    latent_trade_action=fungus_latent_trade,
                    latent_allocation_action=fungus_latent_allocation,
                    physical_action=fungus_physical_action,
                    value=jnp.array(fungus_value),
                    reward=reward[FUNGUS].squeeze(),
                    trade_log_probability=fungus_trade_log_probability,
                    allocation_log_probability=fungus_allocation_log_probability,
                    obs=fungus_obs_batch,
                    info=info[FUNGUS],
                    terminal=done["__all__"].squeeze(),
                )

                runner_state = (train_state, env_state, obs, rng)

                return runner_state, (plant_trajectory, fungus_trajectory)

            # Scan over the number of steps to collect trajectories for parallel envs, per update.
            runner_state, (plant_traj, fungus_traj) = jax.lax.scan(
                _env_step, runner_state, None, config.NUM_STEPS
            )

            # CALCULATE ADVANTAGE
            # Get last observations and apply the policy networks to get the last values.
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(
                last_obs, env.agents, config.NUM_ENVS, config.NUM_ACTORS
            )
            _, plant_last_val = plant_policy.apply(
                train_state[PLANT].params, last_obs_batch[0]
            )
            _, fungus_last_val = fungus_policy.apply(
                train_state[FUNGUS].params, last_obs_batch[1]
            )

            def _calculate_gae(traj_batch, last_val):
                """
                Calculate advantages using Generalized Advantage Estimation (GAE),
                scanning over trajectories. Advantages and targets are used to calculate 
                the loss for the PPO update.

                Returns
                advantages - (NUM_STEPS, NUM_ENVS)
                targets - (NUM_STEPS, NUM_ENVS); one-step TD estimates.
                """
                def _get_advantages(gae_and_next_value, transition):
                    """
                    Calculate the Generalized Advantage Estimate (GAE) for a single transition.
                    The GAE is calculated using the Temporal Difference (TD) error and the next value estimate.
                    Update the GAE using TD error advantage from the "next" step (actually previous value, but reversed)

                    GAMMA - the discount factor.
                    GAE_LAMBDA - the smoothing factor for GAE, varies the bias-variance trade-off.
                        if GAE_LAMBDA = 0, this is equivalent to one-step TD learning (TD(0))
                            - high bias due to uncertainty in value estimates.
                        if GAE_LAMBDA = 1, this is equivalent to Monte Carlo returns (full trajectory)
                            - high variance due to propagating errors.
                    
                    Args:
                        gae_and_next_value: Tuple containing the current GAE and the next value estimate.
                            - gae: The current GAE value.
                            - next_value: The next value estimate for the transition.
                        transition: Transition object containing:
                            - done: Boolean indicating if the episode is done.
                            - value: Value estimate for the current transition.
                            - reward: Reward received for the current transition.

                    Returns:
                        gae_and_next_value: Tuple containing the current GAE and the next value estimate.
                        gae: The calculated GAE for the current transition.
                    """
                    gae, next_value = gae_and_next_value # carry value for scan
                    # Unpack Transition object
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # Calculate Temporal Difference (TD) error
                    delta = reward + config.GAMMA * next_value * (1 - done) - value
                    # TD error + next value estimate
                    gae = (
                        delta
                        + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
                    )
                    return (gae, value), gae

                # Scan backwards over trajectory.
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    xs=traj_batch, # Provides reward, value, done each iteration.
                    reverse=True, # Reverse scan
                    unroll=16, # Limit unroll for computational efficiency
                )
                return advantages, advantages + traj_batch.value

            # Calculate advantages and targets for plant and fungus trajectories.
            # plant_traj and fungus_traj have array-like structures,
            # with shape (NUM_STEPS, NUM_ENVS).
            plant_advantages, plant_targets = _calculate_gae(
                plant_traj, plant_last_val
            )
            fungus_advantages, fungus_targets = _calculate_gae(fungus_traj, fungus_last_val)

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
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        log_probability = (
                            trade_log_probability + allocation_log_probability
                        )
                        old_log_probability = (
                            traj_batch.trade_log_probability
                            + traj_batch.allocation_log_probability
                        )
                        ratio = jnp.exp(log_probability - old_log_probability)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.CLIP_EPS,
                                1.0 + config.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        total_loss = loss_actor + config.VF_COEF * value_loss
                        return total_loss, (value_loss, loss_actor)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(agent_train_state.params, traj_batch, advantages, targets)
                    agent_train_state = agent_train_state.apply_gradients(grads=grads)
                    return agent_train_state, total_loss

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
                rng,
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
            rng = update_fungus_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, (plant_traj, fungus_traj)

        # Scan over update steps.
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, _rng)
        runner_state, (plant_traj, fungus_traj) = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )

        return {
            "runner_state": runner_state,
            "trajectories": (plant_traj, fungus_traj),
        }

    return train
