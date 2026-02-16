
from typing import NamedTuple, Tuple, Dict, List

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
import distrax
import optax


class ActorCritic(nn.Module):
    action_dim: int = 5
    activation: str = 'relu'  # Activation function for the network

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[distrax.Distribution, jnp.ndarray]:
        """Forward pass of the actor-critic model."""
        activation = getattr(jax.nn, self.activation, jax.nn.relu)

        actor_mean = nn.Dense(64)(obs)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)

        actor_log_std = self.param('log_std', constant(0.0), (self.action_dim,)) # state-independent
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=jax.nn.softplus(actor_log_std))

        critic = nn.Dense(64)(obs)
        critic = activation(critic)
        critic = nn.Dense(64)(critic)
        critic = activation(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Trajectory(NamedTuple):
    done: jnp.ndarray # Flag whether agent is done at this step; shape (NUM_STEPS, NUM_ENVS).
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    terminal: jnp.ndarray  # Environment-level termination flag (e.g. global done like done['__all__']); differs from per-agent `done` and is used where we need to know if the entire episode has ended rather than a single agent.


def batchify(
        x: Dict[str, jax.Array], agent_list: List[str],
        num_envs: float, num_actors: float
    ) -> jax.Array:
    # I've adapted this as it was collapsing the envs dimension – I don't know how it
    # worked for their code...
    x_inter = jnp.stack([x[a] for a in agent_list])
    return x_inter.reshape((num_actors, num_envs, -1))

def unbatchify(
        x: jax.Array, agent_list: List[str],
        num_envs: float, num_actors: float
    ) -> Dict[str, jax.Array]:
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(env, config):
    """Factory function to create training function for PPO."""

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
        
        Note: This function assumes a single environment and does not handle multiple environments.
        """
        # Initialize independent tree and fungus networks
        tree_policy = ActorCritic(env.action_spaces["agent_0"].shape[0], activation=config.ACTIVATION)
        fungus_policy = ActorCritic(env.action_spaces["agent_1"].shape[0], activation=config.ACTIVATION)

        rng, tree_rng, fungus_rng = jax.random.split(rng, 3)
        # init_x = jnp.zeros(env.observation_space("agent_0").shape).flatten()
        init_x = jnp.zeros((1, env.observation_spaces["agent_0"].shape[0])) # same for both agents

        tree_tx = optax.adam(learning_rate=config.LR) # Adam optimizer with static learning rate
        fungus_tx = optax.adam(learning_rate=config.LR)

        # Initialize training states
        tree_train_state = TrainState.create(
            apply_fn=tree_policy.apply, # __call__ method of network
            params=tree_policy.init(tree_rng, init_x), # initialised parameters
            tx=tree_tx # optimizer
        )
        fungus_train_state = TrainState.create(
            apply_fn=fungus_policy.apply,
            params=fungus_policy.init(fungus_rng, init_x),
            tx=fungus_tx
        )

        train_state = {"plant": tree_train_state, "fungus": fungus_train_state}

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
                rng, tree_act_rng, fungus_act_rng = jax.random.split(rng, 3)

                # Batchify the last observations for the network from Dict[str, Array] to Array
                # and unpack batched observations for tree and fungus agents 
                obs_batch = batchify(last_obs, env.agents, config.NUM_ENVS, config.NUM_ACTORS)
                tree_obs_batch, fungus_obs_batch = obs_batch[0], obs_batch[1]

                # Get actions from tree and fungus networks
                tree_pi, tree_value = tree_policy.apply(train_state["plant"].params, tree_obs_batch)
                tree_action = tree_pi.sample(seed=tree_act_rng)
                tree_log_prob = tree_pi.log_prob(tree_action)

                fungus_pi, fungus_value = fungus_policy.apply(train_state["fungus"].params, fungus_obs_batch)
                fungus_action = fungus_pi.sample(seed=fungus_act_rng)
                fungus_log_prob = fungus_pi.log_prob(fungus_action)

                # Unbatchify the actions to match the environment's expected input format
                env_act = unbatchify(
                    jnp.stack([tree_action, fungus_action]),
                    env.agents, config.NUM_ENVS, config.NUM_ACTORS
                )

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.NUM_ENVS)
                obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                # Collect Trajectory object
                tree_transition = Trajectory(
                    done['agent_0'].squeeze(),
                    tree_action,
                    jnp.array(tree_value),
                    reward['agent_0'].squeeze(),
                    tree_log_prob,
                    tree_obs_batch,
                    info=info['agent_0'],
                    terminal=done["__all__"].squeeze()
                )
                fungus_transition = Trajectory(
                    done['agent_1'].squeeze(),
                    fungus_action,
                    jnp.array(fungus_value),
                    reward['agent_1'].squeeze(),
                    fungus_log_prob,
                    fungus_obs_batch,
                    info=info['agent_1'],
                    terminal=done["__all__"].squeeze()
                )

                runner_state = (train_state, env_state, obs, rng)

                return runner_state, (tree_transition, fungus_transition)

            # Scan over the number of steps to collect trajectories for parallel envs, per update.
            runner_state, (tree_traj, fungus_traj) = jax.lax.scan(
                _env_step, runner_state, None, config.NUM_STEPS
            )

            # CALCULATE ADVANTAGE
            # Get last observations and apply the policy networks to get the last values.
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config.NUM_ENVS, config.NUM_ACTORS)
            _, tree_last_val = tree_policy.apply(train_state['plant'].params, last_obs_batch[0])
            _, fungus_last_val = fungus_policy.apply(train_state['fungus'].params, last_obs_batch[1])

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

            # Calculate advantages and targets for tree and fungus trajectories.
            # tree_traj and fungus_traj are ExperienceBuffer objects with array-like structures,
            # with shape (NUM_STEPS, NUM_ENVS).
            tree_advantages, tree_targets = _calculate_gae(tree_traj, tree_last_val)
            fungus_advantages, fungus_targets = _calculate_gae(fungus_traj, fungus_last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(agent_train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        """
                        Calculate the loss for the PPO update. Same implementation as in original
                        Schulman et al. (2017) PPO paper (section 5, eq.(9)).
                        
                        Loss = -L_actor + L_value + L_entropy
                        """
                        # RERUN NETWORK
                        pi, value = agent_train_state.apply_fn(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

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
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.VF_COEF * value_loss
                            - config.ENT_COEF * entropy
                        )
                        return total_loss, (value_loss, loss_actor, _)

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

            # Update the tree policy network.
            update_tree_state = (train_state['plant'], tree_traj, tree_advantages, tree_targets, rng)
            update_tree_state, tree_loss_info = jax.lax.scan(
                _update_epoch, update_tree_state, None, config.UPDATE_EPOCHS
            )

            # Update the fungus policy network.
            update_fungus_state = (
                train_state['fungus'], fungus_traj,
                fungus_advantages, fungus_targets,
                update_tree_state[-1]
            )
            update_fungus_state, fungus_loss_info = jax.lax.scan(
                _update_epoch, update_fungus_state, None, config.UPDATE_EPOCHS
            )

            train_state = {'plant': update_tree_state[0], 'fungus': update_fungus_state[0]}
            rng = update_fungus_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, (tree_traj, fungus_traj)

        # Scan over update steps.
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, _rng)
        runner_state, (tree_traj, fungus_traj) = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )

        return {"runner_state": runner_state, "trajectories": (tree_traj, fungus_traj)}

    return train
