
from typing import Dict

import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.algos import ppo


# TODO: Run evaluation episode only until terminal state reached.


def extract_episodes(qoi_traj: jax.Array, done_traj: jax.Array) -> np.ndarray:
    """
    Extracts complete and incomplete episodes from batched trajectory data.

    Args:
        qoi_traj: JAX array of quantity of interest (QOI) with shape (NUM_UPDATES, NUM_STEPS, NUM_ENVS).
        done_traj: JAX array of done flags with shape (NUM_UPDATES, NUM_STEPS, NUM_ENVS).

    Returns:
        np.ndarray of QOI with shape (NUM_ENVS, MAX_EPISODES, MAX_EPISODE_LENGTH).
    """
    num_updates, num_steps, num_envs = qoi_traj.shape
    total_steps = num_updates * num_steps

    # Flatten and convert to numpy for efficient, non-JIT analysis
    qoi_np = np.array(qoi_traj.reshape(total_steps, num_envs))
    done_np = np.array(done_traj.reshape(total_steps, num_envs))

    episode_qoi_per_env = [[] for _ in range(num_envs)]

    # Find the indices of done flags for each environment
    for env_idx in range(num_envs):
        done_indices_env = np.where(done_np[:, env_idx])[0]

        last_done_step = -1
        # Extract rewards for completed episodes
        for done_step_idx in done_indices_env:

            start_step = last_done_step + 1
            episode_qoi = qoi_np[start_step : done_step_idx + 1, env_idx]
            if episode_qoi.size > 0:
                episode_qoi_per_env[env_idx].append(episode_qoi)
            last_done_step = done_step_idx

        # Handle the final, incomplete episode
        if last_done_step + 1 < total_steps:
            remaining_qoi = qoi_np[last_done_step + 1:, env_idx]
            if remaining_qoi.size > 0:
                episode_qoi_per_env[env_idx].append(remaining_qoi)

    max_episode_number = max(len(episodes) for episodes in episode_qoi_per_env)
    max_episode_length = max(len(episode) for env_episodes in episode_qoi_per_env for episode in env_episodes)
    qoi_arr = np.zeros((num_envs, max_episode_number, max_episode_length))

    for env_idx, env_episodes in enumerate(episode_qoi_per_env):
        for episode_idx, episode in enumerate(env_episodes):
            qoi_arr[env_idx, episode_idx, :len(episode)] = episode
            if len(episode) < max_episode_length:
                qoi_arr[env_idx, episode_idx, len(episode):] = np.nan

    return np.array(qoi_arr)

def get_terminal_values(data_episodes: np.ndarray) -> np.ndarray:
    """
    Finds the final value before the episode terminates.
    
    Args:
        data_episodes: np.ndarray of shape (num_envs, num_episodes, episode_length)
                    contains data for each episode, with NaNs marking terminal steps.
    Returns:
        np.ndarray of shape (num_envs, num_episodes) containing terminal values.
    """
    terminal_nans = np.isnan(data_episodes) # Assumes NaNs mark terminal steps.

    # Find index of first terminal nan for each episode (axis=0)
    first_terminal_nan_idx = np.argmax(terminal_nans, axis=-1)

    # If there are no NaNs, set index to the last valid step (episode_length - 1)
    data_episodes_idx = np.where(first_terminal_nan_idx == 0, -1, first_terminal_nan_idx)

    # Last step will have reset the environment, so take the value from the step before.
    data_episodes_idx -= 1

    # Select data at the terminal step for each episode
    terminal_data = np.take_along_axis(
        data_episodes,
        data_episodes_idx[..., np.newaxis],  # Add dimension to match
        axis=-1
    ).squeeze(-1)  # Remove the extra dimension

    return terminal_data

def collect_eval_traj(rng, env: BaseMycorMarl, train_state: Dict[str, TrainState]):
    first_obs, env_state = env.reset(rng)

    def env_step(runner_state, x):
        train_state, env_state, last_obs, rng = runner_state
        rng, plant_act_rng, fungus_act_rng = jax.random.split(rng, 3)

        obs_batch = ppo.batchify(last_obs, env.agents, 1, env.num_agents)
        plant_pi, _ = train_state['plant'].apply_fn(train_state["plant"].params, obs_batch[0])
        fungus_pi, _ = train_state['fungus'].apply_fn(train_state["fungus"].params, obs_batch[1])

        plant_action = plant_pi.sample(seed=plant_act_rng)
        fungus_action = fungus_pi.sample(seed=fungus_act_rng)

        env_act = ppo.unbatchify(
            jnp.stack([plant_action, fungus_action]),
            env.agents, 1, env.num_agents
        )
        env_act = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), env_act)

        rng, rng_step = jax.random.split(rng)
        obs, env_state, _, done, info = env.step(rng_step, env_state, env_act)

        runner_state = (train_state, env_state, obs, rng)
        return runner_state, (env_state, done, info)

    runner_state = (train_state, env_state, first_obs, rng)
    runner_state, (env_state_traj, done_traj, info_traj) = jax.lax.scan(
        env_step, runner_state, None, env.max_episode_steps
    )
    _, final_env_state, _, _ = runner_state
    return final_env_state, env_state_traj, done_traj, info_traj
