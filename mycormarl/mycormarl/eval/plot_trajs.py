
import os
from typing import Tuple, Dict

import numpy as np
import jax
from flax.training.train_state import TrainState
import matplotlib.pyplot as plt

from mycormarl.environments.base_mycor import BaseMycorMarl, State
from mycormarl.algos.ppo import Trajectory
from mycormarl.agents.agent import AgentState
from mycormarl.eval.utils import extract_episodes, get_terminal_values, collect_eval_traj


def plot_mean_return_by_agent(rewards: np.ndarray, opath: str) -> None:
    """
    Plots the mean return per episode for each agent.
    
    Args:
        rewards: A numpy array of shape (num_agents, num_envs, num_episodes, episode_length)
                 containing the rewards for each agent at each timestep.
    """
    num_agents = rewards.shape[0]

    plt.figure(figsize=(10, 6))

    for agent_idx in range(num_agents):
        mean_return_per_episode = np.nanmean(np.nansum(rewards[agent_idx], axis=-1), axis=0)
        plt.plot(mean_return_per_episode, label=f'agent_{agent_idx}')

    plt.xlabel('Episode Number')
    plt.ylabel('Mean Return')
    plt.legend()

    plt.savefig(
        os.path.join(opath, "mean_return_per_episode.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_mean_final_state_vars_per_episode(
        health: np.ndarray,
        biomass: np.ndarray,
        props: np.ndarray,
        phosphorus: np.ndarray,
        sugars: np.ndarray,
        opath: str
) -> None:
    num_agents = health.shape[0]

    _, axs = plt.subplots(3, 2, figsize=(12, 8))

    # Remove last empty subplot.
    axs[2, 1].axis('off')

    for agent_idx in range(num_agents):
        axs[0, 0].plot(
            np.nanmean(health[agent_idx], axis=0),
            label=f'agent_{agent_idx}'
        )
        axs[1, 0].plot(
            np.nanmean(biomass[agent_idx], axis=0),
            label=f'agent_{agent_idx}'
        )
        axs[2, 0].plot(
            np.nanmean(np.nansum(props[agent_idx], axis=-1), axis=0),
            label=f'agent_{agent_idx}'
        )
        axs[0, 1].plot(
            np.nanmean(phosphorus[agent_idx], axis=0),
            label=f'agent_{agent_idx}'
        )
        axs[1, 1].plot(
            np.nanmean(sugars[agent_idx], axis=0),
            label=f'agent_{agent_idx}'
        )

    axs[0, 0].set_ylabel('Final health')
    axs[1, 0].set_ylabel('Final biomass')
    axs[2, 0].set_ylabel('Total propagules generated')
    axs[0, 1].set_ylabel('Final phosphorus')
    axs[1, 1].set_ylabel('Final sugars')

    axs[2, 0].set_xlabel('Episode Number')
    axs[2, 1].set_xlabel('Episode Number')

    for ax in axs.flat[:-1]: # Exclude last empty subplot.
        ax.legend()

    plt.savefig(
        os.path.join(opath, "final_state_vars_per_episode.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_sum_trade_flows(
        p_trade: np.ndarray,
        s_trade: np.ndarray,
        opath: str = None
) -> None:
    num_agents = p_trade.shape[0]

    _, axs = plt.subplots(2, 1, figsize=(12, 8))

    for agent_idx in range(num_agents):
        # Sum over episode length and average over envs, leaving shape (num_episodes,)
        axs[0].plot(
            np.nanmean(np.nansum(p_trade[agent_idx], axis=-1), axis=0),
            label=f'agent_{agent_idx}'
        )
        axs[1].plot(
            np.nanmean(np.nansum(s_trade[agent_idx], axis=-1), axis=0),
            label=f'agent_{agent_idx}'
        )

    axs[0].set_ylabel('Total phosphorus traded')
    axs[1].set_ylabel('Total sugars traded')
    axs[1].set_xlabel('Episode Number')

    for ax in axs.flat:
        ax.legend()

    plt.savefig(
        os.path.join(opath, "mean_resource_trades_per_episode.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_s_p_levels_eval(info: Dict[str, jax.Array], opath: str) -> None:
    """Plot sugar and phosphorus levels over the course of an evaluation episode."""
    _, axs = plt.subplots(2, 1, figsize=(12, 8))
    linewidth = 0.8

    for agent_name, traj in info.items():
        idx = int(agent_name.split('_')[-1]) # Extract agent index from name, e.g. 'agent_0' -> 0
        sugars = traj["avail_sugars"] + traj["growth"] + traj["maintenance"] \
                + traj["reproduction"] + traj["sugars_generated"] + traj["s_trade"]
        phosphorus = traj["avail_phosphorus"] + traj["p_trade"] + (traj["sugars_generated"] * 3)

        axs[0].plot(sugars, label=f'agent_{idx}', linewidth=linewidth)
        axs[0].set_ylabel('Current sugars level')

        axs[1].plot(phosphorus, label=f'agent_{idx}', linewidth=linewidth)
        axs[1].set_ylabel('Current phosphorus level')
        axs[1].set_xlabel('Env Step')

    for ax in axs.flat:
        ax.legend()

    plt.savefig(
        os.path.join(opath, "eval_episode_sugars_phosphorus_levels.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_s_gen_levels_eval(info: Dict[str, jax.Array], opath: str) -> None:
    """Plot generated sugars in line with available phosphorus over evaluation episode."""
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    linewidth = 0.8

    phosphorus = info["agent_0"]["avail_phosphorus"] + info["agent_1"]["p_trade"] - info["agent_0"]["p_trade"]

    ax.plot(info["agent_0"]["sugars_generated"], label='Sugars generated', linewidth=linewidth)
    ax.plot(phosphorus, label='Available phosphorus', linewidth=linewidth)
    ax.set_xlabel('Env Step')

    ax.legend()

    plt.savefig(
        os.path.join(opath, "eval_episode_sugars_generated.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_s_allocations_eval(info: Dict[str, jax.Array], opath: str) -> None:
    """Plot sugar allocation (growth, maintenance, reproduction, trade) for agents over eval episode."""

    num_agents = len(info)
    _, axs = plt.subplots(num_agents, 1, figsize=(12, 4 * max(1, num_agents)))
    axs = np.atleast_1d(axs) # Ensure axs is always an array, even if there's only one agent.
    linewidth = 0.8

    for ax, (agent_name, traj) in zip(axs, info.items()):
        ax.plot(traj["growth"], label='Growth', linewidth=linewidth)
        ax.plot(traj["maintenance"], label='Maintenance', linewidth=linewidth)
        ax.plot(traj["reproduction"], label='Reproduction', linewidth=linewidth)
        ax.plot(traj["s_trade"], label='Trade', linewidth=linewidth)
        ax.set_ylabel('Allocated sugars')
        ax.set_title(f'{agent_name}')
        # ax.set_yscale('symlog') # Use symmetric log scale to handle zero and negative values.

    axs[-1].set_xlabel('Env Step')

    for ax in axs.flat:
        ax.legend()

    plt.savefig(
        os.path.join(opath, "eval_episode_sugar_allocations.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_s_plant_v_p_fungus_allocations_eval(info: Dict[str, jax.Array], opath: str) -> None:
    """Plot trades of exclusive resources for agents over eval episode."""

    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    linewidth = 0.8

    ax.plot(info["agent_0"]["s_trade"], label='S: plant –> fungus', linewidth=linewidth)
    ax.plot(info["agent_1"]["p_trade"], label='P: fungus –> plant', linewidth=linewidth)

    ax.set_xlabel('Env Step')
    ax.set_ylabel('Resource traded')
    ax.set_yscale('symlog') # Use symmetric log scale to handle zero and negative values.
    ax.legend()

    plt.savefig(
        os.path.join(opath, "eval_episode_s_plant_v_p_fungus_trade_allocations.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_health_biomass_eval(tree_traj: jax.Array, fungus_traj: jax.Array, opath: str) -> None:
    """Plot sugar and phosphorus levels over the course of an evaluation episode."""
    _, axs = plt.subplots(2, 1, figsize=(12, 8))
    linewidth = 0.8

    for agent_idx, traj in enumerate((tree_traj, fungus_traj)):
        axs[0].plot(traj.biomass, label=f'agent_{agent_idx}', linewidth=linewidth)
        axs[0].set_ylabel('Current biomass')

        # axs[0].set_yscale('log')

        axs[1].plot(traj.health, label=f'agent_{agent_idx}', linewidth=linewidth)
        axs[1].set_ylabel('Current health level')
        axs[1].set_xlabel('Env Step')

    for ax in axs.flat:
        ax.legend()

    plt.savefig(
        os.path.join(opath, "eval_episode_biomass_health_levels.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_training_trajectories(tree_traj: Trajectory, fungus_traj: Trajectory, opath: str) -> None:
    """Basic plots for training trajectories."""
    # ––– Plot mean return per episode (env mean) for tree and fungus –––
    tree_rewards = extract_episodes(tree_traj.reward, tree_traj.terminal)
    fungus_rewards = extract_episodes(fungus_traj.reward, fungus_traj.terminal)

    plot_mean_return_by_agent(
        np.stack((tree_rewards[:, :-1, :], fungus_rewards[:, :-1, :])),
        opath
    )

    # --- Plot final state values per training episode (env mean) –––
    # Get tree and fungus health terminal values.
    tree_health = extract_episodes(tree_traj.info["health"], tree_traj.terminal)
    tree_health_terminal_values = get_terminal_values(tree_health)
    fungus_health = extract_episodes(fungus_traj.info["health"], fungus_traj.terminal)
    fungus_health_terminal_values = get_terminal_values(fungus_health)

    # Get tree and fungus biomass terminal values.
    tree_biomass = extract_episodes(tree_traj.info["biomass"], tree_traj.terminal)
    tree_biomass_terminal_values = get_terminal_values(tree_biomass)
    fungus_biomass = extract_episodes(fungus_traj.info["biomass"], fungus_traj.terminal)
    fungus_biomass_terminal_values = get_terminal_values(fungus_biomass)

    # Get aggregate propagules generated per episode for tree and fungus.
    tree_props = extract_episodes(tree_traj.info["props_generated"], tree_traj.terminal)
    fungus_props = extract_episodes(fungus_traj.info["props_generated"], fungus_traj.terminal)

    # Get tree and fungus phosphorus terminal values.
    tree_phosphorus = extract_episodes(tree_traj.info["avail_phosphorus"], tree_traj.terminal)
    tree_phosphorus_terminal_values = get_terminal_values(tree_phosphorus)
    fungus_phosphorus = extract_episodes(fungus_traj.info["avail_phosphorus"], fungus_traj.terminal)
    fungus_phosphorus_terminal_values = get_terminal_values(fungus_phosphorus)

    # Get tree and fungus sugars terminal values.
    tree_sugars = extract_episodes(tree_traj.info["avail_sugars"], tree_traj.terminal)
    tree_sugars_terminal_values = get_terminal_values(tree_sugars)
    fungus_sugars = extract_episodes(fungus_traj.info["avail_sugars"], fungus_traj.terminal)
    fungus_sugars_terminal_values = get_terminal_values(fungus_sugars)

    plot_mean_final_state_vars_per_episode(
        np.stack((tree_health_terminal_values, fungus_health_terminal_values)),
        np.stack((tree_biomass_terminal_values, fungus_biomass_terminal_values)),
        np.stack((tree_props, fungus_props)),
        np.stack((tree_phosphorus_terminal_values, fungus_phosphorus_terminal_values)),
        np.stack((tree_sugars_terminal_values, fungus_sugars_terminal_values)),
        opath
    )

    # --- Plot trade flows per training episode (env mean) –––
    tree_p_trade = extract_episodes(tree_traj.info["p_trade"], tree_traj.terminal)
    fungus_p_trade = extract_episodes(fungus_traj.info["p_trade"], fungus_traj.terminal)
    tree_s_trade = extract_episodes(tree_traj.info["s_trade"], tree_traj.terminal)
    fungus_s_trade = extract_episodes(fungus_traj.info["s_trade"], fungus_traj.terminal)

    plot_sum_trade_flows(
        np.stack((tree_p_trade, fungus_p_trade)),
        np.stack((tree_s_trade, fungus_s_trade)),
        opath
    )

def plot_eval_episode(
        eval_traj: State,
        eval_done: Dict[str, jax.Array],
        eval_info: Dict[str, jax.Array],
        opath: str = None
    ) -> None:
    """Basic plots for evaluation episode trajectory."""

    plot_health_biomass_eval(eval_traj.agents[0], eval_traj.agents[1], opath)
    plot_s_p_levels_eval(eval_info, opath)
    plot_s_allocations_eval(eval_info, opath)
    plot_s_plant_v_p_fungus_allocations_eval(eval_info, opath)
    plot_s_gen_levels_eval(eval_info, opath)

def base_eval(
        key: jax.random.PRNGKey,
        env: BaseMycorMarl,
        train_state: TrainState,
        train_traj: Tuple[Trajectory],
        opath: str = None
    ) -> None:
    """
    Basic evaluation function for common plots for trajectories.
    
    Args:
        key: JAX random key(s) for evaluation.
        env: Training environment instance for evaluation episode.
        train_state: Trained agent parameters for evaluation episode.
        train_traj: Tuple of training trajectories for tree and fungus agents.
    """

    plot_training_trajectories(train_traj[0], train_traj[1], opath)

    # Plots for evaluation episode trajectory.
    (env_final_state, eval_traj, eval_done, eval_info) = collect_eval_traj(key, env, train_state)
    plot_eval_episode(eval_traj, eval_done, eval_info, opath)

def plot_mean_return_all_seeds(train_trajs: Tuple[Trajectory], opath: str) -> None:
    """Plot mean return per episode for all seeds on the same plot."""
    num_seeds = train_trajs[0].reward.shape[0]
    # Get number list of colours of length num_seeds for consistent coloring across plots.
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(num_seeds)]

    plt.figure(figsize=(10, 6))

    for i in range(num_seeds):
        plant_rewards = extract_episodes(train_trajs[0].reward[i], train_trajs[0].terminal[i])
        fungus_rewards = extract_episodes(train_trajs[1].reward[i], train_trajs[1].terminal[i])

        plant_mean_return_per_episode = np.nanmean(np.nansum(plant_rewards, axis=-1), axis=0)
        fungus_mean_return_per_episode = np.nanmean(np.nansum(fungus_rewards, axis=-1), axis=0)
        plt.plot(plant_mean_return_per_episode, c=colors[i], label=f'Seed {i}', linewidth=0.8)
        plt.plot(fungus_mean_return_per_episode, c=colors[i], linestyle="--", linewidth=0.8)


    plt.xlabel('Episode Number')
    plt.ylabel('Mean Return')

    # plt.yscale('log')

    plt.legend()

    plt.savefig(
        os.path.join(opath, "mean_return_per_episode_all_seeds.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
