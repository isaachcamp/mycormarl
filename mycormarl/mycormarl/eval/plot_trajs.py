
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_mean_return_by_agent(rewards: np.ndarray) -> None:
    """
    Plots the mean return per episode for each agent.
    
    Args:
        rewards: A numpy array of shape (num_agents, num_envs, num_episodes, episode_length)
                 containing the rewards for each agent at each timestep.
    """
    num_agents = rewards.shape[0]
    opath = os.getcwd()

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

def plot_mean_final_state_vars_per_episode(
        health: np.ndarray,
        biomass: np.ndarray,
        props: np.ndarray,
) -> None:
    num_agents = health.shape[0]
    opath = os.getcwd()

    _, axs = plt.subplots(3, 1, figsize=(12, 8))

    for agent_idx in range(num_agents):
        axs[0].plot(
            np.nanmean(health[agent_idx], axis=0),
            label=f'agent_{agent_idx}'
        )
        axs[1].plot(
            np.nanmean(biomass[agent_idx], axis=0),
            label=f'agent_{agent_idx}'
        )
        axs[2].plot(
            np.nanmean(np.nansum(props[agent_idx], axis=-1), axis=0),
            label=f'agent_{agent_idx}'
        )

    axs[0].set_ylabel('Final health')
    axs[1].set_ylabel('Final biomass')
    axs[2].set_ylabel('Total propagules generated')
    axs[2].set_xlabel('Episode Number')

    for ax in axs.flat:
        ax.legend()

    plt.savefig(
        os.path.join(opath, "final_state_vars_per_episode.png"),
        bbox_inches='tight',
        dpi=300
    )

def plot_sum_trade_flows(
        p_trade: np.ndarray,
        s_trade: np.ndarray,
) -> None:
    num_agents = p_trade.shape[0]
    opath = os.getcwd()

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
