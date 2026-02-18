
import os
import json

import hydra
from omegaconf import DictConfig
import jax
from flax.training import checkpoints

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.algos.ppo import make_train
from mycormarl.eval.plot_trajs import base_eval, plot_mean_return_all_seeds
from mycormarl.eval.utils import get_creation_attributes


RUN_NAME = "basic_twoagent_ppo"
AGENT_TYPES = {"plant": 1, "fungus": 1}
DESCRIPTION = """
    Basic two-agent PPO training with default control parameters.
    Exclusive access to one resource each (P for fungus, S for plant).
    Episode ends when both agents die.
    Reward (+1.5) for accumulated reproduction, punishment (-100) for death.
    No limit on trade flow.
    No constraints on growth or reproduction.
    This run serves as a baseline for future experiments.
"""
SAVE_WEIGHTS = False


@hydra.main(config_path="mycormarl/mycormarl/conf", config_name="env_params")
def main(cfg: DictConfig):

    opath_plots = os.path.join(os.getcwd(), "plots")

    env = BaseMycorMarl(
        agent_types=AGENT_TYPES,
        growth_cost=cfg.control_params.GROWTH_COST,
        reproduction_cost=cfg.control_params.REPRODUCTION_COST,
        maintenance_cost_ratio=cfg.control_params.MAINTENANCE_COST_RATIO,
        p_uptake_max_rate=cfg.control_params.P_UPTAKE_MAX_RATE,
        fungus_p_uptake_efficiency=cfg.control_params.FUNGUS_P_UPTAKE_EFFICIENCY,
        plant_p_uptake_efficiency=cfg.control_params.PLANT_P_UPTAKE_EFFICIENCY,
        max_sugar_gen_rate=cfg.control_params.MAX_SUGAR_GEN_RATE,
        p_cost_per_sugar=cfg.control_params.P_COST_PER_SUGAR,
        trade_flow_constant=cfg.control_params.TRADE_FLOW_CONSTANT,
        max_episode_steps=cfg.control_params.MAX_EPISODE_STEPS,
    )

    key = jax.random.PRNGKey(cfg.rl_params.SEED)
    key_seeds = jax.random.split(key, cfg.rl_params.NUM_SEEDS)
    train = make_train(env, cfg.rl_params)

    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(train))
        out = train_jit(key_seeds)

    train_state, env_state, obs, rng = out['runner_state']
    train_trajs = out['trajectories']

    for i in range(cfg.rl_params.NUM_SEEDS):
        opath_seeds = os.path.join(opath_plots, f"seed{i}")
        os.makedirs(opath_seeds, exist_ok=True)

        train_state_i = jax.tree.map(lambda x: x[i], train_state) # get train state for seed i
        train_trajs_i = jax.tree.map(lambda x: x[i], train_trajs) # get train trajs for seed i

        # Save model weights for seed i if enabled.
        if SAVE_WEIGHTS:
            opath_params = os.path.join(os.getcwd(), "params")
            ckpt_dir = os.path.join(opath_params, f"seed{i}")
            os.makedirs(ckpt_dir, exist_ok=True)

            checkpoints.save_checkpoint(
                os.path.join(ckpt_dir, "plant"),
                train_state_i["plant"],
                int(train_state_i["plant"].step)
            )
            checkpoints.save_checkpoint(
                os.path.join(ckpt_dir, "fungus"),
                train_state_i["fungus"],
                int(train_state_i["fungus"].step)
            )

        # Run evaluation for seed i and save plots.
        base_eval(
            key=key_seeds[i],
            env=env,
            train_state=train_state_i,
            train_traj=train_trajs_i,
            opath=opath_seeds
        )

    plot_mean_return_all_seeds(train_trajs, opath_plots)

    # Collate run details and save to JSON for record-keeping and reproducibility.
    attrs_dict = {}
    attrs_dict['run_name'] = RUN_NAME
    attrs_dict['agent_types'] = AGENT_TYPES
    attrs_dict['description'] = DESCRIPTION
    attrs_dict.update(get_creation_attributes())

    with open(os.path.join(os.getcwd(), "run_details.json"), "w") as f:
        json.dump(attrs_dict, f, indent=4)


if __name__ == "__main__":
    main()
