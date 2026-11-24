"""Run a reproducible two-policy PPO smoke or development training job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mycormarl.algos.ppo import PPOConfig
from mycormarl.training import run_training
from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.random_streams import derive_random_streams
from mycormarl.policy_artifacts import save_policy_artifact


def main() -> None:
    """Train on a small explicit domain and save both policy parameter trees."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("mixed", "plant-only", "fungus-only"),
        default="mixed",
        help="Consumer configuration; the fixed plant/fungus API is retained.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("../mycormarl-data/training/ppo_parameters.msgpack")
    )
    parser.add_argument(
        "--finite-horizon-returns", action="store_true",
        help="Treat the declared episode horizon as the PPO return boundary.",
    )
    parser.add_argument('--run-dir', type=Path, help='Metrics and resumable checkpoint directory.')
    parser.add_argument('--chunk-updates', type=int, default=10)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument('--resume', action='store_true', help='Resume latest checkpoint in run-dir.')
    continuation.add_argument('--resume-from', type=Path, help='Explicit checkpoint, e.g. previous.msgpack.')
    parser.add_argument('--debug-history', action='store_true', help='Save detailed output for each chunk.')
    parser.add_argument('--evaluate-every', type=int, default=0,
                        help='Evaluate at chunk boundaries crossing this update interval; 0 disables.')
    parser.add_argument('--evaluation-episodes', type=int, default=1)
    args = parser.parse_args()
    config = EnvConfig(
        max_steps=args.num_steps,
        dt=0.025,
        soil_radius_cm=1.0,
        soil_depth_cm=1.0,
        radial_interval_cm=0.1,
        depth_interval_cm=0.1,
        consumer_mode=args.mode,
    )
    species = SpeciesParams(plant=PlantTraits(), fungus=FungusTraits())
    env = BaseMycorMarl(config, species)
    ppo = PPOConfig(
        TOTAL_TIMESTEPS=args.total_timesteps,
        NUM_STEPS=args.num_steps,
        NUM_ENVS=args.num_envs,
        NUM_MINIBATCHES=1,
        UPDATE_EPOCHS=1,
        FINITE_HORIZON_RETURNS=args.finite_horizon_returns,
    )
    random_streams = derive_random_streams(args.seed)
    run_dir = args.run_dir or args.output.parent / (args.output.stem + '-run')
    output = run_training(
        env, ppo, seed=args.seed, run_dir=run_dir,
        chunk_updates=args.chunk_updates, resume=args.resume_from or args.resume,
        debug=args.debug_history, evaluate_every=args.evaluate_every,
        evaluation_episodes=args.evaluation_episodes,
    )
    train_state = output["runner_state"][0]
    parameters = {name: state.params for name, state in train_state.items()}
    save_policy_artifact(
        args.output,
        parameters,
        consumer_mode=args.mode,
        random_streams=random_streams,
    )
    print(json.dumps({
        "agents": list(env.agents),
        "consumer_mode": args.mode,
        "output": str(args.output),
        "seed": args.seed,
        "run_dir": str(run_dir),
        "completed_updates": output["completed_updates"],
        "total_timesteps": args.total_timesteps,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
