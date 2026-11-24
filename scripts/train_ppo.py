"""Run a reproducible two-policy PPO smoke or development training job."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from mycormarl.algos.ppo import PPOConfig, make_train
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
    parser.add_argument("--total-timesteps", type=int, default=16384)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument('--env-config', type=Path, help='JSON EnvConfig overrides; episode-steps and mode take precedence.')
    parser.add_argument('--episode-steps', type=int, default=14600,
                        help='Fixed episode horizon, independent of rollout length.')
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--num-minibatches', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--clip-eps', type=float, default=.2)
    parser.add_argument('--gae-lambda', type=float, default=.95)
    parser.add_argument('--vf-coef', type=float, default=.5)
    parser.add_argument('--dry-run', action='store_true', help='Validate and print configuration without training.')
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
    environment = dict(dt=.025, soil_radius_cm=1., soil_depth_cm=1.,
                       radial_interval_cm=.1, depth_interval_cm=.1)
    if args.env_config:
        environment.update(json.loads(args.env_config.read_text()))
    environment.update(max_steps=args.episode_steps, consumer_mode=args.mode)
    config = EnvConfig(**environment)
    species = SpeciesParams(plant=PlantTraits(), fungus=FungusTraits())
    env = BaseMycorMarl(config, species)
    ppo = PPOConfig(
        TOTAL_TIMESTEPS=args.total_timesteps,
        NUM_STEPS=args.num_steps,
        NUM_ENVS=args.num_envs,
        NUM_MINIBATCHES=args.num_minibatches,
        UPDATE_EPOCHS=args.update_epochs,
        LR=args.learning_rate, CLIP_EPS=args.clip_eps,
        GAE_LAMBDA=args.gae_lambda, VF_COEF=args.vf_coef,
        FINITE_HORIZON_RETURNS=args.finite_horizon_returns,
    )
    if any(value < 1 for value in (args.num_envs, args.num_steps, args.episode_steps,
                                   args.update_epochs, args.num_minibatches)):
        parser.error('environment, rollout and optimiser counts must be positive')
    batch_size = args.num_envs * args.num_steps
    if args.total_timesteps < batch_size or args.total_timesteps % batch_size:
        parser.error('total timesteps must contain whole PPO updates')
    make_train(env, ppo, retain_history=False)  # Validate the same PPO contract as execution.
    if args.dry_run:
        print(json.dumps(dict(environment=asdict(config), ppo=asdict(ppo),
                              batch_size=batch_size,
                              minibatch_size=batch_size // args.num_minibatches,
                              seed=args.seed), sort_keys=True))
        return
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
