"""Run a reproducible two-policy PPO smoke or development training job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import serialization
import jax

from mycormarl.algos.ppo import PPOConfig, make_train
from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


def main() -> None:
    """Train on a small explicit domain and save both policy parameter trees."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/ppo_parameters.msgpack")
    )
    args = parser.parse_args()
    config = EnvConfig(
        max_steps=args.num_steps,
        dt=0.025,
        soil_radius_cm=1.0,
        soil_depth_cm=1.0,
        radial_interval_cm=0.1,
        depth_interval_cm=0.1,
        topsoil_depth_cm=1.0,
    )
    species = SpeciesParams(plant=PlantTraits(), fungus=FungusTraits())
    env = BaseMycorMarl(config, species)
    ppo = PPOConfig(
        TOTAL_TIMESTEPS=args.total_timesteps,
        NUM_STEPS=args.num_steps,
        NUM_ENVS=args.num_envs,
        NUM_MINIBATCHES=1,
        UPDATE_EPOCHS=1,
    )
    output = jax.jit(make_train(env, ppo))(jax.random.PRNGKey(args.seed))
    train_state = output["runner_state"][0]
    parameters = {name: state.params for name, state in train_state.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialization.to_bytes(parameters))
    print(json.dumps({
        "agents": list(env.agents),
        "output": str(args.output),
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
