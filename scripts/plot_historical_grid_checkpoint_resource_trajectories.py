"""Plot checkpoint trajectories for plant P intake and fitness-input shares."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from flax import serialization

from mycormarl.checkpoint_evaluation import _actions, _actor_from_checkpoint_metadata
from mycormarl.environments.base_mycor import PLANT
from mycormarl.study import _training_environment


P_LEVELS = (3.0, 5.0, 10.0)
SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.5
UPDATE_SIZE = 128 * 16
CHECKPOINT_SAMPLES = 10
CACHE_VERSION = 2
DEFAULT_CACHE_NAME = "checkpoint-resource-trajectories.json"
SEED_COLOURS = plt.get_cmap("tab10")(np.arange(len(SEEDS)))


def _checkpoint_metrics(evaluation) -> dict[str, float]:
    episode = evaluation.episodes[0]
    trace = episode.trace
    direct = sum(row["uptake_mg"]["plant"] for row in trace)
    indirect = sum(row["transfers"]["plant"]["in"] for row in trace)
    c = sum(float(row["reproductive_resources"]["plant"]["carbon"]) for row in trace)
    p = sum(float(row["reproductive_resources"]["plant"]["phosphorus"]) for row in trace)
    p_factor = p ** (1.0 - ALPHA)
    c_factor = c ** ALPHA
    p_share = p_factor / (c_factor + p_factor) if c_factor + p_factor > 0.0 else 0.5
    return {
        "direct_p": float(direct),
        "indirect_p": float(indirect),
        "total_p": float(direct + indirect),
        "p_fitness_input_share": float(p_share),
    }


def _compact_runner(environment, actor):
    """Compile an evaluation that returns only the requested aggregates."""
    @jax.jit
    def run(parameters, key):
        key, reset_key = jax.random.split(key)
        observations, state = environment.reset(reset_key)

        def step(carry, _):
            key, observations, state = carry
            actions, key = _actions(actor, parameters, observations, "latent-location", key)
            key, step_key = jax.random.split(key)
            observations, state, rewards, _, infos = environment.step_env(
                step_key, state, actions
            )
            plant_info = infos[PLANT]
            c = plant_info["reproduction_c"] / environment.species.plant.gamma_c
            p = plant_info["reproduction_p"] / environment.species.plant.gamma_p
            return (key, observations, state), (
                plant_info["direct_p_uptake_mg"], plant_info["trade_in"],
                c, p,
            )

        _, values = jax.lax.scan(
            step, (key, observations, state), None,
            length=environment.max_episode_steps,
        )
        direct, indirect, c, p = values
        return jnp.asarray((
            jnp.sum(direct), jnp.sum(indirect), jnp.sum(direct + indirect),
            jnp.sum(c), jnp.sum(p),
        ))
    return run


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "mycormarl-checkpoint-resource-trajectories" or payload.get("version") != CACHE_VERSION:
        return {}
    return {
        float(p): {int(seed): points for seed, points in seeds.items()}
        for p, seeds in payload.get("trajectories", {}).items()
    }


def _save_cache(path: Path, trajectories: dict) -> None:
    payload = {
        "format": "mycormarl-checkpoint-resource-trajectories",
        "version": CACHE_VERSION,
        "checkpoint_samples": CHECKPOINT_SAMPLES,
        "fraction_definition": "direct and indirect cumulative plant P intake; P input share is the proposed cumulative Cobb-Douglas input convention",
        "trajectories": {
            str(p): {str(seed): points for seed, points in seeds.items()}
            for p, seeds in trajectories.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect(study_dir: Path, manifest_path: Path, cache_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = json.loads((study_dir / "result-bundle.json").read_text(encoding="utf-8"))
    trajectories = _load_cache(cache_path)
    for p in P_LEVELS:
        trajectories.setdefault(p, {})
        environment = _training_environment(manifest, "mixed", p)
        # One compiled runner is shared by all seeds and checkpoints at this
        # P level. The numerical substeps remain inside lax.scan; the Python
        # loops below only load checkpoint metadata and collect scalars.
        first_checkpoint = next(
            study_dir.glob(f"conditions/mixed-p{p:g}-seed*/checkpoints/checkpoint-*.msgpack")
        )
        first_payload = serialization.msgpack_restore(first_checkpoint.read_bytes())
        runner = _compact_runner(
            environment, _actor_from_checkpoint_metadata(first_payload["metadata"])
        )
        for seed in SEEDS:
            if seed in trajectories[p]:
                continue
            entry = next(
                item for item in bundle["entries"]
                if item["mode"] == "mixed"
                and float(item["initial_p_micromolar"]) == p
                and int(item["seed"]) == seed
            )
            points = []
            checkpoint_paths = sorted((study_dir / "conditions" / f"mixed-p{p:g}-seed{seed}" / "checkpoints").glob("checkpoint-*.msgpack"))
            checkpoints = []
            for checkpoint in checkpoint_paths:
                transitions = int(checkpoint.stem.split("-")[-1])
                payload = serialization.msgpack_restore(checkpoint.read_bytes())
                actor = _actor_from_checkpoint_metadata(payload["metadata"])
                parameters = {
                    agent: payload["runner_state"]["0"][agent]["params"]
                    for agent in ("plant", "fungus")
                }
                checkpoints.append((transitions, parameters, actor))
            if not checkpoints:
                raise ValueError(f"no checkpoints found for P={p:g}, seed={seed}")
            if len(checkpoints) > CHECKPOINT_SAMPLES:
                selected = np.unique(np.linspace(
                    0, len(checkpoints) - 1, CHECKPOINT_SAMPLES, dtype=int
                ))
                checkpoints = [checkpoints[index] for index in selected]
            for transitions, parameters, _ in checkpoints:
                metrics = runner(parameters, jax.random.PRNGKey(seed))
                metrics = [float(value) for value in jax.device_get(metrics)]
                points.append({
                    "update": transitions // UPDATE_SIZE,
                    "direct_p": metrics[0], "indirect_p": metrics[1],
                    "total_p": metrics[2],
                    "p_fitness_input_share": (
                        metrics[4] ** (1.0 - ALPHA)
                        / max(metrics[3] ** ALPHA + metrics[4] ** (1.0 - ALPHA), 1e-12)
                    ),
                    "direct_p_percent": 100.0 * metrics[0] / max(metrics[2], 1e-12),
                    "indirect_p_percent": 100.0 * metrics[1] / max(metrics[2], 1e-12),
                    "total_p_percent": 100.0,
                })
            trajectories[p][seed] = points
            _save_cache(cache_path, trajectories)
    return trajectories


def _plot_metric(trajectories: dict, key: str, ylabel: str, title: str, output: Path,
                 *, percent: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), sharey=False, constrained_layout=True)
    for axis, p in zip(axes, P_LEVELS, strict=True):
        for seed, colour in zip(SEEDS, SEED_COLOURS, strict=True):
            points = trajectories[p][seed]
            x = [point["update"] for point in points]
            y = np.asarray([point[key] for point in points]) * (100.0 if percent else 1.0)
            axis.plot(x, y, marker="o", markersize=2.2, linewidth=1.0,
                      color=colour, label=f"Seed {seed}")
        axis.set_xlabel("Training update")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, loc="best", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_intake(trajectories: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), sharey=True, constrained_layout=True)
    for axis, p in zip(axes, P_LEVELS, strict=True):
        for seed, colour in zip(SEEDS, SEED_COLOURS, strict=True):
            points = trajectories[p][seed]
            x = [point["update"] for point in points]
            for key, linestyle in (("direct_p_percent", "--"), ("indirect_p_percent", ":"), ("total_p_percent", "-")):
                axis.plot(x, [point[key] for point in points], color=colour,
                          linestyle=linestyle, linewidth=1.0,
                          label=f"Seed {seed}" if key == "total_p_percent" else None)
        axis.set_xlabel("Training update")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Plant P intake (%)")
    axes[0].set_ylim(0, 100)
    axes[0].plot([], [], color="#555555", linestyle="--", label="Direct soil uptake")
    axes[0].plot([], [], color="#555555", linestyle=":", label="Indirect fungal transfer")
    axes[0].plot([], [], color="#555555", linestyle="-", label="Total intake")
    axes[-1].legend(frameon=False, loc="best", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    output = args.study_dir / "figures"
    cache = args.cache or args.study_dir / DEFAULT_CACHE_NAME
    trajectories = collect(args.study_dir, args.manifest, cache)
    plot_intake(trajectories, output / "trade-only-ippo-checkpoint-plant-phosphorus-intake.png")
    _plot_metric(
        trajectories, "p_fitness_input_share",
        "Phosphorus share of reproductive inputs (%)",
        "Phosphorus contribution to plant reproductive inputs",
        output / "trade-only-ippo-checkpoint-phosphorus-fitness-contribution.png",
        percent=True,
    )
    print(f"wrote checkpoint resource trajectories to {output}")


if __name__ == "__main__":
    main()
