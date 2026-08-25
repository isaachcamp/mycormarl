"""Run ADR-0015's reproducible 10-µM, plant-only two-arm IPPO study.

The state-only and time-aware arms are intentionally matched except for the
episode-clock observation.  Each checkpoint is a completed 120-day episode;
the runner resumes from the latest checkpoint without replaying it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import traceback
from typing import Any

import jax
from flax import serialization

from mycormarl.algos.ppo import ActorCritic, PPOConfig, make_train
from mycormarl.checkpoint_evaluation import evaluate_policy_summary
from mycormarl.environments.base_mycor import BaseMycorMarl, FUNGUS, PLANT
from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.random_streams import derive_random_streams


HORIZON_DAYS = 120.0
NUMERICAL_TIMESTEP_DAYS = 0.025
INITIAL_SOLUTION_P_UM = 10.0
DECISION_INTERVAL_DAYS = 1.0
EPISODES_PER_SEED = 500
SEEDS = (0, 1, 2, 3, 4)
MAX_WORKERS = 2
EVALUATION_INTERVAL_EPISODES = 25
CONTROL_IDENTIFIERS = (
    "static-growth-10-reproduction-90",
    "static-growth-90-reproduction-10",
    "grow-to-reproduce-day-90",
)
DEPTH_PROFILE = (
    (0.0, 1.0), (5.0, 1.0), (15.0, 0.345), (30.0, 0.17),
    (60.0, 0.103), (100.0, 0.069),
)
DEFAULT_CONTROL_BUNDLE = Path(
    "outputs/episode-time-allocation-plant-only-p-sweep/conditions/p-1000/result-bundle.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/ippo-learning-10-time-aware")


@dataclass(frozen=True)
class StudyArm:
    """The sole intervention: whether both policies receive the episode clock."""

    identifier: str
    include_episode_clock: bool


@dataclass(frozen=True)
class StudyPlan:
    """Complete fixed protocol, with all durations expressed in days."""

    initial_solution_p_um: float
    horizon_days: float
    numerical_timestep_days: float
    decision_interval_days: float
    policy_steps_per_episode: int
    episodes_per_seed: int
    seeds: tuple[int, ...]
    workers: int
    arms: tuple[StudyArm, ...]
    environment: dict[str, Any]
    ppo: dict[str, Any]
    control_identifiers: tuple[str, ...]


def study_plan() -> StudyPlan:
    """Declare the immutable biological and PPO protocol before any launch."""
    return StudyPlan(
        initial_solution_p_um=INITIAL_SOLUTION_P_UM,
        horizon_days=HORIZON_DAYS,
        numerical_timestep_days=NUMERICAL_TIMESTEP_DAYS,
        decision_interval_days=DECISION_INTERVAL_DAYS,
        policy_steps_per_episode=120,
        episodes_per_seed=EPISODES_PER_SEED,
        seeds=SEEDS,
        workers=MAX_WORKERS,
        arms=(StudyArm("state-only", False), StudyArm("time-aware", True)),
        environment={
            "consumer_mode": "plant-only",
            "soil_radius_cm": 20.0,
            "soil_depth_cm": 60.0,
            "radial_interval_cm": 0.1,
            "depth_interval_cm": 0.1,
            "initial_solution_p_depth_profile": DEPTH_PROFILE,
            "action_interface": "rate-action-daily-held",
        },
        ppo={
            "learning_rate": 2.5e-4,
            "update_epochs": 4,
            "num_minibatches": 1,
            "num_envs": 1,
            "num_steps": 120,
            "gae_lambda": 0.95,
            "discount_half_life_days": None,
            "finite_horizon_returns": True,
            "normalize_critic_targets": True,
        },
        control_identifiers=CONTROL_IDENTIFIERS,
    )


def _plan_payload(plan: StudyPlan) -> dict[str, Any]:
    """Make tuple-bearing protocol declarations stable across msgpack resume."""
    return json.loads(json.dumps(asdict(plan)))


def study_environment(
    plan: StudyPlan, arm: StudyArm,
) -> PolicyIntervalMycorMarl:
    """Build one arm's identical plant-only environment and daily policy seam."""
    numerical = BaseMycorMarl(
        EnvConfig(
            max_steps=round(plan.horizon_days / plan.numerical_timestep_days),
            dt=plan.numerical_timestep_days,
            consumer_mode=plan.environment["consumer_mode"],
            include_episode_clock=arm.include_episode_clock,
            soil_radius_cm=plan.environment["soil_radius_cm"],
            soil_depth_cm=plan.environment["soil_depth_cm"],
            radial_interval_cm=plan.environment["radial_interval_cm"],
            depth_interval_cm=plan.environment["depth_interval_cm"],
            initial_solution_p_um=plan.initial_solution_p_um,
            initial_solution_p_depth_profile=plan.environment[
                "initial_solution_p_depth_profile"
            ],
        ),
        SpeciesParams(PlantTraits(), FungusTraits()),
    )
    return PolicyIntervalMycorMarl(
        numerical,
        decision_interval_days=plan.decision_interval_days,
        max_episode_steps=plan.policy_steps_per_episode,
    )


def ppo_config(plan: StudyPlan) -> PPOConfig:
    """Return the common undiscounted finite-horizon PPO configuration."""
    return PPOConfig(
        TOTAL_TIMESTEPS=plan.episodes_per_seed * plan.policy_steps_per_episode,
        RUN_TIMESTEPS=plan.policy_steps_per_episode,
        NUM_STEPS=plan.ppo["num_steps"],
        NUM_ENVS=plan.ppo["num_envs"],
        NUM_MINIBATCHES=plan.ppo["num_minibatches"],
        UPDATE_EPOCHS=plan.ppo["update_epochs"],
        GAE_LAMBDA=plan.ppo["gae_lambda"],
        LR=plan.ppo["learning_rate"],
        DISCOUNT_HALF_LIFE_DAYS=plan.ppo["discount_half_life_days"],
        FINITE_HORIZON_RETURNS=plan.ppo["finite_horizon_returns"],
        NORMALIZE_CRITIC_TARGETS=plan.ppo["normalize_critic_targets"],
    )


def _task_dir(output_dir: Path, arm: StudyArm, seed: int) -> Path:
    return output_dir / "tasks" / f"{arm.identifier}-seed-{seed}"


def _checkpoint_path(task_dir: Path) -> Path:
    """Keep one atomically replaced resume point per arm/seed task."""
    return task_dir / "checkpoint.msgpack"


def _latest_checkpoint(task_dir: Path) -> Path | None:
    checkpoint = _checkpoint_path(task_dir)
    return checkpoint if checkpoint.exists() else None


def _write_checkpoint(
    path: Path, *, plan: StudyPlan, arm: StudyArm, seed: int, episode: int,
    runner_state: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "mycormarl-10um-time-aware-ippo-checkpoint",
        "format_version": 1,
        "plan": _plan_payload(plan),
        "arm": asdict(arm),
        "seed": seed,
        "completed_episodes": episode,
        "runner_state": serialization.to_state_dict(runner_state),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(serialization.msgpack_serialize(payload))
    temporary.replace(path)


def _restore_checkpoint(
    path: Path, *, plan: StudyPlan, arm: StudyArm, seed: int,
    environment: PolicyIntervalMycorMarl, config: PPOConfig,
) -> tuple[int, Any]:
    payload = serialization.msgpack_restore(path.read_bytes())
    if (
        payload.get("format") != "mycormarl-10um-time-aware-ippo-checkpoint"
        or payload.get("format_version") != 1
        or payload.get("plan") != _plan_payload(plan)
        or payload.get("arm") != asdict(arm)
        or payload.get("seed") != seed
    ):
        raise ValueError(f"incompatible study checkpoint: {path}")
    completed = payload.get("completed_episodes")
    if not isinstance(completed, int) or not 0 < completed <= plan.episodes_per_seed:
        raise ValueError(f"invalid completed-episode count in {path}")
    template = jax.jit(make_train(environment, config, derive_random_streams(seed)))(
        jax.random.PRNGKey(seed)
    )["runner_state"]
    return completed, serialization.from_state_dict(template, payload["runner_state"])


def _evaluation_path(task_dir: Path, episode: int) -> Path:
    return task_dir / "evaluations" / f"episode-{episode:05d}.json"


def _write_evaluation(
    path: Path, *, plan: StudyPlan, arm: StudyArm, seed: int, episode: int,
    runner_state: Any, environment: PolicyIntervalMycorMarl, actor: ActorCritic,
) -> dict[str, Any]:
    train_state = runner_state[0]
    record = {
        "arm": arm.identifier,
        "include_episode_clock": arm.include_episode_clock,
        "seed": seed,
        "completed_episodes": episode,
        "cumulative_training_policy_steps": episode * plan.policy_steps_per_episode,
        "evaluation_protocol": "latent-location",
        "evaluation": evaluate_policy_summary(
            environment,
            {agent: train_state[agent].params for agent in (PLANT, FUNGUS)},
            episodes=1,
            protocol="latent-location",
            seed=seed,
            actor=actor,
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def train_arm_seed(plan: StudyPlan, arm: StudyArm, seed: int, output_dir: Path) -> None:
    """Train one seed exactly once per episode, resuming only after its checkpoint."""
    environment = study_environment(plan, arm)
    config = ppo_config(plan)
    streams = derive_random_streams(seed)
    task_dir = _task_dir(output_dir, arm, seed)
    checkpoint = _latest_checkpoint(task_dir)
    completed, runner_state = (0, None) if checkpoint is None else _restore_checkpoint(
        checkpoint, plan=plan, arm=arm, seed=seed, environment=environment, config=config
    )
    actor = ActorCritic(activation=config.ACTIVATION)
    initial_train = jax.jit(make_train(environment, config, streams))
    resumed_train = None if runner_state is None else jax.jit(
        make_train(environment, config, streams, initial_runner_state=runner_state)
    )

    # A crash after a checkpoint but before its scheduled evaluation never
    # repeats an episode; it only reconstructs that deterministic evaluation.
    if completed and (completed % EVALUATION_INTERVAL_EPISODES == 0 or completed == plan.episodes_per_seed):
        evaluation = _evaluation_path(task_dir, completed)
        if not evaluation.exists():
            _write_evaluation(evaluation, plan=plan, arm=arm, seed=seed, episode=completed,
                              runner_state=runner_state, environment=environment, actor=actor)

    while completed < plan.episodes_per_seed:
        if runner_state is None:
            trained = initial_train(jax.random.PRNGKey(seed))
            resumed_train = jax.jit(
                make_train(environment, config, streams, initial_runner_state=trained["runner_state"])
            )
        else:
            trained = resumed_train(jax.random.PRNGKey(seed), runner_state)
        runner_state = trained["runner_state"]
        completed += 1
        _write_checkpoint(_checkpoint_path(task_dir), plan=plan, arm=arm,
                          seed=seed, episode=completed, runner_state=runner_state)
        if completed % EVALUATION_INTERVAL_EPISODES == 0 or completed == plan.episodes_per_seed:
            _write_evaluation(_evaluation_path(task_dir, completed), plan=plan, arm=arm,
                              seed=seed, episode=completed, runner_state=runner_state,
                              environment=environment, actor=actor)


def selected_controls(path: Path, identifiers: tuple[str, ...]) -> list[dict[str, Any]]:
    """Load the deterministic 10-µM controls used for downstream comparison."""
    bundle = json.loads(path.read_text(encoding="utf-8"))
    entries = {entry["policy"]["identifier"]: entry for entry in bundle["entries"]}
    missing = set(identifiers) - set(entries)
    if missing:
        raise ValueError(f"control bundle is missing {sorted(missing)!r}")
    return [
        {
            "identifier": identifier,
            "cumulative_reproductive_fitness": entries[identifier]["cumulative_reproductive_fitness"],
            "final_living_biomass_g": entries[identifier]["final_living_biomass_g"],
        }
        for identifier in identifiers
    ]


def run_study(output_dir: Path, control_bundle: Path) -> None:
    """Run the fixed ten tasks with at most two concurrent training workers."""
    plan = study_plan()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "study-plan.json").write_text(json.dumps({
        "plan": _plan_payload(plan),
        "comparison_controls": selected_controls(control_bundle, plan.control_identifiers),
        "control_bundle": str(control_bundle),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tasks = [(arm, seed) for arm in plan.arms for seed in plan.seeds]
    with ThreadPoolExecutor(max_workers=plan.workers) as executor:
        futures = [
            executor.submit(_run_task_with_failure_record, plan, arm, seed, output_dir)
            for arm, seed in tasks
        ]
        for future in futures:
            future.result()


def _run_task_with_failure_record(plan: StudyPlan, arm: StudyArm, seed: int, output_dir: Path) -> None:
    try:
        train_arm_seed(plan, arm, seed, output_dir)
    except Exception:
        task_dir = _task_dir(output_dir, arm, seed)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--control-bundle", type=Path, default=DEFAULT_CONTROL_BUNDLE)
    args = parser.parse_args()
    run_study(args.output_dir, args.control_bundle)


if __name__ == "__main__":
    main()
