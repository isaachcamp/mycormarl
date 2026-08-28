"""Protocol declarations for the multi-phosphorus reduced-control study."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import traceback
from typing import Any, Callable

import jax
from flax import serialization

from mycormarl.algos.ppo import (
    ActorCritic,
    PPOConfig,
    initialize_runner_state,
    make_train,
)
from mycormarl.checkpoint_evaluation import evaluate_policy_parameters
from mycormarl.environments.base_mycor import BaseMycorMarl, FUNGUS, PLANT
from mycormarl.environments.policy_interval import PolicyIntervalMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.random_streams import derive_random_streams
from mycormarl.trade_only import (
    GROWTH_FRACTION,
    REPRODUCTION_FRACTION,
    TOTAL_BIOLOGICAL_RATE_PER_DAY,
    run_trade_only_baseline,
)


@dataclass(frozen=True)
class TradeOnlyStudyPlan:
    """Immutable protocol for independent P-condition trade-only learning."""

    initial_p_micromolar: tuple[float, ...]
    modes: tuple[str, ...]
    seeds: tuple[int, ...]
    episodes_per_seed: int
    horizon_days: float
    numerical_timestep_days: float
    decision_interval_days: float
    policy_steps_per_episode: int
    evaluation_interval_episodes: int
    workers: int
    environment: dict[str, float]
    fixed_allocation: dict[str, float]
    ppo: dict[str, Any]


def study_plan(
    initial_p_micromolar: tuple[float, ...] = (0.75, 3.0, 10.0),
    *,
    workers: int = 2,
) -> TradeOnlyStudyPlan:
    """Declare a selectable static-allocation cohort without launching it."""
    levels = tuple(float(level) for level in initial_p_micromolar)
    if not levels or any(not math.isfinite(level) or level <= 0.0 for level in levels):
        raise ValueError("initial P levels must be finite and positive")
    if len(set(levels)) != len(levels):
        raise ValueError("initial P levels must be unique")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    return TradeOnlyStudyPlan(
        initial_p_micromolar=levels,
        modes=("plant-only", "mixed"),
        seeds=(0, 1, 2, 3, 4),
        episodes_per_seed=500,
        horizon_days=120.0,
        numerical_timestep_days=0.025,
        decision_interval_days=1.0,
        policy_steps_per_episode=120,
        # One terminal latent-location evaluation keeps the study compact;
        # checkpoints are the resumable progress record.
        evaluation_interval_episodes=500,
        workers=workers,
        environment={
            "soil_radius_cm": 40.0, "soil_depth_cm": 60.0,
            "radial_interval_cm": 0.1, "depth_interval_cm": 0.1,
        },
        fixed_allocation={
            "total_biological_rate_per_day": TOTAL_BIOLOGICAL_RATE_PER_DAY,
            "growth_fraction": GROWTH_FRACTION,
            "reproduction_fraction": REPRODUCTION_FRACTION,
            "storage_rate_per_day": 0.0,
        },
        ppo={
            "trade_only": True,
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
    )


def plan_payload(plan: TradeOnlyStudyPlan) -> dict[str, Any]:
    """Return a stable JSON-compatible protocol declaration for artifacts."""
    return json.loads(json.dumps(asdict(plan)))


def study_tasks(plan: TradeOnlyStudyPlan) -> tuple[dict[str, Any], ...]:
    """Declare one deterministic control and all learned seeds per P level."""
    baseline_seed = plan.seeds[0]
    return tuple(
        {"mode": mode, "initial_p_micromolar": p_micromolar, "seed": seed,
         "kind": "baseline" if mode == "plant-only" else "trade-only-training"}
        for mode in plan.modes for p_micromolar in plan.initial_p_micromolar
        for seed in ((baseline_seed,) if mode == "plant-only" else plan.seeds)
    )


def study_environment(plan: TradeOnlyStudyPlan, p_micromolar: float) -> PolicyIntervalMycorMarl:
    """Build one independent mixed P condition with daily held Rate actions."""
    numerical = BaseMycorMarl(
        EnvConfig(
            max_steps=round(plan.horizon_days / plan.numerical_timestep_days),
            dt=plan.numerical_timestep_days, consumer_mode="mixed",
            initial_solution_p_um=p_micromolar,
            soil_radius_cm=plan.environment["soil_radius_cm"],
            soil_depth_cm=plan.environment["soil_depth_cm"],
            radial_interval_cm=plan.environment["radial_interval_cm"],
            depth_interval_cm=plan.environment["depth_interval_cm"],
        ),
        SpeciesParams(PlantTraits(), FungusTraits()),
    )
    return PolicyIntervalMycorMarl(
        numerical, decision_interval_days=plan.decision_interval_days,
        max_episode_steps=plan.policy_steps_per_episode,
    )


def ppo_config(plan: TradeOnlyStudyPlan) -> PPOConfig:
    """Construct the scalar-trade finite-horizon PPO configuration."""
    return PPOConfig(
        TOTAL_TIMESTEPS=plan.episodes_per_seed * plan.policy_steps_per_episode,
        RUN_TIMESTEPS=plan.policy_steps_per_episode,
        NUM_STEPS=plan.ppo["num_steps"], NUM_ENVS=plan.ppo["num_envs"],
        NUM_MINIBATCHES=plan.ppo["num_minibatches"],
        UPDATE_EPOCHS=plan.ppo["update_epochs"], GAE_LAMBDA=plan.ppo["gae_lambda"],
        LR=plan.ppo["learning_rate"],
        DISCOUNT_HALF_LIFE_DAYS=plan.ppo["discount_half_life_days"],
        FINITE_HORIZON_RETURNS=plan.ppo["finite_horizon_returns"],
        NORMALIZE_CRITIC_TARGETS=plan.ppo["normalize_critic_targets"],
        TRADE_ONLY=True,
    )


def _task_dir(output_dir: Path, p_micromolar: float, seed: int) -> Path:
    return output_dir / "mixed" / f"p-{p_micromolar:g}" / f"seed-{seed}"


def _checkpoint_path(task_dir: Path) -> Path:
    return task_dir / "checkpoint.msgpack"


def _write_checkpoint(path: Path, *, plan: TradeOnlyStudyPlan, p_micromolar: float,
                      seed: int, completed_episodes: int, runner_state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "mycormarl-trade-only-ippo-checkpoint", "format_version": 1,
        "plan": plan_payload(plan), "initial_p_micromolar": p_micromolar,
        "seed": seed, "completed_episodes": completed_episodes,
        "runner_state": serialization.to_state_dict(runner_state),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(serialization.msgpack_serialize(payload))
    temporary.replace(path)


def _restore_checkpoint(path: Path, *, plan: TradeOnlyStudyPlan, p_micromolar: float,
                        seed: int, environment: PolicyIntervalMycorMarl,
                        config: PPOConfig) -> tuple[int, Any]:
    payload = serialization.msgpack_restore(path.read_bytes())
    if (payload.get("format") != "mycormarl-trade-only-ippo-checkpoint"
            or payload.get("format_version") != 1
            or payload.get("plan") != plan_payload(plan)
            or payload.get("initial_p_micromolar") != p_micromolar
            or payload.get("seed") != seed):
        raise ValueError(f"incompatible trade-only checkpoint: {path}")
    completed = payload.get("completed_episodes")
    if not isinstance(completed, int) or not 0 < completed <= plan.episodes_per_seed:
        raise ValueError(f"invalid completed-episode count in {path}")
    template = initialize_runner_state(
        environment, config, derive_random_streams(seed), jax.random.PRNGKey(seed)
    )
    return completed, serialization.from_state_dict(template, payload["runner_state"])


def _write_evaluation(path: Path, *, plan: TradeOnlyStudyPlan, p_micromolar: float,
                      seed: int, completed_episodes: int, runner_state: Any,
                      environment: PolicyIntervalMycorMarl) -> None:
    actor = ActorCritic(trade_only=True)
    record = {
        "initial_p_micromolar": p_micromolar, "mode": "mixed", "seed": seed,
        "completed_episodes": completed_episodes, "evaluation_protocol": "latent-location",
        "fixed_allocation": plan.fixed_allocation,
        "evaluation": _evaluation_payload(evaluate_policy_parameters(
            environment, {agent: runner_state[0][agent].params for agent in (PLANT, FUNGUS)},
            episodes=1, protocol="latent-location", seed=seed, actor=actor,
        )),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _evaluation_payload(evaluation: Any) -> dict[str, Any]:
    """Serialise deterministic traces so summaries can separate commands and flows."""
    return {
        "protocol": evaluation.protocol,
        "episodes": [
            {"initial_state": episode.initial_state, "trace": list(episode.trace),
             "summary": episode.summary}
            for episode in evaluation.episodes
        ],
    }


def train_mixed_seed(
    plan: TradeOnlyStudyPlan,
    p_micromolar: float,
    seed: int,
    output_dir: Path,
    *,
    environment: PolicyIntervalMycorMarl | None = None,
    config: PPOConfig | None = None,
    resumed_update: Callable[..., Any] | None = None,
    status: Callable[[str, int], None] | None = None,
    emit_evaluation: bool = True,
) -> None:
    """Resume one mixed P/seed task only from a completed episode checkpoint.

    ``resumed_update`` is an optional shared, shape-compatible JIT executable.
    It is deliberately passed in by the P-level runner, never stored in a
    checkpoint; the checkpoint remains only model state and provenance.
    """
    environment = environment or study_environment(plan, p_micromolar)
    config = config or ppo_config(plan)
    task_dir = _task_dir(output_dir, p_micromolar, seed)
    checkpoint = _checkpoint_path(task_dir)
    completed, runner_state = (0, None) if not checkpoint.exists() else _restore_checkpoint(
        checkpoint, plan=plan, p_micromolar=p_micromolar, seed=seed,
        environment=environment, config=config,
    )
    if status is not None:
        status("resuming" if runner_state is not None else "initializing", completed)
    if runner_state is None:
        runner_state = initialize_runner_state(
            environment, config, derive_random_streams(seed), jax.random.PRNGKey(seed)
        )
    resumed_train = resumed_update or jax.jit(
        make_train(environment, config, initial_runner_state=runner_state)
    )
    evaluation_path = task_dir / "evaluation.json"
    if (emit_evaluation and completed and _evaluation_due(plan, completed)
            and not evaluation_path.exists()):
        _write_evaluation(evaluation_path, plan=plan, p_micromolar=p_micromolar,
                          seed=seed, completed_episodes=completed,
                          runner_state=runner_state, environment=environment)
    while completed < plan.episodes_per_seed:
        trained = resumed_train(jax.random.PRNGKey(seed), runner_state)
        runner_state = trained["runner_state"]
        completed += 1
        _write_checkpoint(checkpoint, plan=plan, p_micromolar=p_micromolar, seed=seed,
                          completed_episodes=completed, runner_state=runner_state)
        if status is not None:
            status("training", completed)
        if emit_evaluation and _evaluation_due(plan, completed):
            _write_evaluation(evaluation_path, plan=plan, p_micromolar=p_micromolar,
                              seed=seed, completed_episodes=completed,
                              runner_state=runner_state, environment=environment)
    if status is not None:
        status("complete", completed)


def _evaluation_due(plan: TradeOnlyStudyPlan, completed_episodes: int) -> bool:
    return (completed_episodes % plan.evaluation_interval_episodes == 0
            or completed_episodes == plan.episodes_per_seed)


def _has_matching_baseline(output_dir: Path, plan: TradeOnlyStudyPlan) -> bool:
    """Return whether ``output_dir`` already has the declared controls.

    A rejected zero-trade mixed control is still an intentional, completed
    control outcome, so compatibility is defined by its declared condition and
    protocol rather than the bundle's aggregate status.
    """
    plan_path = output_dir / "study-plan.json"
    baseline_path = output_dir / "fixed-allocation-baseline.json"
    try:
        stored_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_entries = {
        (mode, p_micromolar, plan.seeds[0])
        for mode in plan.modes for p_micromolar in plan.initial_p_micromolar
    }
    actual_entries = {
        (entry.get("mode"), entry.get("initial_p_micromolar"), entry.get("seed"))
        for entry in baseline.get("entries", [])
    }
    return (
        stored_plan == plan_payload(plan)
        and baseline.get("format") == "mycormarl-trade-only-baseline"
        and baseline.get("protocol", {}).get("fixed_allocation") == plan.fixed_allocation
        and actual_entries == expected_entries
        and len(baseline.get("entries", [])) == len(expected_entries)
    )


class _StudyProgress:
    """Atomically persist compact task status without per-episode log files."""

    def __init__(self, output_dir: Path, plan: TradeOnlyStudyPlan):
        self._path = output_dir / "progress.json"
        self._lock = threading.Lock()
        self._payload = {
            "format": "mycormarl-trade-only-ippo-progress",
            "format_version": 1,
            "plan": plan_payload(plan),
            "tasks": {
                f"p-{p_micromolar:g}/seed-{seed}": {
                    "initial_p_micromolar": p_micromolar,
                    "seed": seed,
                    "status": "pending",
                    "completed_episodes": 0,
                }
                for p_micromolar in plan.initial_p_micromolar for seed in plan.seeds
            },
        }
        self._write()

    def update(self, p_micromolar: float, seed: int, status: str,
               completed_episodes: int) -> None:
        with self._lock:
            task = self._payload["tasks"][f"p-{p_micromolar:g}/seed-{seed}"]
            task["status"] = status
            task["completed_episodes"] = completed_episodes
            self._write()

    def _write(self) -> None:
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._path)


def _compile_resumed_updater(
    environment: PolicyIntervalMycorMarl, config: PPOConfig, seed: int
) -> Callable[..., Any]:
    """Build one dynamic-state updater for all independent seeds at one P."""
    template = initialize_runner_state(
        environment, config, derive_random_streams(seed), jax.random.PRNGKey(seed)
    )
    return jax.jit(make_train(environment, config, initial_runner_state=template))


def _run_phosphorus_group(
    plan: TradeOnlyStudyPlan,
    p_micromolar: float,
    output_dir: Path,
    progress: _StudyProgress,
    emit_evaluation: bool = True,
) -> None:
    """Run all seed replicates for one P level through one update executable."""
    environment, config = study_environment(plan, p_micromolar), ppo_config(plan)
    updater = _compile_resumed_updater(environment, config, plan.seeds[0])
    for seed in plan.seeds:
        try:
            train_mixed_seed(
                plan, p_micromolar, seed, output_dir,
                environment=environment, config=config, resumed_update=updater,
                emit_evaluation=emit_evaluation,
                status=lambda state, completed, p=p_micromolar, s=seed:
                    progress.update(p, s, state, completed),
            )
        except Exception:
            progress.update(p_micromolar, seed, "failed", 0)
            task_dir = _task_dir(output_dir, p_micromolar, seed)
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise


def run_study(
    output_dir: Path,
    *,
    initial_p_micromolar: tuple[float, ...] = (0.75, 3.0, 10.0),
    workers: int = 2,
) -> None:
    """Reuse matching controls, then run one selectable mixed-P cohort."""
    plan = study_plan(initial_p_micromolar, workers=workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "study-plan.json"
    if plan_path.exists():
        try:
            existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot verify existing study plan: {plan_path}") from error
        if existing_plan != plan_payload(plan):
            raise ValueError(
                "output directory contains a different study plan; choose a new output directory"
            )
    have_baseline = _has_matching_baseline(output_dir, plan)
    plan_path.write_text(
        json.dumps(plan_payload(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not have_baseline:
        baseline = run_trade_only_baseline(
            initial_p_micromolar=plan.initial_p_micromolar, seeds=(plan.seeds[0],),
            days=plan.horizon_days, timestep_days=plan.numerical_timestep_days,
        )
        (output_dir / "fixed-allocation-baseline.json").write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    progress = _StudyProgress(output_dir, plan)
    with ThreadPoolExecutor(max_workers=plan.workers) as executor:
        futures = [
            executor.submit(_run_phosphorus_group, plan, p_micromolar, output_dir, progress)
            for p_micromolar in plan.initial_p_micromolar
        ]
        for future in futures:
            future.result()
