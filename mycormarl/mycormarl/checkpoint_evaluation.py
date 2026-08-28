"""Evaluate saved independent-PPO policies through production interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

from flax import serialization
import jax
import jax.numpy as jnp

from mycormarl.algos.ppo import ActorCritic, latent_to_rate_action
from mycormarl.trade_only import fixed_allocation_rate_action
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.policy_artifacts import (
    ACTOR_INTERFACE_VERSION,
    ENVIRONMENT_STATE_SCHEMA_VERSION,
)


EvaluationProtocol = Literal["latent-location", "sampled-policy"]
TRAINING_CHECKPOINT_FORMAT = "mycormarl-ppo-checkpoint"
TRAINING_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class EvaluationEpisode:
    """One complete evaluation trajectory and independently derived endpoints."""

    initial_state: dict[str, Any]
    trace: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class LatentLocationEvaluation:
    """Primary deterministic evaluation using each actor's latent location."""

    protocol: Literal["latent-location"]
    episodes: tuple[EvaluationEpisode, ...]


@dataclass(frozen=True)
class SampledPolicyEvaluation:
    """Stochastic diagnostic, deliberately distinct from primary outcomes."""

    protocol: Literal["sampled-policy"]
    episodes: tuple[EvaluationEpisode, ...]


CheckpointEvaluation = LatentLocationEvaluation | SampledPolicyEvaluation
_LEGACY_ACTOR_CONFIGURATION = {"activation": "tanh"}


def _actor_from_checkpoint_metadata(metadata: Mapping[str, Any]) -> ActorCritic:
    """Reconstruct the actor architecture that produced checkpoint parameters.

    Phase-1 checkpoints written before actor settings were persisted used the
    production PPO default, ``tanh``. Keeping that compatibility path explicit
    avoids silently evaluating them with the module's unrelated ReLU default.
    """
    configuration = metadata.get("actor_configuration")
    if configuration is None:
        configuration = _LEGACY_ACTOR_CONFIGURATION
    if not isinstance(configuration, Mapping):
        raise ValueError("training checkpoint actor configuration is invalid")
    activation = configuration.get("activation")
    if activation not in {"relu", "tanh"}:
        raise ValueError("training checkpoint actor activation is unsupported")
    return ActorCritic(activation=activation)


def _plain(value: Any) -> Any:
    """Turn JAX values into JSON-compatible immutable Python values."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _scalar(value: Any) -> float:
    return float(jnp.asarray(value).reshape(-1)[0])


def _limitation(info: Mapping[str, Any]) -> str:
    c_limited = _scalar(info["c_deficit"]) > 0.0
    p_limited = _scalar(info["p_deficit"]) > 0.0
    if c_limited and p_limited:
        return "carbon-and-phosphorus"
    if c_limited:
        return "carbon"
    if p_limited:
        return "phosphorus"
    return "none"


def _state_snapshot(state: Any) -> dict[str, Any]:
    """Keep every state quantity needed to recompute reported endpoints."""
    return {
        "raw_biomass": {
            PLANT: _scalar(state.plant_biomass),
            FUNGUS: _scalar(state.fungus_biomass),
        },
        "living_biomass": {
            PLANT: 0.0 if bool(_scalar(state.plant_dead)) else _scalar(state.plant_biomass),
            FUNGUS: 0.0 if bool(_scalar(state.fungus_dead)) else _scalar(state.fungus_biomass),
        },
        "pools": {
            PLANT: {"carbon": _scalar(state.plant_c_pool), "phosphorus": _scalar(state.plant_p_pool)},
            FUNGUS: {"carbon": _scalar(state.fungus_c_pool), "phosphorus": _scalar(state.fungus_p_pool)},
        },
        "soil_inventory_micromol": float(jnp.sum(state.soil_labile_p)),
        "loss_counters": {
            "plant_p_mortality_mg": _scalar(state.cumulative_plant_p_mortality_loss_mg),
            "fungus_p_mortality_mg": _scalar(state.cumulative_fungus_p_mortality_loss_mg),
            "plant_p_maintenance_mg": _scalar(state.cumulative_plant_p_maintenance_loss_mg),
            "fungus_p_maintenance_mg": _scalar(state.cumulative_fungus_p_maintenance_loss_mg),
            "plant_p_reproduction_export_mg": _scalar(state.cumulative_plant_p_reproduction_export_mg),
            "fungus_p_reproduction_export_mg": _scalar(state.cumulative_fungus_p_reproduction_export_mg),
        },
    }


def _relative_growth_rate(initial: float, final: float, days: float) -> float | None:
    if initial <= 0.0 or final <= 0.0 or days <= 0.0:
        return None
    return math.log(final / initial) / days


def _episode_summary(
    initial: dict[str, Any], trace: list[dict[str, Any]], dt_days: float
) -> dict[str, Any]:
    final = trace[-1]["state"]
    duration_days = len(trace) * dt_days
    cumulative_fitness = {
        agent: sum(row["reproductive_fitness"][agent] for row in trace)
        for agent in (PLANT, FUNGUS)
    }
    cumulative_gross_growth = {
        agent: sum(row["gross_growth"][agent] for row in trace)
        for agent in (PLANT, FUNGUS)
    }
    return {
        "duration_days": duration_days,
        "steps": len(trace),
        "cumulative_reproductive_fitness": cumulative_fitness,
        "final_raw_biomass": final["raw_biomass"],
        "final_living_biomass": final["living_biomass"],
        "cumulative_gross_growth": cumulative_gross_growth,
        "episode_rgr_per_day": {
            agent: _relative_growth_rate(
                initial["living_biomass"][agent], final["living_biomass"][agent], duration_days
            )
            for agent in (PLANT, FUNGUS)
        },
        "administratively_truncated": trace[-1]["truncated"],
        "biologically_terminated": {
            agent: any(row["biological_termination"][agent] for row in trace)
            for agent in (PLANT, FUNGUS)
        },
    }


def save_evaluation_artifact(
    path: str | Path,
    evaluation: CheckpointEvaluation,
    *,
    checkpoint: str | Path,
) -> None:
    """Persist the typed result with all trajectory facts used by its summary."""
    artifact = {
        "format": "mycormarl-checkpoint-evaluation",
        "format_version": 1,
        "source_checkpoint": str(checkpoint),
        "protocol": evaluation.protocol,
        "episodes": [
            {
                "initial_state": episode.initial_state,
                "trace": list(episode.trace),
                "summary": episode.summary,
            }
            for episode in evaluation.episodes
        ],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_evaluation_summary_artifact(
    path: str | Path,
    summary: dict[str, Any],
    *,
    checkpoint: str | Path,
) -> None:
    """Persist compact stopping metrics for a non-terminal checkpoint."""
    artifact = {
        "format": "mycormarl-checkpoint-evaluation-summary",
        "format_version": 1,
        "source_checkpoint": str(checkpoint),
        "protocol": "latent-location",
        "summary": summary,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _actions(
    actor: ActorCritic,
    parameters: Mapping[str, Any],
    observations: Mapping[str, Any],
    protocol: EvaluationProtocol,
    key: jax.Array,
) -> tuple[dict[str, Any], jax.Array]:
    actions = {}
    for agent in (PLANT, FUNGUS):
        policy, _ = actor.apply(parameters[agent], observations[agent])
        if protocol == "latent-location":
            trade_latent = policy.trade_loc
            biological_rate_latent = policy.biological_rate_loc
        else:
            key, trade_key, biological_rate_key = jax.random.split(key, 3)
            trade_latent = policy.trade_loc + jnp.exp(policy.trade_log_std) * jax.random.normal(trade_key, policy.trade_loc.shape)
            biological_rate_latent = (
                policy.biological_rate_loc if actor.trade_only else
                policy.biological_rate_loc + jnp.exp(policy.biological_rate_log_std)
                * jax.random.normal(biological_rate_key, policy.biological_rate_loc.shape)
            )
        actions[agent] = (
            fixed_allocation_rate_action(jax.nn.softplus(trade_latent))
            if actor.trade_only else latent_to_rate_action(
                trade_latent, biological_rate_latent
            )
        )
    return actions, key


def _scan_episode(
    environment: BaseMycorMarl,
    actor: ActorCritic,
    parameters: Mapping[str, Any],
    *,
    protocol: EvaluationProtocol,
    key: jax.Array,
) -> tuple[dict[str, Any], tuple[Any, ...], jax.Array]:
    """Run one evaluation horizon in one compiled scan."""
    key, reset_key = jax.random.split(key)
    observations, state = environment.reset(reset_key)
    initial_state = _state_snapshot(state)

    def step(carry: tuple[Any, Any, Any], _: None):
        key, observations, state = carry
        actions, key = _actions(actor, parameters, observations, protocol, key)
        key, step_key = jax.random.split(key)
        observations, state, rewards, dones, infos = environment.step_env(
            step_key, state, actions
        )
        return (key, observations, state), (actions, rewards, infos, state, dones)

    final_carry, trajectory = jax.lax.scan(
        step, (key, observations, state), None, length=environment.max_episode_steps
    )
    return initial_state, trajectory, final_carry[0]


_SUMMARY_RUNNERS: dict[tuple[BaseMycorMarl, EvaluationProtocol, str], Any] = {}


def _compiled_summary_runner(
    environment: BaseMycorMarl,
    protocol: EvaluationProtocol,
    actor: ActorCritic,
) -> Any:
    """Cache the compiled summary rollout for repeated checkpoints."""
    cache_key = (environment, protocol, actor.activation)
    runner = _SUMMARY_RUNNERS.get(cache_key)
    if runner is not None:
        return runner
    def run(parameters: Mapping[str, Any], key: jax.Array):
        key, reset_key = jax.random.split(key)
        observations, state = environment.reset(reset_key)

        def step(carry: tuple[Any, Any, Any], _: None):
            key, observations, state = carry
            actions, key = _actions(actor, parameters, observations, protocol, key)
            key, step_key = jax.random.split(key)
            observations, state, rewards, dones, _ = environment.step_env(
                step_key, state, actions
            )
            return (key, observations, state), (actions, rewards, dones["__all__"])

        final_carry, trajectory = jax.lax.scan(
            step, (key, observations, state), None, length=environment.max_episode_steps
        )
        return final_carry[0], trajectory, final_carry[2]

    runner = jax.jit(run)
    _SUMMARY_RUNNERS[cache_key] = runner
    return runner


def evaluate_policy_parameters(
    environment: BaseMycorMarl,
    parameters: Mapping[str, Any],
    *,
    episodes: int,
    protocol: EvaluationProtocol = "latent-location",
    seed: int = 0,
    actor: ActorCritic | None = None,
) -> CheckpointEvaluation:
    """Evaluate production actor parameters in the production environment."""
    if protocol not in ("latent-location", "sampled-policy"):
        raise ValueError("protocol must be 'latent-location' or 'sampled-policy'")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if set(parameters) != {PLANT, FUNGUS}:
        raise ValueError("parameters must contain exactly plant and fungus actors")

    actor = ActorCritic() if actor is None else actor
    key = jax.random.PRNGKey(seed)
    result_episodes = []
    for _ in range(episodes):
        initial_state, trajectory, key = _scan_episode(
            environment, actor, parameters, protocol=protocol, key=key
        )
        actions_history, rewards_history, infos_history, states_history, dones_history = trajectory
        done_values = _plain(dones_history["__all__"])
        episode_length = next(
            (index + 1 for index, done in enumerate(done_values) if bool(done)),
            environment.max_episode_steps,
        )
        previous_state = initial_state
        trace = []
        for index in range(episode_length):
            actions = jax.tree_util.tree_map(lambda value: value[index], actions_history)
            rewards = jax.tree_util.tree_map(lambda value: value[index], rewards_history)
            infos = jax.tree_util.tree_map(lambda value: value[index], infos_history)
            state = jax.tree_util.tree_map(lambda value: value[index], states_history)
            transitions = infos["transitions"]
            state_snapshot = _state_snapshot(state)
            trace.append({
                "time_days": _scalar(state.step) * environment.config.dt,
                "actions": _plain(actions),
                "realised_actions": {
                    agent: _plain(transitions[agent].realised_action)
                    for agent in (PLANT, FUNGUS)
                },
                "allocation_executed": {
                    agent: bool(transitions[agent].allocation_executed)
                    for agent in (PLANT, FUNGUS)
                },
                "trade_executed": {
                    agent: bool(transitions[agent].trade_executed)
                    for agent in (PLANT, FUNGUS)
                },
                "reproductive_fitness": _plain(rewards),
                "gross_growth": {agent: _scalar(infos[agent]["growth"]) for agent in (PLANT, FUNGUS)},
                "interval_rgr_per_day": {
                    agent: _relative_growth_rate(
                        previous_state["living_biomass"][agent],
                        state_snapshot["living_biomass"][agent],
                        environment.config.dt,
                    )
                    for agent in (PLANT, FUNGUS)
                },
                "limitation": {agent: _limitation(infos[agent]) for agent in (PLANT, FUNGUS)},
                "uptake_mg": {
                    agent: _scalar(infos[agent]["direct_p_uptake_mg"])
                    for agent in (PLANT, FUNGUS)
                },
                "maintenance_p_required_mg": {
                    agent: _scalar(infos[agent]["maintenance_p_required_mg"])
                    for agent in (PLANT, FUNGUS)
                },
                "transfers": {agent: {"proposed_out": _scalar(infos[agent]["proposed_trade_out"]), "out": _scalar(infos[agent]["trade_out"]), "in": _scalar(infos[agent]["trade_in"])} for agent in (PLANT, FUNGUS)},
                "operational_at_start": {agent: bool(transitions[agent].operational_at_start) for agent in (PLANT, FUNGUS)},
                "operational_at_end": {agent: bool(transitions[agent].operational_at_end) for agent in (PLANT, FUNGUS)},
                "biological_termination": {agent: bool(transitions[agent].operational_at_start and not transitions[agent].operational_at_end) for agent in (PLANT, FUNGUS)},
                "truncated": bool(transitions[PLANT].truncated),
                "state": state_snapshot,
            })
            previous_state = state_snapshot
        result_episodes.append(EvaluationEpisode(
            initial_state,
            tuple(trace),
            _episode_summary(initial_state, trace, environment.config.dt),
        ))

    if protocol == "latent-location":
        return LatentLocationEvaluation("latent-location", tuple(result_episodes))
    return SampledPolicyEvaluation("sampled-policy", tuple(result_episodes))


def evaluate_policy_summary(
    environment: BaseMycorMarl,
    parameters: Mapping[str, Any],
    *,
    episodes: int,
    protocol: EvaluationProtocol = "latent-location",
    seed: int = 0,
    actor: ActorCritic | None = None,
) -> dict[str, Any]:
    """Evaluate only stopping metrics without materialising a full trace."""
    if protocol not in ("latent-location", "sampled-policy"):
        raise ValueError("protocol must be 'latent-location' or 'sampled-policy'")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if set(parameters) != {PLANT, FUNGUS}:
        raise ValueError("parameters must contain exactly plant and fungus actors")

    actor = ActorCritic() if actor is None else actor
    key = jax.random.PRNGKey(seed)
    episode_metrics = []
    for _ in range(episodes):
        runner = _compiled_summary_runner(environment, protocol, actor)
        key, trajectory, final_state = runner(parameters, key)
        actions, rewards, done_flags = trajectory
        done_flags = jnp.asarray(done_flags, dtype=bool)
        has_termination = jnp.any(done_flags)
        termination_index = jnp.argmax(done_flags)
        episode_length = jnp.where(
            has_termination, termination_index + 1, done_flags.shape[0]
        )
        active = jnp.arange(done_flags.shape[0]) < episode_length
        episode_metrics.append({
            "fitness": {
                agent: jnp.sum(jnp.where(active, rewards[agent], 0.0))
                for agent in (PLANT, FUNGUS)
            },
            "actions": {
                agent: jnp.sum(
                    jnp.where(active[:, None], actions[agent], 0.0), axis=0
                ) / jnp.maximum(episode_length, 1)
                for agent in (PLANT, FUNGUS)
            },
            "final_raw_biomass": {
                PLANT: _scalar(final_state.plant_biomass),
                FUNGUS: _scalar(final_state.fungus_biomass),
            },
            "final_living_biomass": {
                PLANT: 0.0 if bool(_scalar(final_state.plant_dead)) else _scalar(final_state.plant_biomass),
                FUNGUS: 0.0 if bool(_scalar(final_state.fungus_dead)) else _scalar(final_state.fungus_biomass),
            },
        })
    return {
        "fitness": {
            agent: float(jnp.mean(jnp.asarray([
                item["fitness"][agent] for item in episode_metrics
            ])))
            for agent in (PLANT, FUNGUS)
        },
        "actions": {
            agent: jnp.mean(jnp.stack([
                item["actions"][agent] for item in episode_metrics
            ]), axis=0).tolist()
            for agent in (PLANT, FUNGUS)
        },
        "final_raw_biomass": {
            agent: float(jnp.mean(jnp.asarray([
                item["final_raw_biomass"][agent] for item in episode_metrics
            ])))
            for agent in (PLANT, FUNGUS)
        },
        "final_living_biomass": {
            agent: float(jnp.mean(jnp.asarray([
                item["final_living_biomass"][agent] for item in episode_metrics
            ])))
            for agent in (PLANT, FUNGUS)
        },
    }


def evaluate_checkpoint(
    path: str | Path,
    environment: BaseMycorMarl,
    *,
    episodes: int,
    protocol: EvaluationProtocol = "latent-location",
    seed: int = 0,
) -> CheckpointEvaluation:
    """Evaluate a versioned training checkpoint through its saved actor states."""
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if (
        not isinstance(payload, dict)
        or payload.get("format") != TRAINING_CHECKPOINT_FORMAT
        or payload.get("format_version") != TRAINING_CHECKPOINT_VERSION
        or metadata.get("actor_interface_version") != ACTOR_INTERFACE_VERSION
        or metadata.get("environment_state_schema_version") != ENVIRONMENT_STATE_SCHEMA_VERSION
    ):
        raise ValueError("incompatible training checkpoint")
    try:
        parameters = {
            PLANT: payload["runner_state"]["0"][PLANT]["params"],
            FUNGUS: payload["runner_state"]["0"][FUNGUS]["params"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("training checkpoint does not contain actor parameters") from error
    return evaluate_policy_parameters(
        environment,
        parameters,
        episodes=episodes,
        protocol=protocol,
        seed=seed,
        actor=_actor_from_checkpoint_metadata(metadata),
    )


def evaluate_checkpoint_summary(
    path: str | Path,
    environment: BaseMycorMarl,
    *,
    episodes: int,
    protocol: EvaluationProtocol = "latent-location",
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate checkpoint stopping metrics without building a full artifact."""
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if (
        not isinstance(payload, dict)
        or payload.get("format") != TRAINING_CHECKPOINT_FORMAT
        or payload.get("format_version") != TRAINING_CHECKPOINT_VERSION
        or metadata.get("actor_interface_version") != ACTOR_INTERFACE_VERSION
        or metadata.get("environment_state_schema_version") != ENVIRONMENT_STATE_SCHEMA_VERSION
    ):
        raise ValueError("incompatible training checkpoint")
    try:
        parameters = {
            PLANT: payload["runner_state"]["0"][PLANT]["params"],
            FUNGUS: payload["runner_state"]["0"][FUNGUS]["params"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("training checkpoint does not contain actor parameters") from error
    return evaluate_policy_summary(
        environment,
        parameters,
        episodes=episodes,
        protocol=protocol,
        seed=seed,
        actor=_actor_from_checkpoint_metadata(metadata),
    )
