"""Deterministic static-policy controls for uniform initial-P treatments."""

from __future__ import annotations

from typing import Any
from dataclasses import fields
import math
import time

import jax
import jax.numpy as jnp

from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.actions import rate_action
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_units import MICROMOL_P_TO_MG_P
from mycormarl.soil.phosphate_grid import labile_amount_to_solution_concentration


_AGENTS = (PLANT, FUNGUS)
_POOL_TOLERANCE = 1e-6
_RATE_RTOL, _RATE_ATOL = 1e-7, 1e-10


def _p_inventory(state, species):
    """Soil and organism P in mg, including retained dead structure."""
    soil = float(jnp.sum(state.soil_labile_p) * MICROMOL_P_TO_MG_P)
    organisms = float(
        jnp.sum(state.plant_p_pool) + jnp.sum(state.fungus_p_pool)
        + jnp.sum(state.plant_biomass) * species.plant.gamma_p
        + jnp.sum(state.fungus_biomass) * species.fungus.gamma_p
    )
    return soil, organisms


def _p_exports(state):
    """Actual losses/exports; internal uptake and trade are not exports."""
    return {
        f"{agent}_{name}": float(jnp.sum(getattr(state, f"cumulative_{agent}_p_{suffix}_mg")))
        for name, suffix in (("mortality", "mortality_loss"),
                             ("maintenance", "maintenance_loss"),
                             ("reproduction", "reproduction_export"))
        for agent in _AGENTS
    }


def _static_summary(env, state, actions, seed, initial_soil, initial_total,
                    steps, uptake_values, transfer_values, death_values,
                    growth_values, fitness_values, carbon_fixed,
                    guard_contact, allocation_ok, *, uniform=True):
    """Report one completed rollout with shared qualification and P accounting."""
    mode, p_level = env.config.consumer_mode, env.config.initial_solution_p_um
    reasons = []
    if not bool(allocation_ok):
        reasons.append('allocation integration failed: exhausted budget or non-finite result')
    if bool(guard_contact):
        reasons.append('plant numerical biomass guard contacted')
    uptake = {PLANT: float(uptake_values[0]), FUNGUS: float(uptake_values[1])}
    transfers = {
        "plant_c_out": float(transfer_values[0]),
        "fungus_p_out": float(transfer_values[1]),
    }
    biological_deaths = {PLANT: int(death_values[0]), FUNGUS: int(death_values[1])}

    final_soil, final_pools = _p_inventory(state, env.species)
    p_loss_or_export_counters = _p_exports(state)
    losses = sum(p_loss_or_export_counters.values())
    residual = initial_total - final_soil - final_pools - losses
    if not uniform and env.config.initial_solution_p_depth_profile is None:
        reasons.append("uniform initial P verification failed")
    if float(jnp.min(state.soil_labile_p)) < -_POOL_TOLERANCE:
        reasons.append("negative soil P pool")
    for agent, dead in ((PLANT, state.plant_dead), (FUNGUS, state.fungus_dead)):
        if bool(jnp.any(dead)) and (mode == "mixed" or mode == agent+'-only'):
            reasons.append(f"biological failure: {agent} died")
    if any(
        bool(jnp.any(getattr(state, field) < 0.0))
        for field in ("plant_c_pool", "plant_p_pool", "fungus_c_pool", "fungus_p_pool")
    ):
        reasons.append("negative organism resource pool")
    if abs(residual) > 1e-5 * max(1.0, initial_total):
        reasons.append("resource-accounting failure: P residual is nonzero")
    if initial_soil > 0.0 and final_soil <= 1e-12:
        reasons.append("pathological depletion: soil inventory exhausted")

    return {
        "mode": mode, "initial_p_micromolar": p_level, "seed": seed,
        "status": "rejected" if reasons else "completed",
        "rejection_reasons": reasons,
        "steps": int(steps), "uniform_initial_p": uniform,
        "initial_solution_p_profiled": env.config.initial_solution_p_depth_profile is not None,
        "biomass": {PLANT: float(state.plant_biomass[0]), FUNGUS: float(state.fungus_biomass[0])},
        "final_living_biomass": {
            PLANT: 0.0 if bool(state.plant_dead[0]) else float(state.plant_biomass[0]),
            FUNGUS: 0.0 if bool(state.fungus_dead[0]) else float(state.fungus_biomass[0]),
        },
        "c_pools": {PLANT: float(state.plant_c_pool[0]), FUNGUS: float(state.fungus_c_pool[0])},
        "p_pools": {PLANT: float(state.plant_p_pool[0]), FUNGUS: float(state.fungus_p_pool[0])},
        "uptake": uptake, "transfers": transfers, "biological_deaths": biological_deaths,
        "commanded_rate_actions": {agent: [float(value) for value in actions[agent]] for agent in _AGENTS},
        "cumulative_growth": {PLANT: float(growth_values[0]), FUNGUS: float(growth_values[1])},
        "cumulative_reproductive_fitness": {PLANT: float(fitness_values[0]), FUNGUS: float(fitness_values[1])},
        "cumulative_direct_p_uptake_mg": uptake,
        "cumulative_carbon_fixed": float(carbon_fixed),
        "soil_inventory_initial": initial_soil, "soil_inventory_final": final_soil,
        "p_loss_or_export": losses,
        "p_loss_or_export_counters": p_loss_or_export_counters,
        "p_accounting_residual": residual,
        "terminated": bool(state.terminal),
    }


def _validated_actions(actions):
    if set(actions) != set(_AGENTS):
        raise ValueError("static actions must specify plant and fungus")
    result = {}
    for agent in _AGENTS:
        value = jnp.asarray(actions[agent], dtype=jnp.float32)
        if value.shape != (3,) or not bool(jnp.all(jnp.isfinite(value))) or bool(jnp.any(value < 0)):
            raise ValueError(f"invalid Rate action for {agent}")
        result[agent] = value
    return result


def evaluate_static_allocation(env, *, interval_days, trade_shares,
                               reproductive_shares, seed,
                               target_residual_fraction=.001, total_rate_per_day=None):
    """Run a declared per-interval static strategy, distinct from raw rates.

    The total intensity is an isolated matched-pool turnover reference. Actual
    elemental depletion depends on stoichiometry, trade and other processes.
    Internal numerical substeps hold the resulting physical rates unchanged.
    """
    policy_interval = getattr(env, 'decision_interval_days', env.config.dt)
    if (not math.isfinite(interval_days) or interval_days <= 0.
            or not math.isclose(interval_days, env.config.dt, rel_tol=1e-10)
            or not math.isclose(interval_days, policy_interval, rel_tol=1e-10)):
        raise ValueError('static interval must be positive, finite and match policy/environment intervals')
    if not math.isfinite(target_residual_fraction) or not 0. < target_residual_fraction < 1.:
        raise ValueError('target_residual_fraction must be finite and strictly between zero and one')
    total_rate = -math.log(target_residual_fraction) / interval_days
    if not math.isfinite(total_rate) or total_rate > float(jnp.finfo(jnp.float32).max):
        raise ValueError('total allocation rate must be finite and representable in float32')
    if total_rate_per_day is not None and (
            not math.isfinite(total_rate_per_day) or total_rate_per_day <= 0.
            or not math.isclose(total_rate_per_day, total_rate, rel_tol=_RATE_RTOL, abs_tol=_RATE_ATOL)):
        raise ValueError(f'total_rate_per_day must match the declared interval and residual: {total_rate}')
    for name, shares in (('trade_shares',trade_shares), ('reproductive_shares',reproductive_shares)):
        if set(shares) != set(_AGENTS):
            raise ValueError(f'{name} must specify plant and fungus')
        if any(not math.isfinite(float(v)) or not 0. <= float(v) <= 1. for v in shares.values()):
            raise ValueError(f'{name} must be finite and within [0, 1]')
    actions, strategies = {}, {}
    for agent in _AGENTS:
        p, a = float(trade_shares[agent]), float(reproductive_shares[agent])
        biology = (1-p)*total_rate
        actions[agent] = rate_action(p*total_rate, (1-a)*biology, a*biology)
        strategies[agent] = dict(trade_share=p, reproductive_share=a,
                                reproduction_cost_scale=float(getattr(env.species, agent).reproduction_cost_scale))
    required_substeps = math.ceil(max(float(jnp.sum(action)) for action in actions.values())
                                 * env.config.dt / env.config.allocation_rate_step)
    if required_substeps > env.config.allocation_max_substeps:
        raise ValueError('static allocation requires more substeps than the allocation budget')
    result = evaluate_static_policy(env, actions, seed=seed, record_resource_accounting=True)
    result['static_allocation'] = dict(
        protocol='rate-static-allocation-v1', interval_days=float(interval_days),
        environment_interval_days=float(env.config.dt),
        policy_interval_days=float(policy_interval),
        target_residual_fraction=float(target_residual_fraction),
        total_rate_per_day=total_rate, strategies=strategies,
        supplied_total_rate_per_day=total_rate_per_day,
        total_rate_validation_tolerance=dict(relative=_RATE_RTOL, absolute_per_day=_RATE_ATOL),
        allocation_solver=dict(rate_step=env.config.allocation_rate_step,
                               max_substeps=env.config.allocation_max_substeps),
    )
    return result


def _checkpoint(state, observations, rewards, info):
    return {
        "state": {field.name: getattr(state, field.name) for field in fields(state)},
        "observations": observations,
        "rewards": rewards,
        "transitions": {
            agent: {field.name: getattr(info["transitions"][agent], field.name)
                    for field in fields(info["transitions"][agent])}
            for agent in _AGENTS
        },
        "flows": {
            agent: {name: info[agent][name] for name in (
                "growth", "trade_out", "trade_in", "trade_cancelled",
                "direct_p_uptake_mg", "reproduction_c", "reproduction_p",
                "maint_c_used", "maint_p_used", "c_deficit", "p_deficit",
                "allocation_converged",
            )} for agent in _AGENTS
        },
    }


def evaluate_static_policy(
    env: BaseMycorMarl,
    actions: dict[str, jax.Array],
    *,
    seed: int,
    record_limitation_trace: bool = False,
    record_resource_accounting: bool = False,
    checkpoint_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Execute fixed rates; optionally retain only selected completed steps.

    Checkpoint steps are strictly increasing positive integers within the horizon.
    The initial state and an early final state are included automatically.
    No checkpoint storage is allocated by default.
    """
    actions = _validated_actions(actions)
    if (any(type(step) is not int or not 1 <= step <= env.max_episode_steps
            for step in checkpoint_steps)
            or tuple(sorted(set(checkpoint_steps))) != tuple(checkpoint_steps)):
        raise ValueError("checkpoint_steps must be increasing unique integers within the horizon")
    initial_obs, state = env.reset(jax.random.PRNGKey(seed))
    snapshots = {}
    if checkpoint_steps:
        # Infer buffer shapes without executing or retaining a trajectory.
        obs_shape, state_shape, reward_shape, _, info_shape = jax.eval_shape(
            env.step_env, jax.random.PRNGKey(seed + 1), state, actions)
        template = _checkpoint(state_shape, obs_shape, reward_shape, info_shape)
        snapshots = jax.tree.map(
            lambda value: jnp.zeros((len(checkpoint_steps) + 1,) + value.shape, value.dtype),
            template)
        initial_checkpoint = {
            "state": {field.name: getattr(state, field.name) for field in fields(state)},
            "observations": initial_obs, "rewards": {agent: 0. for agent in _AGENTS},
            "transitions": None, "flows": None,
        }
    initial_soil, initial_p = _p_inventory(state, env.species)
    active_field = state.soil_labile_p > 0.0
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p,
        env.cell_volumes,
        env.config.theta_water,
        env.config.b_p,
    )
    positive_values = concentration[active_field]
    uniform = not positive_values.size or bool(jnp.all(jnp.isclose(positive_values, positive_values[0])))
    initial_total = initial_p + initial_soil

    def rollout(initial_state):
        def condition(carry):
            current_state, steps = carry[:2]
            guard_contact, allocation_ok = carry[-4:-2]
            return (steps < env.max_episode_steps) & ~current_state.terminal & allocation_ok & ~guard_contact

        def advance(carry):
            (previous, steps, uptake, transfers, deaths, growth, fitness,
             carbon_fixed, trace, prior_acquired, accounting, guard_contact,
             allocation_ok, captured, count) = carry
            observations, current, rewards, _, info = env.step_env(
                jax.random.PRNGKey(seed + steps + 1), previous, actions
            )
            plant_info = info[PLANT]
            fungus_info = info[FUNGUS]
            allocation_ok = allocation_ok & plant_info['allocation_converged'] & fungus_info['allocation_converged']
            guard_contact = guard_contact | jnp.any(plant_info['biomass_guard_contact'])
            uptake = uptake + jnp.array([
                plant_info["direct_p_uptake_mg"][0],
                fungus_info["direct_p_uptake_mg"][0],
            ])
            transfers = transfers + jnp.array([
                plant_info["trade_out"][0], fungus_info["trade_out"][0]
            ])
            deaths = deaths + jnp.array([
                info["transitions"][PLANT].operational_at_start
                & ~info["transitions"][PLANT].operational_at_end,
                info["transitions"][FUNGUS].operational_at_start
                & ~info["transitions"][FUNGUS].operational_at_end,
            ], dtype=jnp.int32)
            growth = growth + jnp.array([
                plant_info["growth"][0], fungus_info["growth"][0]
            ])
            fitness = fitness + jnp.array([rewards[PLANT], rewards[FUNGUS]])
            carbon_fixed = carbon_fixed + plant_info["carbon_fixed"][0]
            step_accounting = jnp.zeros((2, 2, 6))
            if record_resource_accounting:
                # Resource axis: C=0, P=1. Destination axis is acquired,
                # maintenance, growth, trade_in, trade_out, reproduction.
                step_accounting = step_accounting.at[:, :, 0].set(jnp.array([
                    [plant_info["carbon_fixed"][0], plant_info["direct_p_uptake_mg"][0] + plant_info["trade_in"][0]],
                    [fungus_info["trade_in"][0], fungus_info["direct_p_uptake_mg"][0]],
                ]))
                step_accounting = step_accounting.at[:, :, 1].set(jnp.array([
                    [plant_info["maint_c_used"][0], plant_info["maint_p_used"][0]],
                    [fungus_info["maint_c_used"][0], fungus_info["maint_p_used"][0]],
                ]))
                step_accounting = step_accounting.at[:, :, 2].set(jnp.array([
                    [plant_info["growth_c_used"][0], plant_info["growth_p_used"][0]],
                    [fungus_info["growth_c_used"][0], fungus_info["growth_p_used"][0]],
                ]))
                step_accounting = step_accounting.at[:, :, 3].set(jnp.array([
                    [0.0, plant_info["trade_in"][0]],
                    [fungus_info["trade_in"][0], 0.0],
                ]))
                step_accounting = step_accounting.at[:, :, 4].set(jnp.array([
                    [plant_info["trade_out"][0], 0.0],
                    [0.0, fungus_info["trade_out"][0]],
                ]))
                step_accounting = step_accounting.at[:, :, 5].set(jnp.array([
                    [plant_info["reproduction_c"][0], plant_info["reproduction_p"][0]],
                    [fungus_info["reproduction_c"][0], fungus_info["reproduction_p"][0]],
                ]))
                accounting = accounting + step_accounting
            acquired = jnp.zeros((2, 2))
            if record_limitation_trace:
                normalized_allocated = jnp.array([
                    plant_info["growth_c_allocated"][0] / env.species.plant.gamma_c,
                    plant_info["growth_p_allocated"][0] / env.species.plant.gamma_p,
                    fungus_info["growth_c_allocated"][0] / env.species.fungus.gamma_c,
                    fungus_info["growth_p_allocated"][0] / env.species.fungus.gamma_p,
                ]).reshape(2, 2)
                normalized_used = jnp.array([
                    plant_info["growth_c_used"][0] / env.species.plant.gamma_c,
                    plant_info["growth_p_used"][0] / env.species.plant.gamma_p,
                    fungus_info["growth_c_used"][0] / env.species.fungus.gamma_c,
                    fungus_info["growth_p_used"][0] / env.species.fungus.gamma_p,
                ]).reshape(2, 2)
                ratios = normalized_allocated / jnp.maximum(normalized_used, 1e-12)
                acquired = jnp.array([
                    plant_info["carbon_fixed"][0],
                    plant_info["direct_p_uptake_mg"][0] + plant_info["trade_in"][0],
                    fungus_info["trade_in"][0],
                    fungus_info["direct_p_uptake_mg"][0],
                ]).reshape(2, 2)
                maintenance_used = jnp.array([
                    plant_info["maint_c_used"][0], plant_info["maint_p_used"][0],
                    fungus_info["maint_c_used"][0], fungus_info["maint_p_used"][0],
                ]).reshape(2, 2)
                maintenance_fraction = jnp.where(steps > 0,
                    maintenance_used / jnp.maximum(prior_acquired, 1e-12), 0.)
                trade_out = jnp.array([plant_info["trade_out"][0], fungus_info["trade_out"][0]])
                trade_in = jnp.array([plant_info["trade_in"][0], fungus_info["trade_in"][0]])
                trade_out_normalized = jnp.array([
                    trade_out[0] / env.species.plant.gamma_c,
                    trade_out[1] / env.species.fungus.gamma_p,
                ])
                trade_in_normalized = jnp.array([
                    trade_in[0] / env.species.plant.gamma_p,
                    trade_in[1] / env.species.fungus.gamma_c,
                ])
                c_capacity = normalized_allocated[:, 0]
                p_capacity = normalized_allocated[:, 1]
                # Positive pressure denotes phosphate limitation: carbon has
                # greater biomass-equivalent growth capacity than phosphate.
                signed_pressure = c_capacity - p_capacity
                limiting_code = jnp.where(
                    (c_capacity <= 1e-12) & (p_capacity <= 1e-12), 0.0,
                    jnp.where(jnp.isclose(c_capacity, p_capacity, rtol=1e-5, atol=1e-12), 3.0,
                              jnp.where(c_capacity < p_capacity, 1.0, 2.0)),
                )
                trace_row = jnp.concatenate([
                    normalized_allocated, normalized_used, ratios,
                    acquired, maintenance_used, maintenance_fraction,
                    jnp.stack((trade_out, trade_in, trade_out_normalized, trade_in_normalized), axis=1),
                    signed_pressure[:, None],
                ], axis=1)
                trace_row = jnp.concatenate([trace_row, limiting_code[:, None]], axis=1)
                trace = trace.at[steps].set(trace_row)
            if checkpoint_steps:
                selected = jnp.any(steps + 1 == jnp.asarray(checkpoint_steps))
                selected |= current.terminal | guard_contact | ~allocation_ok
                captured = jax.lax.cond(
                    selected,
                    lambda buffers: jax.tree.map(
                        lambda buffer, value: buffer.at[count].set(value),
                        buffers, _checkpoint(current, observations, rewards, info)),
                    lambda buffers: buffers, captured)
                count += selected.astype(jnp.int32)
            return (current, steps + 1, uptake, transfers, deaths, growth, fitness,
                    carbon_fixed, trace, acquired, accounting, guard_contact,
                    allocation_ok, captured, count)

        return jax.lax.while_loop(
            condition,
            advance,
            (initial_state, jnp.array(0), jnp.zeros(2), jnp.zeros(2),
             jnp.zeros(2, dtype=jnp.int32), jnp.zeros(2), jnp.zeros(2), jnp.array(0.0),
             jnp.zeros((env.max_episode_steps if record_limitation_trace else 0, 2, 18)), jnp.zeros((2, 2)),
             jnp.zeros((2, 2, 6)), jnp.any(initial_state.plant_biomass >= env.species.plant.biomass_cap), jnp.asarray(True), snapshots, jnp.array(0)),
        )

    compiled = jax.jit(rollout)
    timing = None
    if checkpoint_steps:
        started = time.perf_counter()
        compiled = compiled.lower(state).compile()
        compilation = time.perf_counter() - started
        started = time.perf_counter()
    values = jax.block_until_ready(compiled(state))
    if checkpoint_steps:
        timing = {"compilation": compilation, "execution": time.perf_counter() - started}
    (state, steps, uptake_values, transfer_values, death_values, growth_values,
     fitness_values, carbon_fixed, trace_values, _, accounting_values,
     guard_contact, allocation_ok, captured, count) = values
    result = _static_summary(
        env, state, actions, seed, initial_soil, initial_total, steps,
        uptake_values, transfer_values, death_values, growth_values,
        fitness_values, carbon_fixed, guard_contact, allocation_ok, uniform=uniform)
    if record_resource_accounting:
        destinations = ("acquired", "maintenance", "growth", "trade_in", "trade_out", "reproduction")
        resources = ("c", "p")
        result["resource_accounting"] = {
            agent: {
                resource: {
                    destination: float(accounting_values[index, resource_index, destination_index])
                    for destination_index, destination in enumerate(destinations)
                }
                for resource_index, resource in enumerate(resources)
            }
            for index, agent in enumerate(_AGENTS)
        }
        result["resource_accounting_definition"] = {
            "resources": "C and P amounts in the model's native units",
            "destinations": "cumulative flows observed during each timestep; final pools are reported separately",
            "acquired": "photosynthetic fixation or direct uptake plus incoming trade",
            "growth": "resource consumed for realized structural biomass growth",
            "reproduction": "cost-scaled resource exports, distinct from reproductive output",
            "trade_in_out": "resource transferred between agents",
        }
    if record_limitation_trace:
        labels = ("none", "carbon", "phosphate", "balanced")
        trace = []
        for index in range(int(steps)):
            row = trace_values[index]
            trace.append({
                "step": index + 1,
                "day": (index + 1) * env.config.dt,
                "agents": {
                    agent: {
                        "allocated_c_normalized": float(row[agent_index, 0]),
                        "allocated_p_normalized": float(row[agent_index, 1]),
                        "used_c_normalized": float(row[agent_index, 2]),
                        "used_p_normalized": float(row[agent_index, 3]),
                        "allocated_to_used_c_ratio": float(row[agent_index, 4]),
                        "allocated_to_used_p_ratio": float(row[agent_index, 5]),
                        "limiting_resource": labels[int(row[agent_index, 17])],
                        "acquired_c": float(row[agent_index, 6]),
                        "acquired_p": float(row[agent_index, 7]),
                        "maintenance_c_used": float(row[agent_index, 8]),
                        "maintenance_p_used": float(row[agent_index, 9]),
                        "maintenance_fraction_of_prior_acquired_c": float(row[agent_index, 10]),
                        "maintenance_fraction_of_prior_acquired_p": float(row[agent_index, 11]),
                        "trade_out_raw": float(row[agent_index, 12]),
                        "trade_in_raw": float(row[agent_index, 13]),
                        "trade_out_gamma_normalized": float(row[agent_index, 14]),
                        "trade_in_recipient_gamma_normalized": float(row[agent_index, 15]),
                        "signed_pressure": float(row[agent_index, 16]),
                        "no_realized_growth": bool(float(row[agent_index, 2] + row[agent_index, 3]) <= 1e-12),
                    }
                    for agent, agent_index in ((PLANT, 0), (FUNGUS, 1))
                },
            })
        result["limitation_trace"] = trace
        result["limitation_trace_definition"] = {
            "normalized_units": "g biomass-equivalent resource = allocated_or_used_resource / gamma",
            "limiting_resource": "lower gamma-normalized growth allocation; none if both are zero; balanced if equal",
            "ratio": "gamma-normalized allocated growth resource / gamma-normalized used growth resource",
            "acquired_resources": {
                "plant_c": "photosynthetic carbon fixation",
                "plant_p": "direct soil-P uptake plus fungal P transfer",
                "fungus_c": "plant C transfer",
                "fungus_p": "direct soil-P uptake",
            },
            "maintenance_fraction": "maintenance resource used / corresponding resource acquired in the prior recorded step; the first step has no prior acquisition and is zero",
            "signed_pressure": "C-equivalent allocation minus P-equivalent allocation (positive P-limited, negative C-limited); no_realized_growth is a separate mask",
            "trade": "raw outgoing and incoming resource amounts plus outgoing/recipient gamma-normalized equivalents",
        }
    if checkpoint_steps:
        checkpoints = [initial_checkpoint] + [
            jax.tree.map(lambda value: value[index], captured) for index in range(int(count))]
        result["checkpoints"] = []
        result["timing_seconds"] = timing
        for checkpoint in checkpoints:
            row = jax.tree.map(lambda value: jnp.asarray(value).tolist(), checkpoint)
            row["day"] = row["state"]["step"] * env.config.dt
            result["checkpoints"].append(row)
    return result


def evaluate_batched_static_policy(environments: list[BaseMycorMarl], actions: dict[str, jax.Array], *, seed: int) -> list[dict[str, Any]]:
    """Execute one static action over homogeneous supplied environments."""
    actions = _validated_actions(actions)
    if not environments:
        raise ValueError("batched static rollout requires environments")
    reference = environments[0]
    from dataclasses import replace
    comparable = replace(reference.config, initial_solution_p_um=0.0)
    if any(replace(env.config, initial_solution_p_um=0.0) != comparable or env.species != reference.species for env in environments):
        raise ValueError("batched static rollout requires homogeneous dynamics and species")
    if any(env.config.initial_solution_p_depth_profile is not None for env in environments):
        raise ValueError("batched static controls do not support P depth profiles")
    states = []
    initial_totals = []
    initial_soils = []
    for environment in environments:
        _, state = environment.reset(jax.random.PRNGKey(seed))
        states.append(state)
        initial_soil, initial_p = _p_inventory(state, environment.species)
        initial_soils.append(initial_soil)
        initial_totals.append(initial_soil + initial_p)
    environment = environments[0]
    batch_state = jax.tree.map(lambda *values: jnp.stack(values), *states)
    batch_size = len(states)
    batched_step = jax.vmap(environment.step_env, in_axes=(0, 0, None))

    def retain_active(previous, updated, active):
        shape = (active.shape[0],) + (1,) * (updated.ndim - 1)
        return jnp.where(active.reshape(shape), updated, previous)

    def rollout(initial_state):
        def advance(_, carry):
            current, steps, uptake, transfers, deaths, growth, fitness, carbon_fixed, guard_contact, allocation_ok = carry
            active = ~current.terminal & allocation_ok & ~guard_contact
            keys = jnp.broadcast_to(jax.random.PRNGKey(seed + 1), (batch_size, 2))
            _, updated, rewards, _, info = batched_step(keys, current, actions)
            allocation_ok = allocation_ok & jnp.where(active,
                info[PLANT]['allocation_converged'] & info[FUNGUS]['allocation_converged'], True)
            guard_contact = guard_contact | (active & info[PLANT]['biomass_guard_contact'][:,0])
            next_state = jax.tree.map(
                lambda previous, next_value: retain_active(previous, next_value, active),
                current, updated,
            )
            active_float = active.astype(jnp.float32)
            uptake = uptake + jnp.stack((
                info[PLANT]["direct_p_uptake_mg"][:, 0],
                info[FUNGUS]["direct_p_uptake_mg"][:, 0],
            ), axis=1) * active_float[:, None]
            transfers = transfers + jnp.stack((
                info[PLANT]["trade_out"][:, 0], info[FUNGUS]["trade_out"][:, 0],
            ), axis=1) * active_float[:, None]
            deaths = deaths + jnp.stack((
                info["transitions"][PLANT].operational_at_start
                & ~info["transitions"][PLANT].operational_at_end,
                info["transitions"][FUNGUS].operational_at_start
                & ~info["transitions"][FUNGUS].operational_at_end,
            ), axis=1).astype(jnp.int32) * active[:,None]
            growth = growth + jnp.stack((
                info[PLANT]["growth"][:, 0], info[FUNGUS]["growth"][:, 0],
            ), axis=1) * active_float[:, None]
            fitness = fitness + jnp.stack((rewards[PLANT], rewards[FUNGUS]), axis=1) * active_float[:, None]
            carbon_fixed = carbon_fixed + info[PLANT]["carbon_fixed"][:, 0] * active_float
            return next_state, steps + active.astype(jnp.int32), uptake, transfers, deaths, growth, fitness, carbon_fixed, guard_contact, allocation_ok

        return jax.lax.fori_loop(
            0, environment.max_episode_steps, advance,
            (initial_state, jnp.zeros(batch_size, dtype=jnp.int32),
             jnp.zeros((batch_size, 2)), jnp.zeros((batch_size, 2)),
             jnp.zeros((batch_size, 2), dtype=jnp.int32), jnp.zeros((batch_size, 2)),
             jnp.zeros((batch_size, 2)), jnp.zeros(batch_size),
             initial_state.plant_biomass[:,0] >= environment.species.plant.biomass_cap,
             jnp.ones(batch_size,dtype=bool)),
        )

    state, steps, uptake, transfers, deaths, growth, fitness, carbon_fixed, guard_contact, allocation_ok = jax.jit(rollout)(batch_state)
    entries = []
    for index, (env, initial_soil, initial_total) in enumerate(
        zip(environments, initial_soils, initial_totals, strict=True)
    ):
        current = jax.tree.map(lambda value: value[index], state)
        entries.append(_static_summary(
            env, current, actions, seed, initial_soil, initial_total, steps[index],
            uptake[index], transfers[index], deaths[index], growth[index],
            fitness[index], carbon_fixed[index], guard_contact[index], allocation_ok[index]))
    return entries
