"""Deterministic static-policy controls for uniform initial-P treatments."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp

from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_units import MICROMOL_P_TO_MG_P
from mycormarl.soil.phosphate_grid import labile_amount_to_solution_concentration


_AGENTS = (PLANT, FUNGUS)
_POOL_TOLERANCE = 1e-6
def _species(manifest: dict[str, Any]) -> SpeciesParams:
    declarations = manifest.get("model", {}).get("species", {})
    plant = PlantTraits()
    fungus = FungusTraits()
    for name, values, current in ((PLANT, declarations.get(PLANT, {}), plant), (FUNGUS, declarations.get(FUNGUS, {}), fungus)):
        if not isinstance(values, dict):
            raise ValueError(f"species {name} must be an object")
        unknown = set(values) - set(current.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {name} species traits: {sorted(unknown)}")
        try:
            updated = replace(current, **values)
        except TypeError as error:
            raise ValueError(f"invalid {name} species traits") from error
        if name == PLANT:
            plant = updated
        else:
            fungus = updated
    return SpeciesParams(plant=plant, fungus=fungus)


def _environment(manifest: dict[str, Any], mode: str, p_level: float) -> BaseMycorMarl:
    horizon = manifest["horizon"]
    model = manifest["model"]["environment"]
    config = EnvConfig(
        max_steps=round(horizon["days"] / horizon["timestep_days"]),
        dt=horizon["timestep_days"],
        consumer_mode=mode,
        initial_solution_p_um=p_level,
        initial_solution_p_depth_profile=model.get("initial_solution_p_depth_profile"),
        **{key: model[key] for key in (
            "soil_radius_cm", "soil_depth_cm", "radial_interval_cm",
            "depth_interval_cm", "b_p", "theta_water",
        ) if key in model},
    )
    return BaseMycorMarl(config=config, species=_species(manifest))


def _action(value: Any, agent: str) -> jax.Array:
    try:
        action = jnp.asarray(value, dtype=jnp.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Rate action for {agent}") from error
    if action.shape != (4,) or not bool(jnp.all(jnp.isfinite(action))):
        raise ValueError(f"invalid Rate action for {agent}")
    if bool(jnp.any(action < 0.0)):
        raise ValueError(f"invalid Rate action for {agent}")
    return action


def _actions(policy: Any) -> dict[str, jax.Array]:
    if not isinstance(policy, dict) or any(agent not in policy for agent in _AGENTS):
        raise ValueError("static_policy must declare plant and fungus actions")
    return {agent: _action(policy[agent], agent) for agent in _AGENTS}


def _condition(manifest: dict[str, Any], mode: str, p_level: float, seed: int) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        actions = _actions(manifest["static_policy"])
        env = _environment(manifest, mode, p_level)
    except ValueError as error:
        return {"mode": mode, "initial_p_micromolar": p_level, "seed": seed, "status": "rejected", "rejection_reasons": [str(error)]}

    _, state = env.reset(jax.random.PRNGKey(seed))
    initial_soil = float(jnp.sum(state.soil_labile_p) * MICROMOL_P_TO_MG_P)
    active_field = state.soil_labile_p > 0.0
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p,
        env.cell_volumes,
        env.config.theta_water,
        env.config.b_p,
    )
    positive_values = concentration[active_field]
    uniform = bool(jnp.all(jnp.isclose(positive_values, positive_values[0])))
    initial_p = float(
        jnp.sum(state.plant_p_pool)
        + jnp.sum(state.fungus_p_pool)
        + jnp.sum(state.plant_biomass) * env.species.plant.gamma_p
        + jnp.sum(state.fungus_biomass) * env.species.fungus.gamma_p
    )
    initial_total = initial_p + initial_soil
    record_limitation_trace = bool(manifest.get("record_limitation_trace", False))
    record_resource_accounting = bool(manifest.get("record_resource_accounting", False))

    def rollout(initial_state):
        def condition(carry):
            current_state, steps, *_ = carry
            return (steps < env.max_episode_steps) & ~current_state.terminal

        def advance(carry):
            previous, steps, uptake, transfers, deaths, growth, fitness, carbon_fixed, trace, prior_acquired, accounting = carry
            _, current, rewards, _, info = env.step_env(
                jax.random.PRNGKey(seed + steps + 1), previous, actions
            )
            plant_info = info[PLANT]
            fungus_info = info[FUNGUS]
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
            step_accounting = jnp.zeros((2, 2, 5))
            if record_resource_accounting:
                # Resource axis: C=0, P=1. Destination axis is acquired,
                # maintenance, growth, trade_in, trade_out.
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
                maintenance_fraction = maintenance_used / jnp.maximum(prior_acquired, 1e-12)
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
            return current, steps + 1, uptake, transfers, deaths, growth, fitness, carbon_fixed, trace, acquired, accounting

        return jax.lax.while_loop(
            condition,
            advance,
            (initial_state, jnp.array(0), jnp.zeros(2), jnp.zeros(2),
             jnp.zeros(2, dtype=jnp.int32), jnp.zeros(2), jnp.zeros(2), jnp.array(0.0),
             jnp.zeros((env.max_episode_steps, 2, 18)), jnp.zeros((2, 2)),
             jnp.zeros((2, 2, 5))),
        )

    state, steps, uptake_values, transfer_values, death_values, growth_values, fitness_values, carbon_fixed, trace_values, _, accounting_values = jax.jit(rollout)(state)
    uptake = {PLANT: float(uptake_values[0]), FUNGUS: float(uptake_values[1])}
    transfers = {
        "plant_c_out": float(transfer_values[0]),
        "fungus_p_out": float(transfer_values[1]),
    }
    biological_deaths = {PLANT: int(death_values[0]), FUNGUS: int(death_values[1])}

    final_soil = float(jnp.sum(state.soil_labile_p) * MICROMOL_P_TO_MG_P)
    final_pools = float(
        jnp.sum(state.plant_p_pool)
        + jnp.sum(state.fungus_p_pool)
        + jnp.sum(jnp.where(state.plant_dead, 0.0, state.plant_biomass))
        * env.species.plant.gamma_p
        + jnp.sum(jnp.where(state.fungus_dead, 0.0, state.fungus_biomass))
        * env.species.fungus.gamma_p
    )
    losses = float(sum(float(jnp.sum(getattr(state, field))) for field in (
        "cumulative_plant_p_mortality_loss_mg", "cumulative_fungus_p_mortality_loss_mg",
        "cumulative_plant_p_maintenance_loss_mg", "cumulative_fungus_p_maintenance_loss_mg",
        "cumulative_plant_p_reproduction_export_mg", "cumulative_fungus_p_reproduction_export_mg",
    )))
    residual = initial_total - final_soil - final_pools - losses
    if not uniform and env.config.initial_solution_p_depth_profile is None:
        reasons.append("uniform initial P verification failed")
    if float(jnp.min(state.soil_labile_p)) < -_POOL_TOLERANCE:
        reasons.append("negative soil P pool")
    for agent, dead in ((PLANT, state.plant_dead), (FUNGUS, state.fungus_dead)):
        if bool(jnp.any(dead)) and (mode == "mixed" or agent == PLANT):
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

    p_loss_or_export_counters = {
        "plant_mortality": float(jnp.sum(state.cumulative_plant_p_mortality_loss_mg)),
        "fungus_mortality": float(jnp.sum(state.cumulative_fungus_p_mortality_loss_mg)),
        "plant_maintenance": float(jnp.sum(state.cumulative_plant_p_maintenance_loss_mg)),
        "fungus_maintenance": float(jnp.sum(state.cumulative_fungus_p_maintenance_loss_mg)),
        "plant_reproduction": float(jnp.sum(state.cumulative_plant_p_reproduction_export_mg)),
        "fungus_reproduction": float(jnp.sum(state.cumulative_fungus_p_reproduction_export_mg)),
    }
    result = {
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
    if record_resource_accounting:
        destinations = ("acquired", "maintenance", "growth", "trade_in", "trade_out")
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
                        "acquired_c": float(row[agent_index, 7]),
                        "acquired_p": float(row[agent_index, 8]),
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
    return result


def run_static_controls(manifest: dict[str, Any]) -> dict[str, Any]:
    """Execute every declared mode/P/seed condition with one static action."""
    required = ("horizon", "model", "modes", "initial_p_micromolar", "seeds", "static_policy")
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError("static-control manifest missing: " + ", ".join(missing))
    if set(manifest["modes"]) - {"mixed", "plant-only"}:
        raise ValueError("static controls support only mixed and plant-only modes")
    entries = [
        _condition(manifest, mode, p_level, seed) for mode in manifest["modes"] for p_level in manifest["initial_p_micromolar"] for seed in manifest["seeds"]
    ]
    rejected = sum(entry["status"] == "rejected" for entry in entries)
    return {
        "format": "mycormarl-static-controls",
        "format_version": 1,
        "status": "rejected" if rejected else "complete",
        "entries": entries,
        "completion": {
            "completed": len(entries) - rejected,
            "requested": len(entries)
        }
    }


def run_batched_static_controls(manifest: dict[str, Any]) -> dict[str, Any]:
    """Run one homogeneous static policy over all P levels in a single batch.

    This narrow fast path is for controls such as the trade-only mixed baseline:
    one consumer mode, one seed, no optional traces, and multiple initial-P
    states.  Initial P is used only by ``reset``; all subsequent dynamics share
    the same environment executable, so states can safely be vmapped.
    """
    required = ("horizon", "model", "modes", "initial_p_micromolar", "seeds", "static_policy")
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError("static-control manifest missing: " + ", ".join(missing))
    if len(manifest["modes"]) != 1 or len(manifest["seeds"]) != 1:
        raise ValueError("batched static controls require exactly one mode and one seed")
    if manifest.get("record_limitation_trace") or manifest.get("record_resource_accounting"):
        raise ValueError("batched static controls do not support optional traces")
    mode, seed = manifest["modes"][0], int(manifest["seeds"][0])
    actions = _actions(manifest["static_policy"])
    p_levels = [float(value) for value in manifest["initial_p_micromolar"]]
    environments = [_environment(manifest, mode, p_level) for p_level in p_levels]
    if any(env.config.initial_solution_p_depth_profile is not None for env in environments):
        raise ValueError("batched static controls do not support P depth profiles")
    states = []
    initial_totals = []
    initial_soils = []
    for environment in environments:
        _, state = environment.reset(jax.random.PRNGKey(seed))
        states.append(state)
        initial_soil = float(jnp.sum(state.soil_labile_p) * MICROMOL_P_TO_MG_P)
        initial_p = float(
            jnp.sum(state.plant_p_pool) + jnp.sum(state.fungus_p_pool)
            + jnp.sum(state.plant_biomass) * environment.species.plant.gamma_p
            + jnp.sum(state.fungus_biomass) * environment.species.fungus.gamma_p
        )
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
            current, steps, uptake, transfers, deaths, growth, fitness, carbon_fixed = carry
            active = ~current.terminal
            keys = jnp.broadcast_to(jax.random.PRNGKey(seed + 1), (batch_size, 2))
            _, updated, rewards, _, info = batched_step(keys, current, actions)
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
            ), axis=1).astype(jnp.int32)
            growth = growth + jnp.stack((
                info[PLANT]["growth"][:, 0], info[FUNGUS]["growth"][:, 0],
            ), axis=1) * active_float[:, None]
            fitness = fitness + jnp.stack((rewards[PLANT], rewards[FUNGUS]), axis=1) * active_float[:, None]
            carbon_fixed = carbon_fixed + info[PLANT]["carbon_fixed"][:, 0] * active_float
            return next_state, steps + active.astype(jnp.int32), uptake, transfers, deaths, growth, fitness, carbon_fixed

        return jax.lax.fori_loop(
            0, environment.max_episode_steps, advance,
            (initial_state, jnp.zeros(batch_size, dtype=jnp.int32),
             jnp.zeros((batch_size, 2)), jnp.zeros((batch_size, 2)),
             jnp.zeros((batch_size, 2), dtype=jnp.int32), jnp.zeros((batch_size, 2)),
             jnp.zeros((batch_size, 2)), jnp.zeros(batch_size)),
        )

    state, steps, uptake, transfers, deaths, growth, fitness, carbon_fixed = jax.jit(rollout)(batch_state)
    entries = []
    for index, (p_level, initial_soil, initial_total) in enumerate(
        zip(p_levels, initial_soils, initial_totals, strict=True)
    ):
        final_soil = float(jnp.sum(state.soil_labile_p[index]) * MICROMOL_P_TO_MG_P)
        final_pools = float(
            jnp.sum(state.plant_p_pool[index]) + jnp.sum(state.fungus_p_pool[index])
            + jnp.sum(jnp.where(state.plant_dead[index], 0.0, state.plant_biomass[index])) * environment.species.plant.gamma_p
            + jnp.sum(jnp.where(state.fungus_dead[index], 0.0, state.fungus_biomass[index])) * environment.species.fungus.gamma_p
        )
        counters = {
            "plant_mortality": float(jnp.sum(state.cumulative_plant_p_mortality_loss_mg[index])),
            "fungus_mortality": float(jnp.sum(state.cumulative_fungus_p_mortality_loss_mg[index])),
            "plant_maintenance": float(jnp.sum(state.cumulative_plant_p_maintenance_loss_mg[index])),
            "fungus_maintenance": float(jnp.sum(state.cumulative_fungus_p_maintenance_loss_mg[index])),
            "plant_reproduction": float(jnp.sum(state.cumulative_plant_p_reproduction_export_mg[index])),
            "fungus_reproduction": float(jnp.sum(state.cumulative_fungus_p_reproduction_export_mg[index])),
        }
        losses = sum(counters.values())
        residual = initial_total - final_soil - final_pools - losses
        reasons = []
        if float(jnp.min(state.soil_labile_p[index])) < -_POOL_TOLERANCE:
            reasons.append("negative soil P pool")
        for agent, dead in ((PLANT, state.plant_dead[index]), (FUNGUS, state.fungus_dead[index])):
            if bool(jnp.any(dead)):
                reasons.append(f"biological failure: {agent} died")
        if any(bool(jnp.any(getattr(state, field)[index] < 0.0))
               for field in ("plant_c_pool", "plant_p_pool", "fungus_c_pool", "fungus_p_pool")):
            reasons.append("negative organism resource pool")
        if abs(residual) > 1e-5 * max(1.0, initial_total):
            reasons.append("resource-accounting failure: P residual is nonzero")
        if initial_soil > 0.0 and final_soil <= 1e-12:
            reasons.append("pathological depletion: soil inventory exhausted")
        entries.append({
            "mode": mode, "initial_p_micromolar": p_level, "seed": seed,
            "status": "rejected" if reasons else "completed", "rejection_reasons": reasons,
            "steps": int(steps[index]), "uniform_initial_p": True,
            "initial_solution_p_profiled": False,
            "biomass": {PLANT: float(state.plant_biomass[index, 0]), FUNGUS: float(state.fungus_biomass[index, 0])},
            "final_living_biomass": {
                PLANT: 0.0 if bool(state.plant_dead[index, 0]) else float(state.plant_biomass[index, 0]),
                FUNGUS: 0.0 if bool(state.fungus_dead[index, 0]) else float(state.fungus_biomass[index, 0]),
            },
            "c_pools": {PLANT: float(state.plant_c_pool[index, 0]), FUNGUS: float(state.fungus_c_pool[index, 0])},
            "p_pools": {PLANT: float(state.plant_p_pool[index, 0]), FUNGUS: float(state.fungus_p_pool[index, 0])},
            "uptake": {PLANT: float(uptake[index, 0]), FUNGUS: float(uptake[index, 1])},
            "transfers": {"plant_c_out": float(transfers[index, 0]), "fungus_p_out": float(transfers[index, 1])},
            "biological_deaths": {PLANT: int(deaths[index, 0]), FUNGUS: int(deaths[index, 1])},
            "commanded_rate_actions": {agent: [float(value) for value in actions[agent]] for agent in _AGENTS},
            "cumulative_growth": {PLANT: float(growth[index, 0]), FUNGUS: float(growth[index, 1])},
            "cumulative_reproductive_fitness": {PLANT: float(fitness[index, 0]), FUNGUS: float(fitness[index, 1])},
            "cumulative_direct_p_uptake_mg": {PLANT: float(uptake[index, 0]), FUNGUS: float(uptake[index, 1])},
            "cumulative_carbon_fixed": float(carbon_fixed[index]),
            "soil_inventory_initial": initial_soil, "soil_inventory_final": final_soil,
            "p_loss_or_export": losses, "p_loss_or_export_counters": counters,
            "p_accounting_residual": residual, "terminated": bool(state.terminal[index]),
        })
    rejected = sum(entry["status"] == "rejected" for entry in entries)
    return {"format": "mycormarl-static-controls", "format_version": 1,
            "status": "rejected" if rejected else "complete", "entries": entries,
            "completion": {"completed": len(entries) - rejected, "requested": len(entries)}}
