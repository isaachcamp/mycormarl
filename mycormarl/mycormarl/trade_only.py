"""Fixed-allocation action contracts for the reduced-control trade study."""

from __future__ import annotations

import math

import jax.numpy as jnp

from mycormarl.environments.base_mycor import FUNGUS, PLANT
from mycormarl.static_controls import run_batched_static_controls, run_static_controls


TOTAL_BIOLOGICAL_RATE_PER_DAY = 6.907755
GROWTH_FRACTION = 0.9
REPRODUCTION_FRACTION = 0.1
PLANT_MIXED_TRADE_FRACTION_PER_DAY = 0.05
FUNGUS_MIXED_TRADE_FRACTION_PER_DAY = 0.75


def pool_fraction_to_rate(fraction_per_day: float) -> float:
    """Convert a daily fraction of the current pool to a Rate-action hazard.

    The environment applies a held Rate action as ``1 - exp(-rate * dt)`` of
    the post-maintenance resource pool during each numerical substep.  This
    conversion therefore makes a rate held for one day exchange the declared
    fraction of an otherwise unchanged current pool.
    """
    if not 0.0 <= fraction_per_day < 1.0:
        raise ValueError("pool fraction per day must lie in [0, 1)")
    return -math.log1p(-fraction_per_day)


def fixed_allocation_rate_action(trade_rate: jnp.ndarray) -> jnp.ndarray:
    """Return a normal Rate action with static 90/10 biology and zero storage.

    ``trade_rate`` is already physical and must be non-negative.  PPO maps its
    scalar latent to this quantity with softplus before crossing this seam.
    """
    trade_rate = jnp.asarray(trade_rate, dtype=jnp.float32)
    return jnp.stack((
        trade_rate,
        jnp.full_like(trade_rate, GROWTH_FRACTION * TOTAL_BIOLOGICAL_RATE_PER_DAY),
        jnp.full_like(trade_rate, REPRODUCTION_FRACTION * TOTAL_BIOLOGICAL_RATE_PER_DAY),
        jnp.zeros_like(trade_rate),
    ), axis=-1)


def plant_only_actions() -> dict[str, jnp.ndarray]:
    """Return the fixed plant-only action mapping with an absent fungal partner."""
    return {
        PLANT: fixed_allocation_rate_action(jnp.asarray(0.0)),
        FUNGUS: jnp.zeros(4, dtype=jnp.float32),
    }


def run_trade_only_baseline(
    *, initial_p_micromolar: tuple[float, ...], seeds: tuple[int, ...],
    days: float, timestep_days: float, include_plant_only: bool = True,
) -> dict:
    """Evaluate plant-only and bidirectional current-pool trade controls."""
    no_trade = fixed_allocation_rate_action(jnp.asarray(0.0)).tolist()
    mixed_trade_rates = {
        PLANT: pool_fraction_to_rate(PLANT_MIXED_TRADE_FRACTION_PER_DAY),
        FUNGUS: pool_fraction_to_rate(FUNGUS_MIXED_TRADE_FRACTION_PER_DAY),
    }
    shared = {
        "horizon": {"days": days, "timestep_days": timestep_days},
        "initial_p_micromolar": list(initial_p_micromolar), "seeds": list(seeds),
        "model": {"environment": {}, "species": {"plant": {}, "fungus": {}}},
    }
    plant_only_entries = []
    if include_plant_only:
        plant_only_entries = run_static_controls({
            **shared, "modes": ["plant-only"],
            "static_policy": {PLANT: no_trade, FUNGUS: jnp.zeros(4).tolist()},
        })["entries"]
    mixed = run_batched_static_controls({
        **shared, "modes": ["mixed"],
        "static_policy": {
            PLANT: fixed_allocation_rate_action(jnp.asarray(mixed_trade_rates[PLANT])).tolist(),
            FUNGUS: fixed_allocation_rate_action(jnp.asarray(mixed_trade_rates[FUNGUS])).tolist(),
        },
    })
    entries = plant_only_entries + mixed["entries"]
    rejected = sum(entry["status"] == "rejected" for entry in entries)
    return {
        "format": "mycormarl-trade-only-baseline", "format_version": 1,
        "status": "rejected" if rejected else "complete", "entries": entries,
        "completion": {"completed": len(entries) - rejected, "requested": len(entries)},
        "protocol": {
            "fixed_allocation": {
                "total_biological_rate_per_day": TOTAL_BIOLOGICAL_RATE_PER_DAY,
                "growth_fraction": GROWTH_FRACTION,
                "reproduction_fraction": REPRODUCTION_FRACTION,
                "storage_rate_per_day": 0.0,
            },
            "plant_only_trade_rate_per_day": 0.0,
            "mixed_trade_fraction_of_current_post_maintenance_pool_per_day": {
                PLANT: PLANT_MIXED_TRADE_FRACTION_PER_DAY,
                FUNGUS: FUNGUS_MIXED_TRADE_FRACTION_PER_DAY,
            },
            "mixed_trade_rate_per_day": mixed_trade_rates,
        },
    }
