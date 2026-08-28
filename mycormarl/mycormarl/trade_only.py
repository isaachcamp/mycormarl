"""Fixed-allocation action contracts for the reduced-control trade study."""

from __future__ import annotations

import jax.numpy as jnp

from mycormarl.environments.base_mycor import FUNGUS, PLANT
from mycormarl.static_controls import run_static_controls


TOTAL_BIOLOGICAL_RATE_PER_DAY = 6.907755
GROWTH_FRACTION = 0.9
REPRODUCTION_FRACTION = 0.1


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
    days: float, timestep_days: float,
) -> dict:
    """Evaluate the fixed-allocation plant-only and mixed control contract."""
    fixed = fixed_allocation_rate_action(jnp.asarray(0.0)).tolist()
    shared = {
        "horizon": {"days": days, "timestep_days": timestep_days},
        "initial_p_micromolar": list(initial_p_micromolar), "seeds": list(seeds),
        "model": {"environment": {}, "species": {"plant": {}, "fungus": {}}},
    }
    plant_only = run_static_controls({
        **shared, "modes": ["plant-only"],
        "static_policy": {PLANT: fixed, FUNGUS: jnp.zeros(4).tolist()},
    })
    mixed = run_static_controls({
        **shared, "modes": ["mixed"],
        "static_policy": {PLANT: fixed, FUNGUS: fixed},
    })
    entries = plant_only["entries"] + mixed["entries"]
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
        },
    }
