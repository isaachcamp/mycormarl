"""Reusable fixed-allocation Rate action contracts for reduced-control actors."""

from __future__ import annotations

import math

import jax.numpy as jnp

from mycormarl.environments.base_mycor import FUNGUS, PLANT


TOTAL_BIOLOGICAL_RATE_PER_DAY = 6.907755
GROWTH_FRACTION = 0.9
REPRODUCTION_FRACTION = 0.1


def pool_fraction_to_rate(fraction_per_day: float) -> float:
    """Convert a daily fraction of the current pool to a Rate-action hazard.

    This is the isolated donor-pool conversion: a rate held for one day
    exchanges the declared fraction when no competing processes or inputs
    change that pool. Coupled biological competition changes realised trade.
    """
    if not 0.0 <= fraction_per_day < 1.0:
        raise ValueError("pool fraction per day must lie in [0, 1)")
    return -math.log1p(-fraction_per_day)


def fixed_allocation_rate_action(trade_rate: jnp.ndarray) -> jnp.ndarray:
    """Return three physical rates with the existing fixed 90/10 biology.

    ``trade_rate`` is already physical and must be non-negative.  PPO maps its
    scalar latent to this quantity with softplus before crossing this seam.
    """
    trade_rate = jnp.asarray(trade_rate, dtype=jnp.float32)
    return jnp.stack((
        trade_rate,
        jnp.full_like(trade_rate, GROWTH_FRACTION * TOTAL_BIOLOGICAL_RATE_PER_DAY),
        jnp.full_like(trade_rate, REPRODUCTION_FRACTION * TOTAL_BIOLOGICAL_RATE_PER_DAY),
    ), axis=-1)


def plant_only_actions() -> dict[str, jnp.ndarray]:
    """Return the fixed plant-only action mapping with an absent fungal partner."""
    return {
        PLANT: fixed_allocation_rate_action(jnp.asarray(0.0)),
        FUNGUS: jnp.zeros(3, dtype=jnp.float32),
    }
