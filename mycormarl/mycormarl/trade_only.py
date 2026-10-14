"""Fixed-allocation action contracts for the reduced-control trade study."""

from __future__ import annotations

import math

import jax.numpy as jnp

from mycormarl.environments.base_mycor import FUNGUS, PLANT


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


def run_trade_only_baseline(**kwargs):
    """Temporary legacy forwarding entry point; requires installed REDbyRBD."""
    try:
        from redbyrbd.trade_only import run_trade_only_baseline as run
    except ImportError as error:
        raise ImportError("Named trade baselines moved to REDbyRBD; install that research project.") from error
    return run(**kwargs)
