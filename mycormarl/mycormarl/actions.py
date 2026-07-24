
import chex
import jax.numpy as jnp


def physical_action(
    trade: chex.Numeric,
    growth: chex.Numeric,
    reproduction: chex.Numeric,
    reserve: chex.Numeric,
) -> chex.Array:
    """Construct ``[trade, growth, reproduction, reserve]``.

    Trade is bounded independently; the remaining components are interpreted
    as non-negative biological weights and normalised onto their simplex.
    """
    trade_fraction = jnp.asarray(trade, dtype=jnp.float32)
    trade_fraction = jnp.where(jnp.isfinite(trade_fraction), trade_fraction, 0.0)
    trade_fraction = jnp.clip(trade_fraction, 0.0, 1.0)
    biological_weights = jnp.stack(
        [
            jnp.asarray(growth, dtype=jnp.float32),
            jnp.asarray(reproduction, dtype=jnp.float32),
            jnp.asarray(reserve, dtype=jnp.float32),
        ],
        axis=-1,
    )
    biological_weights = jnp.where(
        jnp.isfinite(biological_weights), biological_weights, 0.0
    )
    biological_weights = jnp.maximum(biological_weights, 0.0)
    total = jnp.sum(biological_weights, axis=-1, keepdims=True)
    biological_allocation = jnp.where(
        total > 0.0,
        biological_weights / total,
        jnp.ones_like(biological_weights) / 3.0,
    )
    return jnp.concatenate(
        [jnp.expand_dims(trade_fraction, axis=-1), biological_allocation],
        axis=-1,
    )
