
import chex
import jax.numpy as jnp


def rate_action(
    trade: chex.Numeric,
    growth: chex.Numeric,
    reproduction: chex.Numeric,
    storage: chex.Numeric,
) -> chex.Array:
    """Construct a non-negative ``d^-1`` Rate action.

    The components are ``[trade, growth, reproduction, storage]``. They are
    independent first-order rates and therefore are neither clipped to one nor
    normalised to a simplex.
    """
    rates = jnp.stack(
        [
            jnp.asarray(trade, dtype=jnp.float32),
            jnp.asarray(growth, dtype=jnp.float32),
            jnp.asarray(reproduction, dtype=jnp.float32),
            jnp.asarray(storage, dtype=jnp.float32),
        ],
        axis=-1,
    )
    rates = jnp.where(jnp.isfinite(rates), rates, 0.0)
    return jnp.maximum(rates, 0.0)
