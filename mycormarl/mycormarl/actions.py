
import chex
import jax.numpy as jnp


def _check_action(action: chex.Array, n_parts: int = 4) -> chex.Array:
    """Convert, sanitize, and clip an action vector.

    This intentionally avoids value-based Python checks so it remains compatible
    with JAX transformations. The final allocation simplex is enforced by
    ``_allocation_from_action``.
    """

    action = jnp.asarray(action, dtype=jnp.float32)
    if action.shape[-1] != n_parts:
        raise ValueError(f"Expected action with {n_parts} components, got shape {action.shape}")

    action = jnp.where(jnp.isfinite(action), action, 0.0)
    return jnp.clip(action, 0.0, 1.0)

def _allocation_from_action(action: chex.Array, n_parts: int = 4) -> chex.Array:
    """Map an action vector to growth / maintenance / reproduction / trade weights.

    The weights are renormalised internally to keep the resource accounting
    stable even when the policy emits arbitrary values. If clipping and
    sanitising leaves no positive allocation, return a uniform allocation.
    """
    a = _check_action(action, n_parts=n_parts)

    total = jnp.sum(a, axis=-1, keepdims=True)
    uniform = jnp.ones_like(a) / n_parts # Return uniform allocation if total is zero.
    return jnp.where(total > 0.0, a / total, uniform)

def constrain_allocation(action: chex.Array, n_parts: int = 4) -> chex.Array:
    """Return a JIT-friendly allocation vector on the simplex."""
    return _allocation_from_action(action, n_parts=n_parts)
