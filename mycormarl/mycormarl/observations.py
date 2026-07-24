
import chex
import jax.numpy as jnp

def _safe_div(numer: chex.Array, denom: chex.Array, eps: float = 1e-8) -> chex.Array:
    return numer / jnp.maximum(denom, eps)

def _normalize_if(norm: bool, x: chex.Array, scale: chex.Array) -> chex.Array:
    return _safe_div(x, scale) if norm else x
