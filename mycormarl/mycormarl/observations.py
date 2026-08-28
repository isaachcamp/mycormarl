
import chex
import jax.numpy as jnp


OBS_EPS = 1e-8


def bounded_saturating_ratio(
    amount: chex.Array, reference: chex.Array
) -> chex.Array:
    """Map a non-negative amount and reference to a finite ratio in [0, 1]."""
    amount = jnp.maximum(amount, 0.0)
    reference = jnp.maximum(reference, 0.0)
    ratio = amount / jnp.maximum(amount + reference, OBS_EPS)
    return jnp.clip(jnp.nan_to_num(ratio, nan=0.0, posinf=1.0), 0.0, 1.0)


def actor_observation(
    *,
    biomass: chex.Array,
    biomass_reference: chex.Array,
    c_pool: chex.Array,
    gamma_c: float,
    p_pool: chex.Array,
    gamma_p: float,
    last_received: chex.Array,
    maintenance_need: chex.Array,
    association: chex.Array,
    operational: chex.Array,
) -> chex.Array:
    """Build one actor's ordered five-feature observation."""
    observation = jnp.concatenate(
        [
            bounded_saturating_ratio(biomass, biomass_reference),
            bounded_saturating_ratio(c_pool, gamma_c * biomass),
            bounded_saturating_ratio(p_pool, gamma_p * biomass),
            bounded_saturating_ratio(last_received, maintenance_need),
            association.astype(jnp.float32),
        ]
    )
    return jnp.where(operational, observation, 0.0).astype(jnp.float32)
