"""Coupled trade and essential-resource allocation, in physical C/P units."""

import jax
import jax.numpy as jnp


def integrate_allocation(pools, rates, costs, reproduction_scales, growth_room,
                         dt, rate_step, max_substeps):
    """Return C, P, cumulative growth, reproduction and donor trade per agent.

    SSPRK3 substeps obey h * maximum outgoing hazard <= rate_step <= 1.
    Each Euler stage is therefore nonnegative. Paired transfer and biological
    extents use the same stage weights, preserving both elemental balances.
    Rates are held constant while pools and the limiting resource are updated.
    """
    required = jnp.maximum(1, jnp.ceil(jnp.max(jnp.sum(rates, axis=1))*dt/rate_step))
    converged = required <= max_substeps
    n = jnp.where(converged, required, 0).astype(jnp.int32)
    h = dt / jnp.maximum(n, 1)
    initial = jnp.concatenate((pools, jnp.zeros((2, 3), dtype=pools.dtype)), axis=1)

    def flux(y):
        available = jnp.min(y[:, :2] / costs, axis=1)
        growth = jnp.minimum(rates[:, 1] * available,
                             jnp.maximum(growth_room-y[:, 2], 0.) / h)
        reproduction = rates[:, 2] * available / reproduction_scales
        used = growth + reproduction_scales * reproduction
        trade = rates[:, 0] * jnp.stack((y[0, 0], y[1, 1]))
        c = -costs[:, 0]*used + jnp.stack((-trade[0], trade[0]))
        p = -costs[:, 1]*used + jnp.stack((trade[1], -trade[1]))
        return jnp.stack((c, p, growth, reproduction, trade), axis=1)

    def advance(_, y):
        first = h * flux(y)
        second = h * flux(y + first)
        third = h * flux(y + .25*(first+second))
        # Increment form also leaves a dormant organism's stocks unchanged.
        return y + (first + second + 4.*third) / 6.

    result = jax.lax.fori_loop(0, n, advance, initial)
    # An exhausted budget is an explicit failed numerical transition, never
    # a silently capped action or an accepted under-resolved approximation.
    return jnp.where(converged, result, jnp.nan), converged, n
