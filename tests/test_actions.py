import jax
import jax.numpy as jnp

from mycormarl.actions import physical_action


def test_physical_action_bounds_trade_independently_of_biological_allocation():
    action = physical_action(
        trade=0.75,
        growth=2.0,
        reproduction=1.0,
        reserve=1.0,
    )

    assert action.dtype == jnp.float32
    assert jnp.allclose(action, jnp.array([0.75, 0.5, 0.25, 0.25]))


def test_physical_action_replaces_non_finite_inputs_with_a_valid_fallback():
    action = physical_action(
        trade=jnp.nan,
        growth=jnp.nan,
        reproduction=jnp.inf,
        reserve=-jnp.inf,
    )

    assert jnp.all(jnp.isfinite(action))
    assert jnp.allclose(action, jnp.array([0.0, 1 / 3, 1 / 3, 1 / 3]))


def test_physical_action_is_jittable_and_vectorises_over_callers():
    make_actions = jax.jit(
        jax.vmap(physical_action, in_axes=(0, 0, 0, 0))
    )

    actions = make_actions(
        jnp.array([-1.0, 0.25, 2.0]),
        jnp.array([1.0, 0.0, 1.0]),
        jnp.array([0.0, 1.0, 1.0]),
        jnp.array([0.0, 0.0, 2.0]),
    )

    assert actions.shape == (3, 4)
    assert jnp.all(jnp.isfinite(actions))
    assert jnp.all((actions[:, 0] >= 0.0) & (actions[:, 0] <= 1.0))
    assert jnp.allclose(jnp.sum(actions[:, 1:], axis=-1), 1.0)
