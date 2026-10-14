import jax
import jax.numpy as jnp

from mycormarl.actions import rate_action


def test_rate_action_preserves_nonnegative_per_day_rates_without_simplex_normalisation():
    action = rate_action(
        trade=1.25,
        growth=2.0,
        reproduction=0.5,
        storage=0.25,
    )

    assert action.dtype == jnp.float32
    assert jnp.allclose(action, jnp.array([1.25, 2.0, 0.5, 0.25]))


def test_rate_action_replaces_non_finite_and_negative_inputs_with_zero_rates():
    action = rate_action(
        trade=jnp.nan,
        growth=jnp.nan,
        reproduction=jnp.inf,
        storage=-jnp.inf,
    )

    assert jnp.all(jnp.isfinite(action))
    assert jnp.allclose(action, jnp.zeros(4))


def test_rate_action_is_jittable_and_vectorises_over_callers():
    make_actions = jax.jit(
        jax.vmap(rate_action, in_axes=(0, 0, 0, 0))
    )

    actions = make_actions(
        jnp.array([-1.0, 0.25, 2.0]),
        jnp.array([1.0, 0.0, 1.0]),
        jnp.array([0.0, 1.0, 1.0]),
        jnp.array([0.0, 0.0, 2.0]),
    )

    assert actions.shape == (3, 4)
    assert jnp.all(jnp.isfinite(actions))
    assert jnp.all(actions >= 0.0)
