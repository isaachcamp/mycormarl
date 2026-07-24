import jax
import jax.numpy as jnp
import pytest

from mycormarl.actions import _allocation_from_action, _check_action, constrain_allocation


def assert_simplex(action, n_parts=4):
    """Assert that an action vector is a valid allocation on the simplex."""
    assert isinstance(action, jax.Array)
    assert action.shape == (n_parts,)
    assert jnp.all(jnp.isfinite(action))
    assert jnp.all(action >= 0.0)
    assert jnp.all(action <= 1.0)
    assert jnp.sum(action) == pytest.approx(1.0)


def test_check_action_clips_to_unit_interval_without_normalising():
    checked = _check_action(jnp.array([-1.0, 0.25, 2.0, 0.5]))

    assert checked.shape == (4,)
    assert jnp.allclose(checked, jnp.array([0.0, 0.25, 1.0, 0.5]))


def test_check_action_replaces_non_finite_values_with_zero():
    checked = _check_action(jnp.array([jnp.nan, jnp.inf, -jnp.inf, 0.5]))

    assert jnp.allclose(checked, jnp.array([0.0, 0.0, 0.0, 0.5]))


def test_check_action_raises_for_wrong_final_dimension():
    with pytest.raises(ValueError):
        _check_action(jnp.array([0.25, 0.25, 0.25]), n_parts=4)


def test_allocation_from_action_preserves_valid_normalized_action():
    action = jnp.array([0.1, 0.2, 0.3, 0.4])

    allocation = _allocation_from_action(action)

    assert_simplex(allocation)
    assert jnp.allclose(allocation, action)


def test_constrain_allocation_preserves_valid_normalized_action():
    action = jnp.array([0.4, 0.3, 0.2, 0.1])

    allocation = constrain_allocation(action)

    assert_simplex(allocation)
    assert jnp.allclose(allocation, action)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (jnp.array([2.0, 1.0, 1.0, 0.0]), jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0])),
        (jnp.array([-1.0, 0.5, 0.5, 0.0]), jnp.array([0.0, 0.5, 0.5, 0.0])),
        (jnp.array([0.0, 2.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0, 0.0])),
        (jnp.array([jnp.nan, 0.2, 0.3, 0.5]), jnp.array([0.0, 0.2, 0.3, 0.5])),
        (jnp.array([jnp.inf, 0.0, 1.0, 1.0]), jnp.array([0.0, 0.0, 0.5, 0.5])),
    ],
)
def test_constrain_allocation_clips_sanitizes_and_renormalizes(action, expected):
    allocation = constrain_allocation(action)

    assert_simplex(allocation)
    assert jnp.allclose(allocation, expected)


@pytest.mark.parametrize(
    "action",
    [
        jnp.array([0.0, 0.0, 0.0, 0.0]),
        jnp.array([-1.0, -2.0, -3.0, -4.0]),
        jnp.array([jnp.nan, jnp.inf, -jnp.inf, jnp.nan]),
    ],
)
def test_constrain_allocation_returns_uniform_when_no_positive_weight_remains(action):
    allocation = constrain_allocation(action)

    assert_simplex(allocation)
    assert jnp.allclose(allocation, jnp.ones(4) / 4)


def test_constrain_allocation_accepts_configurable_action_length():
    allocation = constrain_allocation(jnp.array([2.0, 1.0, 0.0]), n_parts=3)

    assert_simplex(allocation, n_parts=3)
    assert jnp.allclose(allocation, jnp.array([0.5, 0.5, 0.0]))


def test_constrain_allocation_is_jittable():
    action = jnp.array([-1.0, 0.0, 2.0, jnp.nan])

    allocation = jax.jit(constrain_allocation)(action)

    assert_simplex(allocation)
    assert jnp.allclose(allocation, jnp.array([0.0, 0.0, 1.0, 0.0]))
