"""Three latent controls produce the agreed public biological strategy."""

import jax.numpy as jnp
import pytest

from mycormarl.algos.ppo import latent_to_rate_action


def test_latents_separate_total_biological_intensity_from_reproductive_expenditure_share():
    action = latent_to_rate_action(jnp.array(0.), jnp.array([0.,0.]))
    # softplus(0)=ln(2); sigmoid(0)=1/2. One total biological intensity,
    # split equally, rather than two independently transformed hazards.
    assert action == pytest.approx([.6931471806,.3465735903,.3465735903], abs=1e-7)


def test_legacy_four_latent_inputs_are_rejected_instead_of_ignoring_a_scored_variable():
    with pytest.raises(ValueError, match='two biological latents'):
        latent_to_rate_action(jnp.array(0.), jnp.zeros(3))


def test_asymmetric_latents_choose_processing_speed_and_expenditure_share_independently():
    # Inverse transforms of trade .05/day, biology 2/day, reproduction share .8.
    action = latent_to_rate_action(jnp.array(-2.9706281091),
                                   jnp.array([1.8545865421,1.3862943611]))
    assert action == pytest.approx([.05,.4,1.6], abs=2e-7)
