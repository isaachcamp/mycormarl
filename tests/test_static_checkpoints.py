"""Sparse biological evidence through the existing public rollout seam."""

import pytest

from mycormarl.static_rollout import evaluate_static_policy
from test_stoichiometric_actions import pool_environment


def test_static_checkpoints_report_selected_states_rewards_and_transition_facts():
    env = pool_environment(dt=.05, max_steps=6)
    result = evaluate_static_policy(
        env, {'plant': [0., 0., 0.], 'fungus': [0., 0., 0.]},
        seed=7, checkpoint_steps=(1, 3, 6),
    )
    checkpoints = result['checkpoints']
    assert [row['state']['step'] for row in checkpoints] == [0, 1, 3, 6]
    assert [row['day'] for row in checkpoints] == pytest.approx([0., .05, .15, .3])
    assert checkpoints[0]['transitions'] is None
    assert checkpoints[-1]['state']['plant_biomass'] == pytest.approx([1.])
    assert checkpoints[-1]['rewards'] == {'plant': 0., 'fungus': 0.}
    assert checkpoints[-1]['transitions']['plant']['truncated'] is True
    assert checkpoints[-1]['observations']['plant'] == pytest.approx(
        checkpoints[-1]['transitions']['plant']['final_observation'])
    assert result['cumulative_growth']['plant'] == 0.


@pytest.mark.parametrize('steps', [(0,), (7,), (3, 1), (1, 1), (True,), (1.5,)])
def test_static_rollout_rejects_invalid_checkpoint_schedules(steps):
    env = pool_environment(max_steps=6)
    with pytest.raises(ValueError, match='checkpoint_steps'):
        evaluate_static_policy(env, {'plant': [0., 0., 0.], 'fungus': [0., 0., 0.]},
                               seed=7, checkpoint_steps=steps)


def test_checkpoint_capture_includes_early_terminal_state_without_padding():
    env = pool_environment(dt=.05, max_steps=6,
                           plant=dict(initial_c_pool=0., kappa_c=30., death_fraction=.9),
                           fungus=dict(initial_c_pool=0., kappa_c=30., death_fraction=.9))
    result = evaluate_static_policy(
        env, {'plant': [0., 0., 0.], 'fungus': [0., 0., 0.]},
        seed=7, checkpoint_steps=(2, 4, 6))
    assert [row['state']['step'] for row in result['checkpoints']] == [0, 1]
    assert result['checkpoints'][-1]['state']['terminal'] is True


def test_stressed_rollout_cancels_trade_on_death_and_keeps_dead_padding_inactive():
    # With no incoming C, maintenance removes 20% of fungal biomass per step.
    # The historical-max death threshold is crossed at step four, not three.
    env = pool_environment(dt=.25, max_steps=6,
                           plant=dict(initial_c_pool=2.),
                           fungus=dict(initial_biomass=1., gamma_c=.5, initial_c_pool=0.,
                                       initial_p_pool=1., kappa_c=.4, death_fraction=.5))
    result = evaluate_static_policy(
        env, {'plant': [0., 0., 0.], 'fungus': [.4, 0., 0.]},
        seed=7, checkpoint_steps=(3, 4, 5, 6))
    alive, death, padding = result['checkpoints'][1:4]
    assert alive['state']['fungus_biomass'] == pytest.approx([.512], abs=1e-6)
    assert alive['transitions']['fungus']['operational_at_end'] is True
    assert alive['flows']['fungus']['trade_out'][0] > 0.
    assert death['state']['fungus_biomass'] == pytest.approx([.4096], abs=1e-6)
    assert death['transitions']['fungus']['operational_at_start'] is True
    assert death['transitions']['fungus']['operational_at_end'] is False
    assert death['flows']['fungus']['trade_cancelled'] == [True]
    assert death['flows']['fungus']['trade_out'] == [0.]
    assert padding['transitions']['fungus']['operational_at_start'] is False
    assert padding['state']['fungus_biomass'] == death['state']['fungus_biomass']
    assert padding['rewards']['fungus'] == 0.
