"""Public contract tests for reproducible named random streams."""

import jax

from mycormarl.random_streams import (
    RANDOM_STREAM_DERIVATION_VERSION,
    RANDOM_STREAM_NAMES,
    derive_random_streams,
)


def test_named_streams_are_reproducible_and_independent():
    """A master seed deterministically derives one stable key per subsystem."""
    first = derive_random_streams(17)
    second = derive_random_streams(17)

    assert first.derivation_version == RANDOM_STREAM_DERIVATION_VERSION
    assert first.stream_names == RANDOM_STREAM_NAMES
    for stream_name in RANDOM_STREAM_NAMES:
        assert jax.numpy.array_equal(
            first.key(stream_name), second.key(stream_name)
        )
    assert len(
        {
            tuple(map(int, first.key(name))) for name in RANDOM_STREAM_NAMES
        }
    ) == len(RANDOM_STREAM_NAMES)


def test_changing_master_seed_changes_streams_without_rekeying_by_position():
    """Stream identity is its name, so unrelated streams do not share a split chain."""
    baseline = derive_random_streams(17)
    changed = derive_random_streams(18)

    assert all(
        not jax.numpy.array_equal(baseline.key(name), changed.key(name))
        for name in RANDOM_STREAM_NAMES
    )
    assert baseline.key("plant_initialization").shape == (2,)


def test_declared_pairing_is_recorded_as_part_of_the_contract():
    """Paired replicate semantics are explicit metadata, not an accidental key split."""
    streams = derive_random_streams(
        17, paired_streams=("environment_variation", "policy_action_sampling")
    )

    assert streams.paired_streams == (
        "environment_variation",
        "policy_action_sampling",
    )
    assert streams.to_dict() == {
        "derivation_version": RANDOM_STREAM_DERIVATION_VERSION,
        "master_seed": 17,
        "paired_streams": ["environment_variation", "policy_action_sampling"],
        "streams": {
            name: [int(value) for value in streams.key(name)]
            for name in RANDOM_STREAM_NAMES
        },
    }
