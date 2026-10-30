"""Versioned, independently-derived random streams for study replication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import jax


RANDOM_STREAM_DERIVATION_VERSION = "named-prng-v1"
RANDOM_STREAM_NAMES = (
    "plant_initialization",
    "fungal_initialization",
    "policy_action_sampling",
    "environment_variation",
    "minibatch_ordering",
)


def _stream_key(master_seed: int, stream_name: str):
    """Derive one key without consuming or splitting any other stream."""
    digest = hashlib.sha256(
        f"{RANDOM_STREAM_DERIVATION_VERSION}:{master_seed}:{stream_name}".encode(
            "utf-8"
        )
    ).digest()
    words = [
        int.from_bytes(digest[offset : offset + 4], "little")
        for offset in (0, 4)
    ]
    return jax.numpy.asarray(words, dtype=jax.numpy.uint32)


@dataclass(frozen=True)
class RandomStreamContract:
    """Named keys and pairing metadata for one master-seed replicate."""

    master_seed: int
    paired_streams: tuple[str, ...] = ()
    derivation_version: str = RANDOM_STREAM_DERIVATION_VERSION

    @property
    def stream_names(self) -> tuple[str, ...]:
        return RANDOM_STREAM_NAMES

    def key(self, stream_name: str):
        if stream_name not in RANDOM_STREAM_NAMES:
            raise KeyError(f"unknown random stream {stream_name!r}")
        return _stream_key(self.master_seed, stream_name)

    def to_dict(self) -> dict:
        """Return JSON-compatible identities suitable for artifacts/checkpoints."""
        return {
            "derivation_version": self.derivation_version,
            "master_seed": self.master_seed,
            "paired_streams": list(self.paired_streams),
            "streams": {
                name: [int(value) for value in self.key(name)]
                for name in RANDOM_STREAM_NAMES
            },
        }


def derive_random_streams(
    master_seed: int, *, paired_streams: Iterable[str] = ()
) -> RandomStreamContract:
    """Build the reproducible named-stream contract for a master seed."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    paired = tuple(sorted(set(paired_streams)))
    if any(name not in RANDOM_STREAM_NAMES for name in paired):
        raise ValueError("paired_streams contains an unknown random stream")
    return RandomStreamContract(master_seed=master_seed, paired_streams=paired)
