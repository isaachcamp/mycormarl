"""Versioned persistence for independent-PPO policy parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from flax import serialization


POLICY_ARTIFACT_FORMAT = "mycormarl-ippo-policy"
POLICY_ARTIFACT_VERSION = 1
ACTOR_INTERFACE_VERSION = "two-head-latent-v1"
ENVIRONMENT_STATE_SCHEMA_VERSION = "state-v2"


@dataclass(frozen=True)
class PolicyArtifact:
    """A loaded policy bundle and the contracts required to use it safely."""

    parameters: Mapping[str, Any]
    consumer_mode: str
    actor_interface_version: str
    environment_state_schema_version: str
    random_streams: Mapping[str, Any] | None = None


def save_policy_artifact(
    path: str | Path,
    parameters: Mapping[str, Any],
    *,
    consumer_mode: str,
    random_streams: Any = None,
) -> None:
    """Save policy parameters with explicit interface compatibility metadata."""
    payload = {
        "format": POLICY_ARTIFACT_FORMAT,
        "format_version": POLICY_ARTIFACT_VERSION,
        "actor_interface_version": ACTOR_INTERFACE_VERSION,
        "environment_state_schema_version": ENVIRONMENT_STATE_SCHEMA_VERSION,
        "consumer_mode": consumer_mode,
        "parameters": parameters,
    }
    if random_streams is not None:
        payload["random_streams"] = (
            random_streams.to_dict()
            if hasattr(random_streams, "to_dict")
            else random_streams
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialization.msgpack_serialize(payload))


def load_policy_artifact(path: str | Path) -> PolicyArtifact:
    """Load a policy bundle produced for the current public interfaces."""
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    if (
        not isinstance(payload, dict)
        or payload.get("format") != POLICY_ARTIFACT_FORMAT
    ):
        raise ValueError(
            "incompatible unversioned policy artifact; retrain and save with "
            "the current actor interface"
        )
    if payload.get("format_version") != POLICY_ARTIFACT_VERSION:
        raise ValueError(
            "incompatible policy artifact format version; use a matching "
            "MycorMARL release"
        )
    if payload.get("actor_interface_version") != ACTOR_INTERFACE_VERSION:
        raise ValueError(
            "incompatible actor interface version; retrain the saved policy"
        )
    if (
        payload.get("environment_state_schema_version")
        != ENVIRONMENT_STATE_SCHEMA_VERSION
    ):
        raise ValueError(
            "incompatible environment state schema version; saved environment "
            "state and policies must be regenerated"
        )
    return PolicyArtifact(
        parameters=payload["parameters"],
        consumer_mode=payload["consumer_mode"],
        actor_interface_version=payload["actor_interface_version"],
        environment_state_schema_version=payload[
            "environment_state_schema_version"
        ],
        random_streams=payload.get("random_streams"),
    )
