"""Public compatibility contract for saved independent-PPO policies."""

import json
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
from flax import serialization
import pytest

from mycormarl.random_streams import derive_random_streams
from mycormarl.policy_artifacts import (
    ACTOR_INTERFACE_VERSION,
    ENVIRONMENT_STATE_SCHEMA_VERSION,
    POLICY_ARTIFACT_FORMAT,
    load_policy_artifact,
    save_policy_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def test_policy_artifact_round_trip_identifies_current_interfaces(tmp_path):
    """Saved policies state which actor and environment-state contracts they use."""
    output = tmp_path / "policies.msgpack"
    parameters = {
        "plant": {"head": jnp.array([1.0, 2.0])},
        "fungus": {"head": jnp.array([3.0, 4.0])},
    }

    save_policy_artifact(output, parameters, consumer_mode="mixed")
    artifact = load_policy_artifact(output)

    assert artifact.actor_interface_version == ACTOR_INTERFACE_VERSION
    assert (
        artifact.environment_state_schema_version
        == ENVIRONMENT_STATE_SCHEMA_VERSION
    )
    assert artifact.consumer_mode == "mixed"
    assert artifact.random_streams is None
    assert jnp.array_equal(
        artifact.parameters["plant"]["head"], jnp.array([1.0, 2.0])
    )
    assert jnp.array_equal(
        artifact.parameters["fungus"]["head"], jnp.array([3.0, 4.0])
    )


def test_policy_artifact_can_record_named_random_streams(tmp_path):
    """A checkpoint preserves the stream identities used to create policies."""
    output = tmp_path / "policies.msgpack"
    streams = derive_random_streams(17)

    save_policy_artifact(
        output,
        {"plant": {}, "fungus": {}},
        consumer_mode="mixed",
        random_streams=streams,
    )

    assert load_policy_artifact(output).random_streams == streams.to_dict()


def test_loader_rejects_unversioned_legacy_parameter_tree(tmp_path):
    """A raw parameter tree cannot be mistaken for a compatible policy artifact."""
    legacy = tmp_path / "legacy.msgpack"
    legacy.write_bytes(
        serialization.to_bytes(
            {"plant": {"Dense_0": {"kernel": jnp.ones((2, 2))}}}
        )
    )

    with pytest.raises(ValueError, match="incompatible.*unversioned"):
        load_policy_artifact(legacy)


@pytest.mark.parametrize(
    ("metadata_key", "error_contract"),
    (
        ("actor_interface_version", "actor interface"),
        ("environment_state_schema_version", "environment state schema"),
    ),
)
def test_loader_rejects_incompatible_interface_versions(
    tmp_path, metadata_key, error_contract
):
    """Policies tied to old actor or environment-state semantics fail clearly."""
    output = tmp_path / "policies.msgpack"
    save_policy_artifact(
        output, {"plant": {}, "fungus": {}}, consumer_mode="mixed"
    )
    payload = serialization.msgpack_restore(output.read_bytes())
    payload[metadata_key] = "legacy"
    output.write_bytes(serialization.msgpack_serialize(payload))

    with pytest.raises(ValueError, match=f"incompatible {error_contract}"):
        load_policy_artifact(output)


def test_loader_rejects_unknown_policy_artifact_format_version(tmp_path):
    """A future or stale bundle format cannot be interpreted as the current one."""
    output = tmp_path / "policies.msgpack"
    payload = {
        "format": POLICY_ARTIFACT_FORMAT,
        "format_version": 999,
        "actor_interface_version": ACTOR_INTERFACE_VERSION,
        "environment_state_schema_version": ENVIRONMENT_STATE_SCHEMA_VERSION,
        "consumer_mode": "mixed",
        "parameters": {"plant": {}, "fungus": {}},
    }
    output.write_bytes(serialization.msgpack_serialize(payload))

    with pytest.raises(ValueError, match="incompatible policy artifact format"):
        load_policy_artifact(output)


def test_training_cli_selects_consumer_mode_and_writes_versioned_policy(tmp_path):
    """The public trainer preserves single-consumer mode in its saved artifact."""
    output = tmp_path / "plant-only.msgpack"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_ppo.py"),
            "--total-timesteps",
            "2",
            "--num-steps",
            "2",
            "--num-envs",
            "1",
            "--mode",
            "plant-only",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    assert summary["consumer_mode"] == "plant-only"
    assert load_policy_artifact(output).consumer_mode == "plant-only"
