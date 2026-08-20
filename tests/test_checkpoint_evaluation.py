"""Public checkpoint-evaluation contracts."""

import jax
from flax import serialization

from mycormarl.algos.ppo import ActorCritic
from mycormarl.checkpoint_evaluation import (
    LatentLocationEvaluation,
    SampledPolicyEvaluation,
    evaluate_checkpoint,
    evaluate_policy_summary,
    evaluate_policy_parameters,
    save_evaluation_artifact,
    save_evaluation_summary_artifact,
)
from mycormarl.environments.base_mycor import FUNGUS, PLANT, BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits


def _environment():
    return BaseMycorMarl(
        EnvConfig(
            max_steps=2,
            dt=0.05,
            soil_radius_cm=0.2,
            soil_depth_cm=0.2,
            radial_interval_cm=0.1,
            depth_interval_cm=0.1,
        ),
        SpeciesParams(PlantTraits(kappa_c=0.0, kappa_p=0.0), FungusTraits(kappa_c=0.0, kappa_p=0.0)),
    )


def _parameters(environment):
    actor = ActorCritic()
    observations, _ = environment.reset(jax.random.PRNGKey(0))
    return {
        agent: actor.init(jax.random.PRNGKey(index + 1), observations[agent])
        for index, agent in enumerate((PLANT, FUNGUS))
    }


def test_latent_location_evaluation_is_reproducible_at_declared_horizon():
    """The public evaluator transforms actor locations without sampling."""
    environment = _environment()
    parameters = _parameters(environment)

    first = evaluate_policy_parameters(
        environment, parameters, episodes=1, protocol="latent-location", seed=17
    )
    second = evaluate_policy_parameters(
        environment, parameters, episodes=1, protocol="latent-location", seed=17
    )

    assert first == second
    assert first.protocol == "latent-location"
    assert len(first.episodes) == 1
    assert len(first.episodes[0].trace) == 2
    assert first.episodes[0].trace[-1]["truncated"] is True


def test_sampled_diagnostic_has_a_distinct_type_and_complete_recomputable_trace():
    """Stochastic diagnostics cannot be mistaken for primary learned outcomes."""
    environment = _environment()
    parameters = _parameters(environment)

    deterministic = evaluate_policy_parameters(
        environment, parameters, episodes=1, protocol="latent-location", seed=3
    )
    sampled = evaluate_policy_parameters(
        environment, parameters, episodes=1, protocol="sampled-policy", seed=3
    )

    assert isinstance(deterministic, LatentLocationEvaluation)
    assert isinstance(sampled, SampledPolicyEvaluation)
    assert type(deterministic) is not type(sampled)
    row = deterministic.episodes[0].trace[0]
    assert {
        "actions", "reproductive_fitness", "gross_growth", "limitation",
        "uptake_mg", "maintenance_p_required_mg", "transfers", "operational_at_start",
        "operational_at_end", "biological_termination", "truncated", "state",
    } <= row.keys()
    assert "uptake_sufficient" not in row
    assert {
        "raw_biomass", "living_biomass", "pools", "soil_inventory_micromol",
        "loss_counters",
    } <= row["state"].keys()
    summary = deterministic.episodes[0].summary
    assert {
        "cumulative_reproductive_fitness", "final_raw_biomass",
        "final_living_biomass", "cumulative_gross_growth", "episode_rgr_per_day",
    } <= summary.keys()


def test_checkpoint_evaluation_and_artifact_preserve_the_complete_trace(tmp_path):
    """A versioned trainer checkpoint produces a standalone scientific artifact."""
    environment = _environment()
    parameters = _parameters(environment)
    checkpoint = tmp_path / "checkpoint.msgpack"
    checkpoint.write_bytes(serialization.msgpack_serialize({
        "format": "mycormarl-ppo-checkpoint",
        "format_version": 1,
        "metadata": {
            "actor_interface_version": "two-head-latent-v1",
            "environment_state_schema_version": "state-v2",
        },
        "runner_state": {"0": {agent: {"params": parameters[agent]} for agent in (PLANT, FUNGUS)}},
    }))

    result = evaluate_checkpoint(checkpoint, environment, episodes=1, seed=8)
    artifact = tmp_path / "evaluation.json"
    save_evaluation_artifact(artifact, result, checkpoint=checkpoint)

    assert artifact.exists()
    assert '"protocol": "latent-location"' in artifact.read_text(encoding="utf-8")


def test_checkpoint_summary_artifact_includes_final_biomass_for_post_analysis(tmp_path):
    """Compact checkpoint JSONs retain the biomass endpoints needed for plots."""
    environment = _environment()
    summary = evaluate_policy_summary(
        environment, _parameters(environment), episodes=1, seed=8,
    )
    artifact = tmp_path / "evaluation-summary.json"
    save_evaluation_summary_artifact(artifact, summary, checkpoint="checkpoint.msgpack")

    saved = __import__("json").loads(artifact.read_text(encoding="utf-8"))["summary"]
    assert set(saved["final_raw_biomass"]) == {PLANT, FUNGUS}
    assert set(saved["final_living_biomass"]) == {PLANT, FUNGUS}
    assert saved["final_living_biomass"][PLANT] > 0.0


def test_survivor_continues_after_partner_death_until_the_declared_horizon():
    """Biological death is not confused with the administrative horizon."""
    environment = BaseMycorMarl(
        EnvConfig(
            max_steps=2, dt=0.05, soil_radius_cm=0.2, soil_depth_cm=0.2,
            radial_interval_cm=0.1, depth_interval_cm=0.1,
        ),
        SpeciesParams(
            PlantTraits(kappa_c=0.0, kappa_p=0.0),
            FungusTraits(kappa_c=100.0, kappa_p=0.0),
        ),
    )

    result = evaluate_policy_parameters(
        environment, _parameters(environment), episodes=1, seed=5
    )

    first, second = result.episodes[0].trace
    assert first["biological_termination"][FUNGUS] is True
    assert first["truncated"] is False
    assert second["operational_at_start"][PLANT] is True
    assert second["operational_at_start"][FUNGUS] is False
    assert second["truncated"] is True
