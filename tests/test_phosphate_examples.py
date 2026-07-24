"""Contracts for the small reproducible phosphate uptake examples."""

import pytest

from mycormarl.phosphate_examples import MODES, run_example


@pytest.mark.parametrize("mode", MODES)
def test_example_closes_soil_to_consumer_phosphate_balance(mode):
    """Each public example conserves P across soil loss and pool credit."""
    result = run_example(mode)

    assert result["soil_loss_mg"] > 0.0
    assert result["balance_error_mg"] == pytest.approx(0.0, abs=1e-9)
    assert result["minimum_soil_labile_p_micromol"] >= 0.0


def test_single_consumer_examples_credit_only_the_present_consumer():
    """Disabled consumers cannot receive uptake in plant/fungus-only runs."""
    plant = run_example("plant-only")
    fungus = run_example("fungus-only")

    assert plant["plant_uptake_mg"] > 0.0
    assert plant["fungus_uptake_mg"] == pytest.approx(0.0, abs=1e-12)
    assert fungus["fungus_uptake_mg"] > 0.0
    assert fungus["plant_uptake_mg"] == pytest.approx(0.0, abs=1e-12)


def test_mixed_example_credits_both_consumers_and_is_reproducible():
    """The mixed scenario exercises competition with deterministic output."""
    first = run_example("mixed")
    second = run_example("mixed")

    assert first == second
    assert first["plant_uptake_mg"] > 0.0
    assert first["fungus_uptake_mg"] > 0.0
    assert 0.0 <= first["continuous_weight_mean"] <= 1.0


def test_example_rejects_unknown_mode():
    """The public runner fails clearly on an unsupported scenario name."""
    with pytest.raises(ValueError, match="mode"):
        run_example("unknown")
