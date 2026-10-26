"""Public reference-check command detects changed biological evidence."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / 'scripts' / 'biological_reference.py'
FIXTURES = ROOT / 'tests' / 'fixtures' / 'biological'


@pytest.mark.parametrize('field,value', [('plant_biomass', [99.]), ('plant_dead', [True])])
def test_reference_command_rejects_changed_checkpoint_without_overwriting_it(tmp_path, field, value):
    reference = json.loads((FIXTURES / 'mixed.json').read_text())
    reference['result']['checkpoints'][1]['state'][field] = value
    path = tmp_path / 'mixed.json'
    original = json.dumps(reference)
    path.write_text(original)
    result = subprocess.run(
        [sys.executable, str(COMMAND), 'check', '--scenario', 'mixed',
         '--reference-dir', str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'mixed.checkpoints[1].state.' + field in result.stdout
    assert path.read_text() == original


@pytest.mark.parametrize('scenario', ['mixed', 'plant-only', 'fungus-only', 'stressed'])
def test_reference_command_preserves_each_biological_scenario(scenario):
    result = subprocess.run([sys.executable, str(COMMAND), 'check', '--scenario', scenario],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('scenario', ['mixed', 'plant-only', 'fungus-only', 'stressed'])
def test_batched_static_summaries_match_the_frozen_biological_reference(scenario):
    from mycormarl.environments.base_mycor import BaseMycorMarl
    from mycormarl.fungus.traits import FungusTraits
    from mycormarl.params import EnvConfig, SpeciesParams
    from mycormarl.plant.traits import PlantTraits
    from mycormarl.static_rollout import evaluate_batched_static_policy

    fixture = json.loads((FIXTURES / f'{scenario}.json').read_text())
    declaration = fixture['scenario']
    env = BaseMycorMarl(EnvConfig(**declaration['config']), SpeciesParams(
        PlantTraits(**declaration['species']['plant']),
        FungusTraits(**declaration['species']['fungus'])))
    actual = evaluate_batched_static_policy([env], declaration['actions'], seed=7)[0]
    for field, value in actual.items():
        expected = fixture['result'][field]
        if isinstance(value, dict) and all(isinstance(v, (float, int)) for v in value.values()):
            assert value == pytest.approx(expected, rel=2e-6, abs=1e-8), field
        elif isinstance(value, float):
            assert value == pytest.approx(expected, rel=2e-6, abs=1e-8), field
        else:
            assert value == expected, field
