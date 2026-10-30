"""Check or explicitly regenerate the small biological preservation suite.

Run with ``uv run python scripts/biological_reference.py check``.
Regeneration requires an explanation and never happens during checking.
"""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import jax
import numpy as np

from mycormarl.environments.base_mycor import BaseMycorMarl
from mycormarl.fungus.traits import FungusTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_units import MICROMOL_P_TO_MG_P
from mycormarl.static_rollout import evaluate_static_policy


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / 'tests' / 'fixtures' / 'biological'
SCENARIOS = ('mixed', 'plant-only', 'fungus-only', 'stressed')
TOLERANCE = {'relative': 2e-6, 'absolute': 1e-8}


def scenario(name):
    config = EnvConfig(max_steps=12, dt=.25, soil_radius_cm=2., soil_depth_cm=2.,
                       radial_interval_cm=.5, depth_interval_cm=.5,
                       initial_solution_p_um=10.)
    species = SpeciesParams(
        PlantTraits(initial_c_pool=.02, initial_p_pool=.08),
        FungusTraits(initial_c_pool=.005, initial_p_pool=.01))
    if name in ('plant-only', 'fungus-only'):
        config = config.replace(consumer_mode=name)
    actions = {'plant': [.3, .2, .1], 'fungus': [.4, .15, .1]}
    if name == 'stressed':
        species = species.replace(fungus=species.fungus.replace(
            initial_c_pool=0., kappa_c=.4, death_fraction=.5))
        actions['plant'][0] = 0.
    return config, species, actions


def differences(expected, actual, path, tolerance):
    """Yield field-specific differences, keeping discrete facts exact."""
    if type(expected) is not type(actual):
        yield f'{path}: type {type(expected).__name__} != {type(actual).__name__}'
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            yield f'{path}: fields {sorted(expected)} != {sorted(actual)}'
        for key in expected.keys() & actual.keys():
            yield from differences(expected[key], actual[key], f'{path}.{key}', tolerance)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            yield f'{path}: length {len(expected)} != {len(actual)}'
        else:
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                yield from differences(left, right, f'{path}[{index}]', tolerance)
    elif isinstance(expected, float):
        if not (math.isfinite(expected) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=tolerance['relative'],
                                 abs_tol=tolerance['absolute'])):
            yield f'{path}: expected {expected:.9g}, got {actual:.9g}, delta {actual-expected:.3g}'
    elif expected != actual:
        yield f'{path}: expected {expected!r}, got {actual!r}'


def validate(name, config, species, result):
    """Independent whole-system P ledger at every retained checkpoint."""
    totals = []
    for row in result['checkpoints']:
        state = row['state']
        total = float(np.sum(state['soil_labile_p'])) * MICROMOL_P_TO_MG_P
        for agent in ('plant', 'fungus'):
            total += float(np.sum(state[f'{agent}_p_pool']))
            total += float(np.sum(state[f'{agent}_biomass'])) * getattr(species, agent).gamma_p
            for suffix in ('mortality_loss', 'maintenance_loss', 'reproduction_export'):
                total += float(np.sum(state[f'cumulative_{agent}_p_{suffix}_mg']))
        totals.append(total)
        assert np.min(state['soil_labile_p']) >= 0., 'negative soil inventory'
        for agent in ('plant', 'fungus'):
            for resource in ('c', 'p'):
                assert min(state[f'{agent}_{resource}_pool']) >= 0., 'negative free pool'
            if config.consumer_mode not in ('mixed', agent + '-only'):
                geometry = 'root_length_density' if agent == 'plant' else 'hyphae_length_density'
                for field in (f'{agent}_biomass', f'{agent}_c_pool', f'{agent}_p_pool', geometry):
                    assert np.all(np.asarray(state[field]) == 0.), f'absent {agent}: {field}'
                assert state[f'{agent}_dead'] == [True]
                assert row['observations'][agent] == [0.] * 5
                assert row['rewards'][agent] == 0.
                if row['transitions'] is not None:
                    assert not row['transitions'][agent]['allocation_executed']
                    assert not row['transitions'][agent]['trade_executed']
                    assert row['flows'][agent]['direct_p_uptake_mg'] == [0.]
        if row['transitions'] is not None:
            assert all(row['flows'][a]['allocation_converged'] for a in ('plant', 'fungus'))
    assert max(abs(value-totals[0]) for value in totals) <= 1e-7 * max(1., totals[0]), 'P ledger'
    assert [row['state']['step'] for row in result['checkpoints']] == [0, 2, 4, 6, 8, 10, 12]
    stressed = name == 'stressed'
    if stressed:
        assert result['rejection_reasons'] == ['biological failure: fungus died']
        assert result['biological_deaths'] == {'plant': 0, 'fungus': 1}
        assert result['p_loss_or_export_counters']['fungus_mortality'] > 0.
        assert result['checkpoints'][1]['state']['fungus_biomass'][0] < species.fungus.initial_biomass
        death = result['checkpoints'][2]
        assert death['state']['fungus_dead'] == [True]
        assert death['flows']['fungus']['c_deficit'][0] > 0.
        assert death['flows']['fungus']['trade_cancelled'] == [True]
        assert death['flows']['fungus']['trade_out'] == [0.]
        for row in result['checkpoints'][2:]:
            assert row['state']['fungus_dead'] == [True]
            assert np.all(np.asarray(row['state']['hyphae_length_density']) == 0.)
            assert row['flows']['fungus']['direct_p_uptake_mg'] == [0.]
            assert row['rewards']['fungus'] == 0.
    else:
        assert result['status'] == 'completed', result['rejection_reasons']
    for agent in ('plant', 'fungus'):
        if config.consumer_mode not in ('mixed', agent + '-only'):
            continue
        if stressed and agent == 'fungus':
            continue
        assert result['cumulative_growth'][agent] > 0., f'{agent} growth not exercised'
        assert result['cumulative_reproductive_fitness'][agent] > 0., f'{agent} reproduction not exercised'
        assert result['uptake'][agent] > 0., f'{agent} uptake not exercised'
    if stressed:
        assert result['transfers']['fungus_p_out'] > 0., 'pre-death trade not exercised'
    elif config.consumer_mode == 'mixed':
        assert min(result['transfers'].values()) > 0., 'bilateral trade not exercised'
    else:
        assert list(result['transfers'].values()) == [0., 0.]


def run(name):
    config, species, actions = scenario(name)
    started = time.perf_counter()
    result = evaluate_static_policy(
        BaseMycorMarl(config, species), actions, seed=7,
        record_resource_accounting=True, checkpoint_steps=(2, 4, 6, 8, 10, 12))
    seconds = time.perf_counter() - started
    timing = result.pop('timing_seconds')
    timing['total'] = seconds
    validate(name, config, species, result)
    declaration = dict(config=asdict(config), species=asdict(species), actions=actions,
                       seed=7, checkpoint_steps=[2, 4, 6, 8, 10, 12])
    return declaration, result, timing


def provenance(reason):
    sources = sorted((ROOT / 'src' / 'mycormarl').rglob('*.py'))
    return {
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        'action_contract': 'ADR-0020 / #79–#85: three-rate stoichiometric allocation',
        'reason': reason, 'python': platform.python_version(), 'platform': platform.platform(),
        'dependencies': {name: version(name) for name in ('jax', 'jaxlib', 'numpy', 'flax', 'chex')},
        'backend': jax.default_backend(), 'x64': jax.config.x64_enabled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'regenerate'))
    parser.add_argument('--scenario', choices=SCENARIOS, action='append')
    parser.add_argument('--reference-dir', type=Path, default=REFERENCE_DIR)
    parser.add_argument('--reason', help='required explanation for a reviewed baseline update')
    args = parser.parse_args()
    if args.command == 'regenerate' and not args.reason:
        parser.error('regenerate requires --reason; checking never updates fixtures')
    failures = []
    for name in args.scenario or SCENARIOS:
        path = args.reference_dir / f'{name}.json'
        expected = json.loads(path.read_text()) if args.command == 'check' else None
        declaration, result, timing = run(name)
        if expected is not None:
            # Historical all-external fixtures predate the optional IRM trait.
            # Interpret its documented zero default without rewriting evidence.
            expected['scenario']['species']['fungus'].setdefault('irm_fraction', 0.0)
            failures.extend(differences(expected['scenario'], declaration, name + '.scenario', TOLERANCE))
            failures.extend(differences(expected['result'], result, name, expected['tolerance']))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(schema=1, scenario=declaration, result=result,
                                           tolerance=TOLERANCE, provenance=provenance(args.reason)),
                                       indent=2, allow_nan=False) + '\n')
        print(f'{name}: {args.command}, compile={timing["compilation"]:.3f}s, '
              f'execution={timing["execution"]:.3f}s, total={timing["total"]:.3f}s')
    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
