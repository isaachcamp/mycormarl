"""Exercise the batch wrapper using scheduler/GPU stubs and the real sweep CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/slurm/batching_audit.sbatch'


@pytest.mark.parametrize(('device','arguments','exit_code'), [
    ('gpu', ['--dry-run'], 0),
    ('gpu', ['--dry-run', '--env-config', 'CUSTOM_ENVIRONMENT'], 0),
    ('cpu', ['--dry-run'], 1),
    ('unavailable', ['--dry-run'], 1),
    ('gpu', ['--not-an-audit-option'], 2),
])
def test_batch_wrapper_saves_outputs_and_propagates_failures(tmp_path, device, arguments, exit_code):
    binaries = tmp_path/'bin'
    binaries.mkdir()
    for name, content in {
        'srun': '#!/bin/sh\n[ "$1" != "--unbuffered" ] || shift\nexec "$@"\n',
        'nvidia-smi': '#!/bin/sh\nprintf "simulated GPU diagnostic\\n"\n',
    }.items():
        path = binaries/name
        path.write_text(content)
        path.chmod(0o755)
    (tmp_path/'jax.py').write_text(
        'import os\n'
        'class Device:\n'
        '    id=0\n'
        '    platform=os.environ["TEST_DEVICE"]\n'
        '    device_kind="simulated device"\n'
        'def devices():\n'
        '    if os.environ["TEST_DEVICE"] == "unavailable":\n'
        '        raise RuntimeError("Backend cuda is not in the list of known backends: cpu, tpu")\n'
        '    return [Device()]\n'
    )
    # Only the GPU preflight uses the stub. Configuration generation and the
    # sweep CLI run with the real interpreter and installed project modules.
    python_wrapper = binaries/'python-wrapper'
    python_wrapper.write_text(
        '#!/bin/sh\n'
        'case "${2:-}" in\n'
        '  */runtime.json) export PYTHONPATH="$TEST_JAX_PATH" ;;\n'
        'esac\n'
        'exec "$TEST_REAL_PYTHON" "$@"\n'
    )
    python_wrapper.chmod(0o755)
    custom = tmp_path/'custom-environment.json'
    custom.write_text(json.dumps(dict(soil_radius_cm=30., soil_depth_cm=60.,
                                      radial_interval_cm=.1, depth_interval_cm=.1)))
    arguments = [str(custom) if item == 'CUSTOM_ENVIRONMENT' else item for item in arguments]
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    output = tmp_path/'outputs with spaces'
    environment = dict(os.environ, SLURM_JOB_ID='1234', SLURM_SUBMIT_DIR=str(ROOT),
                       MYCORMARL_PYTHON=str(python_wrapper), TEST_REAL_PYTHON=sys.executable,
                       TEST_JAX_PATH=str(tmp_path), SLURM_TMPDIR=str(scratch),
                       TEST_DEVICE=device, PATH=str(binaries)+os.pathsep+os.environ['PATH'])
    result = subprocess.run(['bash', str(SCRIPT), str(output), *arguments],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr + result.stdout
    assert not list(scratch.iterdir()), 'temporary configuration/cache must be removed'
    job = output/'job-1234'
    assert (job/'exit-code.txt').read_text().strip() == str(exit_code)
    runtime = json.loads((job/'runtime.json').read_text())
    assert runtime['jax_platforms'] == 'cuda'
    if device == 'unavailable':
        assert runtime['devices'] == []
        assert 'known backends' in runtime['backend_error']
    else:
        assert runtime['devices'][0]['platform'] == device
    assert (job/'source-revision.txt').read_text().strip()
    assert 'simulated GPU diagnostic' in (job/'nvidia-smi.txt').read_text()
    log = (job/'job.log').read_text()
    if exit_code == 0:
        manifest = json.loads((job/'audit/manifest.json').read_text())
        assert len(manifest['runs']) == 18
        grid = manifest['environment_overrides']
        if '--env-config' in arguments:
            assert grid == json.loads(custom.read_text())
        else:
            assert grid['soil_radius_cm'] == 50.
            assert grid['soil_depth_cm'] == 100.
            assert grid['radial_interval_cm'] == .05
            assert grid['depth_interval_cm'] == .05
            assert grid['dt'] == .025
        temporary_config = Path(runtime['audit_arguments'][1])
        assert runtime['audit_arguments'][0] == '--env-config'
        assert not temporary_config.exists()
        assert '"runs": 18' in log
        # A second invocation must not overwrite the same job's retained files.
        duplicate = subprocess.run(['bash', str(SCRIPT), str(output), '--dry-run'],
                                   env=environment, capture_output=True, text=True)
        assert duplicate.returncode != 0
        assert (job/'exit-code.txt').read_text().strip() == '0'
    elif device == 'unavailable':
        assert 'jax[cuda12]==' in log
        assert 'uv pip install' in log
        assert not (job/'audit').exists()
    elif device == 'cpu':
        assert 'requires exactly one visible JAX GPU' in log
        assert not (job/'audit').exists()
    else:
        assert 'unrecognized arguments' in log
