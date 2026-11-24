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
    ('cpu', ['--dry-run'], 1),
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
        'def devices(): return [Device()]\n'
    )
    output = tmp_path/'outputs with spaces'
    environment = dict(os.environ, SLURM_JOB_ID='1234', SLURM_SUBMIT_DIR=str(ROOT),
                       MYCORMARL_PYTHON=sys.executable, PYTHONPATH=str(tmp_path),
                       TEST_DEVICE=device, PATH=str(binaries)+os.pathsep+os.environ['PATH'])
    result = subprocess.run(['bash', str(SCRIPT), str(output), *arguments],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr + result.stdout
    job = output/'job-1234'
    assert (job/'exit-code.txt').read_text().strip() == str(exit_code)
    runtime = json.loads((job/'runtime.json').read_text())
    assert runtime['jax_platforms'] == 'cuda'
    assert runtime['devices'][0]['platform'] == device
    assert (job/'source-revision.txt').read_text().strip()
    assert 'simulated GPU diagnostic' in (job/'nvidia-smi.txt').read_text()
    log = (job/'job.log').read_text()
    if exit_code == 0:
        manifest = json.loads((job/'audit/manifest.json').read_text())
        assert len(manifest['runs']) == 18
        assert '"runs": 18' in log
        # A second invocation must not overwrite the same job's retained files.
        duplicate = subprocess.run(['bash', str(SCRIPT), str(output), '--dry-run'],
                                   env=environment, capture_output=True, text=True)
        assert duplicate.returncode != 0
        assert (job/'exit-code.txt').read_text().strip() == '0'
    elif device == 'cpu':
        assert 'requires exactly one visible JAX GPU' in log
        assert not (job/'audit').exists()
    else:
        assert 'unrecognized arguments' in log
