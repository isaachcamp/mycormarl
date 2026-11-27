#!/bin/bash
#SBATCH --job-name=mycormarl-batching
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-batching-%j.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf 'Usage: sbatch %s OUTPUT_ROOT [sweep_ppo.py arguments...]\n' "$0" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
repo_dir="${MYCORMARL_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$repo_dir"
repo_dir="$(pwd -P)"
python_bin="${MYCORMARL_PYTHON:-$repo_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    printf 'Python is not executable: %s (set MYCORMARL_PYTHON)\n' "$python_bin" >&2
    exit 2
fi
output_root="$1"
shift
mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd -P)"
job_dir="$output_root/job-$SLURM_JOB_ID"
mkdir "$job_dir"
scratch_dir=""
cleanup() {
    task_exit_code=$?
    if [[ -n "$scratch_dir" ]]; then
        rm -rf -- "$scratch_dir"
    fi
    printf '%s\n' "$task_exit_code" > "$job_dir/exit-code.txt"
    exit "$task_exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Job log: %s/job.log\n' "$job_dir"
exec > "$job_dir/job.log" 2>&1

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
scratch_dir="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/mycormarl-batching-$SLURM_JOB_ID.XXXXXX")"
export MPLCONFIGDIR="$scratch_dir/matplotlib-cache"

# Serialize production defaults without constructing the environment/grid.
# CUDA remains required for the preflight and the actual audit processes.
JAX_PLATFORMS=cpu "$python_bin" - "$scratch_dir/production-environment.json" <<'PYCONFIG'
from dataclasses import asdict
import json
from pathlib import Path
import sys
from mycormarl.params import EnvConfig

Path(sys.argv[1]).write_text(json.dumps(asdict(EnvConfig()), indent=2) + '\n')
PYCONFIG
# A throughput audit does not run full-horizon policy evaluations by default.
# Later user arguments can explicitly override either default.
set -- --env-config "$scratch_dir/production-environment.json" \
    --evaluation-interval-timesteps 0 "$@"

printf 'Audit output: %s\n' "$job_dir"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$job_dir/started-at-utc.txt"
git rev-parse HEAD > "$job_dir/source-revision.txt"
git status --short > "$job_dir/worktree-status.txt"
nvidia-smi > "$job_dir/nvidia-smi.txt"
"$python_bin" - "$job_dir/runtime.json" "$@" <<'PY'
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import jax

backend_error = None
callback_cpus = []
try:
    devices = jax.devices()
    callback_cpus = jax.local_devices(backend="cpu")
except RuntimeError as error:
    devices = []
    backend_error = str(error)
metadata = dict(
    python=sys.version, executable=sys.executable, host=platform.node(),
    slurm_job_id=os.environ['SLURM_JOB_ID'],
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    jax_platforms=os.environ['JAX_PLATFORMS'],
    packages={name: version(name) for name in ('jax','jaxlib','flax','optax','jaxmarl')},
    devices=[dict(id=d.id, platform=d.platform, kind=d.device_kind) for d in devices],
    audit_arguments=sys.argv[2:], backend_error=backend_error,
    callback_cpu_devices=len(callback_cpus),
)
for name in ('jax-cuda12-plugin', 'jax-cuda12-pjrt', 'jax-cuda13-plugin', 'jax-cuda13-pjrt'):
    try:
        metadata['packages'][name] = version(name)
    except PackageNotFoundError:
        metadata['packages'][name] = None
Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2) + '\n')
if backend_error:
    command = (f'uv pip install --python {shlex.quote(sys.executable)} '
               f'"jax[cuda12]=={version("jax")}"')
    raise SystemExit(
        f'CUDA backend initialization failed: {backend_error}\n'
        'Check CUDA-enabled JAX and the NVIDIA driver. If GPU extras are missing '
        f'and the Linux driver is >=525, add CUDA 12 support before submission:\n  {command}\n'
        'CUDA 13 requires driver >=580. See https://docs.jax.dev/en/latest/installation.html'
    )
if len(devices) != 1 or devices[0].platform != 'gpu':
    raise SystemExit('This audit requires exactly one visible JAX GPU')
if not callback_cpus:
    raise SystemExit('This audit requires a local CPU device for numerical callbacks')
print(f'JAX GPU: {devices[0].device_kind}; callback CPUs: {len(callback_cpus)}', flush=True)
PY

srun --unbuffered "$python_bin" scripts/sweep_ppo.py "$@" --output-dir "$job_dir/audit"
