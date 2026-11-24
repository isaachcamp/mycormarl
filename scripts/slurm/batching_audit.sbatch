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
trap 'task_exit_code=$?; printf "%s\n" "$task_exit_code" > "$job_dir/exit-code.txt"' EXIT
printf 'Job log: %s/job.log\n' "$job_dir"
exec > "$job_dir/job.log" 2>&1

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$job_dir/matplotlib-cache"

printf 'Audit output: %s\n' "$job_dir"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$job_dir/started-at-utc.txt"
git rev-parse HEAD > "$job_dir/source-revision.txt"
git status --short > "$job_dir/worktree-status.txt"
nvidia-smi > "$job_dir/nvidia-smi.txt"
"$python_bin" - "$job_dir/runtime.json" "$@" <<'PY'
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import jax

devices = jax.devices()
metadata = dict(
    python=sys.version, executable=sys.executable, host=platform.node(),
    slurm_job_id=os.environ['SLURM_JOB_ID'],
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    jax_platforms=os.environ['JAX_PLATFORMS'],
    packages={name: version(name) for name in ('jax','jaxlib','flax','optax','jaxmarl')},
    devices=[dict(id=d.id, platform=d.platform, kind=d.device_kind) for d in devices],
    audit_arguments=sys.argv[2:],
)
Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2) + '\n')
if len(devices) != 1 or devices[0].platform != 'gpu':
    raise SystemExit('This audit requires exactly one visible JAX GPU')
print(f'JAX GPU: {devices[0].device_kind}', flush=True)
PY

srun --unbuffered "$python_bin" scripts/sweep_ppo.py "$@" --output-dir "$job_dir/audit"
