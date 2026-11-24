# Run the batching audit on JASMIN Orchid

[scripts/slurm/batching_audit.sbatch](../scripts/slurm/batching_audit.sbatch)
runs the existing batching sweep sequentially on one GPU and retains its
outputs in a fresh directory for the Slurm job.

The directives use `partition=orchid`, `account=orchid`, `qos=orchid` and
`gres=gpu:1`, following the [Orchid submission guidance](https://help.jasmin.ac.uk/docs/batch-computing/orchid-gpu-cluster/).
Initial requests are four CPUs, 32 GB host memory and two hours. These are
starting requests, not measured production requirements; override them with
`sbatch --cpus-per-task=... --mem=... --time=...` after measuring the workload.
The ordinary Orchid QoS permits up to 24 hours.

## Environment

Before submission, prepare a Python >=3.12 environment with this checkout
installed and CUDA-enabled JAX compatible with the node's NVIDIA driver. Follow
[JAX's installation guidance](https://docs.jax.dev/en/latest/installation.html).
The script uses `<checkout>/.venv/bin/python` by default and does not install
packages or change CUDA modules during the job. To select another interpreter:

```bash
export MYCORMARL_PYTHON=/absolute/path/to/environment/bin/python
```

Submit from the repository root on a JASMIN science server. The checkout,
Python environment, configuration inputs and output directory must be visible
on the compute node. `MYCORMARL_REPO_DIR` can select a different checkout.

The wrapper forces `JAX_PLATFORMS=cuda` and checks for exactly one visible JAX
GPU before starting the sweep. It preserves Slurm's `CUDA_VISIBLE_DEVICES`.
JAX's [platform configuration](https://docs.jax.dev/en/latest/config_options.html#platforms)
raises an error when a requested backend cannot initialise, preventing silent
CPU measurements. GPU/driver diagnostics and Python package versions are saved.

## Submit

The first argument is an output root on persistent storage. Remaining arguments
are forwarded to `sweep_ppo.py`; the wrapper supplies the job-specific
`--output-dir` itself.

```bash
# Replace this with a persistent directory you can write on JASMIN.
audit_root=/path/to/mycormarl-data/batching-audit
mkdir -p "$audit_root"
sbatch --output="$audit_root/slurm-%j.out" \
  scripts/slurm/batching_audit.sbatch "$audit_root"
```

The scheduler log records the path to the detailed job log. Without the
`--output` override above, the scheduler creates `slurm-batching-<jobid>.out`
in the submission directory; the audit outputs still go under the supplied
output root.

For representative measurements, pass the same qualified environment to all
cases:

```bash
sbatch --output="$audit_root/slurm-%j.out" \
  scripts/slurm/batching_audit.sbatch "$audit_root" \
  --env-config /path/to/qualified-environment.json \
  --episode-steps 14600 --seeds 42 43 44
```

Without `--env-config`, the sweep uses its small development soil grid.
See the [batching guide](ppo-batching-audit.md) for its defaults and the fixed-
batch/fixed-rollout comparisons. The script leaves those experiment defaults
unchanged. For a short execution check, append:

```bash
--suite fixed-rollout --case n4-t128 --seeds 42 \
--total-timesteps 1024 --episode-steps 4 \
--evaluation-interval-timesteps 512
```

Appending `--dry-run` instead creates the eighteen-run default manifest without
training; inside Slurm it still performs the GPU preflight. A planning preview
can also be run locally with the Python sweep CLI without allocating a GPU.

## Saved output

Each submission creates `<output-root>/job-<SLURM_JOB_ID>/` containing:

- `job.log`: wrapper output and all sweep stdout/stderr, including failure messages.
- `source-revision.txt`, `worktree-status.txt`, `started-at-utc.txt`: source and start provenance.
- `nvidia-smi.txt`, `runtime.json`: driver/GPU details, Python/package versions,
  visible JAX device, Slurm job identity and forwarded audit arguments.
- `exit-code.txt`: exit status when the shell's exit trap runs; forced process
  termination may prevent this file from being written.
- `audit/`: sweep manifest, environment snapshot, per-run metrics/checkpoints,
  evaluation policies/summaries and `results.json` as successful runs complete.

A nonzero sweep exit propagates to the Slurm job. Existing job directories are
never overwritten. The script does not automatically resume or requeue old
sweeps: the two [known resume limitations](ppo-batching-audit.md#known-resume-limitations)
remain deferred. Completed artifacts and checkpoints stay on persistent storage
if a later subrun fails or the allocation ends.

Local verification uses `bash -n` and an integration harness with simulated
Slurm/GPU commands and the real sweep CLI in dry-run mode. Actual Orchid
submission, CUDA execution and resource sizing must be verified on JASMIN.
