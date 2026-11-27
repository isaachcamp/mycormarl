# Run the batching audit on JASMIN Orchid

[scripts/slurm/batching_audit.sbatch](../scripts/slurm/batching_audit.sbatch)
runs the existing batching sweep sequentially on one GPU and retains its
outputs in a fresh directory for the Slurm job. It generates the current
production `EnvConfig` automatically: 50 cm soil radius, 100 cm depth and
0.05 cm spacing (1,000 × 2,000 cells). No manually created JSON is needed.
This Slurm job measures training throughput by default; optional full-horizon
policy evaluation is disabled (`--evaluation-interval-timesteps 0`).

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

If JAX reports `Backend 'cuda' is not in the list of known backends:
['cpu', 'tpu']`, the environment has no registered CUDA backend. Plain `jax`
in the project lock does not request the GPU extras. On JASMIN, install CUDA
support in that same environment before submission. For a Linux NVIDIA driver
>=525, use CUDA 12 while preserving the installed JAX version:

```bash
audit_jax_version=$(.venv/bin/python -c 'from importlib.metadata import version; print(version("jax"))')
uv pip install --python .venv/bin/python "jax[cuda12]==${audit_jax_version}"
```

CUDA 13 instead requires driver >=580, as documented by
[JAX](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).
Check the saved `nvidia-smi.txt` for the allocation's driver. Use the selected
interpreter path instead of `.venv/bin/python` if it differs. CUDA extras are
not in the project lock: an exact `uv sync` can remove them; reinstall them
after syncing, or use `uv sync --frozen --inexact` to retain extra packages.
Unsetting the requested CUDA platform would only allow CPU fallback; it does
not add GPU support. GPU verification must happen on a GPU node/allocation.

Submit from the repository root on a JASMIN science server. The checkout,
Python environment, configuration inputs and output directory must be visible
on the compute node. `MYCORMARL_REPO_DIR` can select a different checkout.

The wrapper sets `JAX_PLATFORMS=cuda,cpu`: CUDA is the default for training,
while the CPU backend remains available for `jax.debug.callback` numerical
qualification checks. It verifies exactly one visible GPU and an available
local CPU device before starting the sweep, and preserves Slurm's
`CUDA_VISIBLE_DEVICES`. Setting only `cuda` disables the callback's required
CPU backend and causes training to fail even when the GPU is detected.
JAX's [platform configuration](https://docs.jax.dev/en/latest/config_options.html#platforms)
raises an error when a requested backend cannot initialise, preventing silent
CPU measurements. GPU/driver diagnostics and Python package versions are saved.

## Submit

The first argument is an output root on persistent storage. Remaining arguments
are forwarded to `sweep_ppo.py`; the wrapper supplies the job-specific
`--output-dir` itself. User arguments can override the wrapper's production
configuration and throughput-only evaluation setting.

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

The default submission already supplies production defaults to every case.
To use a different study configuration, explicitly replace the generated
configuration:

```bash
sbatch --output="$audit_root/slurm-%j.out" \
  scripts/slurm/batching_audit.sbatch "$audit_root" \
  --env-config /path/to/qualified-environment.json \
  --episode-steps 14600 --seeds 42 43 44
```

The Slurm wrapper serialises `EnvConfig()` in a temporary directory, without
constructing a soil grid. The actual audit still requires CUDA. Its copied
configuration is retained in `audit/manifest.json` and, for an executing sweep,
`audit/environment.json`. An explicit `--env-config` replaces the generated
file; unspecified fields in that custom file follow the Python training CLI's
fallbacks, so specify all grid fields for a custom domain.

See the [batching guide](ppo-batching-audit.md) for the fixed-batch/fixed-rollout
comparisons. The standalone Python sweep still uses the small development grid
without `--env-config`; the Slurm wrapper now supplies production defaults.
It disables policy evaluation by default, preserving all training numerical
checks. Compilation/execution timings, memory diagnostics, training metrics and
checkpoints are still saved; `learning_curve` is empty without evaluations.

To opt into learning curves, pass `--evaluation-interval-timesteps 4096` with an
appropriate, qualified episode horizon. The default horizon is 365 days; an
evaluation policy can reach the 50 g plant biomass numerical guard over that
horizon and correctly abort. This is not a GPU error or biological saturation.
A throughput audit does not require those evaluation episodes. Training still
aborts if it contacts the guard; disabling evaluation does not accept invalid
training transitions.

For a short execution check on the production grid, append:

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
- `nvidia-smi.txt`, `runtime.json`: driver/GPU details, Python/package versions
  (including installed CUDA plugins), visible JAX device, Slurm job identity and
  forwarded audit arguments. CUDA initialisation errors are saved in the runtime
  JSON and reported with a version-matched installation command.
- `exit-code.txt`: exit status when the shell's exit trap runs; forced process
  termination may prevent this file from being written.
- `audit/`: sweep manifest, environment snapshot, per-run metrics/checkpoints,
  evaluation policies/summaries and `results.json` as successful runs complete.

The generated input JSON and Matplotlib cache live in a job-owned temporary
directory under `SLURM_TMPDIR`, `TMPDIR` or `/tmp`. The exit handler removes
that directory on success or ordinary failure, and on handled INT/TERM signals.
A forced kill can bypass cleanup. Audit configurations and results in the
persistent output directory are retained.

A nonzero sweep exit propagates to the Slurm job. Existing job directories are
never overwritten. The script does not automatically resume or requeue old
sweeps: the two [known resume limitations](ppo-batching-audit.md#known-resume-limitations)
remain deferred. Completed artifacts and checkpoints stay on persistent storage
if a later subrun fails or the allocation ends.

Local verification uses `bash -n` and an integration harness with simulated
Slurm/GPU commands and the real sweep CLI in dry-run mode. Actual Orchid
submission, CUDA execution and resource sizing must be verified on JASMIN.
