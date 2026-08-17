# Study manifest and result-bundle guide

**Status:** Implemented walking-skeleton contract

This guide describes how to prepare and run a study through the public
[`run_study`](../../mycormarl/mycormarl/study.py) interface introduced for
GitHub issue #33. It covers the current manifest schema, output identity,
resume rules, and completed-bundle immutability.

The current runner supports only the `walking-skeleton` stage. That stage is a
non-scientific contract fixture: it exercises manifest validation, condition
enumeration, persistence, provenance, resume, and reporting, but it does not
train policies or emit biological endpoints. Later study stages should extend
this same path rather than introduce separate orchestration contracts.

## Manifest validity and execution completion are different concepts

A manifest is either valid or invalid. It does not have an `incomplete` or
`complete` status.

- A **valid manifest** contains every required declaration and passes all
  preflight checks.
- An **invalid or incomplete manifest** is missing a declaration or contains an
  unsupported or inconsistent value. It fails before an output directory is
  created or study work starts.
- `incomplete` and `complete` are states of the persisted **result bundle**.
  They describe how much of a valid manifest's condition matrix has executed.

This distinction matters when resuming work: the manifest must remain complete
and compatible while its result bundle may temporarily be incomplete.

## Minimal manifest

The runner accepts a path to a JSON object. A minimal two-mode fixture looks
like this:

```json
{
  "schema_version": 1,
  "stage": "walking-skeleton",
  "model": {
    "environment": {
      "soil_radius_cm": 1.0,
      "soil_depth_cm": 1.0
    },
    "species": {
      "plant": {},
      "fungus": {}
    }
  },
  "horizon": {
    "days": 0.05,
    "timestep_days": 0.025
  },
  "modes": ["mixed", "plant-only"],
  "initial_p_micromolar": [0.3],
  "seeds": [7],
  "training": {
    "total_timesteps": 2,
    "checkpoint_interval_timesteps": 1
  },
  "evaluation": {
    "protocol": "latent-location",
    "episodes": 1
  },
  "output": {
    "directory": "outputs/studies",
    "identity": "tiny-fixture"
  }
}
```

Run it from Python with:

```python
from mycormarl.study import run_study

result = run_study("path/to/manifest.json")
print(result.bundle_path)
print(result.summary_path)
```

## Required declarations

| Field | Current contract |
|---|---|
| `schema_version` | Must be integer `1`. Other versions are incompatible. |
| `stage` | Must currently be `walking-skeleton`. A scientific stage is rejected until its implementation exists. |
| `model.environment` | Must be a JSON object containing the complete environment configuration required by the intended stage. The walking skeleton checks the object boundary but does not interpret model parameters. |
| `model.species` | Must be a JSON object containing the complete species configuration required by the intended stage. The walking skeleton checks the object boundary but does not interpret model parameters. |
| `horizon.days` | Must be finite and greater than zero. |
| `horizon.timestep_days` | Must be finite and greater than zero. `days / timestep_days` must be a whole number of transitions. |
| `modes` | A non-empty list of unique values selected from `mixed` and `plant-only`. |
| `initial_p_micromolar` | A non-empty list of unique, finite, strictly positive initial solution-P levels. |
| `seeds` | A non-empty list of unique, non-negative integer master-seed IDs. Boolean values are not seed IDs. |
| `training.total_timesteps` | A positive integer training budget. |
| `training.checkpoint_interval_timesteps` | A positive integer no greater than `total_timesteps`. |
| `evaluation.protocol` | Must currently be `latent-location`. |
| `evaluation.episodes` | A positive integer. |
| `output.directory` | A non-empty path string naming the common output root. |
| `output.identity` | A stable study name, not a path. It must start with an ASCII letter or digit and may then contain letters, digits, `.`, `_`, or `-`. |

All scientific choices that could change an execution should be written into
the manifest rather than supplied through unrecorded runtime state. Later
stages may add required nested declarations, such as qualification parents or
stopping rules, while retaining these top-level boundaries.

## Condition matrix

The requested condition matrix is the Cartesian product of:

```text
modes × initial_p_micromolar × seeds
```

The example therefore requests two conditions:

```text
(mixed,      0.3 micromolar, seed 7)
(plant-only, 0.3 micromolar, seed 7)
```

Each result entry is identified by this three-part condition key. Duplicate
values on any manifest axis are rejected because they would create ambiguous
or repeated condition identities.

## Output layout

The runner writes beneath `output.directory / output.identity`:

```text
outputs/studies/
└── tiny-fixture/
    ├── result-bundle.json
    └── summary.md
```

`result-bundle.json` is the machine-readable source of truth. It contains:

- result format and format version;
- separate scientific study and reproducible execution identities;
- the parsed content of the submitted manifest;
- software and interface provenance;
- one entry per requested condition;
- requested and completed counts; and
- overall `incomplete` or `complete` status.

`summary.md` is derived from the saved bundle. It reports the study identity,
stage, completion status, completed-condition count, Git commit, and result
execution identity. If a completed bundle remains intact but its summary is
lost, rerunning the same manifest reconstructs only the summary and does not
rewrite the bundle.

The result-bundle format version is `2`. Version 1 bundles predate the identity
split and strict environment provenance and are rejected rather than silently
treated as equivalent.

## Study identity, execution identity, and provenance

Before any output is written, the runner computes two SHA-256 identities.

The **study identity** hashes the canonical scientific manifest. It includes
every top-level manifest declaration except `output`. Output location and
naming are storage concerns, so changing `output.directory` or
`output.identity` does not redefine the scientific question.

The **execution identity** hashes the study identity together with the complete
execution provenance. It therefore identifies one reproducible realization of
that study under a specific committed source and runtime environment.

The recorded provenance contains:

- the full Git commit hash;
- `git_dirty: false`;
- the SHA-256 hash of the repository's `uv.lock`;
- the MycorMARL package version;
- Python, JAX, and JAXlib versions;
- the manifest and result-format versions;
- the actor-interface and environment-state schema versions; and
- the execution kind.

Changing model configuration, horizon, modes, P levels, seeds, training, or
evaluation changes both identities. Changing the Git commit, dependency lock,
runtime versions, or an interface version preserves the study identity but
changes the execution identity.

Package or semantic versions complement the commit hash; they do not replace
it. A version such as `0.1.0` identifies a human-facing release line, while the
full commit identifies the exact committed source revision.

## Clean-checkout requirement

The commit hash identifies committed source only. Before execution, the runner
checks tracked modifications and untracked files with Git. Any dirty state
fails before the output directory is created. Formal runs must therefore be
started after committing or stashing all changes. Git-ignored output paths,
including the repository's `outputs/` directory, do not make the checkout
dirty.

The runner also fails if it cannot resolve the Git repository or `HEAD`, or if
the repository's `uv.lock` cannot be read. It never records an unknown source
or dependency version. The lock hash documents the intended complete Python
dependency solution, while the separately recorded Python, JAX, and JAXlib
versions expose the most important active runtime versions.

This contract deliberately does not support formal dirty-tree executions. If
exploratory dirty-tree runs are needed later, they should use a separately
labelled non-reproducible mode or archive and hash a reviewed patch; they must
not claim the clean commit as their exact source.

## Bundle compatibility

An existing bundle is compatible only when all of the following agree:

1. result format and format version;
2. scientific study identity;
3. execution identity;
4. full execution provenance;
5. both identities recomputed from the manifest and provenance embedded in the
   bundle; and
6. the declared condition inventory, statuses, and completion counts.

Unreadable, unversioned, internally inconsistent, or stale-interface bundles
are rejected. If the selected output directory already contains files but no
result bundle carrying a compatible execution identity, the runner also
rejects the run; it will not claim or join orphaned checkpoints or outputs.

## Fresh execution

For a fresh execution, the selected output directory must either not exist or
be empty. The runner:

1. reads and validates the complete manifest;
2. verifies the clean checkout and computes provenance and both identities;
3. enumerates the condition matrix;
4. executes each condition through the current stage implementation;
5. saves the versioned result bundle; and
6. reads that saved bundle to derive `summary.md`.

For `walking-skeleton`, execution records each condition as `completed` with
`execution_kind: contract-fixture`. These entries prove orchestration behavior
only and must not be interpreted as learned IPPO or biological outcomes.

## Resuming an incomplete result bundle

A compatible incomplete bundle has:

- overall `status: "incomplete"`;
- zero or one entry for each requested condition;
- only `pending` or `completed` entry statuses;
- no condition outside the current manifest matrix; and
- a `completion` object whose completed count matches its entries and whose
  requested count matches the complete manifest matrix.

When the same valid manifest is run again, the runner preserves every
completed entry exactly as stored. It executes pending or absent conditions,
then writes the combined bundle and regenerates the summary. Completed entries
are selected by `(mode, initial_p_micromolar, seed)` and are not recreated or
replaced during resume.

Changing the scientific manifest while work is incomplete changes the study
and execution identities. Changing the commit, dependency lock, runtime, or
active interface versions changes the execution identity. Either change is
incompatible and cannot join the existing bundle. Return to the exact recorded
execution environment to resume, or preserve the old bundle and choose a new
`output.identity` for a new execution.

## Reusing a completed result bundle

A bundle may claim `status: "complete"` only when:

- every requested condition appears exactly once;
- every entry has `status: "completed"`; and
- completed and requested counts both equal the size of the manifest matrix.

After compatibility and inventory checks pass, running the same manifest
returns the existing bundle and summary paths without opening either artifact
for replacement. This makes a completed result bundle immutable through the
public runner.

The overwrite protections are:

- **Same study and execution identities:** reuse the completed bundle without
  writing.
- **Same bundle, missing summary:** rebuild only the derived summary.
- **Different manifest under the same output identity:** reject because the
  study identity differs.
- **Same study under different code, dependencies, runtime, or interfaces:**
  reject because the execution identity differs.
- **Corrupt completion claim or condition inventory:** reject rather than
  aggregate or repair completed machine data.
- **Existing files without a compatible bundle:** reject rather than join the
  directory.

The runner never silently replaces a completed bundle with results from a
different execution. To revise a completed study, preserve the old bundle and
choose a new manifest `output.identity`.

## Setup checklist

Before starting a study:

1. Choose a supported stage and record all stage-specific scientific inputs in
   `model`, `training`, and `evaluation`.
2. Confirm that horizon and timestep produce an integral transition count.
3. Declare unique modes, P levels, and master seed IDs.
4. Choose a stable output identity that describes this frozen design.
5. Commit the source, manifest if tracked, and all other changes; verify that
   the Git checkout is clean.
6. Sync the environment from the committed `uv.lock`.
7. Ensure the selected output directory is new, empty, or contains only the
   compatible resumable bundle for this exact execution.
8. Preserve the manifest with the result bundle; do not edit either after a
   completed run.
9. Treat `result-bundle.json` as canonical and generate human reporting from
   it rather than from untracked runtime state.

The executable contract tests are in
[`tests/test_study_runner.py`](../../tests/test_study_runner.py).
