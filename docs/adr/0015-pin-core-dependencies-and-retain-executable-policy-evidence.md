# ADR-0015: Pin core dependencies and retain executable policy evidence

**Status:** Accepted; archive hosting and routine build budget resolved in P5; upload and snapshot tags remain cutover work

## Context

The combined research repository must stay consistent with the version of the
biological environment and algorithms used for its results. The publication
evidence bundle need not contain every trajectory and training checkpoint when
derived evidence can be regenerated, but policy weights alone do not identify
the environment, actor construction, or evaluation protocol required to
execute them.

## Decision

Record the research repository's core dependency as an exact Git commit using
uv's dependency specification, committed in `pyproject.toml` and `uv.lock`.
Record core source revision, research source revision, and the execution
environment separately in scientific provenance. Changing the pin is a
deliberate research-repository change; it does not retroactively change the
identity of earlier results.

For publication, retain study summary results and selected policy network
weights, together with the requirements needed to execute the policies: model and organism
configuration, actor construction and policy context, interface versions,
evaluation protocol, and dependency/runtime information. Document training
seeds and their random-stream derivation, plus evaluation and experimental
sampling seeds where applicable, so others can repeat the protocol from
scratch. Seed documentation does not promise bitwise-identical retraining
across different hardware or numerical runtimes.

For reporting, regenerate derived numerical tables, trajectories, and figures
through the showyourwork! workflow rather than preserving all generated
outputs. Keep additional figure inputs when they cannot feasibly be
regenerated from the retained evidence. Full optimizer and environment
checkpoints are not a default publication archival requirement; restarting
training from documented seeds is distinct from resuming a partially completed
run.

Preserve a separate, richer study exploration record, including per-checkpoint
data and other detailed run evidence needed for follow-up scientific analysis
without rerunning simulations. Do not discard such data before sufficient
exploration. Preparing a publication evidence bundle does not authorize
deleting exploration records. Retention or pruning of those records is a
separate decision.

Store exploration records outside Git checkouts in a configurable data
directory. The research repository records run identities and locations.
Preserve existing records during extraction; relocating data is not a
prerequisite for splitting the code. Publication evidence bundles are selected
exports from those retained records. Exact storage and public archive
locations remain to be selected.

Retain compact per-checkpoint evaluation metrics used by reported learning
curves even when those curves' intermediate policy weights or full training
states are not included in the publication evidence bundle.

Expensive recomputation is provided through documented Slurm scripts declaring
and requesting compute resources, as established in ADR-0014. Permit
inexpensive policy evaluation from retained weights in the ordinary
showyourwork! build. Retain outputs of expensive evaluations and provide their
Slurm scripts. Measure actual evaluation costs before setting the routine
build budget.

Preserve report and submission source snapshots with fixed Git tags. Continue
the paper on an active development branch, using upgrade branches to test
deliberate core pin changes and permanent branches only when maintaining an
older version independently. Identify the report-producing source,
configuration, and results before tagging its snapshot.

## Documentation across repositories

When the repositories are created, document the relevant version-control and
dependency-pinning workflow in both core and research. The user excludes the
animation repository from this additional version-control documentation
requirement. This does not alter animation ownership under ADR-0014.

Carry relevant ADR decisions and the repository's domain-document conventions
into each new repository consistently. Keep `CONTEXT.md` as the established
glossary location and `docs/adr/` for architectural decisions. Core owns model
and algorithm contracts; research owns scientific protocols and
interpretation. Preserve traceability to originating ADRs when moving a
decision and avoid contradictory copies of the same decision. Detailed
documentation distribution and migration mechanics remain to be
planned.

## Sources

- Repository-boundary discussion, 13 October 2026: the user accepted uv-based
  core pinning, requested consistent version-control notes and ADRs in the new
  repositories, and selected summary results plus executable policy weights
  and documented seeds instead of retaining all generated results.
- The user subsequently accepted inexpensive evaluation in the ordinary
  build, compact learning-curve evidence, and fixed report tags, and clarified
  that richer per-checkpoint exploration records must remain available until
  sufficiently explored.
- [uv Git dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#git).
- [showyourwork! Zenodo integration](https://show-your.work/en/latest/zenodo/).

## P5 implementation — 14 October 2026

The user selected private GitHub release assets initially, followed by a public
Zenodo deposit when the paper is ready. The checksum-indexed selected archive
retains all30 final reported policy networks, their original actor/model/seed
requirements and learning-curve records. Full exploration checkpoints remain
intact. Original learned core95d36e1 is an isolated execution runtime, never
relabelled as the current dependency pin. Ordinary build CPU budget is900s,
with30 bounded actor-inference conditions, four contract-evaluation steps, no
training or full historical rollout. Source/tag/descriptor mappings and actual
release upload are completed at cutover; no release tag or DOI is fabricated.
Slurm estimates explicitly request CPU/memory/time and retain strict identity
checks. See REDbyRBD docs/reproducibility for executable requirements.
