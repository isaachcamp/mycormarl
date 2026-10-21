# ADR-0014: Organize repositories around the core and scientific studies

**Status:** Accepted; initial repository extraction and cutover completed.
Implementation status clarified 21 October 2026; the original decision and
source identities are retained.

## Context

The project is becoming difficult to navigate as biological model code,
algorithms, experimental protocols, analysis, and animations accumulate in one
repository. The primary reason for splitting is clear ownership and easier
independent development, alongside reproducibility. Physical disk usage is a
separate concern.

## Decision

Keep the reusable biological environment and algorithm implementations in the
core `mycormarl` repository. Extract animation scripts into a separate
repository. Organize downstream research repositories around coherent
scientific questions or publications, allowing several related experiments
within one study rather than assigning one repository to each current script
or screen.

Initially keep fungal foraging, resource-pressure exploration, and simple MARL
studies in one combined research repository. Fungal foraging contributes to the
same scientific story, potentially as an appendix. Resource-pressure screens
document parameter effects and associated uncertainty within the wider model
exploration. This gives three repositories initially: core, combined research,
and animations. Future independent scientific stories can justify further
research repositories.

Name the combined research repository `REDbyRBD`, expanded as "Resource exchange
dynamics by resource budgetary demands". This name identifies the shared
scientific story rather than tying the repository to the core package's name
or a single experimental method. The animation repository's selected name
is `mycormarl-animations`.

Use the following opening in the research README:

> **REDbyRBD — Resource exchange dynamics by resource budgetary demands**
>
> Reproducible studies of how plant and fungal resource budgets shape resource
> exchange.

Retain the existing repository's history without a history-rewriting
extraction. Start the research and animation repositories with extraction
commits that identify the originating source revision. The user intends to
squash-merge the current development branch into the original repository to
simplify its history. Capture report-producing revisions with tags before
removing branch references, and distinguish the original report source from
the later squash and extraction commits in provenance. Squash merging does
not authorize rewriting existing main history or replacing old result
identities. Future working branches should follow the applicable naming
convention.

Core retains reusable policy persistence, checkpoint evaluation, random-stream
handling, and generic rollout facilities. Study repositories own named
protocols, treatment matrices, stopping criteria, acceptance checks, and
scientific interpretation. Extract the experimental portions of `study.py`
rather than retaining that mixed module wholesale as a core framework.

The animation repository owns scenes, animation-specific data preparation,
rendering configuration, and visual validators. Reusable biological
calculations remain in core, and scientific figure preparation belongs with
the research study. A study may consume a versioned animation output.

Each study pins an immutable core commit initially. Record the study source
revision, core source revision, and dependency environment separately.
Dependency upgrades are deliberate; existing archived results retain their
original provenance.

Assess each core update for scientific impact. Regenerate affected evidence as
a new result version while preserving earlier results with their original
model identity. Historical comparisons may be retained when their model
versions are explicit; old results must not be relabelled as outputs of the
updated core. Changes that do not affect numerical results need not trigger
recomputation.

Use showyourwork! for downstream figure/article workflows. The ordinary build
regenerates analysis and figures from archived results. Expensive simulation
or training recomputation is supplied through documented Slurm submission
scripts that declare and request the required compute resources; an automatic
training rerun in the ordinary article build is not required.

## Consequences

Study-specific orchestration and standalone scientific scripts belong
downstream; reusable facilities remain in core. Initial extraction and
independent-package verification are recorded in the
[cutover provenance](../reproducibility/repository-cutover.json).
Future study/API extensions still follow this ownership boundary rather than
reintroducing named scientific protocols into core.

ADR-0015 records the accepted dependency-pinning, result-retention, and
cross-repository documentation conventions, fixed report tags, and the
distinction between exploration records and publication evidence bundles.
Routine bounded policy evaluation was permitted in the initial report build;
the historical budget and archive requirements are recorded in ADR-0015.
Current downstream build behaviour is governed by its own versioned protocol.

## Sources

- Repository-boundary discussion, 13 October 2026: the user accepted ownership
  and navigability as the primary goal, scientific-question/publication
  boundaries, and archived-result figure builds, with Slurm scripts as the
  provision for expensive recomputation.
- The same discussion subsequently settled one combined research repository,
  reusable core facilities, exact core pinning with separate provenance, and
  ownership of complete animation workflows.
- [showyourwork! introduction](https://show-your.work/en/latest/intro/).
- [showyourwork! Zenodo integration](https://show-your.work/en/latest/zenodo/).
