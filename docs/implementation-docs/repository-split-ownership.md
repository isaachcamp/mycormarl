# Repository split ownership map

This is the proposed extraction inventory for
[the implementation plan](repository-split-plan.md). All source paths below
are relative to the current repository. Validate individual destinations when
compiling the relevant phase; this map does not authorize code moves.

## Core mycormarl

Retain package identity `mycormarl` and its existing layout initially.

| Current path/group | Destination and treatment |
|---|---|
| `environments/`, `plant/`, `fungus/` under `mycormarl/mycormarl/` | Core biological environment and geometry/traits; retain. |
| `actions.py`, `transition.py`, `observations.py`, `params.py`, `state.py`, `growth.py`, `maintenance.py` | Core contracts; retain. |
| `soil/phosphate_units.py`, `phosphate_grid.py`, `phosphate_diffusion.py`, `phosphate_uptake.py`, `soil.py` | Core numerical biology; retain without duplicated kernels downstream. |
| `algos/ppo.py`, `policy_artifacts.py`, `random_streams.py` | Core algorithm/persistence/RNG contracts; retain and declare actual dependencies. |
| `checkpoint_evaluation.py` | Keep generic policy/checkpoint execution and summary contracts; separate named-study interpretation if found. Expose only needed supported seams. |
| `static_controls.py` | Split: generic static rollout facility in core; scientific manifests/gates/condition selection in REDbyRBD. |
| `trade_only.py` | Split: reusable action conversion/constraint mechanics in core; named baseline and study settings in REDbyRBD. Preserve the current fixed-allocation policy context in persisted artifacts. |
| `soil/phosphate_qualification.py`, `soil/deep_soil_qualification.py` | Core numerical-verification machinery and invariant evidence; retain. |
| `plant_growth_qualification.py`, `domain_qualification.py` | Split by purpose: reusable numerical/model checks stay core; empirical carrot reference and named study domain acceptance belong to research. Avoid duplicate checks. |
| `phosphate_examples.py`, `scripts/phosphate_examples.py`, `scripts/phosphate_qualification.py`, `scripts/train_ppo.py`, `main.py` | Core examples, numerical qualification, and minimal training launcher; retain/relink to public APIs. |

The package paths in the table are under `mycormarl/mycormarl/` unless
explicitly written otherwise. Module moves must preserve public environment
and saved-policy semantics. Removing scientific orchestration is not a
mandate to redesign every generic utility.

## REDbyRBD

Use a local research helper package if needed, with an import name distinct
from `mycormarl`. Publication scripts live under `src/scripts/`; exploration
entry points have their own documented area and share appropriate helpers.

| Current path/group | Treatment |
|---|---|
| `study.py`, `trade_only_study.py` | Move named protocols/orchestration/results/provenance to research after extracting required core facilities. |
| `resource_pressure_screen.py`, `resource_pressure_analysis.py` | Research parameter exploration, sampling, and interpretation. |
| `soil/absorber_diagnostic.py`, `soil/depletion_gradient_diagnostic.py`, `soil/hyphal_density_sweep.py` | Research foraging/uptake diagnostic experiments, importing core kernels. |
| `scripts/absorber_construction_carbon_diagnostic.py`, `depletion_gradient_diagnostic.py`, `hyphal_density_sweep.py`, `plot_hyphal_density_sweep.py` | Research diagnostic generation/plots. |
| `scripts/run_resource_pressure_canonical_screen.py`, `run_resource_pressure_factorial_plant_boundary.py`, corresponding `analyse_resource_pressure_*.py` | Research static screens and analysis. |
| `scripts/run_trade_only_ippo.py`, `analyse_trade_only.py`, `run_historical_grid_fixed_trade_control.py`, `refresh_historical_grid_evaluations.py` | Research training/control/evaluation entry points. |
| `scripts/plot_historical_grid_*.py`, `plot_episode_time_allocation_p_sweep.py`, `plot_low_p_gamma_analysis.py`, `plot_proposed_fitness_contribution.py`, `plot_static_controls.py`, `plot_static_control_trajectory.py`, `plot_stylized_trajectory.py`, `plot_trade_feedback_diagnostic.py` | Research figure/exploration scripts. Separate data generation from rendering where needed. |
| `scripts/plot_uptake_far_field_decay.py` | Research figure. Replace its video-helper dependency with its own preparation using shared core kernels. |
| `scripts/plant_growth_qualification.py` | Research empirical/reference qualification launcher; reusable checks remain core. |
| `docs/studies/`, resource-pressure/study manifests in `docs/qualification/` | Research scientific configuration. Preserve historical/current distinctions. |
| Study specs and runner/resource-screen guides in `docs/implementation-docs/` | Research requirements and execution docs, relinked to moved code. |
| Existing Overleaf project source | Import into manuscript area with styles, bibliography, and asset mapping; preserve original snapshot before live sync. |

Retain obsolete/historical experiment files with clear status or preserve them
in the original tagged history; no unreviewed removal of scientifically useful
evidence. The phase inventory must account for every file, including scripts
not used by the current report.

## mycormarl-animations

| Current script | Treatment |
|---|---|
| `manim_soil_root_mycelium.py`, `manim_colony_uptake.py`, `manim_phosphate_uptake.py` | Animation scenes. |
| `geometry_growth_video.py`, `hyphal_branching.py` | Animation/visual geometry preparation and rendering. |
| `colony_uptake_video_data.py`, `phosphate_uptake_video_data.py` | Animation-specific preparation. Retain biological calculations through core imports. |
| `validate_colony_uptake_video.py`, `validate_phosphate_uptake_video.py` | Visual/data validators. |
| Scene-related configuration/docs | Move with the scene that uses them. |

`scripts/__init__.py` is infrastructure: create needed downstream packages
rather than copying it as a scientific module. `__pycache__`, installed
egg-info, virtual environments, and generated media are not extraction inputs.

## Tests

- Core: tests for units, phosphate state/diffusion/uptake, geometry, reset pools,
  environment, Rate actions, transitions, observations, policy intervals, PPO,
  policy artifacts, RNG, and generic checkpoint evaluation/rollout contracts.
- REDbyRBD: `test_study_runner.py`, `test_resource_pressure_screen.py`,
  `test_absorber_diagnostic.py`, `test_depletion_gradient_diagnostic.py`,
  `test_hyphal_density_sweep.py`, plus named protocol/baseline/empirical
  qualification assertions from mixed test modules.
- Animations: `test_colony_animation.py`, `test_colony_uptake_video.py`,
  `test_phosphate_uptake_video.py`, `test_geometry_growth_video.py`.
- Split `test_trade_only.py`, `test_static_controls.py`,
  `test_plant_growth_qualification.py`, `test_review_repairs.py`, and any CLI
  assertions by implementation owner; inspect tests rather than assigning only
  by filename. Keep core numerical qualification tests with core.
- Add installed-package integration checks and publication-bundle rebuild
  checks across repositories; avoid tests that merely assert files moved.

## Documentation and ADR ownership

Core keeps model/growth/phosphate/module-map/algorithm interface docs, the
parameter register and references underpinning default parameters, numerical
qualification, and historical source provenance. REDbyRBD keeps scientific
study specs, analysis/uncertainty interpretation, empirical study checks,
publication workflows, manuscript, and exploratory study references.
Inspect each `docs/research/` document: a reference supporting a default model
parameter belongs with core even when a paper cites it. Link instead of
maintaining contradictory independent parameter registers.

| Existing ADR IDs | Ownership approach |
|---|---|
| 0001–0004 | Core model/algorithm/action/geometry contracts. |
| 0005–0007, 0013 | Research diagnostic decisions. Preserve 0005's superseded status and links; any core-kernel constraints remain referenced by core. |
| 0008 | Research scientific endpoint interpretation; core PPO objective docs link to the distinction. |
| 0009–0012 | Core consumer-mode/reset/guard/action contracts, with research-side protocol references as needed. |
| 0014–0015 | Original cross-repository decisions remain traceable in core; downstream ADR indexes reference originating IDs/revisions and record local consequences. |

Retain original IDs/decision dates/status in migrated records and update code
links. Each repository has a local ADR index defining owners and cross-repo
references. Split glossary terms by domain scope, keeping shared biological
definitions authoritative in core. Research adds its study/exploration/
publication terms. All repos use the existing single-context agent convention.

## Data and migration records

Detailed outputs/media stay outside new source checkouts or at their existing
locations referenced by the run catalog. Document original report source,
core/research commits, lock hashes, Overleaf snapshot checksum/revision,
figure/run/artifact mapping, and the squash/extraction commit relationship.
Publication exports contain selected evidence with checksum manifests;
exporting does not prune source runs. Git extraction commits contain source
provenance, not large generated data.

Before final removal from core, audit tracked files against this map and
record an explicit owner or retained historical status for every item.
