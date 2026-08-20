# Phase 1: uniform-P association-response pilot specification

**Status:** Reviewed proposal; implementation has not started.

## Purpose

Test whether the learned plant benefit from access to an established AM
association increases as initial phosphorus availability declines. This pilot
is the range-finding stage of Phase 1. It qualifies the environmental range,
training procedure, and provisional response shape for a subsequent denser
Phase 1 response map; it does not establish a universal P threshold or
globally optimal strategy.

Phase 1 addresses the response-tendency goal:

> Demonstrate that learned plant benefit from access to an established AM
> association increases with increasing P limitation.

A separate Phase 2 will address the threshold goal:

> Determine whether there exists a `P_thresh` above which the complete
> `plant-only` mode is favoured over access to an established AM association.

This wording deliberately refers to the complete mode contrast rather than
cessation of trade within `mixed`.

This study follows:

- [ADR-0008](../adr/0008-distinguish-optimized-reproductive-fitness-from-observed-biomass-response.md),
  which defines endpoints and evaluation language; and
- [ADR-0009](../adr/0009-represent-am-engagement-using-complete-consumer-modes.md),
  which defines the `mixed` versus `plant-only` contrast.

Carbon-scale and high-P interpretation are reviewed in
[`carbon-limitation-growth-rates-and-high-phosphorus-response.md`](../research/carbon-limitation-growth-rates-and-high-phosphorus-response.md).
The species-matched growth benchmark and parameter audit are in
[`carrot-growth-biomass-cap-and-carbon-fixation.md`](../research/carrot-growth-biomass-cap-and-carbon-fixation.md).

## Scientific contrast

At each initial solution-P level `P`, train policies independently in:

- `mixed`: co-train plant and fungal IPPO policies; and
- `plant-only`: train the plant without a fungal organism.

Define primary association advantage as:

```text
Delta_AM(P) = mean_fitness_mixed(P) - mean_fitness_plant_only(P)
```

Report biomass mycorrhizal growth response as:

```text
MGR(P) = 100 * (mean_biomass_mixed(P) / mean_biomass_plant_only(P) - 1)
```

Do not average seedwise biomass ratios. Record MGR as missing, without an
epsilon denominator, if mean `plant-only` biomass is zero.

## Fixed pilot design

- Initial solution-P grid: `0.1`, `0.3`, `1.0`, and `3.0 µM`.
- Uniformity: homogeneous concentration within the configured P-bearing region
  at reset; normal finite-inventory diffusion, depletion, and uptake afterward.
- Horizon: 120 days, or 4,800 transitions at `dt = 0.025 day`.
- Day 120 is administrative truncation, not biological death.
- Continue after one partner dies while the other remains operational; stop
  early only if both organisms are dead.
- Training replication: five independent master seed IDs per P level and mode,
  giving 40 initial runs.
- Train every policy from scratch at its fixed P level; do not train one
  P-generalist policy in this phase.

The four P levels and 120-day horizon are range-finding assumptions. Reassess
them before the denser Phase 1 response map if they fail to span limitation
through relative sufficiency or omit the relevant cost, benefit, depletion,
reproduction, or mortality timescale. Any revision must be justified
independently of whether it strengthens the hypothesized response.

## Preflight qualification

### Plant growth and carbon scale

Before interpreting learned biomass or any P-response contrast, qualify the
default plant carbon budget and biomass limit against age-matched observations
for the represented plant scenario. This is a prerequisite to the 40-run
pilot, not a post-hoc explanation of its results.

Use a `plant-only`, high-P vegetative-growth control that reserves no resources
for reproduction or fungal transfer, together with an analytical carbon-only
upper bound. Report:

- early-, middle-, and late-window relative growth rate;
- total living dry biomass through time and at the 120-day endpoint;
- gross fixed C, standing maintenance, structural-growth C, and free-pool
  change in common carbon units;
- the start-up contribution from the initial free C and P pools;
- the effective whole-plant fixation term `kleaf * amass`; and
- distance from the configured biomass cap through time.

Compare these quantities with observations matched as closely as possible for
species, age, photoperiod, irradiance, temperature, cultivation regime, and
dry-mass definition. Do not diagnose carbon fixation from a learned policy's
biomass alone: the PPO objective can allocate resources to reproduction and
therefore need not maximize vegetative growth.

For the initial qualification, retain `amass=0.05` as the 450-PAR reference-
day value and evaluate fixed `kleaf=0.30`, `0.45`, and `0.60`. Compare biomass
and RGR over windows aligned explicitly with the `Forto` 40, 60, 80, 100, and
120 DAS observations. Treat `kleaf=0.60` as an age-averaged sensitivity, not
an accepted constant trait. If the intended light regime differs, test
`amass=0.06` and `0.07` as named higher-irradiance sensitivities rather than as
generic corrections for low growth.

Use `biomass_cap=50 g DM` as the provisional non-binding numerical guard and
`biomass_observation_reference=50 g DM` as the independent actor-input scale.
The guard was selected above the `25--35 g DM` biological trajectory reference:
it is about `1.43` times the unobserved fitted `Forto` asymptote and about twice
the largest independent cultivar-mean endpoint recovered in the source review.
Reject any qualification trajectory that contacts the guard; do not interpret
contact as a physiological plateau or use it to establish a high-P threshold.

If the control is materially inconsistent with the empirical envelope, revise
the physiological parameters or model form and repeat this qualification
before running the P grid. Parameter choices must be justified independently
of whether they create the expected AM response. Treat the biomass cap as a
scenario-specific biological parameter only if it is independently supported;
otherwise retain it as an explicitly non-binding numerical guard and do not
interpret contact with it as a growth plateau.

### Static-policy controls

Run deterministic valid static plant and fungal policies at every P level
before PPO. Check for pathological depletion, premature death, accounting
failures, and obviously uninformative treatment ranges.

### Scientific soil domain

Do not use the `1 × 1 cm` development domain as scientific evidence and do not
assume the `50 × 100 cm` production domain is computationally feasible for 40
training runs. Benchmark candidate domains and choose the smallest one that:

- avoids material root and fungal contact with outer and lower boundaries in
  representative 120-day trajectories;
- records a defensible total initial P inventory;
- preserves depletion behavior and P-response ordering when enlarged; and
- has recorded training throughput and memory use.

### Training stopping rule

Before inspecting treatment-response signs, predeclare identical:

- minimum and maximum training-transition budgets;
- checkpoint intervals;
- deterministic evaluation windows; and
- a scale-aware fitness plateau tolerance (absolute floor plus relative
  window scale) and a policy-action summary tolerance.

Choose numerical tolerances from blinded development learning curves without
calculating `Delta_AM`. A run cannot stop before the minimum budget. Retain and
label a run that reaches the maximum without satisfying the rule as
unconverged; do not silently accept or selectively extend it. If optimizer or
training settings change, rerun the complete affected comparison block.

## Checkpoints and evaluation

Save regular, resumable checkpoints containing policy parameters, optimizer
state, training transitions, seed, configuration, and schema/version metadata.

Evaluate every checkpoint primarily through one deterministic 120-day
latent-location trajectory. As a secondary diagnostic, run sampled-policy
trajectories to characterize residual policy-distribution sensitivity; keep
this within-policy variation separate from between-training-seed variation.

Track at least:

- cumulative plant and fungal reproductive fitness;
- final raw and living biomass;
- cumulative gross growth;
- interval and whole-episode plant relative growth rate;
- C- versus P-limited realised growth, calculated from the two structural-
  biomass equivalents offered to growth;
- direct plant P uptake relative to maintenance and to the P required to match
  the available carbon allocation;
- requested and realized actions;
- proposed and realized bilateral transfers;
- free carbon and phosphorus pools;
- biological death and administrative truncation; and
- total soil-P inventory and available P-loss/export counters.

## Seed design

Use the same five master seed IDs in both modes as a fixed paired-block design,
and use each five-seed block consistently at every P level. This establishes
the operational precedent that comparisons between modes or environmental
treatments use aligned replicate identifiers, even when a relatively
deterministic model provides little variance reduction. Pairing does not
change either treatment mean or MGR.

Report marginal mode distributions, seed-matched scatter and differences, and
descriptive cross-mode covariance. Do not make confirmatory paired inference
from five seed pairs. Later phases should retain paired blocking by default,
introduce named random streams so that the intended coupling is explicit, and
report whether pairing actually induces positive cross-treatment covariance.
Pairing evaluation environments or held-out heterogeneous P fields is a
separate and stronger requirement: every evaluated policy must encounter the
same comparison set. See
[`paired-random-seeds-for-policy-comparisons.md`](../research/paired-random-seeds-for-policy-comparisons.md).

## Pilot interpretation

Plot `Delta_AM(P)` over the complete predeclared grid on a logarithmic P axis.
Pilot support requires low-P association advantage to exceed high-P advantage
and an ordered overall tendency toward greater benefit under P limitation. Do
not require every adjacent noisy point estimate to be strictly monotonic.

- A sign change brackets a region for later concentration refinement; it is
  not a precise threshold.
- A flat, reversed, or seed-dominated response does not support the proposed
  mechanism.
- Report uncertainty and every seed-level outcome.
- Report biomass MGR beside fitness and allow the endpoints to disagree.
- Do not select or exclude P levels after observing which ones strengthen the
  trend.
- Emphasize the response curve rather than one significance test.

## Denser Phase 1 response map

Completing the 40-run pilot does not complete Phase 1. After qualifying the P
range, domain, training budget, and observed between-seed variance, predeclare
and execute a denser P grid over the qualified range. Its design must:

- preserve paired seed blocking and the same mode contrast;
- include enough additional P levels to resolve the response shape rather than
  only the endpoints;
- retain pilot levels where scientifically and computationally valid so that
  the stages remain comparable;
- set replication from the pilot variance and desired precision; and
- choose grid limits and spacing using qualification results without selecting
  concentrations merely because they strengthen the expected trend or sign
  crossing.

The exact dense-grid levels and replication are qualification-dependent Phase
1 parameters and must be fixed before inspecting dense-grid treatment
contrasts.

## Transition from Phase 1 to Phase 2

An ordered decline in `Delta_AM(P)` supports the Phase 1 response-tendency
hypothesis but is not sufficient to begin threshold estimation. Phase 2 is
warranted only if the denser Phase 1 response map identifies a credible sign
bracket: at least one qualified lower-P treatment has positive association
advantage and at least one qualified higher-P treatment has negative
association advantage, with the following checks satisfied:

- the apparent signs are not created by unconverged runs, biological failure,
  domain contact, or one anomalous seed pair;
- seed-level outcomes and uncertainty are compatible with a genuine crossing,
  while acknowledging that five pairs are not confirmatory evidence;
- plant growth-rate trajectories are compared with a relevant empirical
  envelope and do not reveal a gross scale mismatch;
- the `plant-only` response becomes weakly responsive to further P, providing
  evidence that the upper side of the bracket is relatively P-sufficient;
- C- and P-limitation diagnostics show that increasing P actually moves the
  plant away from P limitation; and
- the crossing is not an artefact of the unsupported plant biomass cap or of
  exhaustion of the finite domain inventory.

If the denser grid shows only a trend, report the completed Phase 1 response
without calling it a threshold. If it produces a credible bracket, predeclare
a separate Phase 2 threshold-estimation design with locally refined P levels,
appropriate replication, and a threshold uncertainty method before inspecting
the Phase 2 results.

## Interpretation limits

- `mixed` represents access to an established association, not simulated AM
  formation.
- A living PPO actor cannot request exact zero trade through the current
  sigmoid transform. Near-zero trade is not cessation or zero colonisation.
- Mixed outcomes include adaptation by both independent learners and are not a
  response to a fixed fungal strategy.
- Results are learned IPPO outcomes under the qualified domain and protocol,
  not global optima.
- Five seeds are sufficient for range finding and failure detection, not final
  inference. Set main-study replication from observed variance and required
  precision around any sign crossing.

## Completion criteria

### Range-finding pilot

- The preflight controls and domain qualification pass.
- All 40 requested runs are complete or explicitly labeled unconverged.
- Checkpoint plateau evidence and policy-action stability are reported.
- Deterministic evaluation outputs are reproducible from saved checkpoints.
- All primary and secondary endpoints, seed-level outcomes, and limitations
  are present in a machine-readable result bundle and concise report.
- The result either supports the predicted low-P tendency, fails to support it,
  or identifies an unresolved training/range failure without post-hoc treatment
  changes.

### Phase 1

- Pilot qualification has been converted into a predeclared dense-grid design.
- The denser response map is complete, including all paired seed-level
  outcomes and the same physiological and numerical diagnostics.
- The Phase 1 result establishes the supported response shape and either
  identifies a credible sign bracket for Phase 2 or reports that no credible
  bracket was found over the qualified range.
