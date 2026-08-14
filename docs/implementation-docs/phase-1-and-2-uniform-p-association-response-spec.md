# Phase 1 and Phase 2 uniform-P association-response study

**Status:** Ready for implementation

**Tracker:** [GitHub issue #31](https://github.com/isaachcamp/mycormarl/issues/31)

## Problem Statement

The project needs to determine how the learned plant benefit from access to an
established arbuscular-mycorrhizal association changes along a uniform initial
soil-phosphorus gradient, and whether there is credible evidence for an
initial solution-P concentration above which the complete `plant-only` mode is
favoured over the complete `mixed` mode.

The current model can compare these modes, but it does not yet provide a
study-level workflow that qualifies plant growth scale, soil domain, policy
training, paired replication, checkpoint evaluation, response mapping, and
threshold bracketing as one reproducible experiment. Without that workflow, a
sign change could be manufactured or misidentified through an unsupported
biomass ceiling, implausible growth, P-inventory exhaustion, domain contact,
unconverged policy learning, stochastic seed imbalance, or post-hoc selection
of P concentrations.

The scientific goals must remain distinct:

- **Phase 1:** determine whether learned plant association advantage tends to
  increase with increasing P limitation, first through a range-finding pilot
  and then through a denser response map.
- **Phase 2:** only after Phase 1 identifies a credible sign bracket, determine
  whether there exists a `P_thresh` above which the complete `plant-only` mode
  is favoured over access to an established AM association.

The comparison is not an in-episode colonisation or trade gate. `mixed`
contains an established living fungal partner, reserves, and uptake geometry;
`plant-only` contains no fungus. Finite sigmoid trade actions cannot be exactly
zero, so near-zero transfer in `mixed` cannot be described as zero
colonisation or complete cessation of trade.

## Solution

Provide one versioned study-runner interface that executes and analyses the
uniform-P association-response programme in declared stages. A study design
manifest selects growth qualification, static controls, domain qualification,
Phase 1 range finding, the denser Phase 1 map, or Phase 2 local refinement. The
runner emits immutable, machine-readable result bundles and concise reports
containing the complete scientific configuration, policy provenance,
checkpoints, paired seed identifiers, evaluation outputs, acceptance checks,
and interpretation status.

Before PPO response runs, qualify the represented carrot growth scale using a
high-P `plant-only` vegetative control and an analytical carbon-only bound.
Use `25--35 g DM` as the favourable trajectory reference, not as a hard cap.
Use `biomass_cap = 50 g DM` as a provisional numerical guard and
`biomass_observation_reference = 50 g DM` as an independent policy-input
scale. Any trajectory contacting the guard fails qualification.

Run the Phase 1 pilot at `0.1`, `0.3`, `1.0`, and `3.0 micromolar` initial
solution P for 120 days (`4,800` transitions at `dt = 0.025 day`) in separately
trained `mixed` and `plant-only` modes. Use the same five master seed IDs in
both modes and at every P level, giving 40 pilot training runs. Evaluate saved
policies primarily with deterministic latent-location trajectories and report
sampled-policy trajectories only as a secondary sensitivity diagnostic.

After qualifying the range, domain, training budgets, and variance, freeze a
denser Phase 1 design before inspecting its treatment contrasts. Phase 1 ends
with a response map and an explicit decision that a credible Phase 2 sign
bracket either exists or does not exist. Phase 2 is permitted only when at
least one qualified lower-P treatment has positive association advantage and
at least one qualified higher-P treatment has negative association advantage,
with physiological, numerical, and learning artefacts excluded. A separate
Phase 2 manifest must freeze its local P grid, replication, and threshold-
uncertainty method before Phase 2 outcomes are inspected.

The primary endpoint is the difference in mode-level mean cumulative plant
reproductive fitness:

```text
Delta_AM(P) = mean_fitness_mixed(P) - mean_fitness_plant_only(P)
```

Final living biomass and cumulative gross growth are secondary endpoints. The
greenhouse-comparable mycorrhizal growth response is:

```text
MGR(P) = 100 * (mean_biomass_mixed(P) / mean_biomass_plant_only(P) - 1)
```

MGR is missing if mean `plant-only` biomass is zero. Seedwise biomass ratios
must not be averaged.

## User Stories

1. As a plant–fungus modeller, I want Phase 1 and Phase 2 represented as separate study stages, so that a response tendency is not conflated with threshold estimation.
2. As a plant–fungus modeller, I want one versioned study interface, so that every stage uses consistent configuration, provenance, and output contracts.
3. As a researcher, I want a preserved design manifest for every execution, so that the study can be reproduced without reconstructing command-line choices.
4. As a researcher, I want uniform P to mean a uniform reset condition followed by normal finite-inventory dynamics, so that the treatment is not mistaken for a replenished reservoir.
5. As a researcher, I want `mixed` and `plant-only` trained separately, so that association advantage represents the complete established-association contrast.
6. As a researcher, I want fungal biomass, reserves, geometry, uptake, and both transfer directions absent in `plant-only`, so that the off treatment is not a fungal-competition treatment.
7. As a researcher, I want near-zero trade reported quantitatively rather than called zero colonisation, so that claims match the continuous sigmoid action interface.
8. As a researcher, I want a high-P vegetative plant control before policy training, so that implausible carbon and biomass scales cannot be explained post hoc by PPO allocation.
9. As a researcher, I want the analytical carbon-only ceiling reported beside simulated growth, so that resource and policy effects can be separated from the source-strength limit.
10. As a researcher, I want growth compared at 40, 60, 80, 100, and 120 days after sowing, so that endpoint agreement cannot hide an implausible trajectory.
11. As a researcher, I want windowed RGR reported over the same age intervals as the empirical benchmark, so that early and late growth limitations remain visible.
12. As a researcher, I want gross fixation, maintenance, structural-growth C, and free-pool changes reported in common units, so that the carbon budget can be audited.
13. As a researcher, I want the contribution from initial free C and P pools quantified, so that start-up biomass is not mistaken for sustained growth.
14. As a researcher, I want `kleaf = 0.30`, `0.45`, and `0.60` evaluated during initial qualification, so that the fixed leaf-fraction mismatch is exposed before the P study.
15. As a researcher, I want higher `amass` values used only as named irradiance sensitivities, so that irradiance and ontogenetic allocation are not conflated.
16. As a researcher, I want `25--35 g DM` treated as a biological trajectory reference, so that a fitted asymptote is not misrepresented as a carrying capacity.
17. As a researcher, I want a `50 g DM` numerical biomass guard, so that the model has finite protection with headroom above observed carrot trajectories.
18. As a researcher, I want cap contact to fail qualification, so that a numerical guard cannot manufacture a high-P plateau or sign crossing.
19. As a policy-learning researcher, I want the biomass observation reference independent of the growth guard, so that cap changes do not silently rescale actor inputs.
20. As a researcher, I want deterministic static-policy controls at every pilot P level, so that pathological depletion, death, and accounting failures are caught before PPO.
21. As a researcher, I want candidate soil domains benchmarked for boundary contact, inventory, response ordering, runtime, and memory, so that development geometry is not used as scientific evidence.
22. As a researcher, I want one scientific domain frozen before response signs are inspected, so that geometry is not selected to support the hypothesis.
23. As a policy-learning researcher, I want identical minimum and maximum training budgets across a comparison block, so that one treatment does not receive selective optimization effort.
24. As a policy-learning researcher, I want checkpoint intervals and plateau tolerances declared before treatment contrasts are examined, so that stopping decisions remain blind to `Delta_AM`.
25. As a policy-learning researcher, I want maximum-budget runs retained and labelled unconverged, so that failed learning is not silently treated as biological evidence.
26. As a policy-learning researcher, I want resumable checkpoints containing policy and optimizer state, so that long training runs can continue without losing provenance.
27. As a policy-learning researcher, I want checkpoint metadata to include seed, mode, P level, transitions, configuration, and interface versions, so that incompatible policies cannot enter an analysis block.
28. As a researcher, I want deterministic latent-location evaluation as the primary policy evaluation, so that learned outcomes are reproducible without conflating Gaussian sampling with biological bet hedging.
29. As a researcher, I want sampled-policy evaluation kept as a separate diagnostic, so that residual distribution sensitivity can be measured without changing the primary estimand.
30. As a researcher, I want the same five master seed IDs used across modes and P levels in the pilot, so that paired-block operational precedent is established.
31. As a researcher, I want marginal distributions, paired differences, paired scatter, and cross-mode covariance reported, so that the value of seed pairing remains visible.
32. As a researcher, I want no confirmatory paired inference from five pilot pairs, so that range-finding replication is not overstated.
33. As a future study maintainer, I want named independent random streams, so that initialization, action sampling, environment variation, and minibatch order can be paired or separated deliberately.
34. As a researcher, I want all pilot concentrations retained in the report, so that no P treatment is removed after its response is known.
35. As a researcher, I want `Delta_AM(P)` displayed across the complete grid on a logarithmic P axis, so that response tendency and possible sign changes are interpretable.
36. As a researcher, I want an ordered overall tendency rather than strict adjacent monotonicity as the Phase 1 pilot criterion, so that noisy learned outcomes are not judged by an unrealistic rule.
37. As a researcher, I want biomass MGR reported beside reproductive-fitness advantage, so that optimized fitness and greenhouse-style growth response can disagree transparently.
38. As a researcher, I want every seed-level outcome and uncertainty summary retained, so that aggregate curves cannot conceal training failures or anomalous replicates.
39. As a researcher, I want a machine-readable pilot result bundle and concise human report, so that later dense-grid design can use recorded evidence rather than manual reconstruction.
40. As a researcher, I want the denser Phase 1 grid and replication frozen after qualification but before its treatment contrasts are inspected, so that refinement remains prospective.
41. As a researcher, I want pilot P levels retained in the dense map where valid, so that range-finding and main response stages remain comparable.
42. As a researcher, I want dense-grid replication informed by pilot variance and desired precision, so that computation is allocated to the intended scientific uncertainty.
43. As a researcher, I want Phase 1 to end even when no sign bracket exists, so that a supported response tendency can be reported without forcing threshold estimation.
44. As a researcher, I want a credible-bracket gate before Phase 2, so that threshold refinement cannot begin from a merely suggestive curve.
45. As a researcher, I want bracket signs rejected when created by unconverged runs, death, domain contact, cap contact, or inventory exhaustion, so that numerical failure is not interpreted biologically.
46. As a researcher, I want the upper bracket side to show weak additional plant-only response to P, so that it has evidence of relative P sufficiency.
47. As a researcher, I want C- versus P-limitation diagnostics to move away from P limitation across the bracket, so that external P concentration has the assumed mechanistic effect.
48. As a researcher, I want Phase 2 concentrations, replication, and uncertainty method predeclared in a separate manifest, so that threshold analysis is not tuned after observing local outcomes.
49. As a researcher, I want Phase 2 to refine only inside a qualified bracket, so that it estimates a local crossing rather than extrapolating a universal threshold.
50. As a researcher, I want Phase 2 to report the threshold estimate together with its supported interval or an explicit non-estimable result, so that uncertainty and failure remain first-class outcomes.
51. As a researcher, I want the analysis to refuse a threshold claim when qualified signs do not persist, so that Phase 2 can falsify the initial bracket.
52. As a modeller, I want every result described as a learned IPPO outcome, so that the study does not claim a global optimum or game-theoretic equilibrium.
53. As a modeller, I want mixed-mode adaptation by both learners documented, so that association advantage is not mistaken for a response to a fixed fungal strategy.
54. As a researcher, I want complete soil-P inventory and P-loss accounting in every evaluation, so that resource conservation failures invalidate the result.
55. As a researcher, I want biological death separated from administrative day-120 truncation, so that evaluation and bootstrap semantics remain correct.
56. As a repository maintainer, I want artifact schemas and policy interfaces versioned, so that stale runs fail clearly instead of entering a current result bundle.
57. As a repository maintainer, I want repeated execution with the same manifest and deterministic fixture to reproduce the same analysis artifact, so that orchestration changes are regression-testable.
58. As a repository maintainer, I want incompatible or incomplete artifacts rejected before aggregation, so that summaries contain only comparable runs.
59. As a repository maintainer, I want partially completed study blocks resumable without overwriting valid outputs, so that expensive scientific runs remain recoverable.
60. As a future researcher, I want all qualification failures and deferred choices recorded, so that later heterogeneous-P phases inherit an auditable experimental precedent.

## Implementation Decisions

- Expose one study-level runner and versioned design-manifest contract. Stages
  share artifact, provenance, validation, and reporting semantics rather than
  becoming unrelated scripts.
- Treat the study runner and its emitted result bundle as the primary public
  testing seam. Small fixtures may shorten horizons, grids, and training
  budgets, but must exercise the same configuration and analysis path as a
  scientific run.
- Preserve environment/PPO separation. The study layer configures and invokes
  the existing environment, independent PPO trainer, policy artifact contract,
  and actor mapping; scientific orchestration does not enter core environment
  dynamics.
- Represent study stages explicitly: growth qualification, static-policy
  controls, domain qualification, Phase 1 pilot, Phase 1 dense map, Phase 2
  design freeze, and Phase 2 refinement.
- Require every manifest to declare its stage, schema version, environment and
  species parameters, P grid, modes, seed IDs, horizon, numerical timestep,
  policy-training configuration, evaluation configuration, output location,
  and parent qualification-artifact identifiers where applicable.
- Use `mixed` and `plant-only` as complete consumer modes. Do not create an
  in-episode AM-engagement Bernoulli action and do not retain fungal state in
  the off mode.
- Train from scratch at each P level and mode. Phase 1 does not train a
  P-generalist policy.
- Use an initially uniform solution-P concentration within the declared
  P-bearing region. Normal finite-inventory diffusion, uptake, and depletion
  proceed after reset.
- Fix the Phase 1 pilot grid at `0.1`, `0.3`, `1.0`, and `3.0 micromolar`, the
  horizon at 120 days, `dt` at `0.025 day`, and the pilot seed count at five
  paired master IDs per mode and P level.
- Treat day 120 as administrative truncation. Continue a surviving organism
  after partner death and stop early only when both organisms are dead.
- Use the current `50 g DM` plant biomass numerical guard and independent
  `50 g DM` biomass observation reference. Any guard-contacting qualification
  or study trajectory is invalid for scientific inference.
- Growth qualification uses `plant-only`, high P, and a deterministic
  vegetative allocation that sends nothing to fungal transfer or reproduction.
  It reports the analytical carbon-only upper bound and the realized budgets.
- Initial growth qualification keeps `amass = 0.05` for the named 16-hour,
  approximately 450-PAR reference day and compares fixed `kleaf` values
  `0.30`, `0.45`, and `0.60`. Values `0.06` and `0.07` for `amass` are allowed
  only as explicitly named higher-irradiance sensitivities.
- Reject progression when plant growth is materially inconsistent with the
  empirical trajectory envelope. Any resulting physiological model revision
  must be justified independently of the desired AM response and requalified
  before the P grid proceeds.
- Domain qualification chooses the smallest scientifically acceptable domain,
  not the smallest runnable domain. It records boundary contact, initial P
  inventory, response-order stability under enlargement, runtime, and memory.
- Training stopping rules have common minimum and maximum transition budgets,
  checkpoint intervals, deterministic evaluation windows, and plateau
  tolerances across each comparison block. Rules are frozen before
  `Delta_AM` is inspected.
- A maximum-budget run that does not satisfy the stopping rule remains in the
  bundle with status `unconverged`. Selective extension of individual runs is
  prohibited; changed optimizer settings require rerunning the affected block.
- Checkpoints are resumable and contain policy parameters, optimizer state,
  training transitions, seed and named random-stream metadata, environment and
  species configuration, mode, P level, and schema/interface versions.
- Primary checkpoint evaluation is one deterministic latent-location policy
  trajectory at the scientific horizon. Sampled-policy replicates are a
  secondary diagnostic stored separately from between-training-seed outcomes.
- Preserve the same master seed IDs across modes and P levels. Treat paired
  seeds as a design block, not as a change to mode-level estimands. Pairing
  diagnostics include marginals, matched differences, scatter, and covariance.
- Introduce named random streams before confirmatory use of seed pairing.
  Training-seed pairing may be retained only with its covariance behaviour
  reported; paired evaluation environments remain mandatory.
- Store transition- or interval-level data sufficient to reconstruct fitness,
  biomass, gross growth, RGR, limitation status, uptake sufficiency, actions,
  transfers, pools, mortality, truncation, soil inventory, and loss counters.
- Calculate primary `Delta_AM` from mode-level mean cumulative plant
  reproductive fitness. Calculate MGR from the ratio of mode-level mean final
  living biomass; do not average seedwise ratios and do not epsilon-adjust a
  zero plant-only denominator.
- Preserve raw and living biomass separately. Report the primary fitness
  endpoint even when it disagrees with biomass response.
- The Phase 1 pilot tests an ordered overall decline in association advantage
  with increasing P, not strict monotonicity between every adjacent estimate.
  All predeclared levels remain in analysis.
- The denser Phase 1 manifest is derived from accepted pilot qualification,
  including range, variance, precision, domain, and training budgets, then
  frozen before dense-grid contrasts are inspected. Exact levels and
  replication are therefore data-informed but prospective.
- A credible Phase 2 bracket requires a qualified positive lower-P
  `Delta_AM` and qualified negative higher-P `Delta_AM`, supported across seed
  outcomes and uncertainty and not explained by nonconvergence, mortality,
  domain or cap contact, finite-inventory exhaustion, or one anomalous pair.
- The upper bracket side must also show weak response of `plant-only` growth to
  further P and diagnostics consistent with movement away from P limitation.
- Phase 2 has its own frozen manifest. It must declare local levels inside and
  including the qualified bracket, replication justified from Phase 1
  variance, and one threshold uncertainty method before local outcomes are
  inspected. The implementation must not silently choose a method from the
  observed Phase 2 curve.
- Threshold analysis reports either a local zero-crossing estimate with its
  predeclared uncertainty summary or `not estimable`. It does not extrapolate
  outside the qualified bracket or claim a universal species threshold.
- Result bundles are immutable by study identity. Resume may fill missing runs
  and checkpoints only when manifest and interface identities match; an
  incompatible run cannot overwrite or join the bundle.
- Human reports are derived from machine-readable artifacts, ensuring plotted
  values, tables, completion status, and scientific decisions share one source.

## Testing Decisions

- Test external study behaviour through the public study runner and emitted
  artifacts. Avoid tests tied to internal helper structure or exact learned
  policy parameters.
- Use one small deterministic end-to-end fixture that executes the complete
  manifest-to-result path for each supported stage. Scientific horizons and
  training budgets may be reduced, but mode, seed, checkpoint, evaluation,
  aggregation, and provenance semantics must remain production-identical.
- Verify the growth-qualification fixture reports carbon-budget closure,
  windowed RGR, endpoint biomass, initial-pool contribution, limitation state,
  and distance to the 50 g guard.
- Verify a guard-contacting fixture fails qualification and cannot be promoted
  to a Phase 1 or Phase 2 input.
- Verify changing `biomass_cap` does not change actor biomass observations when
  `biomass_observation_reference` is held fixed.
- Verify static controls conserve C and P according to existing loss/export
  semantics and reject negative pools, invalid actions, and incompatible modes.
- Verify domain qualification rejects boundary-contacting or response-order-
  changing candidates and records runtime, memory, and total initial P.
- Verify manifests reject duplicate seeds, unsupported modes, non-positive P,
  inconsistent horizons, missing budgets, incompatible schemas, and incomplete
  parent qualification identifiers.
- Verify checkpoint resume reproduces an uninterrupted small training run at
  the declared deterministic seam and refuses configuration drift.
- Verify a maximum-budget non-plateau run is labelled `unconverged` and remains
  visible rather than being dropped or selectively extended.
- Verify deterministic latent-location evaluation is reproducible and is
  distinguished in schema and summaries from sampled-policy evaluation.
- Verify paired seed IDs are complete and aligned across modes and P levels;
  missing or duplicated members make a comparison block incomplete.
- Verify `Delta_AM` is the difference of mode-level mean fitness and MGR is the
  ratio of mode-level mean biomass. Include unequal seedwise values that would
  catch accidental averaging of ratios.
- Verify zero mean plant-only biomass produces missing MGR with an explicit
  reason rather than an epsilon-adjusted value.
- Verify raw seed outcomes, paired differences, marginal summaries, covariance,
  and uncertainty inputs round-trip through the result bundle.
- Verify Phase 1 tendency classification uses the complete predeclared grid and
  does not require strict adjacent monotonicity.
- Verify dense-grid execution refuses an unfrozen design or an unqualified
  pilot parent and preserves retained pilot levels declared in the manifest.
- Verify the credible-bracket gate rejects crossings caused by unconverged runs,
  cap/domain contact, biological failure, inventory exhaustion, or absent
  evidence of reduced P limitation.
- Verify Phase 2 refuses to execute without a qualified bracket, a frozen local
  grid, replication, and a named uncertainty method.
- Verify threshold analysis never extrapolates outside the qualified bracket,
  reports `not estimable` when signs do not persist, and keeps the predeclared
  uncertainty method in result provenance.
- Verify reports and plots are generated from the saved result bundle rather
  than recalculating from untracked runtime state.
- Reuse existing prior art: environment transition and mass-balance tests,
  actor-observation contract tests, policy-artifact round trips and
  incompatibility rejection, deterministic phosphate qualification runners,
  provenance-aware artifact writers, and public CLI subprocess tests.
- Do not test that PPO discovers a predetermined trade policy or sign crossing.
  Learning success is a scientific result governed by declared convergence and
  acceptance criteria, not a deterministic software invariant.
- Run focused study, policy-artifact, observation, environment, and accounting
  suites for each ticket. Before scientific execution, run the full repository
  suite and explicitly distinguish new failures from documented pre-existing
  failures.

## Out of Scope

- Spatially heterogeneous initial P fields, correlation-length qualification,
  and the proposed later variability study.
- Entry-triggered resource sampling, temporal P replenishment, mineralisation,
  occlusion, organic-P cycling, or other exogenous P dynamics.
- A learned or mechanistic in-episode AM-engagement/colonisation gate.
- Colonisation initiation costs, persistence costs, establishment delay,
  intraradical structures, or colonisation-dependent uptake geometry.
- A fungus-present but transfer-disabled treatment; the off mode is complete
  `plant-only`.
- Claims that finite sigmoid trade reaches exact zero.
- Claims of globally optimal strategies, Nash equilibria, universal species
  thresholds, or evolutionary bet hedging.
- Reproductive-fitness ratios or geometric-mean reproductive fitness.
- A P-generalist policy trained across concentrations in Phase 1 or Phase 2.
- Selecting a hard limiting-factor threshold merely to create a sign crossing.
- Adding nitrogen, water, canopy self-shading, phenology, or another external
  limitation unless an independently scoped growth-qualification failure
  demonstrates that a model revision is required.
- Full ontogenetic leaf-area or leaf-mass dynamics. The current specification
  qualifies fixed `kleaf` sensitivities and may stop before the P study if they
  are inadequate.
- Treating the `50 g DM` guard, the `35.05 g` fitted Forto asymptote, or any
  single endpoint as a mechanistic carrying capacity.
- Confirmatory biological inference from the five-pair range-finding pilot.
- Choosing exact dense Phase 1 levels, Phase 1 main-study replication, Phase 2
  local levels, or the threshold uncertainty method before their prerequisite
  qualification evidence exists. The workflow and freeze gates are in scope;
  those values remain deliberately qualification-dependent.

## Further Notes

### Preserved growth-scale evidence

The favourable field benchmark for carrot cultivar `Forto` is represented by
the fitted dry-mass trajectories:

```text
leaf DM = 6.84 / (1 + exp[-0.062 * (DAS - 98.00)])
storage-root DM = 28.21 / (1 + exp[-0.088 * (DAS - 113.91)])
```

Their sum gives the following reference values:

| DAS | Leaf DM (g) | Storage-root DM (g) | Total DM (g) | Leaf fraction | Following 20-day RGR (d^-1) |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.183 | 0.042 | 0.225 | 0.812 | 0.0656 |
| 60 | 0.592 | 0.243 | 0.836 | 0.709 | 0.0647 |
| 80 | 1.688 | 1.358 | 3.046 | 0.554 | 0.0596 |
| 100 | 3.632 | 6.410 | 10.042 | 0.362 | 0.0420 |
| 120 | 5.447 | 17.797 | 23.244 | 0.234 | -- |

The observed day-120 shoot-plus-storage-root endpoint was approximately
`23.26 g DM`. The arithmetic sum of the independently fitted organ asymptotes
is `35.05 g`, but this was not an observed plant, excludes fine roots, and is
approached only later in the equations. Independent field cultivar means were
approximately `21.63--24.65 g whole-plant DM`; controlled-pot endpoints were
approximately `9.5--14.4 g` and `13.03 g DM`. The project therefore uses
`25--35 g DM` as a favourable trajectory reference and `50 g DM` only as a
numerical guard.

With the current fixed values `kleaf = 0.30`, `amass = 0.05`,
`kappa_c = 0.007`, and `gamma_c = 0.402`, the sustained carbon-only ceiling is:

```text
kleaf * amass = 0.015 g C g^-1 plant DM d^-1
net C before growth = 0.015 - 0.007 = 0.008 g C g^-1 DM d^-1
maximum sustained carbon-only RGR = 0.008 / 0.402 = 0.0199 d^-1
```

If each initial free pool supplies one structural-biomass equivalent, an
idealized growth-only upper-bound trajectory starting from `0.01 g` reaches
approximately:

```text
0.02 * exp(0.0199 * 120) = 0.218 g DM
```

This is far below the favourable Forto trajectory. The evidence points first
to the fixed whole-plant leaf fraction rather than an unsupported doubling of
leaf-level `amass`: the fitted Forto leaf fraction declines from about `0.81`
at day 40 to `0.23` at day 120. A fixed `kleaf` near `0.60` produces an
age-averaged carbon-only ceiling near `0.057 d^-1`, close to the Forto
40--120-day mean RGR of approximately `0.058 d^-1`, but it remains a
qualification sensitivity rather than an accepted ontogenetic model.

### Preserved P-scale diagnostic

At the current default initial `0.01 g` plant biomass and the full default
domain, the initial direct-root uptake probe gave approximately:

| Initial solution P (micromolar) | Initial uptake (mg P g^-1 DM d^-1) |
|---:|---:|
| 0.1 | 0.00582 |
| 0.3 | 0.01746 |
| 1.0 | 0.05805 |
| 3.0 | 0.17294 |

The carbon-balanced structural-P need at the current carbon ceiling is roughly
`0.0392 mg P g^-1 DM d^-1`, giving an instantaneous initial crossover near
`0.7 micromolar`. This is a scale diagnostic, not an AM-association threshold:
depletion, allocation, fungal exchange, mortality, domain inventory, and
learned policies all change the realized trajectory.

### Interpretation guardrails

- The surplus-C hypothesis does not imply that plants are universally or
  permanently carbon-unlimited. Source versus sink limitation is contextual
  and ontogenetic.
- An arbitrary external limiting-factor threshold chosen to induce a desired
  `P_thresh` would make the inference circular.
- Growth qualification must reproduce a plausible trajectory and declining
  RGR, not merely a final biomass value.
- Positive low-P and negative high-P association advantage define a local
  bracket only after every qualification gate passes.
- A Phase 2 result may legitimately be `not estimable`; this is preferable to
  forcing a threshold from an unsupported curve.
