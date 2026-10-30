# Three-rate allocation and workflow qualification

Verified model source baseline: `451586489d884a250be8fbbf2e34f1f8773834eb` (`codex/stoichiometric-three-action`), 22 October 2026. This is the implemented action baseline for later biological simplification. Historical calibration and coupled-grid reports retain their original model identity.

The final complete suite passed **487 tests in 159.77 s** with one existing JAXopt deprecation warning. No tests were skipped and no tolerance was widened. [Retained test output](three-rate-test-results.txt) records the complete command result.

```sh
MPLCONFIGDIR=/tmp/mycormarl-matplotlib .venv/bin/python -m pytest -q
MPLCONFIGDIR=/tmp/mycormarl-matplotlib .venv/bin/python -m scripts.three_rate_qualification
MPLCONFIGDIR=/tmp/mycormarl-matplotlib .venv/bin/python scripts/static_allocation_examples.py
```

The [machine-readable qualification](three-rate-allocation.json) includes every fixture configuration, trait, physical action, reference, resource residual, timing, dependency version and generator hash. [Static examples](three-rate-static-examples.json) and the [two-step PPO artifact smoke](three-rate-ppo-smoke.json) are separate records. The smoke is interface evidence rather than a scientific learned-policy result.

## Six independent allocation references

Pool-only fixtures use explicit initial C/P stocks, unit growth costs, one day, and disabled maintenance, photosynthesis and uptake. Both organisms begin with biomass 1. A has no trade; B has surplus plant C; C has equal trade/biological intensities; D has reciprocal trade and a limiting-resource switch; E changes the expenditure split; F doubles plant reproductive costs. Reference vectors are the independently computed DOP853 illustrative values retained in the agreed case studies. This runner calls the production public environment and contains no second biological engine.

Tolerances were declared before acceptance: reference absolute/relative `2e-5` (maximum of the two bounds), each elemental ledger absolute `4e-6`, paired transfer `1e-7`. The tighter isolated analytical, cost-scale, guard and grouping checks are retained in the test suite.

| Case | Plant growth | Reproductive output | Final plant C | Final plant P | C transferred | Maximum reference error |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.474090517 | 0.158030167 | 0.367879301 | 0.367879301 | 0.000000000 | 1.4e-07 |
| B | 0.237045214 | 0.079015069 | 0.551819086 | 0.183939710 | 1.132120609 | 7.57e-08 |
| C | 0.324249297 | 0.108083099 | 0.135335192 | 0.567667603 | 0.432332397 | 9.11e-08 |
| D | 0.413335353 | 0.137778446 | 0.381850630 | 0.715921819 | 1.067035556 | 2.54e-06 |
| E | 0.108083099 | 0.324249297 | 0.135335192 | 0.567667603 | 0.432332397 | 9.11e-08 |
| F | 0.324249297 | 0.054041550 | 0.135335192 | 0.567667603 | 0.432332397 | 9.11e-08 |

All final resource stocks are nonnegative. Maximum reference error is `2.544e-06`; maximum per-organism C/P ledger residual is `4.47e-08`. Each donor debit equals its counterpart credit. F retains the same expenditure and free stocks as C while producing half its reproductive output. Reproduction leaves parental biomass unchanged in the dedicated tests.

## Fixed-command refinement and fast static allocation

D keeps the same physical commands `[1,.75,.25]` for both organisms, initial stocks, costs and one-day duration. Only the internal outgoing-hazard step bound changes.

| Rate-step bound | Allocation substeps | Plant C absolute reference error |
|---:|---:|---:|
| 0.1 | 20 | 1.5089e-05 |
| 0.02 | 100 | 1.5406e-06 |
| 0.005 | 400 | 9.1657e-09 |

All five recorded plant observables converge together: the maximum reference error falls from `6.427e-05` at the coarse bound to `7.831e-08` at the finest bound, within the declared fine `2e-6` check. The coarse `.1` setting is a refinement candidate and misses the default matrix accuracy criterion; stability alone does not qualify accuracy.

The declared `.05 day`, `epsilon=.001` static experiment derives `K=138.1551055796/day`. Plant trade share `.05` and reproductive share zero execute `[6.907755374908447, 131.24734497070312, 0.0]`. The solver uses `346` internal substeps, yielding growth `0.949050009`, C transfer `0.049950004`, final C `0.000999998` and final P `0.050949998`.

The P remainder is legitimate complementary surplus. K specifies nominal isolated-pool turnover; it does not compel both elements to deplete to epsilon. Internal refinement holds K and all physical rates fixed. Recomputing K after changing the outer experiment interval changes processing speed. General fixed-rate references and learned policies remain unrestricted; no new trade-only strategy is introduced.

## Workflow and lifecycle coverage

| Requirement | Public evidence |
|---|---|
| Essential C/P use, cost scales for both species, zero stocks/actions, physical transfers with unequal costs, guard conservation and six references | [Allocation tests](../../tests/test_stoichiometric_actions.py) |
| Same numerical partition under different grouping; early execution before death; absent partners; shortened/aligned/frozen horizons; JIT/vmap and PPO validity | [Interval tests](../../tests/test_three_rate_intervals.py), [existing interval tests](../../tests/test_decision_intervals.py), [auto-reset tests](../../tests/test_autoreset_contract.py) |
| Static declarations, exact share endpoints, interval-dependent K, metadata, raw reference seam and refinement | [Static experiment tests](../../tests/test_static_allocation.py), [static rollout tests](../../tests/test_static_rollout.py) |
| Three Gaussian latents, physical transform, finite PPO update, latent scoring and entropy masks | [Actor tests](../../tests/test_ppo_actor.py), [transform tests](../../tests/test_three_rate_policy.py) |
| Compatible save/load, explicit legacy rejection, deterministic/sampled evaluation and physical-time reporting | [Artifact tests](../../tests/test_policy_artifacts.py), [evaluation tests](../../tests/test_checkpoint_evaluation.py) |
| Solver-budget and guard rejection in static, evaluation, compiled PPO, qualification and benchmark workflows | [Failure tests](../../tests/test_numerical_workflow_failures.py), [static tests](../../tests/test_static_rollout.py) |
| Fixed reduced-control rates retained under the new schema, including same-environment actor switching | [Reduced-control tests](../../tests/test_trade_only.py) |

Growth blocked by the numerical guard spends no resource and is not redirected to reproduction. Guard contact is an any-step qualification event even when later maintenance lowers biomass. Static runs stop and report rejection; evaluation/qualification raise; synchronized jitted PPO raises a runtime error through one end-of-chunk qualification check. Invalid results are not accepted as successful trajectories. Requested commands remain visible when trade or biology cannot execute.

High-rate existing analytical tests at `1000/day` exposed float32 ledger drift. Compensated summation fixes it without stock clipping or rate caps. An integer-boundary auto-reset test also exposed eager/JIT reassociation in substep selection; precomputing the time-scale factor restores consistent execution diagnostics.

## Compilation and warmed short-run timing

Wall-clock timings include explicit `jax.block_until_ready`; five warmed repetitions follow the first call. Pool cases repeat the same public step from an identical initial state and return the full observations/state/reward/info payload. The coupled benchmark carries state plus numerical qualification flags through successive small-grid steps. These payloads and workloads differ; their timings are not a speed ranking.

| Workload | Compile plus first call (s) | Warmed step (microseconds) |
|---|---:|---:|
| Pool case A | 0.3310 | 83.38 |
| Pool case D | 0.2595 | 70.97 |
| Fast static, 346 allocation substeps | 0.2598 | 76.53 |
| Coupled full step, 8 cells | 0.2361 | 48.65 |

Runtime: Python `3.12.9`, platform `macOS-26.6.2-arm64-arm-64bit`, device `TFRT_CPU_0`, `jax_enable_x64=False`. Package versions: `jax 0.9.0`, `jaxlib 0.9.0`, `numpy 2.3.5`, `flax 0.12.3`, `optax 0.2.6`, `pytest 9.0.2`.

These are reduced-domain numerical checks and short CPU timings. They do not measure annual production training, peak process memory, or all spatial/depleted/seasonal regimes. The retained older spatial qualification was not rerun as a new ecological study.

## Supported numerical contract and remaining boundaries

Allocation is SSPRK3 with compensated float32 updates. Internal substeps satisfy the configured outgoing-hazard bound up to floating-point roundoff; default bound `.02` and budget `65536` were used for the case matrix. The limiter is reevaluated at each stage. Budget exhaustion or non-finite integrated state/output returns `allocation_converged=False` and a non-finite failed transition, which the public workflows reject. Reproductive expenditure is integrated directly and output is derived once afterward. Both-species tests at cost scales `1e37`/`1e38` verify expenditure is preserved even when output increments underflow. Output outside float32 range explicitly fails qualification; a tiny rounded output never erases resolved resource spending. No action is silently capped and no negative stock is repaired by clipping. This is a qualified fixed-step solver, not an adaptive error estimator or an accuracy guarantee for arbitrary parameters. New regimes require their own checks.

Policy grouping invariance applies to identical commands on the identical numerical partition. Changing feedback frequency can change commands. Refining the outer environment step still changes maintenance/death and source/soil splitting; the full model is not timestep-exact. Convergence fixtures pin physical initial pools because unspecified reset stocks still depend on dt. Final-substep trade-history observation semantics remain unchanged.

The [successor ADR](../adr/0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md) records the implemented baseline and compatibility break. [Bug notes](../implementation-docs/action-application-bug-notes.md) distinguish resolved findings from deferred reset/observation questions. Existing parent issues for biological simplification and the historical rate migration were not changed.
