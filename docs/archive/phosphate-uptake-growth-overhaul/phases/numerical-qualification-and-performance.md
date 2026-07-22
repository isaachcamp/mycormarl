# Numerical convergence and performance qualification

## Phase goal

Quantify the numerical sensitivity, diagnostic behaviour, and computational
cost of the implemented phosphate model, then select the coarsest spatial grid
and largest biological timestep supported by the provisional 5% criterion.

## Walking skeleton

Run one small deterministic fixed-geometry soil experiment through the real
P4/P5 evolution kernel and emit machine-readable metrics. Extend that runner
across concentration, consumer mode, timestep, and grid matrices. Separately
run a deterministic coupled trajectory for uptake, free P pools, biomass, and
spatial extent. Finally
JIT and time reduced and target-grid soil kernels, estimate live array memory,
and project a one-year run without executing all annual steps.

## Scope

- Initial solution-P scenarios `0.1, 0.3, 1, 3, 10 µM`.
- Root-only, fungus-only, and mixed fixed-geometry uptake scenarios.
- Candidate biological timesteps `0.025, 0.05, 0.1, 0.2, 0.4 day` over the
  same physical horizon, with existing diffusion subcycling active.
- Reduced-domain grid comparison at `0.1, 0.05, 0.025 cm` over identical
  physical geometry and initial conditions.
- Deterministic coupled trajectories for plant/fungal uptake, uptake shares,
  final soil inventory, free P pools, biomass, and root/fungal extent.
- `C_s/C_b`, overlap weight, cap frequency, uptake, consumer shares, soil and
  extended P-balance metrics.
- Sensitivity of the provisional transition controls at `p = 1, 2, 4` and a
  small documented `T_ref` set.
- JIT compilation, steady per-step runtime, estimated array memory, and
  one-year cost projection on reduced and full default grids.

## Non-scope

- Empirical calibration or claims of ecological validation.
- Running the 0.025 cm grid over the full `50 x 100 cm` domain; that would be
  approximately eight million cells and is not needed to estimate convergence.
- A complete one-year simulation; annual cost is projected from timed steps.
- GPU/accelerator comparison or distributed execution.
- Colony-age transition, midpoint-growth geometry, or a new uptake substep
  trigger unless current results fail the agreed criterion.
- Resolution of the separately recorded maintenance-P accounting issue.

## Qualification fixtures and metrics

- **Soil fixture:** a `2 x 2 cm` axisymmetric cylinder with P initially in its
  upper `1 cm`, fixed physically defined root/hyphal density regions, and a
  two-day horizon. This retains the topsoil diffusion front while keeping the
  finest grid to `80 x 80 = 6,400` cells.
- **Modes:** root-only, fungus-only, and mixed fields use exactly zero density
  for absent consumers. Mixed density includes the nominal saturated hyphal
  density so its overlap diagnostic retains physical provenance.
- **Coupled fixture:** small initial biomasses and one fixed action vector per
  partner, repeated over an identical horizon and seed. Actions and traits are
  recorded with results; it is run across both grid and timestep candidates,
  and outputs are not interpreted as calibrated ecology.
- **Primary convergence outputs:** cumulative root uptake, fungal uptake,
  total uptake, final soil inventory, root/fungal uptake shares, biomass, and
  root/fungal spatial extent.
- **Diagnostic outputs:** mean/min active-cell `C_s/C_b`, mean/max `w_cont`,
  capped cell-update fraction, cumulative mortality/reproduction losses, CFL
  ceiling, and substep count.
- **Balance tolerance:** relative error no greater than `1e-5` where the
  unresolved maintenance-P pathway is inactive or explicitly excluded.
- **Selection tolerance:** no primary output may change by more than 5% versus
  the next finer grid/timestep reference when its magnitude is scientifically
  non-negligible. Near-zero outputs use a documented absolute scale guard.

## Dependencies and existing stack

- Verified P0–P5 amount, geometry, diffusion, uptake, competition, and
  diagnostic primitives.
- Existing JAX/Flax/JaxMARL stack; no new package is required.
- JSON and Markdown result artifacts generated deterministically from the same
  qualification runner.

## Granular task checklist

- [v] **P6.1 — Diagnostic contracts and deterministic runner.**
  - Expose one reusable calculation of sparse/continuous requests, `C_s/C_b`,
    overlap weight, cap mask, and accepted uptake without duplicating physics.
  - Build fixed-geometry and coupled deterministic runners with explicit units,
    horizon, densities/actions, and reproducible seed.
  - Test diagnostic consistency against the production uptake transaction and
    exact soil-to-pool balance.
- [v] **P6.2 — Concentration, timestep, and transition sensitivity.**
  - Run all concentration scenarios and three consumer modes.
  - Run all candidate timesteps over an identical physical horizon.
  - Quantify `p = 1, 2, 4` and provisional `T_ref` sensitivity.
  - Record convergence tables and whether an uptake-accuracy trigger appears
    necessary in addition to the diffusion CFL trigger.
- [v] **P6.3 — Spatial-grid and coupled-output convergence.**
  - Run `0.1`, `0.05`, and `0.025 cm` reduced-domain grids.
  - Compare uptake, shares, inventory, biomass, and spatial extents.
  - Select or reject a production grid using the 5% next-finer rule.
- [v] **P6.4 — Runtime, compilation, memory, and annual projection.**
  - Benchmark compilation and warmed soil steps on reduced and target grids.
  - Record backend/device, precision, cell counts, live-state bytes, and an
    explicit model-memory estimate rather than claiming process RSS attribution.
  - Project annual step count, runtime, and update work for the selected `dt`.
- [v] **P6.5 — Results artifact, review, and verification.**
  - Generate durable JSON and concise Markdown reports with configuration,
    metrics, selection decision, commands, and limitations.
  - Run focused/full tests and the phase review/fix cycle.

## Tests to add or run

- Add diagnostic and qualification contracts in
  `tests/test_phosphate_qualification.py`.
- Retain all P0–P5 scientific and integration tests.
- Test deterministic repeated results, metric units/shapes, request/acceptance
  equivalence, balance error, comparison arithmetic, and annual projection.
- Run the complete suite before and after phase review.

## Verification checklist

- [v] Qualification fixtures and results are reproducible and fully specified.
- [v] Diagnostics reuse production equations and match accepted transactions.
- [v] All five concentration scenarios and three consumer modes complete.
- [v] Candidate timestep comparisons use identical physical horizons.
- [v] Grid comparisons use identical physical domains and fields.
- [v] Balance error remains within `1e-5` in applicable scenarios.
- [v] Transition sensitivity states the nominal `w_cont` consequence clearly.
- [v] Grid/timestep selection follows the 5% rule without parameter retuning.
- [v] Target-grid compile/runtime/memory and annual projection are recorded.
- [v] Focused/full suites pass and review has no blocking finding.

## Phase exit criteria

- A committed-format JSON artifact and human-readable report reproduce every
  qualification result from explicit configuration.
- The report selects a production grid/timestep or clearly rejects selection
  with the failing metric and required follow-up.
- Stability, conservation, diagnostic interpretation, memory, and projected
  annual cost are documented without claiming empirical model validity.
- Every task and check is `[v]` after the phase review/fix pass.

## Phase notes

- Phase status: **verified** on 2026-07-21.
- The full-domain cell counts are approximately 500,000 at 0.1 cm, 2,000,000
  at 0.05 cm, and 8,000,000 at 0.025 cm. Fine-grid scientific convergence is
  therefore separated from full-domain performance qualification.
- The maintenance-P accounting issue remains unchanged; coupled balance
  reporting must identify rather than conceal its effect if maintenance is
  active.
- P6.1 red evidence: the diagnostic test module failed during collection
  because the qualification module and shared transaction did not exist.
  Production and qualification now share `blended_uptake_transaction`; an
  offline wrapper retains exact post-diffusion substep diagnostics while the
  JIT environment API remains state-only.
- Scenario evidence: all `0.1, 0.3, 1, 3, 10 µM` cases and root-only,
  fungus-only, and mixed modes completed. Maximum fixed-soil relative P-balance
  error was `1.537e-6`; maximum coupled extended-balance error was `6.880e-7`.
  No scientific matrix case was inventory-capped.
- Selection evidence: `0.1 cm` differs from `0.05 cm` by at most `0.254%` in
  fixed-soil outputs and `3.030%` in coupled outputs, so the spatial candidate
  passes. Once coupled uptake and free-P pools were added, every coarser
  timestep failed: the maximum changes were `99.512%`, `99.025%`, `98.054%`,
  and `102.176%` for `0.05`, `0.1`, `0.2`, and `0.4 day`, respectively,
  dominated by endpoint free-P pools. Defaults are therefore `dt=0.025 day`,
  `max_steps=14600`, and `0.1 cm`. The timestep is the finest tested fallback,
  not proof of convergence.
- Transition evidence: at `T_ref=1 day`, changing `p` from `1` to `2` to `4`
  changes mean `w_cont` from `0.1533` to `0.03172` to `0.001072`. Because
  `T_ref` also sets sparse propagation distance, total uptake need not vary
  monotonically with it; this is documented as parameter sensitivity rather
  than numerical convergence.
- Target CPU evidence on `TFRT_CPU_0`, macOS ARM64: the `500 x 1000` grid has
  500,000 cells; the final run measured approximately `0.0023 s` per warmed
  soil/full deterministic step. The explicit core-array estimate is about
  `49.6 MiB` (concrete state/caches plus 18 float32 cell-array equivalents),
  not process RSS. At 14,600 steps, the latest measured deterministic annual
  projection is roughly 70 seconds plus compilation, excluding training,
  learned-policy inference, output, and transfers. Exact platform-specific values remain in
  [`../../../qualification/phosphate-numerical-qualification.json`](../../../qualification/phosphate-numerical-qualification.json).
- Phase review found and fixed two blocking qualification gaps: coupled
  timestep convergence is now required before choosing `dt`, and extended
  coupled P balance is now measured. It also found that `EnvConfig.max_steps`
  was ignored in favour of a separate 256-step default; environment
  construction now uses the validated configured value unless explicitly
  overridden, preventing annual runs from terminating early.
- The post-P7 audit extended the coupled selection tuple to include plant and
  fungal uptake, uptake shares, final soil inventory, and free P pools. This
  invalidated the earlier `0.4 day` selection and triggered the conservative
  `0.025 day` fallback described above.
- Final focused diagnostic/qualification checks passed; the second complete
  suite passed `194 passed, 1 warning`. `git diff --check`, bytecode
  compilation, and function/test documentation coverage passed. The warning
  remains the unchanged upstream JAXopt deprecation.
- Residual risks: convergence uses a two-day reduced domain and one fixed
  coupled policy, so it does not prove accuracy for every long-horizon MARL
  trajectory. Benchmark timings are device-specific, warm, and exclude
  training/output. The 18-array memory estimate is explicit but XLA fusion and
  allocator behaviour mean it is not peak RSS. Empirical validity is untested.

### Post-P7 audit correction

- The fixture description now matches the implemented and recorded two-day
  `HORIZON_DAYS`.
- Scenario rows now include `maximum_continuous_weight` and
  `diffusion_cfl_seconds`, in addition to mean weight and substep count. The
  canonical JSON and Markdown reports were regenerated after this correction.
