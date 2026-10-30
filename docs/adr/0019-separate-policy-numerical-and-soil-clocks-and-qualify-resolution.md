# ADR-0019: Separate policy, numerical and soil clocks and qualify resolution

**Status:** Accepted; records implemented numerical contracts and the
21 October 2026 selection of the qualified `0.05 cm` default.

The historical separate outgoing/incoming allocation ordering below is
superseded by [ADR-0020](0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md).
Maintenance/death and post-allocation source, geometry and soil boundaries
remain. The retained older coupled qualification keeps its original model
identity; new allocation evidence is reported separately.

## Context and decision

Changing resolution must approximate the same declared physical experiment,
rather than silently changing its action schedule, starting resource stock or
duration. Diffusion stability alone cannot qualify a coupled biological
trajectory. Distinguish three clocks:

- The policy decision interval holds a Rate action across an integral number
  of numerical environment steps.
- `EnvConfig.dt` is the numerical environment interval in days. Each step
  pays maintenance, determines death and bilateral trade availability,
  allocates post-maintenance/post-outgoing-trade pools, then credits incoming
  trade and photosynthate. Post-allocation living biomass determines geometry
  before soil uptake; accepted P is first usable in the next numerical step.
- Soil diffusion/uptake substeps use seconds and fixed post-growth geometry.
  Every substep diffuses first, refreshes concentration and applies uptake.
  Precompute the exact cellwise explicit diffusion ceiling, apply the `0.8`
  safety factor and choose equal substeps covering the numerical interval.
  Geometry-only uptake coefficients are held over that interval, while
  concentration-dependent requests are refreshed in every soil substep.

This records the existing outer splitting. It does not endorse every ordering
as timestep-independent physiology or authorise a continuous
maintenance/source/soil reordering. ADR-0020 subsequently selected a coupled
three-rate allocation solver and admitted incoming trade during allocation.

Use an exactly uniform production grid whose requested intervals divide the
physical extents. Reject invalid intervals with a valid alternative rather
than silently shortening a final cell. Pure geometric helpers may accept
explicit nonuniform edges; that does not change the production-grid contract.

Select numerical resolution using both fixed-geometry soil observables and
coupled trajectories at a fixed physical policy schedule. Require agreement
within the declared 5% criterion with both next-finer and finest tested
references, using explicit absolute scales near zero. Include consumer uptake,
shares, inventory, biomass, spatial extent and coupled endpoint free pools;
preserve identical initial physical stocks. Do not retune scientific
parameters to manufacture convergence.

Use `0.05 cm` radial/depth intervals and `0.025 day` numerical steps, matching
the retained qualification selection. The user explicitly selected the finer
spatial default on 21 October 2026. The default `50 cm × 100 cm` domain then
has `1000 × 2000` cells, four times the earlier `0.1 cm` grid without changing
physical volume or uniform initial P. These are supported numerical choices
for the tested fixture, not universal resolution guarantees.

## Trade-offs and consequences

Policy feedback frequency, numerical integration and soil positivity controls
can now be discussed separately. More soil substeps do not refine growth,
maintenance or geometry exposure. Full-step exposure of new absorbing surface
and nonlinear front-cell uptake still introduce approximation error; the
finite-inventory cap prevents overdraw but does not prove uptake accuracy.
Hold commands and physical conditions fixed when refining integration.

Qualification is numerical evidence, not ecological calibration. The retained
reduced-domain, two-day matrix and deep-soil check do not cover all annual
policies, death boundaries or severely depleted runs; no standard scientific
matrix row was inventory-capped. New regimes need targeted convergence checks.
The finer full-domain default increases compute and memory requirements.
Historical warmed-step timings and annual projections are device- and
configuration-specific, exclude learning/output, and are not measured annual
production runs. Explicit array-memory estimates are not peak process RSS.

## Evidence

The [phosphate guide](../phosphate-model.md#numerical-qualification-and-limitations)
and [coupling guide](../model-overview.md#coupling-pipeline) give equations and
limitations. Selection comes from the retained
[JSON comparisons](../qualification/phosphate-numerical-qualification.json)
and [report tables](../qualification/phosphate-numerical-qualification.md),
not the earlier archive's coarser-grid pass or finest-tested fallback.
[Diffusion tests](../../tests/test_phosphate_diffusion.py),
[qualification tests](../../tests/test_phosphate_qualification.py),
[interval tests](../../tests/test_decision_intervals.py) and
[default-grid inventory tests](../../tests/test_labile_phosphate_state.py)
provide executable checks. See
[ADR-0012](0012-adopt-rate-actions-for-held-policy-intervals.md) for held rates
and [ADR-0016](0016-use-finite-inventory-buffered-phosphate-transport-and-uptake.md)
for the soil model.
