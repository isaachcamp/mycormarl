# P2 — Realised growth to conservative root and fungal geometry

## Phase goal

Make reset and every biological step derive physically unit-labelled root and
external-hyphal length-density fields from current structural biomass, so
realised stoichiometric growth changes absorbing geometry before soil uptake.

## Walking skeleton

Species traits convert plant and fungal dry biomass into total root and hyphal
length. Pure axisymmetric distributors turn those totals into `(n_r, n_z)`
length-density fields. Reset applies the conversion to initial biomass; each
step applies it again after realised growth and maintenance-related biomass
loss, before the fail-closed P3 soil stage.

## Scope

- Atomic migration from `carbon_per_growth`/`phosphorus_per_growth` to
  organism-specific `gamma_c`/`gamma_p` traits.
- Provisional *Daucus carota* root fraction and SRL defaults.
- Bisot-style fungal dry-biomass-to-carbon-to-tissue-volume-to-length
  conversion.
- Stacked-disc root distribution with uniform within-disc
  `lambda_root` and depth-dependent radii.
- Saturated hemispherical fungal occupancy with volume-averaged front cells.
- Initial and post-growth environment integration.
- Explicit zero-biomass, biomass-decline, and domain-capacity behaviour.

## Non-scope

- Phosphate uptake, diffusion, sparse/continuous blending, or P-pool credit.
- Root or hyphal age, turnover schedules, or inactive structural fractions.
- Hyphae, roots, spores, or intraradical fungal structures outside the
  simulated soil domain.
- Calibration of provisional trait values.

## Dependencies and existing stack

- Verified P0 unit contracts and P1 interval/maxima axisymmetric geometry.
- Existing JAX/Flax immutable state and JaxMARL environment conventions.
- No new dependency or framework.

## Scientific contracts

- Root length: `L_root = biomass_plant * k_root * SRL`; `k_root` is applied
  exactly once.
- Hyphal length:
  `L_h = biomass_fungus * gamma_c / (M_C * pi * r_h^2)`.
- When geometry fits inside the domain,
  `sum(lambda_i * V_i) = L_total` within documented float32 tolerance.
- A fungal colony that reaches a boundary remains capped at `lambda_sat`.
  Represented external length is `lambda_sat` times the exact intersection
  volume between its implied hemisphere and the soil domain. It approaches
  saturated domain capacity only once the hemisphere covers the full domain;
  structure outside the modelled soil has no in-domain absorbing surface.
- For depth layer `k`, assign `L_k = L_root * w_k` from the differences of
  `F(d) = 1 - beta^d`, then calculate
  `R_k = sqrt(L_k / (pi * lambda_root * dz_k))`. Thus `lambda_root` is uniform
  within every occupied disc while deeper layers with less length expand more
  slowly. Volume-averaged front cells may lie between zero and `lambda_root`.
- When a calculated root radius exceeds the outer soil radius, represent the
  disc-domain intersection at `lambda_root`; root length outside the modelled
  soil is clipped rather than redistributed across depths.
- Density is recomputed from current post-growth biomass, so maintenance loss
  contracts represented structure and zero biomass produces exactly zero
  density.

## Granular task checklist

- [v] **P2.1 — Migrate traits and implement biomass-to-length contracts.**
  - Rename growth costs atomically to `gamma_c` and `gamma_p`.
  - Add agreed plant and fungal defaults with explicit units in docstrings.
  - Implement pure root and hyphal length conversions and zero limits.
  - Verify the agreed per-gram reference calculations and JIT compatibility.
- [v] **P2.2 — Complete conservative axisymmetric density construction.**
  - Distribute root length through stacked discs without applying `k_root`
    twice.
  - Convert total hyphal length to a saturated hemispherical radius and
    volume-average partial front cells.
  - Verify integral conservation while geometry fits, monotonic extent, zero
    biomass, shortened grid boundary cells, and explicit domain clipping.
- [v] **P2.3 — Integrate initial and post-growth geometry.**
  - Initialise both density fields from initial biomass.
  - Recompute both fields from post-growth/post-maintenance biomass before the
    soil step.
  - Verify immediate growth credit, density contraction after biomass loss,
    and unchanged environment interfaces.
- [v] **P2.4 — Remove superseded placeholders and document the P2 pipeline.**
  - Remove obsolete geometry formulas, dead state assumptions, and stale
    exports.
  - Update function/test documentation and planning references.
  - Confirm no old growth-cost names remain.
- [v] **P2.5 — Revise stacked discs to uniform density and depth-varying radii.**
  - Restore provisional `root_length_density = 1.0 cm cm^-3` as the scalar
    `lambda_root` inside occupied discs.
  - Derive one radius per depth layer from its beta-weighted length and actual
    layer thickness.
  - Preserve volume-averaged radial front cells and explicit domain clipping.
  - Update the geometry video so the maximum root-disc radius and fungal
    hemisphere radius advance by one radial grid interval per frame.

## Tests to add or run

- Add focused P2 unit and invariant tests for traits, length conversions,
  spatial integrals, clipping, reset, and post-growth sequencing.
- Retain the P0/P1 geometry and amount-state tests.
- Run focused tests after each red/green increment.
- Run the complete suite before phase verification and after review fixes.

## Verification checklist

- [v] Per-gram root and fungal reference lengths match independent arithmetic.
- [v] `k_root` is applied exactly once.
- [v] Root length is conserved while all discs fit; clipped discs match their
  analytical intersections with the radial domain.
- [v] Fungal length is conserved before boundary contact and equals the
  analytical hemisphere–domain intersection after contact, approaching
  saturated domain capacity when the domain is fully covered.
- [v] Partial cells, zero biomass, monotonic extent, and biomass decline pass.
- [v] Reset and post-growth geometry use the same pure conversion path.
- [v] Growth geometry is updated before the soil stage.
- [v] Focused and complete suites pass.
- [v] Phase review finds no P3 uptake or P4 diffusion implemented early.
- [v] Root density is uniform at `lambda_root` inside every disc, layer radii
  decrease with beta-weighted depth, and represented length satisfies the
  contained/clipped analytical contract.
- [v] The regenerated video shows non-uniform radial root expansion and stays
  within the 1,000-frame limit.

## Phase exit criteria

- Reset and integrated steps expose finite non-negative root and hyphal
  density fields derived from current biomass and documented traits.
- Spatial integrals satisfy the conservation/capacity contracts above.
- Trait naming and units are unambiguous throughout code and tests.
- All tasks and verification checks are `[v]`, followed by phase review/fix.

## Phase notes

- Phase status: **verified**, including P2.5, on 2026-07-21.
- The P1 grid is configured by maximum radius/depth and interval sizes; P2
  geometry must use generated edges and cell volumes, including shortened
  final cells.
- P2.1 red evidence: focused test collection failed because the new
  unit-labelled conversion functions did not exist.
- P2.1 green evidence: trait defaults, per-gram conversions, zero limits, and
  JIT compatibility passed `4 passed, 1 warning`.
- P2.2 evidence: conservation on shortened grid cells, partial fungal-front
  occupancy, monotonic extent, zero biomass, and saturated-domain clipping
  passed with the retained P1 geometry tests: `22 passed, 1 warning`.
- P2.3 red evidence: reset produced zero density and the step called obsolete
  placeholder signatures; the integration tests failed at both boundaries.
- P2.3 green evidence: initial geometry, immediate post-growth geometry before
  soil processing, and maintenance-loss contraction passed `3 passed, 1
  warning`; the broader P1/P2/environment set passed `46 passed, 1 warning`.
- P2.4 red/review evidence: new checks exposed that the private fungal
  intersection helper returned volume divided by `pi`, and that an oversized
  explicit root disc normalised length over soil that was not represented.
  Both regression checks pass after restoring physical volume units and
  clipping root-disc radius to the domain.
- P2.4 cleanup removed superseded formulas, added fail-fast trait validation,
  completed public exports, and expanded the function/test reference. Before
  the final verification pass, the complete suite passed `89 passed, 1
  warning`.
- Second verification pass: focused P1/P2/environment tests passed `58 passed,
  1 warning`; the post-review complete suite passed `91 passed, 1 warning`.
  Independent arithmetic reproduced `15,769.266 cm root g^-1 plant dry mass`,
  `5,511,859.501 cm hypha g^-1 fungal dry mass`, and a `24.984 cm` saturated
  hemispherical radius for one gram of fungal biomass. Documentation coverage,
  stale-symbol audit, and `git diff --check` all passed.
- Phase completion review found and fixed the missing `pi` in the private
  physical-volume helper and root-length loss for oversized explicit discs.
  No blocking findings remain; P3 uptake and P4 diffusion are still absent.
- P2.5 was added after visual inspection showed that the beta distribution
  changed density but not radial shape: one disc radius had been broadcast to
  every depth. The revised contract uses uniform `lambda_root` and derives a
  separate radius for each depth layer.
- P2.5 red evidence: the new depth-specific geometry tests initially failed at
  collection because `root_disc_radii_from_biomass` did not yet exist.
- P2.5 verification: root-focused tests passed `10 passed, 1 warning`; the
  combined geometry/environment set passed `65 passed, 1 warning`; and the
  complete post-review suite passed `98 passed, 1 warning`. The regenerated
  GIF has 100 frames at 1200 by 600 pixels, visibly tapers the root discs with
  depth, and advances the largest root radius and fungal radius by one radial
  interval per frame. Documentation coverage, stale-interface search, and
  `git diff --check` passed. The warning remains the existing JAXopt
  deprecation notice.
