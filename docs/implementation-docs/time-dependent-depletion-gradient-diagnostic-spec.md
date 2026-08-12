## Problem Statement

The implemented sparse phosphate-uptake closure uses the fixed configured
reference time `T_ref` to set the distance reached by an unresolved radial
depletion gradient. Existing construction-carbon surfaces therefore do not
show how the sparse closure separates from the continuous closure as that
gradient develops through time.

We need a simple diagnostic that isolates this time dependence without
running the full model, allowing absorber growth, depleting the bulk reservoir,
or implying that global simulation time is suitable for production dynamics.

## Solution

Add a standalone fixed-reservoir experiment that calls the uptake closures for
fixed absorber geometries. Hold bulk phosphate concentration and all
non-geometry transport and uptake parameters fixed. Replace `T_ref` only in
the diagnostic sparse effective-radius calculation with elapsed experiment
time `t`:

    R_soil = 1 / sqrt(pi * lambda)
    R_eff(t) = r_a + min(
        sqrt(D_app * t),
        max(R_soil - r_a, 0),
    )
    k(t) = (r_a * J_max / D_flux) * log(R_eff(t) / r_a)

`R_eff(t)` is the effective outer radius of the unresolved phosphate
depletion gradient around a fixed absorbing cylinder. It is not a root,
hyphal, or colony growth radius. At `t=0`, `R_eff=r_a` and `k=0`, so the sparse
and continuous closures coincide.

Run the experiment from zero to 30 days and mark the current one-day `T_ref`
with a vertical reference line. Use two fixed absorber radii:

- plant-scale radius: `r_a=1e-2 cm`;
- fungus-scale radius: `r_a=5e-4 cm`.

At each radius use the same three length densities so that density effects can
be read independently within each radius panel:

- low: `lambda=1 cm cm^-3`;
- intermediate: `lambda=100 cm cm^-3`;
- high: `lambda=2000 cm cm^-3`.

Use one documented plant-trait baseline for `C_b`, `D_app`, `D_flux`, `J_max`,
and `K_m` in every curve. The radius labels describe geometric scales, not
native plant and fungal parameter bundles. Construction economics and
`gamma_c` do not enter this diagnostic.

Produce one two-panel figure:

- panels: plant-scale and fungus-scale absorber radius;
- y-axis: instantaneous P uptake rate per unit absorber length versus time;
- colour: length density;
- line style: time-dependent sparse or continuous closure.

The continuous per-unit-length curves are exactly density-independent when the
other traits are held fixed. Show this overlap as one clearly labelled
continuous reference where drawing three coincident lines would obscure the
plot.

Label each rate panel compactly with the 30-day cumulative total uptake by the
represented cell for every density and closure combination:

    U_30d = integral from 0 to 30d of uptake_rate(t) dt.

“Per density” means one integral is reported for each selected length density;
the integral is not divided by length density.

## User Stories

1. As a closure researcher, I want to see the sparse uptake rate depart from the continuous rate as the depletion gradient travels outward.
2. As a closure researcher, I want uptake rate per unit length so that increased represented length is not mistaken for a change in local closure behaviour.
3. As a model researcher, I want each curve labelled with its 30-day cumulative total uptake so that the cell-scale consequence of each density remains visible without a redundant cumulative-trajectory panel.
4. As a model researcher, I want identical densities at two fixed radii so that density effects can be interpreted independently within each radius panel.
5. As a reader, I want the current one-day `T_ref` marked against a longer 30-day trajectory.
6. As a reader, I want the 30-day integrated uptake labelled for each density and closure.
7. As a maintainer, I want an isolated closure diagnostic with no environment, growth, policy, or multi-agent execution.
8. As a reviewer, I want global experiment time clearly distinguished from colony propagation and production-model cohort age.
9. As an author, I want a publication-quality figure suitable for inclusion in a scientific manuscript.

## Implementation Decisions

- Add a pure diagnostic calculation of `R_eff(t)` and `k(t)`; do not change the production sparse-closure API or semantics.
- Evaluate sparse and continuous requests directly in a fixed-geometry harness.
- Hold the fixed reservoir at the documented default bulk concentration.
- Hold non-geometry parameters at one documented plant-trait baseline across both radii and all densities.
- Use the repository's existing territory radius, surface-concentration, kinetics, area, unit-conversion, and uptake-request conventions.
- Use a fixed representative cell volume and report it in output metadata.
- Calculate instantaneous rate consistently from timestep uptake and calculate the 30-day cumulative endpoint with the documented timestep quadrature/sum.
- Include `t=0` explicitly and handle the `k=0` sparse/continuous identity without a numerical singularity.
- Produce machine-readable time-series output containing time, radius, density, closure, `R_soil`, `R_eff`, `k`, surface concentration, per-length rate, and total rate, plus a summary table containing each 30-day cumulative total uptake.
- Make plotting consume the time-series output rather than recompute scientific quantities.
- Produce a publication-quality figure with legible final-size typography, colourblind-accessible encodings, unambiguous units and panel labels, and vector output (`PDF` or `SVG`) alongside a high-resolution raster preview.

## Testing Decisions

- Test `R_eff(0)=r_a`, `k(0)=0`, and equality of sparse and continuous rates at `t=0`.
- Test monotonic outward travel of `R_eff` until the density-derived territory limit and constancy thereafter.
- Test that fixed absorber radius, length density, represented length, reservoir concentration, and non-geometry traits do not evolve.
- Test density independence of continuous uptake per unit length.
- Test proportional density scaling of continuous total uptake at fixed radius and cell volume.
- Test sparse per-length trajectories against direct evaluation of `R_eff(t)`, `k(t)`, surface concentration, and Michaelis--Menten flux.
- Test the endpoint labels against the machine-readable 30-day integrals.
- Test both configured radii, all three densities, the 30-day endpoint, and the one-day reference marker.
- Check generated plots for the two expected radius panels, finite values, units, curve labels, endpoint integrals, and reference annotation.
- Verify publication outputs at their intended final dimensions, including text and annotation legibility, absence of clipping or overlap, distinguishability without relying on colour alone, and successful vector-file generation.

## Out of Scope

- Running or modifying `BaseMycorMarl`, organism growth, policy training, or multi-agent dynamics.
- Root, hyphal, colony, or absorber-radius growth.
- Finite-inventory depletion or soil-grid diffusion.
- Construction carbon, `gamma_c`, or economic normalization.
- Changing the production meanings or defaults of `T_ref`, `D_app`, `D_flux`, `k`, or the sparse/continuous blend.
- Adding a production configuration switch for time-dependent uptake.
- Representing absorber cohort age, mixed-age tissue, turnover, or recolonization in the full model.
- Treating global experiment time as an acceptable substitute for cohort age in production runs.

## Further Notes

This diagnostic intentionally uses global experiment time because all absorber
geometry is present at the start and remains fixed. A future production-model
design should use cohort age rather than global simulation time; that work
requires its own issue and modelling decision.

Related construction-carbon diagnostic: #26.
