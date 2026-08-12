## Problem Statement

The model currently has no figure that relates phosphate uptake to the plant-side carbon required to construct absorbing tissue across absorber radius and length density. Raw uptake rises naturally with length density because more absorbing surface is present, so raw surfaces alone cannot distinguish construction efficiency from scale.

The sparse closure also contains a concentration-like resistance `k` whose dependence on absorber radius, length density, effective territory, and flux diffusivity is difficult to interpret from existing plots. We need a reproducible diagnostic that makes these relationships visible while preserving the distinction between buffered propagation and amount-flux transport.

## Solution

Add a standalone qualification and plotting workflow for an isolated,
single-cell absorber experiment. Call the uptake and geometry closures
directly; do not construct or step `BaseMycorMarl`, run organism growth, or run
the multi-agent model.

Sweep absorber radius logarithmically from `1e-4` to `3e-2` cm and absorber length density logarithmically from `1e-1` to `1e4 cm cm^-3`. For each geometry, run two experiments over the common configured uptake reference horizon `T_ref`:

1. A fixed-bulk-concentration reservoir experiment that holds `C_b` at the model default and isolates intrinsic transport and uptake performance.
2. A finite-inventory experiment initialized with the same default soil phosphate state and allowed to deplete.

Produce an efficiency figure (integrated P uptake per construction carbon and maximum P uptake rate per construction carbon), a scale figure (the corresponding absolute integrated P uptake and maximum uptake rate), and a separate depletion-timescale figure showing `t_1%`, the first time surface concentration falls below one percent of its initial value, for finite-inventory experiments.

Do not add a separate length-normalized surface. Under the plant-economics
mapping used by the sweep, construction carbon is a constant multiple of
absorber length, so uptake per construction carbon and uptake per unit length
contain the same geometric structure and differ only by a constant scale
factor. Time-dependent growth of the unresolved depletion gradient is a
separate diagnostic; this workflow evaluates the implemented closure with its
fixed configured `T_ref`.

Construction carbon is plant-side and analytical. For target absorber length `L=lambda*V_i`, derive plant root biomass using the existing root fraction and specific-root-length relation, then calculate structural carbon with plant `gamma_c`. Do not include above-ground growth, maintenance, reproduction, or unrelated biomass.

The surfaces use plant-side economics. Mark plant-native, fungus-geometry-under-plant-economics, and fungus-native-economics reference cases. The fungus-native point uses fungal biomass-to-hypha conversion, fungal `gamma_c`, and fungal uptake traits; it is an explicitly labelled off-surface comparison.

## User Stories

1. As a plant-model researcher, I want integrated phosphate uptake per construction carbon over absorber radius and length density, so that I can compare lifetime return on construction investment.
2. As a plant-model researcher, I want maximum phosphate uptake rate per construction carbon, so that I can compare short-term uptake power per unit construction investment.
3. As a plant-model researcher, I want absolute integrated uptake on a companion figure, so that higher uptake caused only by more absorbing surface is not mistaken for greater efficiency.
4. As a plant-model researcher, I want absolute maximum uptake rate on a companion figure, so that the scale of the uptake response remains visible.
5. As a sparse-closure researcher, I want radius and length-density sweeps to expose uptake dependence on `k`.
6. As a sparse-closure researcher, I want buffered propagation diffusivity and amount-flux diffusivity roles preserved.
7. As a model user, I want a fixed-bulk-concentration experiment to isolate geometry and kinetics from finite soil inventory.
8. As a model user, I want a finite-inventory experiment to see realized uptake under depletion.
9. As a model user, I want the two conditions shown separately.
10. As a model user, I want both conditions to use a common `T_ref`.
11. As a model user, I want a separate `t_1%` diagnostic to assess whether `T_ref` is representative.
12. As a model user, I want `t_1%` measured only in finite-inventory experiments.
13. As a plant researcher, I want construction carbon computed from absorbing length only.
14. As a plant researcher, I want the plant default point marked.
15. As a plant-fungus researcher, I want fungus geometry evaluated with plant economics.
16. As a plant-fungus researcher, I want a fungus-native economics point.
17. As a reader, I want the fungus-native point visually distinguished as off-surface.
18. As a researcher, I want default points annotated with geometry and metric values.
19. As a researcher, I want machine-readable output for every grid point and marker.
20. As a researcher, I want deterministic runs under explicit model defaults.
21. As a maintainer, I want the workflow to reuse existing uptake and geometry APIs.
22. As a maintainer, I want reduced-domain or single-cell qualification settings.
23. As a maintainer, I want invalid scientific inputs rejected clearly.
24. As a reviewer, I want limiting cases tested.
25. As a reviewer, I want surface concentration and resistance diagnostics exposed.
26. As a maintainer, I want the diagnostic to call isolated closure kernels rather than run the full environment or organism model.
27. As a reader, I want the absence of a separate length-normalized surface explained rather than mistaken for an omitted diagnostic.
28. As an author, I want publication-quality figures suitable for inclusion in a scientific manuscript.

## Implementation Decisions

- Add one high-level experiment runner for the geometry sweep, with reservoir and finite-inventory modes.
- Implement the runner as a closure-level, fixed-geometry harness; do not instantiate or step `BaseMycorMarl`.
- Reuse existing uptake requests, sparse resistance `k`, diffusivities, cell-volume conventions, Michaelis-Menten kinetics, and unit conversions.
- Reuse existing plant root-length and fungal hyphal-length conversion formulas for native markers.
- Keep the experiment outside the production transition contract.
- Use model defaults for phosphate, transport, kinetics, timestep, and `T_ref`, with explicit qualification overrides.
- Use `T_ref` as the common horizon for efficiency and scale figures.
- Sum timestep uptake for integrated P and take the maximum recorded rate for maximum uptake.
- Compute plant construction carbon from `L=lambda*V_i`, root biomass `L/(k_root*SRL)`, and plant `gamma_c`.
- Record that `uptake/C_construction = (uptake/L) * (k_root*SRL/gamma_c)` under the plant-economics sweep; do not produce a duplicate length-normalized surface.
- Compute the fungus-native marker with fungal conversion, `gamma_c`, and uptake traits; label it separately.
- Hold `C_b` fixed in reservoir mode. Initialize default soil P and allow depletion in finite-inventory mode.
- Define `t_1%` as the first sampled time with `C_s/C_s(0) <= 0.01`; return an explicit not-reached result otherwise.
- Use logarithmic grids `1e-4`–`3e-2` cm radius and `1e-1`–`1e4 cm cm^-3` density.
- Provide units, labels, marker annotations, and machine-readable tabular output containing geometry, experiment mode, economics mode, uptake, cost, `k`, surface concentration, and `t_1%` diagnostics.
- Make plotting consume tabular results rather than recompute scientific quantities.
- Produce publication-quality figures with legible final-size typography, colourblind-accessible encodings, unambiguous units and panel labels, and vector output (`PDF` or `SVG`) alongside a high-resolution raster preview.
- Document the workflow alongside the phosphate qualification documentation and ADR.

## Testing Decisions

- Test construction carbon against the existing plant root-length mapping.
- Test the algebraic equivalence of construction-carbon and length normalization up to the documented constant factor.
- Test that normalization excludes above-ground growth, maintenance, and reproduction.
- Test deterministic fixed-reservoir uptake and finite-inventory depletion/conservation.
- Test integrated uptake and maximum-rate summaries against recorded timestep data.
- Test first-threshold and not-reached `t_1%` behavior.
- Test zero/near-zero density, territory scaling, radius effects, and invalid-input handling.
- Test native/economic marker parameter selection and labels.
- Test complete tabular output schema and finite grid values.
- Test the experiment runner seam rather than private plotting details, following existing phosphate qualification, sparse uptake, diffusion, geometry, unit-contract, and conservation tests.
- Check generated plots for expected files, finite data, units, grid shapes, and marker presence.
- Verify publication outputs at their intended final dimensions, including text and annotation legibility, absence of clipping or overlap, distinguishability without relying on colour alone, and successful vector-file generation.

## Out of Scope

- Changing production uptake equations, `D_app`, `D_flux`, `k`, or `T_ref`.
- Running the full environment, organism growth, policy code, or multi-agent dynamics.
- Replacing fixed `T_ref` with elapsed time in the sparse effective-radius calculation; that belongs to the separate time-dependent depletion-gradient diagnostic.
- Adding a separate uptake-per-length sweep surface that duplicates the construction-carbon surface up to a constant factor.
- Adding time-dependent depletion-front state.
- Including above-ground, maintenance, reproduction, or whole-organism carbon budgets.
- Treating the fungus-native point as part of the plant-economics surface.
- Policy training, biological calibration, or a general plotting framework.
- Using fixed-reservoir results as evidence of finite-inventory conservation.
- Replacing common `T_ref` horizons with geometry-specific horizons.

## Further Notes

Plant defaults are `r_a=1e-2 cm`, `lambda=1 cm cm^-3`; fungus defaults are `r_a=5e-4 cm`, `lambda=2000 cm cm^-3`. The timescale reference uses fungal geometry at `lambda=2000 cm cm^-3` and fungal absorber radius.

The plant-economics fungus marker is a counterfactual plant-side result; the fungus-native marker measures fungal P gained per fungal carbon. The diagnostic illuminates construction cost, uptake scale, sparse resistance, and the adequacy of `T_ref`; it does not establish a universal biological optimum.

Related time-dependent depletion-gradient diagnostic: #27.
