# ADR-0016: Use finite-inventory buffered phosphate transport and uptake

**Status:** Accepted; retrospective record of the implemented model, recorded
21 October 2026.

## Context and decision

The original phosphate placeholder did not distinguish concentration from
inventory consistently. The replacement must connect soil supply, absorbing
surface and organism P gain without creating resources, while remaining
affordable on a JAX grid. Use a finite-inventory axisymmetric `r-z` soil model
with the following contracts:

- Store total instantaneously labile inorganic P in µmol per cell. Derive
  solution concentration through the zero-intercept linear buffer
  `M = C * V * (theta_water + b_p)`; do not keep concentration as a second
  independently mutable state. Organism P pools use mg, with one explicit
  conversion at the accepted-uptake boundary.
- Transport P through conservative internal-face finite-volume diffusion.
  Use `D_l * theta_water * f_l` in amount-flux conductances; buffering already
  enters the amount-to-concentration conversion. Use buffered apparent
  diffusivity for depletion propagation and overlap, avoiding double
  retardation in the amount flux.
- Close the surface, bottom, outer radius and symmetry axis to external P
  flux. There is no implicit replenishment when an organism enters a cell.
- Approximate sub-cell uptake with analytical cylindrical sparse resistance
  and continuous Michaelis–Menten demand, rather than a radial PDE or
  iterative surface-concentration solve for every absorber and cell.
- Blend sparse and continuous requests with one local hyphal-density-derived
  overlap weight shared by roots and fungi. With no hyphae, plant uptake is
  sparse. `T_ref` is a static exposure parameter, not numerical step duration
  or tracked absorber age.
- Evaluate both requests from the same post-diffusion concentration, blend
  them, then proportionally cap their sum once against cell inventory. Debit
  soil and credit each organism from the same accepted amounts.

## Trade-offs and consequences

Explicit amount state and paired transfers give a testable conservation
ledger. Axisymmetry, homogeneous buffering and analytical uptake avoid the
cost of individual branching networks, three-dimensional ownership maps and
resolved local depletion profiles. In exchange, the model cannot resolve
angular patches or sub-cell root–fungus interference. The proportional cap is
an inventory rule, not a spatial solution of that interference.

No advective transport, changing water content, nonlinear/kinetic sorption,
mineralisation, precipitation or external supply is represented. These are
scope choices, not evidence that such processes are negligible. Solution and
labile P cannot be equated to an extraction assay without an observation
model. `T_ref`, the blend exponent, and empirical transport/kinetic parameters
remain subject to calibration and sensitivity analysis.

## Evidence and related decisions

Equations, units, parameter provenance and limits are in the
[phosphate model](../phosphate-model.md). The
[production soil transaction](../../src/mycormarl/soil/soil.py) shares
uptake physics with qualification; tests cover
[buffered state](../../tests/test_labile_phosphate_state.py),
[diffusion](../../tests/test_phosphate_diffusion.py),
[sparse limits](../../tests/test_sparse_phosphate_uptake.py), and
[accepted uptake/pool credit](../../tests/test_environment_phosphate_uptake.py).
[ADR-0010](0010-represent-spatial-p-variability-in-the-initial-soil-inventory.md)
distinguishes reset treatments;
[ADR-0019](0019-separate-policy-numerical-and-soil-clocks-and-qualify-resolution.md)
records numerical scheduling and qualification.
