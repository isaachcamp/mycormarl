# Carbon transport costs for fungal and fine-root growth

Selected model proposal, 17 November 2026. The fungal and plant figures describe
one linear carbon-delivery cost law with different distance geometries. This is
a modelling specification with illustrative numerical checks, not a production
implementation or a calibrated physiological law.

- [Fungal growth transport](growth-transport-figure.md):
  [PDF](fungal-growth-transport-costs.pdf),
  [SVG](fungal-growth-transport-costs.svg),
  [generator](generate_growth_transport.py),
  [numerical checks](growth-transport-verification.json).
- [Plant growth transport](plant-growth-transport-figure.md):
  [PDF](plant-growth-transport-costs.pdf),
  [SVG](plant-growth-transport-costs.svg),
  [generator](generate_plant_growth_transport.py),
  [numerical checks](plant-growth-transport-verification.json).

## Shared cost law and scope

For organism `o`, price realised carbon delivery to new absorbing tissue as

\[
\Delta C_{\mathrm{tr},o}=b_o\,\Delta C_{G,o}\,\bar d_o,
\qquad
\bar d_o=\frac{\int_{\mathcal A_o}\lambda_{\mathrm{new},o}(x)d_o(x)\,dV}
{\int_{\mathcal A_o}\lambda_{\mathrm{new},o}(x)\,dV}.
\]

Here `Delta C_G,o` is delivered construction carbon [g C], `b_o` is additional
carbon expenditure per unit carbon delivered per cm [cm⁻¹], and `A_o` is tissue
added during the current numerical step. Plant and fungal coefficients are
separate and unset. The rate form is `Cdot_transport = b * F_C * mean_distance`.

Length weighting assumes equal carbon demand per unit newly constructed length
within each organism. If demand varies spatially, weight by actual delivered
carbon instead. Use axisymmetric volume weights `dV = 2*pi*r dr dz`, rather than
uniform meridional pixel weights. Do not multiply an already budgeted
construction amount by shell volume again.

Assume constant effective transport velocity across routes, times and delivery
rates. Its effect is absorbed into `b`; velocity and transit time are unresolved.
Constant expenditure per unit carbon per unit distance is a separate constitutive
assumption, not a consequence of constant velocity alone. Coefficients require
calibration or sensitivity analysis, including zero.

Internal conducting capacity is non-limiting under
[ADR-0021](../../adr/0021-treat-internal-transport-capacity-as-non-limiting.md).
Lower interior density causes no independent throughput limit, delay or
compensatory carbon surcharge. Explicit fungal polyphosphate-translocation cost
is zero. Uptake, P exchange, structural P requirements, conservation and ordinary
maintenance continue unchanged. This charge covers construction delivery;
maintenance and reproductive delivery are outside its scope.

## Fungal growth shell and nearest-root distance

Represent the persistent root-containing footprint as stacked discs:

\[
\mathcal R=\bigcup_k\{(r,\varphi,z):0\le r\le R_k,\ z_k\le z\le z_{k+1}\}.
\]

It includes implicit conducting roots where fine roots have died. Every cell
is exchange-available while the plant is operational, without a colonisation
state, under
[ADR-0022](../../adr/0022-separate-fine-root-mortality-from-persistent-backbone-connectivity.md).
For external fungal growth from `R_t` to `R_(t+dt)`, use

\[
\mathcal A_{f,t}=\{x:z\ge0,\ R_t\le\|x\|\le R_{t+\Delta t}\},
\qquad w_t=R_{t+\Delta t}-R_t.
\]

Combine the nearest footprint gap with unresolved individual-root spacing:

\[
d_f(x)=\delta(x)+\ell_\lambda,
\qquad \ell_\lambda=\frac{2}{3\sqrt{\pi\lambda_{\rm root}}},
\]
\[
\delta(r,z)=\min_k\sqrt{[r-R_k]_+^2+
\operatorname{dist}(z,[z_k,z_{k+1}])^2}.
\]

The nearest disc need not occupy the same depth layer. The figure uses this
exact continuum distance; distance to the nearest root-containing numerical
cell is its grid approximation and needs spatial refinement checks.

Use reference `PlantTraits.root_length_density` for `lambda_root`, independently
of mortality-depleted density. The spacing closure assumes circular territories
of radius `(pi*lambda_root)^(-1/2)` with mean radial distance two-thirds of that
radius. It is a statistical approximation, not a measured universal law or a
resolved distance inside one numerical cell. Reference density 1 cm/cm³ gives
0.3761 cm.

If the whole growth shell lies inside roots, `delta=0` and the mean is
`ell_lambda`. If part extends beyond roots, integrate its positive gap while
retaining the **whole new shell** in the denominator. These are the two fungal
panels. Construction carbon covers new ERM only, excluding the static IRM share
of total fungal growth, maintenance and reproductive exports.

## Plant growth bands and stem distance

Fine-root growth adds an annulus to each depth-dependent disc:

\[
\mathcal A_{p,t}=\bigcup_k\{(r,\varphi,z):
R_{k,t}\le r\le R_{k,t+\Delta t},\ z_k\le z<z_{k+1}\},
\qquad d_p(r,z)=\sqrt{r^2+z^2}.
\]

The endpoint is the central stem base/root collar at `(0,0)`. Straight-line
distance is the explicit proxy in the plant figure; it does not resolve vascular
branching, tortuosity or aboveground stem length. There is no nearest-root spacing
term because growth sites are already roots. Construction carbon moves outward
along the same distance shown by the front-to-stem dimension arrow. Delivered
carbon covers fine-root construction, not all whole-plant growth or trade.

## Mortality, contact and implementation requirements

The intended oldest-first mortality rule preserves the outer front while the
organism is operational. Fine-root cohorts can disappear completely; the
implicit backbone preserves distal stem connectivity. Any fungal floor must
permit exact prescribed biomass loss without silently forgiving loss or
restoring front retreat. These are planned changes, not production behaviour;
see the [mortality revision](../../implementation-docs/age-dependent-absorber-death-revision.md).

An interior fungal cavity does not directly change new-shell distance weights.
It can affect maintenance and root contact. Losing every root contact disables
bilateral exchange without immediate fungal death; a finite distance does not
establish connectivity or restore contact. Absence/death gates remain.

Zero growth gives zero construction-related transport charge. Do not evaluate
an empty-shell ratio. Shell width is the current step's realised growth, with
no additional width parameter. It shrinks under timestep refinement as a
numerical construction increment; qualify cumulative costs over matched
physical durations as well as spatial distance integration.

Keep route/demand construction, the replaceable energetic law and the shared
resource-budget transaction separate. Fund construction and transport together;
do not debit transport after spending all available C or charge delivered
construction carbon twice. Payer, exact delivery basis and integration timing
still require design in [ticket #99](https://github.com/isaachcamp/mycormarl/issues/99).

The fungal generator checks containment, mixed overlap, quadrature refinement
and a closed-form radial mean. The plant generator checks depth-allocation
weights, distance bounds and agreement between analytically reduced and
independent tensor quadrature. Linked figure notes and JSON record dimensions
and results. These checks verify the illustrations, not physiological parameters
or production resource accounting.
