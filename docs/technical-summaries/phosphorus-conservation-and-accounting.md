# Phosphorus conservation and accounting

## Scope

This summary explains what “conservative” means in the phosphate implementation,
from transfers between soil cells to the wider soil–plant–fungus system. It
distinguishes an exact numerical property from a complete biological accounting
claim. Carbon accounting is outside scope.

## Executive summary

The code is conservative at more than one level, but it is **not yet valid to
call every trajectory a closed, whole-system phosphorus balance**.

| Level | Conservation contract | Current status |
|---|---|---|
| Soil diffusion | P leaving one cell enters its neighbour | Conservative on the closed domain |
| Root–fungus competition | Accepted uptake cannot exceed the cell inventory | Conservative cell by cell |
| Soil-to-organism uptake | Soil P removed in µmol is converted once and credited to free organism P in mg | Conservative |
| Trade | Fungal P transferred out is added to plant P | Conservative internal transfer |
| Growth | Free P becomes structural P represented by biomass × \(\gamma_P\) | Conservative if inferred structural P is included |
| Reproduction | P leaves the living pools | Accounted as cumulative export |
| Mortality | Structural P leaves with lost biomass | Accounted as cumulative loss |
| Maintenance | Free P is spent, but has no explicit receiving pool or loss counter | **Accounting gap** |

The P-accounting stringency therefore serves a broader purpose than checking
diffusion. It tests whether every modeled P transfer has an equal destination or
an explicit export. At present, the extended whole-system claim is deliberately
restricted to trajectories with maintenance P use disabled.

## Where conservation sits in the pipeline

```text
labile P in soil cells (µmol)
    │
    ├─ diffusion: equal and opposite internal-face transfers
    │
    └─ uptake: simultaneous root/fungus requests capped by cell inventory
             │
             └─ exact µmol→mg conversion
                    │
                    ├─ plant free-P pool
                    └─ fungus free-P pool
                           │
                           ├─ trade: internal transfer
                           ├─ growth: implicit structural P
                           ├─ reproduction: recorded export
                           ├─ mortality: recorded loss
                           └─ maintenance: currently unrecorded destination
```

The canonical soil state is an amount, `soil_labile_p` in µmol P per cell, while
organism pools and biological loss/export counters use mg P
([`state.py`, lines 15–29](../mycormarl/mycormarl/state.py#L15-L29)).

## 1. Conservative diffusion between cells

For an internal face between cells \(i\) and \(j\), the update has the form

\[
M_i^{n+1}=M_i^n-F_{ij}\Delta t,\qquad
M_j^{n+1}=M_j^n+F_{ij}\Delta t.
\]

Consequently, the face contribution cancels when the two cells are summed:

\[
\Delta M_i+\Delta M_j=0.
\]

The implementation constructs each radial and vertical transfer once, subtracts
it from one cell, and adds the identical value to its neighbour
([`phosphate_diffusion.py`, lines 117–132](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L117-L132)).
There are no exterior-face flux terms, so the represented soil boundary is
no-flux. Thus, with uptake disabled,

\[
\sum_i M_i^{n+1}=\sum_i M_i^n
\]

apart from floating-point round-off.

This is the conventional finite-volume conservation mechanism: a shared face
flux is counted with opposite signs in adjacent control volumes. The explicit
CFL ceiling is a separate requirement. Conservation alone does not prevent a
large timestep from producing negative cell amounts; the timestep bound protects
positivity and stability
([`phosphate_diffusion.py`, lines 135–155](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L135-L155)).

The tests check both the domain sum and non-negativity at the configured safety
limit ([`test_phosphate_diffusion.py`, lines 270–286](../tests/test_phosphate_diffusion.py#L270-L286)),
and repeat the balance through the subcycled environment loop
([`test_phosphate_diffusion.py`, lines 410–424](../tests/test_phosphate_diffusion.py#L410-L424)).

## 2. Conservative competition for finite cell inventory

Root and fungal uptake requests are first calculated independently, including
the sparse/continuous blend. Competition is then applied once against the shared
labile amount in each cell
([`phosphate_uptake.py`, lines 325–354](../mycormarl/mycormarl/soil/phosphate_uptake.py#L325-L354)).

For cell \(i\), let \(M_i\) be available P and \(R_i,F_i\) the two requests:

\[
A_i=\min(M_i,R_i+F_i).
\]

When demand is non-zero, the plant receives

\[
A_{R,i}=A_i\frac{R_i}{R_i+F_i},
\]

and the fungus receives the remainder,

\[
A_{F,i}=A_i-A_{R,i}.
\]

The remaining soil inventory is \(M'_i=M_i-A_i\). Therefore,

\[
M_i-M'_i=A_{R,i}+A_{F,i}
\]

by construction. Assigning fungal uptake as the remainder is intentional: it
avoids a second independently rounded multiplication and gives the strongest
available finite-precision closure
([`phosphate_uptake.py`, lines 357–381](../mycormarl/mycormarl/soil/phosphate_uptake.py#L357-L381)).
The JIT-compiled symmetry and cellwise balance contract is tested in
[`test_continuous_phosphate_uptake.py`, lines 141–153](../tests/test_continuous_phosphate_uptake.py#L141-L153).

## 3. Conservative transfer from soil to free organism pools

Accepted root and fungal amounts are summed, converted from µmol P to mg P, and
credited to the corresponding free-P pools in the same state replacement that
stores the reduced soil inventory
([`soil.py`, lines 127–151](../mycormarl/mycormarl/soil/soil.py#L127-L151)).
In a common unit, the transaction satisfies

\[
\mu\sum_i(M_i-M'_i)
=\Delta P_{\mathrm{plant,free}}+\Delta P_{\mathrm{fungus,free}},
\]

where \(\mu\) is the exact conversion used by the code from µmol P to mg P.

The end-to-end test explicitly forms “soil P + plant free P + fungus free P”
before and after a step with no biological P allocation
([`test_environment_phosphate_uptake.py`, lines 260–282](../tests/test_environment_phosphate_uptake.py#L260-L282)).

## 4. The extended biological P ledger

Free organism pools alone are insufficient once growth occurs. Structural P is
not stored in a separate state array; it is inferred from dry biomass:

\[
P_{\mathrm{struct},o}=\gamma_{P,o}G_o,
\]

where \(G_o\) is plant or fungal biomass and \(\gamma_{P,o}\) is its structural
P cost in mg P g\(^{-1}\) dry biomass. Growth deducts exactly
\(G_{\mathrm{growth}}\gamma_P\) from the free-P pool
([`base_mycor.py`, lines 487–505](../mycormarl/mycormarl/environments/base_mycor.py#L487-L505);
[`base_mycor.py`, lines 589–603](../mycormarl/mycormarl/environments/base_mycor.py#L589-L603)).

The diagnostic extended inventory is therefore

\[
\begin{aligned}
P_{\mathrm{extended}}={}&
\mu\sum_i M_i
+P_{p,\mathrm{free}}+P_{f,\mathrm{free}}\\
&+\gamma_{P,p}G_p+\gamma_{P,f}G_f\\
&+P_{\mathrm{mortality,cumulative}}
+P_{\mathrm{maintenance,cumulative}}
+P_{\mathrm{reproduction,cumulative}}.
\end{aligned}
\]

This is the quantity used by the coupled qualification scenario
([`phosphate_qualification.py`, lines 260–275](../scripts/phosphate_qualification.py#L260-L275)).
Fungal-to-plant P trade is internal: the same `fungus_p_trade_out` is added to
the plant pool and subtracted from the fungal pool
([`base_mycor.py`, lines 287–295](../mycormarl/mycormarl/environments/base_mycor.py#L287-L295);
[`base_mycor.py`, lines 349–357](../mycormarl/mycormarl/environments/base_mycor.py#L349-L357)).
Reproduction, mortality, and paid maintenance P are not treated as disappearing
silently; cumulative counters retain their amounts in the ledger
([`base_mycor.py`, lines 358–373](../mycormarl/mycormarl/environments/base_mycor.py#L358-L373)).

The extended-balance qualification test verifies closure to relative tolerance
\(10^{-5}\)
([`test_phosphate_qualification.py`, lines 263–279](../tests/test_phosphate_qualification.py#L263-L279)).
Its fixture sets both maintenance coefficients to zero
([`phosphate_qualification.py`, lines 43–61](../scripts/phosphate_qualification.py#L43-L61);
[`phosphate_qualification.py`, lines 275–279](../scripts/phosphate_qualification.py#L275-L279)).
This isolates growth and uptake sensitivity; maintenance-active closure is
covered separately by focused environment tests.

## 5. How paid maintenance P closes the extended ledger

For both organisms, `maint_p_used` is deducted automatically from the
start-of-step free pool
([`base_mycor.py`, lines 519–545](../mycormarl/mycormarl/environments/base_mycor.py#L519-L545);
[`base_mycor.py`, lines 617–643](../mycormarl/mycormarl/environments/base_mycor.py#L617-L643)).
The amount actually paid is accumulated in the species-specific
`cumulative_*_p_maintenance_loss_mg` counter. Unmet demand is not added to this
counter; it drives deterministic biomass loss instead. Treating paid P as an
external maintenance/turnover loss closes the extended diagnostic ledger
without claiming recycling into a biological or soil compartment.

## Relationship to the literature

- Eymard, Gallouët and Herbin’s finite-volume treatment provides the numerical
  foundation for local conservation: fluxes are balanced over control-volume
  faces. The implementation adopts this standard shared-face cancellation
  pattern. DOI:
  [10.1016/S1570-8659(00)07005-8](https://doi.org/10.1016/S1570-8659(00)07005-8).
- Schnepf and Roose model phosphate removal by roots and external mycorrhizal
  hyphae, including a volumetric hyphal sink. The present implementation adapts
  that uptake setting to a finite cell inventory and makes root–fungus
  competition an explicit conservative transaction. DOI:
  [10.1111/j.1469-8137.2006.01771.x](https://doi.org/10.1111/j.1469-8137.2006.01771.x).
- Vance, Uhde-Stone and Allan review the biological distinction between P
  acquisition, internal use, remobilisation, and conservation. That distinction
  motivates keeping soil, free, structural, exported, and lost P conceptually
  separate rather than treating “organism P” as one undifferentiated pool. DOI:
  [10.1046/j.1469-8137.2003.00695.x](https://doi.org/10.1046/j.1469-8137.2003.00695.x).

## Assumptions and limitations

- Soil exterior boundaries are closed; no fertiliser input, leaching, or
  prescribed boundary concentration enters the balance.
- Mortality P is treated as lost from the represented system, not recycled into
  soil.
- Reproduction P is an export, not a persistent offspring pool.
- Structural P is inferred from biomass and fixed \(\gamma_P\), rather than
  stored independently or allowed variable tissue stoichiometry.
- Sorbed and solution P are combined in the canonical labile amount through
  instantaneous linear buffering; diffusion redistributes this total labile
  inventory using solution concentration as the driving potential.
- Numerical conservation is subject to floating-point round-off and the stated
  test tolerances.
- Paid maintenance P is treated as an external loss rather than recycled into
  soil or another biological compartment.

## Practical interpretation

If “conservative” appears beside the diffusion kernel, read it narrowly as
closed-domain redistribution without creation or destruction. If it appears
beside competing uptake, read it as exact allocation of a finite cell inventory.
For a whole simulation, use the extended ledger and state its boundary and
export conventions explicitly. Under the current code, add the qualification
that maintenance P has an unresolved fate.
