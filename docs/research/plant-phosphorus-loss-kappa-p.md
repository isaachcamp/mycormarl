# Evidence review for plant phosphorus loss (`kappa_p`)

**Research date:** 4 August 2026
**Scope:** A literature-backed or explicitly derived default for the plant
`kappa_p` parameter in MycorMARL.

## Recommendation

Use **`kappa_p = 0 mg P g⁻¹ whole-plant dry biomass d⁻¹`** as the default,
which preserves P conservation in the absence of an explicit loss process.
Use `0.001–0.002 mg P g⁻¹ DM d⁻¹` as small positive sensitivities for
herbivory, leakage, and unrecovered tissue turnover; `0.005` remains an
exploratory high-turnover case.

The selected value is not a measured carrot maintenance rate. No primary
carrot study recovered a constant irreversible P-loss flux, and the available
experiments record minimal identifiable irrecoverable loss. Zero is therefore
the default; positive values remain explicit approximations for small
ecological export pathways omitted by the current model.

This is a recommendation about model semantics, not a claim that real plants
never lose P. Real losses occur mainly with tissue death, abscission, root
turnover, herbivory, and harvest. They should be coupled to the amount of
biomass lost and reduced by P resorption or local litter/root recycling.
Internal retranslocation and biochemical P recycling are not losses.

Test `0`, `0.001`, `0.002`, and `0.005`. The `0.005` case represents unusually
high, poorly resorbed tissue turnover; it is not a carrot-derived default.

## 1. Runtime contract and exclusions

The required contract is

`P_loss = kappa_p * standing whole-plant dry biomass * dt`,

with `kappa_p` in mg P g⁻¹ whole-plant DM d⁻¹. Structural or otherwise frozen
P is already represented by `gamma_p = 1.92 mg P g⁻¹ DM`; growth
immobilization therefore belongs to `growth * gamma_p`, not `kappa_p`.

The following are not valid evidence for this loss term:

- xylem/phloem movement among leaves, roots, and storage root;
- P liberated from senescing cells and retranslocated to living tissues;
- temporary root phosphate efflux followed by reuptake;
- organic-acid or phosphatase exudation that mobilizes soil P;
- P incorporated into new biomass;
- growth respiration or carbon maintenance;
- P removed only at crop harvest.

Unlike carbon, phosphorus is not oxidized to provide maintenance energy.
Cellular turnover can release phosphate for reuse, so there is no necessary
P analogue of maintenance respiration.

## 2. Direct carrot evidence: accumulation and redistribution, not a constant loss

No primary *Daucus carota* study was found that measured a sustained,
irreversible phosphate leak per unit standing whole-plant dry mass.

The closest crop-scale evidence points in the opposite direction. A time-course
study of carrot macronutrient absorption found P still accumulating in both
leaves and storage roots through the crop cycle; at 147 d after sowing the
reported amounts were 1.19 kg P ha⁻¹ in leaves and 2.72 kg P ha⁻¹ in roots
([Fernández-Pérez et al. 2023](https://doi.org/10.17584/rcch.2023v17i3.16508)). A separate
time-course study of cultivar Forto sampled leaves and roots from 40 to 120 d
and found 87.4 mg P plant⁻¹ at the end, 86.1% in the root
([Cecílio Filho et al.](https://acervodigital.unesp.br/handle/11449/75622)).
These measurements demonstrate uptake and changing allocation. They do not
resolve gross uptake from efflux, nor do they identify an irreversible sink.

Nilsson's three-year carrot study found that storage-root P concentration was
not changed by sowing date through 137 d
([Nilsson 1987](https://doi.org/10.1017/S0021859600079508)). This is compatible
with regulated P homeostasis, but again supplies no loss coefficient.

The carrot growth model of Reid explicitly models leaf senescence and dry-mass
remobilization but states that nutrient stresses require future development;
it supplies no P-loss parameter
([Reid 2019](https://doi.org/10.1080/01140671.2019.1588134)). Consequently,
carrot leaf turnover cannot be converted to `kappa_p` without an additional
measurement of green- versus senesced-tissue P and the fate of shed tissue.

## 3. Why efflux and senescence do not justify continuous irreversible loss

Phosphate can cross roots in both directions. In intact maize seedlings,
double-isotope measurements found efflux/influx ratios of about 0.68 at
0.2 µM external P and 0.08 at 2.0 µM. Nevertheless, plants depleted their
solutions, showing that efflux was one component of a regulated bidirectional
flux rather than automatically an irreversible plant loss
([Elliott et al. 1984](https://doi.org/10.1104/pp.76.2.336)). Transferring such
a short-term maize flux to carrot as a biomass-proportional daily sink would
also require root surface area, external P, and proof that released P cannot be
reabsorbed by the plant or its AM fungus.

Senescence likewise contains strong internal recovery. In a primary
Arabidopsis experiment, more than 90% of membrane-lipid P—over one-third of
cellular P—was recycled from leaves during senescence, and manipulating the
responsible phospholipases altered movement of P to young tissues
([Yang et al. 2024](https://doi.org/10.1186/s13059-024-03348-x)). This is not a
carrot resorption percentage, but it directly demonstrates why loss of living
leaf structure cannot be equated with loss of all its P.

P left in abscised carrot leaves or dead fine roots may be unavailable to the
same individual on the modeled timescale. That is a genuine ecological loss
only if the model boundary excludes litter mineralization and root/AMF
recovery. It occurs with turnover, not continuously while biomass is intact.

## 4. What the selected and former values imply

The selected `kappa_p = 0.001` removes `0.001 / 1.92 = 0.000521 d⁻¹`, or
**0.0521% of a structural-P equivalent per day**. At constant biomass, that is
5.21% over 100 d. This is a scale interpretation of the abstraction, not a
claim that structural P itself is removed.

For comparison, the former `kappa_p = 0.002` removes a
structural-P equivalent fraction

`f_irrev = kappa_p / gamma_p = 0.002 / 1.92 = 0.001042 d⁻¹`,

or **0.104% of standing-biomass-equivalent P every day**. At constant biomass,
that is 10.4% of its structural-P equivalent over 100 d and 15.6% over 150 d.
This comparison does not mean structural P is actually removed—the runtime
takes P from the free pool—but it makes the assumed sink interpretable.

For a turnover-based proxy,

`kappa_p = gamma_p * tau * (1 - R_P) * (1 - L_recovery)`,

where `tau` is tissue turnover in g lost g⁻¹ whole-plant DM d⁻¹, `R_P` is the
fraction resorbed internally before tissue loss, and `L_recovery` is the
fraction of residual litter/root P recovered inside the model boundary.

If litter recovery is zero, reproducing the former `0.002` requires:

| Assumed P resorption `R_P` | Required gross tissue turnover `tau` | Approximate tissue fraction turned over in 100 d |
|---:|---:|---:|
| 0% | 0.104% d⁻¹ | 9.9% |
| 50% | 0.208% d⁻¹ | 18.8% |
| 65% | 0.298% d⁻¹ | 25.8% |
| 80% | 0.521% d⁻¹ | 40.6% |

These are scenario calculations, not measured carrot rates. They show that
`0.002` is plausible only as a proxy for appreciable, unrecovered tissue
turnover. It is not supported as a basal biochemical maintenance loss.

A transparent sensitivity envelope can be generated by assuming turnover of
`0.1–0.5% d⁻¹`, P resorption of `50–90%`, and no litter recovery:

`kappa_p = 1.92 * (0.001–0.005) * (0.1–0.5)`

`= 0.000192–0.0048 mg P g⁻¹ DM d⁻¹`.

Rounded, this motivates the proposed **`0–0.005` exploratory envelope**, with
`0–0.002` as the ordinary sensitivity range. The turnover and resorption
bounds are explicit assumptions because matched carrot measurements were not
recovered.

## 5. Preferred model change

The biologically cleaner implementation is:

1. Replace the lumped positive `kappa_p` with measured export processes when
   suitable data become available; retain zero as the conservation limit.
2. When biomass `Delta B` senesces or turns over, release
   `gamma_p * Delta B * R_P` to the plant free-P pool.
3. Send `gamma_p * Delta B * (1 - R_P)` to litter/soil P, or out of the model
   only when that boundary choice is intentional.
4. Allow soil mineralization and root/AMF uptake to recover litter P when those
   processes are represented.
5. Keep growth immobilization as `gamma_p * growth` and harvest export as an
   event, not a daily maintenance coefficient.

This makes P conservation auditable and allows leaf, fine-root, and whole-plant
death to have different resorption and recovery. A constant positive
`kappa_p` can remain as an optional numerical abstraction for unmodeled
leaching or herbivory, but its name and documentation should say so.

## Remaining decision points

1. Decide whether dead tissue remains inside the simulated soil system or is
   exported. This determines whether unresorbed P is delayed recycling or true
   loss.
2. Measure carrot green and naturally senesced leaf P concentrations and leaf
   mass loss over time; similarly quantify fine-root turnover and P recovery.
3. Decide whether AMF recapture of root/rhizosphere P occurs inside the model
   boundary before interpreting root efflux as loss.
4. Until those data exist, retain zero as the default and report any positive
   `kappa_p` as an explicit loss sensitivity.
