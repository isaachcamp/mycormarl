# Three canonical plant-C / fungus-P limitation conditions

## Executive conclusion

Conditions 084, 125, and 135 are the only three rows in the canonical
360-condition screen classified as plant C-limited and fungus P-limited. They
share a pronounced exchange asymmetry: plant→fungus trade is low (0.063–0.093),
whereas fungus→plant trade is high (0.655–0.787). The fungus therefore exports
roughly 8–10 times more of its trade-controlled resource than the plant exports
of its own.

This asymmetry provides a coherent mechanism, but not a sufficient causal
explanation. Fungus→plant trade removes fungal free P before fungal allocation,
while the plant receives that P for a later pool update. That can make fungal P
the limiting input while relieving plant P limitation and exposing plant C as
the remaining bottleneck. Low plant→fungus trade simultaneously limits the
fungus's C income, creating a coupled C/P feedback. The screen is a static
multivariate design, so the interpretation requires matched follow-up
experiments before it can be called causal.

## Parameter combinations and observed regimes

| Condition | Plant κC | Fungus κC | Fungus γP | Initial P (µM) | Initial fungus biomass | Plant→fungus trade | Fungus→plant trade | Plant P fraction | Plant C fraction | Fungus P fraction | Fungus C fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 084 | 0.0769 | 0.0249 | 1.565 | 0.370 | 32.92 | 0.0931 | 0.734 | 0.466 | 0.534 | 0.583 | 0.417 |
| 125 | 0.0369 | 0.6453 | 0.610 | 0.295 | 37.91 | 0.0628 | 0.655 | 0.425 | 0.575 | 0.877 | 0.123 |
| 135 | 0.3622 | 0.3791 | 1.851 | 0.615 | 80.97 | 0.0780 | 0.787 | 0.196 | 0.804 | 0.678 | 0.322 |

The fractions are calculated only over realised-growth timesteps; the screen
calls a resource limiting when its fraction exceeds 0.5
([analysis code](../../scripts/analyse_resource_pressure_canonical_screen.py#L168-L182)).
The underlying records are [084](../../outputs/resource-pressure-canonical-screen/result-bundle-checkpoints/condition-084.json), [125](../../outputs/resource-pressure-canonical-screen/result-bundle-checkpoints/condition-125.json), and [135](../../outputs/resource-pressure-canonical-screen/result-bundle-checkpoints/condition-135.json).

## Why the trade asymmetry can produce this regime

The implementation converts each trade action to a continuous-time fraction
and applies it to the owner's post-maintenance free pool. Plant trade is taken
from plant C and fungal trade from fungal P
([`base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py#L368-L387)).
The pools are then updated before growth: plant C is reduced by plant trade and
fungal P by fungal trade
([`base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py#L415-L429)).
The transfers are credited in the subsequent state update: fungal P enters the
plant P pool, and plant C enters the fungal C pool
([`base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py#L442-L450)).

This gives four linked effects:

1. **Direct fungal P drain.** A high fungus→plant fraction removes a large
   share of fungal free P before fungal growth. This is especially consequential
   when initial P is low or fungal γP is high, because growth requires more P per
   unit structural biomass.
2. **Plant P relief.** The same transfer supplies plant P. If plant P ceases to
   be the smaller growth resource, the plant's C requirement becomes the active
   bottleneck; this is a shift in the identity of the limiting resource, not
   necessarily evidence that P transfer harmed the plant.
3. **Weak fungal C replenishment.** Low plant→fungus trade sends relatively
   little C to the fungus. Fungal C stress can reduce fungal growth and future
   uptake capacity, making the fungus less able to replace exported P.
4. **Maintenance competition.** Maintenance is paid before trade and growth,
   with C and P requirements proportional to biomass and κC/κP
   ([`_pay_maintenance`](../../mycormarl/mycormarl/environments/base_mycor.py#L645-L673)).
   A high initial fungal biomass therefore creates a large early standing cost;
   condition 135 (80.97 biomass units) is the clearest example.

The realised fluxes support this interpretation: cumulative fungal P exports
are 0.626, 0.437, and 0.620 mg in 084, 125, and 135 respectively—about 79%,
89%, and 87% of each condition's direct fungal P uptake. C exports from the
plant are only 0.0569, 0.0569, and 0.0326 g. These are outcomes of the fixed
screen policies, not imposed conservation targets, and show why the action
fractions translate into a large net P flow toward the plant.

Growth itself is limited by the lower γ-normalised C or P allocation
([allocation code](../../mycormarl/mycormarl/environments/base_mycor.py#L676-L718));
therefore a high P export can switch the plant away from P limitation while the
fungus remains P-constrained.

## Temporal evidence

The regime develops over time rather than appearing uniformly from reset.
Across all three conditions, the plant is almost entirely C-limited during the
first quarter of realised growth. Fungal P limitation generally appears later:

| Condition | Fungal pattern over the run | First realised fungal P-limited timestep |
|---|---|---:|
| 084 | 21% P in the first quarter → 69% in the middle → 75% in the final quarter | Day 14.0 |
| 125 | 53% P in the first quarter → 98% in the middle → 90% in the final quarter | Day 16.7 |
| 135 | No realised fungal growth initially; then 91% P in the middle, falling to 30% late as C limitation returns | Day 23.6 |

The high-AMF case therefore does **not** show immediate fungal P limitation. It
first experiences a large maintenance/available-C burden, then enters a strong
fungal-P-limited phase, and eventually becomes predominantly C-limited again.
This ordering is consistent with a dynamic depletion/feedback process, but the
three cases also differ in κC, γP, initial P, and trade values.

## What is and is not established

The trade pattern is unusual relative to the full screen: the three plant→fungus
values fall around the 9th–29th percentiles, while the fungus→plant values fall
around the 52nd–96th percentiles. Nevertheless, 56/360 conditions satisfy the
simple rule plant trade ≤0.1 and fungal trade ≥0.65, so the asymmetry is not
sufficient by itself.

The canonical protocol explicitly describes this as a multivariate landscape;
it recommends matched one- or two-factor follow-ups for causal boundaries
([screen plan](../implementation-docs/resource-pressure-static-screen-plan.md#interpretation-limits)).
The most informative next test would hold each condition's non-trade factors
fixed and perturb the two trade fractions independently, while recording
windowed C/P limitation, free-pool trajectories, actual trade fluxes, biomass,
and uptake. A second test should vary initial AMF biomass at fixed trade and
γP to determine whether it delays the onset of fungal P limitation as suggested
by condition 135.
