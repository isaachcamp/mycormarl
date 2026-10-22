# ADR-0018: Account for structural phosphorus and explicit biological exports

**Status:** Accepted; retrospective record of implemented accounting and reset
contracts, recorded 21 October 2026.

The reproductive objective subsequently selected by
[ADR-0020](0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md)
is realised stoichiometric output with cost-scaled paired exports. The
structural, mortality, loss and reset accounting boundaries here remain accepted.

## Context and decision

Soil-only conservation cannot establish whole-system P accounting: growth
moves P into structure, trade moves P between organisms, and biological losses
leave free or structural pools. Keep an explicit common-unit ledger rather
than silently discarding those terms or inventing a recycling compartment.

- Growth uses fixed organism-specific essential C/P stoichiometry. Infer
  structural P as stored dry biomass times `gamma_p`; charge only realised
  growth. In particular, plant growth blocked by the numerical biomass guard
  spends neither structural C nor P.
- Pay automatic standing-biomass C maintenance and optional irreversible
  free-P loss from start pools. A payment deficit removes biomass according
  to the larger C- or P-equivalent deficit, capped at active biomass.
- Interpret `kappa_p` as irreversible free-P loss, not structural
  immobilisation or a measured physiological P-maintenance demand. Both
  organisms default to zero; a positive value is a declared sensitivity.
  Record only paid P in the legacy-named maintenance-loss counters, and
  structural P removed with actual biomass loss in mortality-loss counters.
- Treat reproductive P as an explicit export. Keep all three export counters
  separate; none creates soil P, litter, offspring or a receiving pool.
  At adoption this accounting boundary did not select a reproductive objective;
  ADR-0020 subsequently superseded the earlier Cobb–Douglas reward.
- Make biological death absorbing after the post-maintenance historical-
  maximum biomass threshold. Suppress later activity while retaining residual
  stored biomass and free pools for accounting. Do not discard or recycle
  those residual amounts merely because the organism is non-operational.
- At reset, an unspecified free pool funds one automatic payment:
  `initial_biomass * kappa_c * dt` for C and
  `initial_biomass * kappa_p * dt` for P. Zero demand gives zero free stock.
  A numeric initial pool is an explicit scenario override; absent partners
  have zero biomass, pools and uptake geometry.

## Trade-offs and consequences

The extended P ledger includes soil, both free pools, implicit structural P
and cumulative mortality, irreversible-loss and reproduction exports. Uptake
and trade are internal transfers and must not be counted again as exports.
The ledger includes retained dead state, whereas uptake geometry includes
only living biomass. Compare pre-reset states with fixed stoichiometry.

This accounts for exported P without pretending to model its subsequent fate.
Litter, mineralisation, remobilisation, demographic reproduction and recovery
would change the system boundary and require new pools, transfers and tests.
Carbon fixation is an external input; atmospheric carbon is not a finite
state pool. Growth currently consumes structural C without an extra
construction-efficiency/growth-respiration cost. Such a cost must remain
distinct from standing-biomass maintenance if introduced.

Deriving reset pools from `dt` avoids an arbitrary structural-biomass-sized
free-resource endowment, but changes initial free inventory when numerical
`dt` changes. Convergence fixtures must therefore pin identical initial
physical pools explicitly. Neither this reset choice nor the current outer
maintenance/death boundary is a claim of timestep-exact physiology.

## Evidence and related decisions

The [model overview ledger](../model-overview.md#phosphorus-accounting) and
[growth model](../growth-model.md) give units and equations. The
[environment](../../src/mycormarl/environments/base_mycor.py) and
[state](../../src/mycormarl/state.py) implement the transactions and
counters. [Resource-transaction tests](../../tests/test_base_mycor_refactor.py),
[reset tests](../../tests/test_labile_phosphate_state.py), and
[uptake/mortality tests](../../tests/test_environment_phosphate_uptake.py)
protect these contracts. See
[ADR-0009](0009-represent-am-engagement-using-complete-consumer-modes.md) for
association modes and
[ADR-0011](0011-separate-the-plant-biomass-guard-from-policy-observation-scaling.md)
for the numerical growth guard.
