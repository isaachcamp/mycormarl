# Architectural decision ownership

Core owns model/algorithm contracts ADR-0001–0004 and ADR-0009–0012, and the
cross-repository decisions ADR-0014/0015. Keep the original IDs, dates and status.
Research diagnostic/endpoint decisions ADR-0005–0008 and ADR-0013 live in
[REDbyRBD](https://github.com/isaachcamp/REDbyRBD/tree/main/docs/adr).
Its index records original revisions and the superseded status of ADR-0005.
Each repository uses CONTEXT.md and docs/adr; new evidence never relabels old
source identities. See ../reproducibility/version-control.md.

## Accepted mortality and transport assumptions

- [ADR-0021](0021-treat-internal-transport-capacity-as-non-limiting.md): internal
  transport capacity is non-limiting; thinning does not add a throughput cap or
  compensatory carbon surcharge. Energetic transport costs remain separate.
- [ADR-0022](0022-separate-fine-root-mortality-from-persistent-backbone-connectivity.md):
  fine-root cohorts may die completely, with an implicit persistent conducting
  backbone and exchange-available historical footprint. Fungal contact loss
  disables bilateral trade without immediately killing the organisms.

These are accepted modelling assumptions for planned work, not implemented
replacements for ADR-0004. The fungal mortality geometry and contact timing
remain explicit design gates in [specification #89](https://github.com/isaachcamp/mycormarl/issues/89).
