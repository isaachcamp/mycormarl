# Architectural decision ownership

Core owns model/algorithm decisions and the cross-repository ownership and
evidence contracts. Preserve existing IDs and original decision context;
record implementation updates or supersession explicitly rather than
renumbering records or relabelling historical evidence. New ADRs 0016–0019
record existing model decisions retrospectively, with their recording date.

| ADR | Decision |
|---|---|
| [0001](0001-keep-environment-independent-of-ppo.md) | Keep the environment independent of PPO |
| [0002](0002-use-a-typed-transition-contract.md) | Typed, algorithm-independent transition facts |
| [0003](0003-require-valid-physical-actions-at-the-environment-boundary.md) | Valid commands executed unchanged at the environment boundary |
| [0004](0004-derive-uptake-geometry-from-current-biomass.md) | Memoryless geometry reconstructed from living biomass |
| [0009](0009-represent-am-engagement-using-complete-consumer-modes.md) | Reset-level AM association through complete consumer modes |
| [0010](0010-represent-spatial-p-variability-in-the-initial-soil-inventory.md) | Complete initial P fields; prescribed profiles versus equal-inventory variability |
| [0011](0011-separate-the-plant-biomass-guard-from-policy-observation-scaling.md) | Separate numerical biomass guard and observation scale |
| [0012](0012-adopt-rate-actions-for-held-policy-intervals.md) | Implemented four-rate action and held policy intervals |
| [0014](0014-organize-repositories-around-core-and-scientific-studies.md) | Core, research and animation ownership |
| [0015](0015-pin-core-dependencies-and-retain-executable-policy-evidence.md) | Exact core pins, evidence identity and separate exploration retention |
| [0016](0016-use-finite-inventory-buffered-phosphate-transport-and-uptake.md) | Finite buffered soil inventory, diffusion, uptake and competition |
| [0017](0017-represent-absorbers-as-fine-root-discs-and-an-external-hyphal-hemisphere.md) | Fine-root discs, external-hyphal hemisphere and domain clipping |
| [0018](0018-account-for-structural-phosphorus-and-explicit-biological-exports.md) | Structural P, biological exports, death and reset resource accounts |
| [0019](0019-separate-policy-numerical-and-soil-clocks-and-qualify-resolution.md) | Separate clocks, explicit numerical splitting and qualified resolution |

The assumption catalogue, equations, units and parameter provenance belong in
the [model overview](../model-overview.md), [growth model](../growth-model.md),
[phosphate model](../phosphate-model.md) and
[parameter register](../model-parameter-register.md). ADRs explain the
consequential choices, alternatives and limitations, rather than duplicating
every parameter or treating all current assumptions as calibrated facts.

The three-rate allocation/reproduction design has no successor ADR yet.
Its [implementation specification](https://github.com/isaachcamp/mycormarl/issues/79)
remains on execution hold; [ticket #85](https://github.com/isaachcamp/mycormarl/issues/85)
must add that ADR when the migration is implemented. The current record is
ADR-0012; this documentation work does not supersede it or start the migration.

Research diagnostic/endpoint decisions ADR-0005–0008 and ADR-0013 live in
[REDbyRBD](https://github.com/isaachcamp/REDbyRBD/tree/main/docs/adr).
Its index records original revisions and ADR-0005's supersession by ADR-0007.
Research protocols own study-specific endpoints, evaluation and field-sampling
details; link to them rather than creating contradictory core copies.

Each repository uses `CONTEXT.md` and `docs/adr`. New evidence never relabels
old source identities. See [source version conventions](../reproducibility/version-control.md).
