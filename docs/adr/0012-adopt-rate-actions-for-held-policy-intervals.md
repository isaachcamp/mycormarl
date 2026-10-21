# ADR-0012: Adopt rate actions for held policy intervals

## Status

Accepted; the four-component Rate action is implemented. The historical
fraction/simplex context below describes the model it replaced. This record
does not declare the later three-rate proposal implemented or supersede the
current storage contract.

## Context

The previous Physical action was `[trade, growth, reproduction, reserve]`: a
trade fraction plus a growth/reproduction/reserve simplex. When one policy
decision is held over several numerical substeps, applying a fraction to each
updated free pool makes biological change depend on numerical resolution.
The one-day fixed-policy numerical qualification exposed this coupling through
uptake, biomass-derived geometry, and subsequent uptake feedback.

## Decision

Replace the Physical action with a **Rate action** whose four components are
non-negative first-order rates in `d^-1` for trade, growth, reproduction, and
storage.

- Growth, reproduction, and storage compete as rates of free-pool outflow;
  their total rate is the outflow hazard and their relative rates split that
  outflow.
- Storage initially means retention in the existing free C/P pools. It does
  not create a new storage compartment or mobilisation process.
- Newly acquired soil P, photosynthate, and received trade participate in the
  held rate from the next numerical substep.
- The policy-facing interval wrapper holds a Rate action constant while the
  numerical environment integrates it at its own timestep.
- Validity and PPO likelihood alignment remain requirements at the environment
  boundary, preserving the intent of ADR-0003.

## Consequences

This is an intentional breaking change to action units, validation, policy
transforms, static controls, transition provenance, qualification fixtures,
and trained-policy compatibility. It removes the allocation simplex and must
be introduced atomically with rate-aware tests and qualification.

## Implementation evidence and later work

The [action constructor](../../src/mycormarl/actions.py),
[policy-interval wrapper](../../src/mycormarl/environments/policy_interval.py),
[rate/interval tests](../../tests/test_decision_intervals.py), and
[qualification report](../qualification/phosphate-numerical-qualification.md)
record the implemented four-rate model. The original tracking
[issue #49](https://github.com/isaachcamp/mycormarl/issues/49) remains open;
its tracker state is separate from this source-level implementation status.

The approved [three-rate specification #79](https://github.com/isaachcamp/mycormarl/issues/79)
is on execution hold. Per the 21 October 2026 instruction, its successor ADR
will be written when that migration is implemented, through
[final integration ticket #85](https://github.com/isaachcamp/mycormarl/issues/85).
Until then, this remains the record of the implemented four-rate contract.
