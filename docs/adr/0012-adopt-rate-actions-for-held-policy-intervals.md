# ADR-0012: Adopt rate actions for held policy intervals

## Status

Partially superseded by
[ADR-0020](0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md)
on 22 October 2026. Its three-rate model replaces the four-component/storage
contract below. The held-rate physical-time principle remains accepted.

## Context

The previous Physical action was `[trade, growth, reproduction, reserve]`: a
trade fraction plus a growth/reproduction/reserve simplex. When one policy
decision is held over several numerical substeps, applying a fraction to each
updated free pool makes biological change depend on numerical resolution.
The one-day fixed-policy numerical qualification exposed this coupling through
uptake, biomass-derived geometry, and subsequent uptake feedback.

## Original decision (four-component/storage portions superseded)

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

The original four-rate source baseline was `1ec2804`; its
[qualification archive](../qualification/rate-action-49/phosphate-numerical/phosphate-numerical-qualification.md)
retains that model's provenance. The original tracking
[issue #49](https://github.com/isaachcamp/mycormarl/issues/49) remains open;
its tracker state is separate from this source-level implementation status.

The [three-rate specification #79](https://github.com/isaachcamp/mycormarl/issues/79)
was subsequently implemented. ADR-0020 records simultaneous trade/biology,
stoichiometric reproductive output, implicit retention and the separate static
experiment contract. Current constructor and interval tests exercise that
successor model; historical four-rate outcomes are not equivalent evidence.
