# ADR-0012: Adopt rate actions for held policy intervals

## Status

Accepted for implementation through a follow-up issue.

## Context

The current Physical action is `[trade, growth, reproduction, reserve]`: a
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
