# ADR-0003: Require valid physical actions at the environment boundary

**Status:** Accepted

## Context

PPO computes a likelihood for the action generated from its latent sample. If
`BaseMycorMarl` clips, sanitises, projects, or renormalises that action, the
executed action may differ from the action scored by PPO. Silent repair would
therefore hide interface defects and corrupt the policy-gradient ratio.

JAX runtime value checks inside the compiled environment loop would also add
complexity to a path whose normal callers can satisfy the constraints by
construction.

## Decision

Every caller must provide a valid physical action before invoking
`BaseMycorMarl`.

- PPO uses the agreed sigmoid and centred-simplex transforms.
- Heuristics, tests, examples, and qualification scripts use one shared public
  action-construction helper.
- `BaseMycorMarl` consumes the physical action unchanged.
- The compiled environment does not clip, replace non-finite values, project
  to the simplex, renormalise, or provide a fallback allocation.
- Optional validation may be provided as an explicit debug/test helper outside
  the compiled training path. Invalid input is a contract violation, not a
  recoverable environment event.

Because the environment never repairs an action, `Transition` does not need an
`action_was_projected` field.

## Consequences

- The PPO likelihood always describes the action the environment executes.
- Invalid-action bugs fail focused contract tests instead of being hidden.
- All repository action call sites must migrate to the shared constructor.
- Direct external callers are responsible for satisfying the documented
  bounds and simplex invariant.

