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

- PPO constructs the implemented per-day Rate action using softplus transforms
  and scores its sampled Gaussian latents. The former sigmoid/simplex mapping
  was replaced by the rate migration in ADR-0012.
- Heuristics, tests, examples, and qualification scripts use one shared public
  action-construction helper.
- `BaseMycorMarl` consumes the physical action unchanged.
- The compiled environment does not clip, replace non-finite values, project
  an action, renormalise, or provide a fallback allocation.
- Optional validation may be provided as an explicit debug/test helper outside
  the compiled training path. Invalid input is a contract violation, not a
  recoverable environment event.

Because the environment never repairs an action, `Transition` does not need an
`action_was_projected` field.

The public non-policy `rate_action` constructor has a separate, permissive
input contract: negative and non-finite input values become zero rates before
the environment is called. This construction-time sanitisation is not an
environment repair and is not part of PPO's latent transform. Scientific
callers requiring invalid-input rejection must check their raw inputs before
using that constructor. The unchanged-command guarantee begins at the
environment boundary; it does not promise that the constructor rejects
malformed input.

## Consequences

- The PPO likelihood always describes the action the environment executes.
- Invalid environment inputs are contract violations; a caller must not rely
  on compiled execution to diagnose them.
- All repository action call sites must migrate to the shared constructor.
- Direct external callers are responsible for satisfying the documented
  finite, non-negative per-day rate contract. No sum-to-one constraint applies.

## Evidence

- [Action constructor](../../mycormarl/mycormarl/actions.py) and
  [constructor contracts](../../tests/test_actions.py).
- [PPO rate transform](../../mycormarl/mycormarl/algos/ppo.py) and
  [environment action contracts](../../tests/test_base_mycor_refactor.py).

