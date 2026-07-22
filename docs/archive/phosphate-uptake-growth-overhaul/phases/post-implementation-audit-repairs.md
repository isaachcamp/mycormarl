# Post-implementation audit repairs

## Scope

A delegated whole-plan audit after P7 found integration defects outside the
well-tested phosphate kernels. This record distinguishes the original P7 gate
from the subsequent repair and verification cycle.

## Repairs

- [v] Limit plant growth to remaining biomass capacity before charging
  stoichiometric C and P; test cap-active structural-P accounting.
- [v] Restore `scripts/phosphate_examples.py` and exercise it and `main.py` as
  subprocess entry points.
- [v] Standardise PPO and environment identifiers on `plant` and `fungus`,
  expose `env.agents`, and exercise a complete PPO update. Remove the obsolete
  evaluation package rather than adapting it to the replacement state model.
- [v] Add typed `PPOConfig` and `scripts/train_ppo.py`; verify a two-step CLI
  training job writes policy parameters.
- [v] Add explicit `mixed`, `plant-only`, and `fungus-only` consumer modes.
  Independent modes retain the two-agent API and permanently mask the absent
  partner's state, actions, trade, geometry, uptake, and reward.
- [v] Validate initial pools, maintenance rates, death fractions, plant
  biomass capacity, and photosynthetic controls before state construction.
- [v] Correct the P6 horizon description, emit maximum cellwise regime weight
  and the diffusion CFL ceiling, regenerate qualification artifacts, and mark
  the completed axisymmetric migration as resolved.
- [v] Make death absorbing for initially active organisms and suppress all
  later biology, geometry, uptake, trade, and reward while retaining biomass
  as historical state.
- [v] Return flat `(4,)` observations matching the declared observation space.
- [v] Add coupled plant/fungal uptake, shares, final soil inventory, and free-P
  pools to the P6 convergence outputs and selection gate.
- [v] Remove the unused `reward_scaling` configuration field.

## Verification

- Regression tests first failed at each audited boundary.
- Focused repair checks cover absorbing death, observation shape, PPO, and
  coupled qualification accounting.
- The final complete suite passed `221 passed, 1 warning`; bytecode compilation
  and `git diff --check` also passed.
- The PPO launcher completed a deterministic two-step smoke job with seed 42.
- The P6 qualification matrix and target benchmark regenerated successfully.
  The selected grid remains `0.1 cm`; after adding coupled P outputs, no
  coarser timestep passed, so `0.025 day` is the finest-tested fallback.
- The warning is the unchanged upstream JAXopt deprecation.

## Remaining scientific limitations

Maintenance-P fate, empirical calibration, local colony age, and sub-cell
sparse root–fungus interference remain explicitly unresolved. The repair does
not expand the scientific validity claimed by P6.
