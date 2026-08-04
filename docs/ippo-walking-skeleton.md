# Independent PPO walking skeleton

MycorMARL provides one independent actor–critic for `plant` and one for
`fungus`. This is a development walking skeleton for checking interfaces and
numerical execution. A successful smoke run is not evidence of convergence,
cooperation, robustness, or biological optimality.

## Environment interface

`BaseMycorMarl` always exposes the fixed agent mapping `plant`, `fungus`.
`EnvConfig.consumer_mode` accepts `mixed`, `plant-only`, and `fungus-only`.
In a single-consumer mode the absent counterpart remains in the mapping but is
non-operational, with zero observation, action effects, reward, biomass, pools,
geometry, and uptake.

Each operational actor receives a finite `float32` vector in `[0, 1]` with
features in this order:

1. biomass relative to its species-specific reference;
2. free C relative to structural C need;
3. free P relative to structural P need;
4. most recently received trade relative to next-step maintenance need;
5. whether a living bilateral association is present.

The received-trade feature is reconstructed from `State.plant_last_p_received`
or `State.fungus_last_c_received`; there is no separate observation-only
memory. Dead and absent agents receive the all-zero vector.

## Physical actions and transition order

The environment accepts a Physical action
`[trade, growth, reproduction, reserve]`. Trade is independently bounded in
`[0, 1]`; growth, reproduction, and reserve are non-negative and sum to one.
Non-policy callers should construct actions with
`mycormarl.actions.physical_action`. PPO maps one Gaussian trade latent through
a sigmoid and two Gaussian allocation latents through a centred simplex
transform. `BaseMycorMarl` executes valid inputs unchanged.

Each step pays automatic C and P maintenance before learned allocation.
Maintenance deficit can remove biomass and cause absorbing biological death.
If either partner dies while paying maintenance, bilateral trade is cancelled;
otherwise outgoing trade is removed before allocation. Growth, reproduction,
and reserve are then applied separately to the remaining C and P. Incoming
trade, photosynthate, and soil P uptake become available only on the next step.

## Transition and PPO boundaries

Every completed environment step returns a typed, algorithm-independent
`Transition` for both agents under `info["transitions"]`. It records requested
and realised actions, operational status at beginning and end,
whether allocation and trade executed, administrative truncation, and the
final pre-reset observation.

The PPO layer converts those facts into actor/critic validity masks and
separate bootstrap and GAE-trace controls. Biological death terminates an
agent and uses zero bootstrap. Administrative truncation bootstraps from the
final pre-reset observation and stops the GAE trace. Already dead or absent
padding is excluded from actor and critic learning.

## Training and saved policies

Run a reproducible JIT-compiled development job with:

```bash
uv run python scripts/train_ppo.py \
  --mode mixed \
  --total-timesteps 256 \
  --num-steps 128 \
  --num-envs 1 \
  --seed 42
```

Use `--mode plant-only` or `--mode fungus-only` for the corresponding smoke
checks, and `--num-envs 2` to exercise vectorised training. The output bundle
contains both policy parameter trees plus policy-format, actor-interface, and
environment-state schema versions. Load it through
`mycormarl.policy_artifacts.load_policy_artifact`; unversioned raw parameter
trees and incompatible versions fail explicitly.

This migration changed the action meaning, observation shape, policy tree,
`Trajectory`, and environment `State`. Regenerate all earlier policy and saved
environment-state artifacts; no automatic conversion is safe because the old
and new action vectors have the same shape but different meanings.
