# Allocation case studies

21 October 2026. Illustrations of the accepted three-variable sampling and simultaneous-flux design. These are isolated continuous-time calculations, not rollouts of the existing environment or trained policies. No production code changes are made.

## Model and units

Define normalised pools `x=C/gamma_C`, `y=P/gamma_P`. These quantities have growth-output-equivalent units: one unit of either element supports one unit of growth when its complement is present. Both illustrative organisms use the same cost coefficients so that normalised transfers can be shared directly. Real plant/fungus costs can differ and require debit/credit conversion in physical C/P units.

Use `q=min(x,y)`, total biological hazard `kb`, reproductive expenditure share `a`, and reproductive cost multiplier `s`:

```text
G = (1-a) * kb * q
R = a * kb * q / s
biological consumption of each normalised element = G + s*R = kb*q
plant C trade = kt_plant * x_plant
fungus P trade = kt_fungus * y_fungus
```

All fluxes are evaluated in one coupled ODE. Growth and reproduction compete with trade simultaneously. Received trade is continuously available to the recipient. Growth and reproductive output are integrated alongside the free pools and transfers. Reproductive output is exported and does not become parental biomass. No maintenance, photosynthesis, soil uptake, biomass caps, death, spatial grid, or capacity feedback is included. This isolates allocation behavior without deciding the outstanding maintenance/death rules.

Each calculation lasts one day. Commands are held fixed, representing a possible sampled action, not a sample from a fitted actor. All cases have `kb=1/day`. The plant exports C. Except in D, its partner is a passive recipient that retains transferred C and sends nothing back. In D the fungus starts at `(x,y)=(0.2,2)`, uses `kt=kb=1/day`, `a=0.25`, `s=1`, and returns P. Plant starting pools and remaining controls are shown in the figure and CSV.

## Cases

- **A — Balanced pools, no trade.** Starts with `(1,1)`. The matched pools turn over at 1/day: 0.632 units of each are consumed, split into 0.474 growth and 0.158 reproduction. The rest is finite-rate retention.
- **B — Carbon surplus, C trade.** Starts with `(2,0.5)`. P limits biology throughout the day. C export is 1.132, growth is 0.237, and reproductive output is 0.079. P consumption is exactly the same as a no-trade counterfactual with these starting pools and biological rate: trade does not reserve an artificial fraction of P.
- **C — Balanced pools, C trade.** Starts with `(1,1)`. C trade and biological expenditure have equal instantaneous hazards while C is limiting. Each consumes 0.432 C units; 0.135 C remains. Only 0.432 P is consumed, leaving 0.568. That extra P is a genuine complementary-resource surplus.
- **D — Complementary partners.** The plant starts C-rich/P-poor at `(2,0.2)` and the fungus has the reverse imbalance. The plant receives 1.067 P units while transferring 1.067 C units. Plant growth is 0.413, versus 0.095 if both trade rates were zero with the same starting pools. The limiting element changes during the run. Reciprocal transfer is independently commanded, not enforced by a matching-price rule.
- **E — More reproductive investment.** Same starting stocks and trade as C, but `a=0.75`. Growth becomes 0.108 and reproductive output 0.324. Total biological use, trade, and ending pools match C because both biological destinations have the same C:P ratio and this illustration excludes acquisition feedback from growth.
- **F — Double reproductive cost.** Same as C with `s=2`. Resource allocation and growth match C; reproductive output halves from 0.108 to 0.054. The multiplier changes output per resource expenditure, not processing speed.

In the stacked budgets, bar length is initial stock plus received resource, with destinations growth, reproduction, outgoing trade, and ending free stock. Thus D's plant P bar includes incoming trade. Reproductive use is `s * output`, so F has the same reproductive-resource segment as C despite half the reward. The second figure shows free pools and cumulative outputs over time for B–D.

## Reproduction and verification

Run from the repository root:

```sh
.venv/bin/python docs/implementation-docs/action-allocation-case-studies/generate.py
```

The reference calculation uses SciPy DOP853, relative tolerance `1e-11`, absolute tolerance `1e-13`, and maximum internal step 0.0025 days. Repeating with 0.00125-day maximum steps verifies refinement. The selected production JAX integrator remains a separate decision.

Checks cover nonnegative pools, each organism's C/P budget at all saved times, paired trade, analytical isolated growth and P-limited growth, invariant resource use under reproductive-cost changes, and the expected output change. The generation run had maximum budget residual `3.11e-15` and maximum state/refinement difference `5.44e-11` normalised units. See `verification.json` for machine-readable results.

Outputs: `summary.csv`, `trajectories.json`, `allocation-budgets.png`/`.pdf`, and `allocation-dynamics.png`/`.pdf`. The script regenerates all outputs.
