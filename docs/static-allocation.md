# Static allocation experiments

New allocation screens use `mycormarl.static_rollout.evaluate_static_allocation`.
Declare an interval in days equal to both the policy interval and
`EnvConfig.dt`, a target residual fraction `epsilon` (default `0.001`), and
plant/fungal trade and reproductive expenditure shares. All shares are finite
and within `[0, 1]`; both organisms need an explicit declaration, including an
absent counterpart. Each organism's `reproduction_cost_scale` must be finite
and positive (default one).

The experiment derives total intensity `K = -log(epsilon) / interval_days`.
For trade share `p` and reproductive expenditure share `a`, its physical
`[trade, growth, reproduction]` rates are
`[p*K, (1-a)*(1-p)*K, a*(1-p)*K]`, in inverse days. A growth-only biological
control sets `a=0`. At `0.05` days, the default `K` is
`138.1551055796 / day`. An optional supplied `total_rate_per_day` must match
within relative tolerance `1e-7` or absolute tolerance `1e-10 / day`.

For an already constructed environment `env`:

```python
from mycormarl.static_rollout import evaluate_static_allocation

result = evaluate_static_allocation(
    env,
    interval_days=0.05,
    target_residual_fraction=0.001,
    trade_shares={"plant": 0.05, "fungus": 0.1},
    reproductive_shares={"plant": 0.0, "fungus": 0.25},
    seed=7,
)
```

The result records declarations, derived `K`, shares and reproductive cost
scales under `static_allocation`, physical commands under
`commanded_rate_actions`, and realised growth, reproductive output, transfers,
final pools and resource accounting. Reproductive resource exports are
cost-scaled expenditures; cumulative reproductive fitness measures output.
Internal solver controls are recorded separately from the physical strategy.

Run `.venv/bin/python scripts/static_allocation_examples.py` for a four-case
pool-only screen with explicit initial stocks and disabled maintenance and
acquisition. The retained [example results](qualification/three-rate-static-examples.json)
include complete environment/trait declarations and output ledgers.

The residual target describes nominal isolated-pool turnover. It does not
guarantee that each elemental stock falls to `epsilon` of its starting value.
An essential resource can limit biology and leave its complementary element
unused. Trade also credits the counterpart during allocation; photosynthesis
and soil uptake supply later resources under the existing outer update order.
Retained surplus and finite-rate residue remain free resources.

Changing the declared outer interval and recomputing `K` intentionally changes
physical processing speed. Refining the internal allocation solver instead
holds the three commanded rates fixed. Likewise, numerical convergence work
must hold physical commands, initial pools, forcing and duration fixed.
`evaluate_static_policy` remains the unrestricted explicit-rate reference seam
for those comparisons. Learned policies and existing reduced-control callers
do not inherit this screen-specific `K` constraint.
