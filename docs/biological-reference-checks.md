# Biological reference checks

Run all four scenarios from the repository root:

```sh
uv run python scripts/biological_reference.py check
```

Check one scenario with `--scenario mixed`, `--scenario plant-only`,
`--scenario fungus-only` or `--scenario stressed`. The default fixture directory
is `tests/fixtures/biological`; `--reference-dir` allows comparison with a copy.
Checking returns nonzero on differences and never writes reference files.
Failures name the scenario, checkpoint index, state/transition field and value.

The baseline is source revision `9a68fa4`, following implemented ADR-0020 and
closed #79–#85. It uses `[trade, growth, reproduction]` per-day rates,
simultaneous allocation and trade, and stoichiometric reproductive output.
This deliberately replaces the four-component action assumptions from the
original discussion. The fixtures also record exact source-file hashes,
complete configuration and organism traits, commands, seed, Python/dependency
versions, backend and precision mode. Checkpoints use the existing production
fixed-action rollout; there is no separate reference biological integrator.

Each scenario runs 12 steps of .25 days on a 4 × 4 soil grid. Evidence is saved
at reset, steps 2/4/6/8/10 and the final step 12: five intermediate snapshots,
not a per-step state history. It includes full state/geometry/soil fields,
observations, rewards, typed transition facts, selected flows and endpoint
summaries. The initial snapshot has no completed transition. Requested sparse
capture also includes an early final state if execution stops before the horizon.

The sufficient mixed run exercises both organisms' growth, reproduction,
uptake and bilateral trade. Single-consumer runs independently check absent
biomass, pools, geometry, uptake, observations, reward and execution flags.
The stressed mixed run removes the plant's C transfer and starts the fungus
without free C. Fungal maintenance causes shrinkage and death at step four;
the plant continues. This expected-death run remains correctly classified as
a rejected *living qualification* trajectory. The reference checks that exact
reason rather than changing the qualification rules. Focused tests check the
precise death/cancellation boundary separately from evenly spaced snapshots.

The check compares discrete facts exactly and floats with relative `2e-6` /
absolute `1e-8` tolerance. Pre-cleanup repeats were exactly identical on the
recorded CPU runtime. The small allowance permits float32 arithmetic
reassociation; it is not a scientific uncertainty interval or a cross-device
reproducibility guarantee. An independent checkpoint ledger checks soil P,
free and structural organism P (including retained dead state), and cumulative
mortality/maintenance/reproductive exports. Timing is printed separately and
excluded from fixture comparison. Measured baseline timings are in
`docs/qualification/biological-reference-baseline.json`.

## Changing a reference

Investigate a failed comparison before updating anything. A simplification
should preserve the reference. A deliberate model change needs an explained,
reviewed baseline update identifying the changed contract and source revision.
After that decision, regenerate explicitly:

```sh
uv run python scripts/biological_reference.py regenerate --reason 'Explain the agreed model change and source revision'
```

Use `--scenario` to update only affected scenarios. Review numerical fixture,
declaration and provenance differences together; do not widen tolerances to
make an unexplained difference pass. Keep earlier scientific evidence under
its original identity. These checks characterize selected current behaviour;
they complement core tests and do not prove scientific validity everywhere.
