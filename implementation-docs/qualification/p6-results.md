# P6 phosphate qualification results

## Outcome

Selected interval: `0.1 cm`; selected biological timestep: `0.025 day`.
Grid had a passing coarser candidate: `True`; timestep had a passing larger candidate: `False`.

The selection uses a 5% next-finer/next-smaller comparison on fixed-geometry uptake, final inventory, consumer shares, and coupled uptake, free P pools, biomass, and extents. It is numerical qualification, not empirical validation.

## Balance and diagnostic ranges

- Maximum relative P-balance error: `1.537e-06`.
- Mean continuous-weight range: `0` to `0.0317241`.
- Maximum cellwise continuous weight: `0.031724`.
- Diffusion CFL ceiling range: `40465.9` to `647456` seconds.
- Capped-demand fraction range: `0` to `0`.
- Maximum coupled extended-P balance error: `6.880e-07`.

## Concentration response (mixed mode)

| Initial µM | Total uptake (µmol) | Mean root C_s/C_b | Mean fungal C_s/C_b |
|---:|---:|---:|---:|
| 0.1 | 0.0311587 | 0.136866 | 0.477539 |
| 0.3 | 0.0926934 | 0.137395 | 0.481413 |
| 1 | 0.30008 | 0.139278 | 0.495022 |
| 3 | 0.828489 | 0.144934 | 0.533844 |
| 10 | 2.06886 | 0.168363 | 0.654209 |

## Timestep convergence

| Candidate day | Reference day | Worst fixed-soil change | Coupled change | Pass |
|---:|---:|---:|---:|:---:|
| 0.05 | 0.025 | 0.056% | 99.512% | no |
| 0.1 | 0.05 | 0.112% | 99.025% | no |
| 0.2 | 0.1 | 0.225% | 98.054% | no |
| 0.4 | 0.2 | 0.453% | 102.176% | no |

## Grid convergence

| Candidate cm | Reference cm | Worst fixed-soil change | Coupled change | Pass |
|---:|---:|---:|---:|:---:|
| 0.05 | 0.025 | 0.255% | 0.053% | yes |
| 0.1 | 0.05 | 0.254% | 3.030% | yes |

## Transition sensitivity (mixed mode)

| T_ref (day) | p | Mean w_cont | Total uptake (µmol) |
|---:|---:|---:|---:|
| 1 | 1 | 0.153265 | 0.334462 |
| 1 | 2 | 0.031724 | 0.30008 |
| 1 | 4 | 0.001072 | 0.291357 |
| 0.25 | 2 | 0.002044 | 0.322045 |
| 4 | 2 | 0.343925 | 0.371658 |

## Performance

- Reduced grid: `400` cells, compile+first step `0.229 s`, warmed step `0.000039 s`.
- Target grid: `500 x 1000` = `500000` cells.
- Target soil compile+first step: `0.247 s`; warmed soil step: `0.003091 s`.
- Target full-step incremental compile+first step, measured after the soil benchmark: `0.408 s`; warmed full step: `0.004761 s`.
- Estimated core working arrays: `49.6 MiB`, comprising concrete state/cached arrays plus `18` float32 cell-array equivalents. This is a formula-based estimate, not peak process RSS; XLA fusion may reduce actual temporary storage.
- Projected year: `14600` steps, `45.13 s` soil-only and `69.52 s` for the deterministic full step, excluding compilation, learned-policy inference, training, and output.
- The target environment is configured with `max_steps=14600` so the projected year is not truncated by the episode limit.

## Interpretation and limitations

- The static transition is sensitive to `T_ref` and `p`; the JSON artifact contains the complete scenario rows.
- `T_ref` changes both the overlap weight and the sparse propagation radius, so its total-uptake response need not be monotonic; it remains a provisional model parameter rather than a numerical tuning control.
- No scientific matrix row was inventory-capped. Fixed-soil outputs changed by less than 0.5% across the timestep candidates, but endpoint coupled free-P pools changed by approximately 98–102%; none of the coarser timestep candidates passed the 5% gate.
- 0.025 day is the finest tested timestep and was selected as the fallback. Because it has no finer reference, this result does not demonstrate timestep convergence; a finer follow-up study or a revised scientifically justified endpoint metric is required.
- Reduced-domain convergence retains a topsoil diffusion front but cannot reproduce every full-domain spatial scale.
- Coupled actions are fixed at `[trade=0.25, growth=0.75, maintenance=0, reproduction=0]`; maintenance costs are disabled only in this qualification fixture so the unresolved maintenance-P fate cannot contaminate balance interpretation.
- Annual runtime is projected from both warmed soil-only and deterministic full-environment steps. MARL training, learned-policy inference, output, and accelerator transfer costs are excluded.
- The complete machine-readable tables and exact platform metadata are in `p6-results.json`.

