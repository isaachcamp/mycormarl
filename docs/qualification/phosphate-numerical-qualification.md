# Phosphate numerical qualification results

## Outcome

Selected interval: `0.1 cm`; selected fixed-geometry soil-solver timestep: `0.05 day`.
Grid had a passing coarser candidate: `True`; timestep had a passing larger candidate: `True`.

The numerical timestep selection uses a 5% next-smaller comparison on fixed-geometry soil uptake, final inventory, and consumer shares. Coupled fixed-action trajectories are reported separately as action-frequency sensitivity diagnostics and do not select the numerical timestep because `dt` is also the policy decision interval until issue #24 is implemented. Grid convergence continues to include coupled endpoint pools. This is numerical qualification, not empirical validation.

## Balance and diagnostic ranges

- Maximum relative P-balance error: `1.290e-06`.
- Mean continuous-weight range: `0` to `0.837801`.
- Maximum cellwise continuous weight: `0.837802`.
- Diffusion CFL ceiling range: `40465.9` to `647456` seconds.
- Capped-demand fraction range: `0` to `0`.
- Maximum coupled extended-P balance error: `8.771e-07`.

## Concentration response (mixed mode)

| Initial µM | Total uptake (µmol) | Mean root C_s/C_b | Mean fungal C_s/C_b |
|---:|---:|---:|---:|
| 0.1 | 0.273521 | 0.136711 | 0.505429 |
| 0.3 | 0.81804 | 0.136932 | 0.507153 |
| 1 | 2.69592 | 0.137762 | 0.513558 |
| 3 | 7.78669 | 0.140606 | 0.534699 |
| 10 | 21.5078 | 0.155740 | 0.626332 |

## Timestep convergence

| Candidate day | Reference day | Worst fixed-soil solver change | Coupled action-frequency change | Solver pass |
|---:|---:|---:|---:|:---:|
| 0.05 | 0.025 | 3.081% | 0.527% | yes |
| 0.1 | 0.05 | 6.414% | 0.977% | no |
| 0.2 | 0.1 | 13.927% | 1.654% | no |
| 0.4 | 0.2 | 33.022% | 3.297% | no |

## Grid convergence

| Candidate cm | Reference cm | Worst fixed-soil change | Coupled change | Pass |
|---:|---:|---:|---:|:---:|
| 0.05 | 0.025 | 1.212% | 0.589% | yes |
| 0.1 | 0.05 | 1.345% | 2.564% | yes |

## Transition sensitivity (mixed mode)

| T_ref (day) | p | Mean w_cont | Total uptake (µmol) |
|---:|---:|---:|---:|
| 1 | 1 | 0.694444 | 2.63571 |
| 1 | 2 | 0.837801 | 2.69592 |
| 1 | 4 | 0.963872 | 2.74118 |
| 0.25 | 2 | 0.244046 | 2.39651 |
| 4 | 2 | 0.988042 | 2.74911 |

## Performance

- Reduced grid: `400` cells, compile+first step `0.227 s`, warmed step `0.000038 s`.
- Target grid: `500 x 1000` = `500000` cells.
- Target soil compile+first step: `0.259 s`; warmed soil step: `0.003285 s`.
- Target full-step incremental compile+first step, measured after the soil benchmark: `0.375 s`; warmed full step: `0.003563 s`.
- Estimated core working arrays: `49.6 MiB`, comprising concrete state/cached arrays plus `18` float32 cell-array equivalents. This is a formula-based estimate, not peak process RSS; XLA fusion may reduce actual temporary storage.
- Projected year: `7300` steps, `23.98 s` soil-only and `26.01 s` for the deterministic full step, excluding compilation, learned-policy inference, training, and output.
- The target environment is configured with `max_steps=7300` so the projected year is not truncated by the episode limit.

## Interpretation and limitations

- The static transition is sensitive to `T_ref` and `p`; the JSON artifact contains the complete scenario rows.
- `T_ref` changes both the overlap weight and the sparse propagation radius, so its total-uptake response need not be monotonic; it remains a provisional model parameter rather than a numerical tuning control.
- No scientific matrix row was inventory-capped. Coupled endpoint free-P pools remain reported in the JSON artifact as timestep-scaling diagnostics.
- 0.05 day is the largest tested timestep that passed the fixed-geometry soil-solver gate.
- Reduced-domain convergence retains a topsoil diffusion front but cannot reproduce every full-domain spatial scale.
- After biomass-consistent pool initialisation, all tested coupled grid comparisons pass the 5% gate for the established-organism fixture; this is qualification evidence for the tested reduced domain, not universal spatial validation.
- The coupled fixture uses 0.01 g plant biomass and 0.0001 g living external fungal biomass; each free C/P pool starts at one structural-biomass equivalent and automatic maintenance costs are disabled.
- Coupled Physical actions are fixed at `[trade=0.25, growth=1, reproduction=0, reserve=0]`. Their timestep comparisons change both numerical resolution and action frequency, so they remain diagnostics pending issue #24 and cannot establish solver convergence.
- Annual runtime is projected from both warmed soil-only and deterministic full-environment steps. MARL training, learned-policy inference, output, and accelerator transfer costs are excluded.
- The complete machine-readable tables and exact platform metadata are in `phosphate-numerical-qualification.json`.
