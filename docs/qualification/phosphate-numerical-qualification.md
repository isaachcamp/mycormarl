# Phosphate numerical qualification results

## Outcome

Selected interval: `0.05 cm`; selected fixed-geometry soil-solver timestep: `0.025 day`.
Grid had a passing coarser candidate: `True`; timestep had a passing larger candidate: `True`.
Deep-soil confinement and extended-P balance pass: `True`.

Numerical timestep and spatial selection both require 5% agreement with the next-smaller discretisation and the finest tested discretisation. Timestep comparisons cover fixed-geometry soil observables and coupled trajectories under a fixed 1-day policy decision interval; candidates that do not divide that interval are ineligible. Grid convergence includes coupled endpoint pools. This is numerical qualification, not empirical validation.

## Balance and diagnostic ranges

- Maximum relative P-balance error: `1.142e-06`.
- Mean continuous-weight range: `0` to `0.837803`.
- Maximum cellwise continuous weight: `0.837802`.
- Diffusion CFL ceiling range: `40465.9` to `1.03593e+07` seconds.
- Capped-demand fraction range: `0` to `0`.
- Maximum coupled extended-P balance error: `2.112e-07`.

## Concentration response (mixed mode)

| Initial µM | Total uptake (µmol) | Mean root C_s/C_b | Mean fungal C_s/C_b |
|---:|---:|---:|---:|
| 0.1 | 0.274747 | 0.136711 | 0.505433 |
| 0.3 | 0.821694 | 0.136933 | 0.507164 |
| 1 | 2.70779 | 0.137767 | 0.513598 |
| 3 | 7.81914 | 0.140624 | 0.534832 |
| 10 | 21.5708 | 0.155829 | 0.626835 |

## Timestep convergence

| Candidate day | Next-smaller ref day | Worst fixed-soil change | Coupled fixed-policy change | Finest ref day | Worst fixed-soil change | Coupled fixed-policy change | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.025 | 0.0125 | 0.187% | 3.195% | 0.0125 | 0.187% | 3.195% | yes |
| 0.05 | 0.025 | 0.376% | 6.604% | 0.0125 | 0.564% | 10.009% | no |
| 0.1 | 0.05 | 0.755% | 13.874% | 0.0125 | 1.323% | 25.273% | no |
| 0.2 | 0.1 | 1.525% | 28.958% | 0.0125 | 2.869% | 61.548% | no |

## Grid convergence

| Candidate cm | Next-smaller ref cm | Worst fixed-soil change | Coupled change | Finest ref cm | Worst fixed-soil change | Coupled change | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.05 | 0.025 | 0.210% | 2.222% | 0.025 | 0.210% | 2.222% | yes |
| 0.1 | 0.05 | 0.180% | 4.348% | 0.025 | 0.390% | 6.667% | no |
| 0.2 | 0.1 | 0.107% | 2.036% | 0.025 | 0.496% | 6.667% | no |
| 0.4 | 0.2 | 20.033% | 20.000% | 0.025 | 20.430% | 20.000% | no |

## Deep-soil integration check

- Confinement and extended-P balance pass: `True`.
- Maximum fungal density wholly outside the colony: `0` cm cm^-3.
- Maximum fungal uptake request wholly outside the colony: `0` micromol P per cell.
- Maximum relative extended-P balance error: `7.410e-08`.

## Transition sensitivity (mixed mode)

| T_ref (day) | p | Mean w_cont | Total uptake (µmol) |
|---:|---:|---:|---:|
| 1 | 1 | 0.694444 | 2.64709 |
| 1 | 2 | 0.837801 | 2.70779 |
| 1 | 4 | 0.963872 | 2.75346 |
| 0.25 | 2 | 0.244046 | 2.40627 |
| 4 | 2 | 0.988042 | 2.76147 |

## Performance

- Reduced grid: `1600` cells, compile+first step `0.139 s`, warmed step `0.000059 s`.
- Target benchmark skipped by command-line option.

## Interpretation and limitations

- The static transition is sensitive to `T_ref` and `p`; the JSON artifact contains the complete scenario rows.
- `T_ref` changes both the overlap weight and the sparse propagation radius, so its total-uptake response need not be monotonic; it remains a provisional model parameter rather than a numerical tuning control.
- No scientific matrix row was inventory-capped. Coupled endpoint free-P pools remain reported in the JSON artifact as timestep-scaling diagnostics.
- 0.025 day is the largest tested timestep that passed both fixed-soil and coupled fixed-policy gates.
- Reduced-domain convergence retains a topsoil diffusion front but cannot reproduce every full-domain spatial scale.
- After biomass-consistent pool initialisation, all tested coupled grid comparisons pass the 5% gate for the established-organism fixture; this is qualification evidence for the tested reduced domain, not universal spatial validation.
- The coupled fixture uses 0.01 g plant biomass and 0.0001 g living external fungal biomass; each free C/P pool starts at one structural-biomass equivalent and automatic maintenance costs are disabled.
- Coupled Physical actions are fixed at `[trade=0.25, growth=1, reproduction=0, reserve=0]` and held for a fixed 1-day policy interval. Their timestep comparisons now isolate numerical resolution and are required for timestep selection.
- Annual runtime is projected from both warmed soil-only and deterministic full-environment steps. MARL training, learned-policy inference, output, and accelerator transfer costs are excluded.
- The complete machine-readable tables and exact platform metadata are in `phosphate-numerical-qualification.json`.
