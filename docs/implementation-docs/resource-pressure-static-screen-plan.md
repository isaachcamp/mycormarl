# Issue #47: canonical resource-pressure screen

## Purpose

Identify how resource-pressure parameters control vegetative plant and fungal
biomass, uptake sources, and C/P limitation under fixed full-growth policies.
This is a static-policy screen, not a PPO-training or reproductive-fitness
study.

## Canonical protocol

- Sampling: a fresh, reproducible **360-condition continuous Latin hypercube**
  (seed `3047`). A changed sample size alters LHS strata, so it is a new design
  rather than an extension of the prior 300-condition screen.
- Domain: axisymmetric soil cylinder with radius `40 cm` and depth `60 cm`.
- Grid: radial and depth intervals both `0.2 cm`.
- Horizon: `80 d`, with `0.05 d` environment steps.
- Mode: mixed plant–AMF association with uniform initial solution P.
- Policies: sampled bilateral trade followed by full growth allocation
  (`[trade, 1, 0, 0]`); reproduction and reserve allocations are zero.
- Both plant and fungal `kappa_p` values are fixed at zero.
- Checkpoints and a compact progress manifest are written after each condition;
  the reconstructable combined JSON bundle is disabled.

## Sampled factors

All factors are continuous and independently Latin-hypercube sampled.

| Factor | Range | Scale |
| --- | --- | --- |
| Plant `kappa_c` multiplier | `0.01–0.693145×` default | logarithmic |
| Fungus `kappa_c` multiplier | `0.01–0.693145×` default | logarithmic |
| Initial AMF biomass multiplier | `1–100×` default | logarithmic |
| Initial solution P | `0.1–1.0 µM` | linear |
| Plant→fungus trade fraction | `0.05–0.20` | linear |
| Fungus→plant trade fraction | `0.50–0.80` | linear |
| Fungal `gamma_p` | `0.5–2.0 mg P g⁻¹ DM` | linear |

## Diagnostics and outcomes

Every condition records final plant and fungal biomass, cumulative growth,
direct P uptake, fungal P transfer, plant carbon fixation, final geometry, and
timestep resource-pressure diagnostics. The latter record gamma-normalized C
and P allocations/use, the limiting resource, C/P acquisition, maintenance,
growth, and trade flows. A limitation regime is defined when a resource limits
more than half of realized-growth timesteps.

The primary endpoint is final living plant biomass. Fungal biomass, direct and
indirect plant P uptake, and plant/fungal C- and P-limitation fractions are
secondary endpoints.

## Execution and provenance

- Canonical manifest:
  `docs/qualification/resource-pressure-canonical-screen-manifest.json`.
- Canonical runner: `scripts/run_resource_pressure_canonical_screen.py`.
- Canonical analysis entry point:
  `scripts/analyse_resource_pressure_canonical_screen.py`.
- Active rerun output: `outputs/resource-pressure-canonical-screen/`.

The earlier discrete 120-condition, paired low-P gamma-P, and 300-condition
continuous screens remain historical result/report provenance only. Their
manifests and dedicated plotting scripts are retired; they must not be used as
current #47 configurations.

## Interpretation limits

- The within-step reserve action does not guarantee next-step maintenance,
  because resource acquisition is pooled after allocation.
- This screen estimates a multivariate response landscape. It can suggest main
  effects and interactions, but matched one- or two-factor follow-ups are
  needed for clean causal contrasts or precise regime boundaries.
