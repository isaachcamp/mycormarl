# ADR-0010: Represent spatial P variability in the initial soil inventory

**Status:** Accepted

## Context

The variability study needs heterogeneous phosphorus supply without adding an
unqualified P-cycling process. Generating P only when roots or hyphae enter a
cell would leave total inventory undefined at reset, suppress diffusion before
entry, and couple resource creation to organism geometry. It could also reward
fungal exploration by construction.

The axisymmetric grid stores amount per annular cell. Equal-concentration cells
have unequal inventories because cell volume increases radially, so unweighted
array statistics do not describe the represented soil volume.

## Decision

The baseline reset field is `initial_solution_p_um` applied uniformly to every
cell of the configured domain. An explicit depth profile or spatial-variability
treatment replaces that uniform initial field; it must generate the complete
heterogeneous solution-P field at reset. Such a field must:

- be non-negative;
- be spatially correlated in physical `(r, z)` coordinates;
- preserve the exact configured total initial labile-P inventory across the
  represented domain;
- preserve the volume-weighted mean concentration;
- use cell-volume weights for realized mean, variance, and coefficient of
  variation;
- be reproducible from recorded field-generation parameters and a field ID.

Use a bounded symmetric marginal as the primary variability treatment. Avoid
creating negative concentrations and repairing them by clipping. A positive
right-skewed marginal such as lognormal represents a distinct hotspot-form
sensitivity rather than the same treatment at greater variance.

Train `mixed` and `plant-only` policies against the same distribution of
fields. Evaluate all saved policies on the same held-out field IDs, disjoint
from training fields. Conceptually, the reset-level association choice occurs
before the particular field realization is known.

Do not use entry-triggered P sampling in the initial study. Do not add temporal
replenishment, mineralisation noise, or another exogenous P-cycling process
until the initial-field mechanism has been assessed.

## Consequences

- Total P and mode comparisons remain defined independently of organism entry.
- A finite topsoil-only initial inventory is not a supported configuration;
  vertical heterogeneity must be expressed as an explicit depth profile.
- Variability is a property of represented soil volume rather than numerical
  array indices.
- Paired held-out fields provide a fair environmental comparison even if
  training seeds are not paired.
- A positive expected association advantage under variable fields is an
  insurance or risk-buffering result under the current arithmetic-mean reward
  objective, not proof that the policy optimizes evolutionary bet hedging.
- `(r, z)` patches represent axisymmetric annuli, not unrestricted localized
  three-dimensional hotspots, and must be described accordingly.

## Related study

The provisional correlation scale, variability grid, and qualification rules
belong to
[`spatial-p-heterogeneity-pilot-spec.md`](../implementation-docs/spatial-p-heterogeneity-pilot-spec.md),
not to this durable decision.
