
"""Integrated fixed-geometry soil phosphate evolution."""

import chex
import jax
import jax.numpy as jnp

from mycormarl.params import SpeciesParams
from mycormarl.soil.phosphate_diffusion import apparent_diffusivity_cm2_s
from mycormarl.soil.phosphate_grid import labile_amount_to_solution_concentration
from mycormarl.soil.phosphate_diffusion import diffuse_labile_amount
from mycormarl.soil.phosphate_units import days_to_seconds, micromol_p_to_mg_p
from mycormarl.soil.phosphate_uptake import (
    PhosphateUptakeDiagnostics,
    blended_uptake_transaction,
    continuous_regime_weight,
    hyphal_overlap_time_seconds,
    sparse_uptake_resistance,
)
from mycormarl.state import State


def uptake_geometry_coefficients(
    state: State,
    species: SpeciesParams,
    diffusion_coefficient_cm2_s: float,
    theta_water: float,
    impedance_factor: float,
    b_p: float,
    reference_time_days: float,
    transition_exponent: float,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Return root/fungal sparse resistances and their shared blend weight.

    These arrays depend on absorber geometry and static soil parameters, not
    on solution concentration. The caller computes them once after biological
    growth and reuses them throughout all fixed-geometry soil substeps.
    Root density sets only root sparse resistance; the shared regime weight is
    derived exclusively from hyphal density.
    """
    d_flux = (
        jnp.asarray(diffusion_coefficient_cm2_s)
        * jnp.asarray(theta_water)
        * jnp.asarray(impedance_factor)
    )
    d_app = apparent_diffusivity_cm2_s(
        diffusion_coefficient_cm2_s,
        theta_water,
        impedance_factor,
        b_p,
    )
    root_resistance = sparse_uptake_resistance(
        state.root_length_density,
        species.plant.root_radius,
        species.plant.jmax,
        d_flux,
        d_app,
        reference_time_days,
    )
    fungus_resistance = sparse_uptake_resistance(
        state.hyphae_length_density,
        species.fungus.hyphal_radius,
        species.fungus.jmax,
        d_flux,
        d_app,
        reference_time_days,
    )
    overlap_time = hyphal_overlap_time_seconds(
        state.hyphae_length_density,
        species.fungus.hyphal_radius,
        d_app,
    )
    continuous_weight = continuous_regime_weight(
        overlap_time,
        reference_time_days,
        transition_exponent,
    )
    return root_resistance, fungus_resistance, continuous_weight


def _apply_blended_uptake(
    state: State,
    dt_days: float,
    species: SpeciesParams,
    cell_volumes_cm3: chex.Array,
    theta_water: float,
    b_p: float,
    root_sparse_resistance: chex.Array,
    fungus_sparse_resistance: chex.Array,
    continuous_weight: chex.Array,
) -> State:
    """Apply the complete concentration-dependent uptake transaction.

    Current canonical amount is converted to bulk solution concentration.
    Root and fungal sparse requests use their cached resistances; continuous
    requests use the same bulk field. Both consumer pairs are interpolated by
    the shared hyphal-overlap weight, after which the existing inventory cap
    is applied exactly once. Only accepted µmol P leave soil, and those amounts
    cross the unit boundary to the corresponding organism pools in mg P.
    """
    updated_state, _ = _apply_blended_uptake_with_diagnostics(
        state,
        dt_days,
        species,
        cell_volumes_cm3,
        theta_water,
        b_p,
        root_sparse_resistance,
        fungus_sparse_resistance,
        continuous_weight,
    )
    return updated_state


def _apply_blended_uptake_with_diagnostics(
    state: State,
    dt_days: float,
    species: SpeciesParams,
    cell_volumes_cm3: chex.Array,
    theta_water: float,
    b_p: float,
    root_sparse_resistance: chex.Array,
    fungus_sparse_resistance: chex.Array,
    continuous_weight: chex.Array,
) -> tuple[State, PhosphateUptakeDiagnostics]:
    """Apply blended uptake and return exact cellwise qualification observables."""
    concentration = labile_amount_to_solution_concentration(
        state.soil_labile_p,
        cell_volumes_cm3,
        theta_water,
        b_p,
    )
    remaining, accepted_root, accepted_fungus, diagnostics = blended_uptake_transaction(
        state.soil_labile_p,
        concentration,
        state.root_length_density,
        state.hyphae_length_density,
        cell_volumes_cm3,
        species,
        dt_days,
        root_sparse_resistance,
        fungus_sparse_resistance,
        continuous_weight,
    )
    plant_gain_mg = micromol_p_to_mg_p(jnp.sum(accepted_root))
    fungus_gain_mg = micromol_p_to_mg_p(jnp.sum(accepted_fungus))
    return state.replace(
        soil_labile_p=remaining,
        plant_p_pool=state.plant_p_pool + plant_gain_mg,
        fungus_p_pool=state.fungus_p_pool + fungus_gain_mg,
    ), diagnostics


def soil_diffusion_uptake_substep(
    state: State,
    dt_days: float,
    species: SpeciesParams,
    cell_volumes_cm3: chex.Array,
    theta_water: float,
    b_p: float,
    radial_conductance: chex.Array,
    vertical_conductance: chex.Array,
    root_sparse_resistance: chex.Array,
    fungus_sparse_resistance: chex.Array,
    continuous_weight: chex.Array,
) -> State:
    """Apply diffusion then blended uptake over one stable soil substep.

    Geometry and face conductances are fixed over the biological step.
    Diffusion first updates canonical amount; uptake then derives a fresh
    solution concentration from that amount, applies shared competition, and
    accumulates accepted P in the organism pools.
    """
    updated_state, _ = soil_diffusion_uptake_substep_with_diagnostics(
        state,
        dt_days,
        species,
        cell_volumes_cm3,
        theta_water,
        b_p,
        radial_conductance,
        vertical_conductance,
        root_sparse_resistance,
        fungus_sparse_resistance,
        continuous_weight,
    )
    return updated_state


def soil_diffusion_uptake_substep_with_diagnostics(
    state: State,
    dt_days: float,
    species: SpeciesParams,
    cell_volumes_cm3: chex.Array,
    theta_water: float,
    b_p: float,
    radial_conductance: chex.Array,
    vertical_conductance: chex.Array,
    root_sparse_resistance: chex.Array,
    fungus_sparse_resistance: chex.Array,
    continuous_weight: chex.Array,
) -> tuple[State, PhosphateUptakeDiagnostics]:
    """Apply one production substep and retain its post-diffusion diagnostics.

    Qualification uses this entry point to observe the exact concentration at
    which uptake occurred. The ordinary production wrapper discards the
    diagnostics, so environment state and its JaxMARL-facing API remain
    unchanged.
    """
    diffused_amount = diffuse_labile_amount(
        state.soil_labile_p,
        cell_volumes_cm3,
        theta_water,
        b_p,
        radial_conductance,
        vertical_conductance,
        days_to_seconds(dt_days),
    )
    diffused_state = state.replace(soil_labile_p=diffused_amount)
    return _apply_blended_uptake_with_diagnostics(
        diffused_state,
        dt_days,
        species,
        cell_volumes_cm3,
        theta_water,
        b_p,
        root_sparse_resistance,
        fungus_sparse_resistance,
        continuous_weight,
    )


def evolve_soil_p(
    state: State,
    dt_days: float,
    species: SpeciesParams,
    cell_volumes_cm3: chex.Array,
    theta_water: float,
    b_p: float,
    radial_conductance: chex.Array,
    vertical_conductance: chex.Array,
    n_substeps: int,
    diffusion_coefficient_cm2_s: float,
    impedance_factor: float,
    reference_time_days: float,
    transition_exponent: float,
) -> State:
    """Subcycle complete fixed-geometry diffusion and uptake transactions.

    The equal substep durations sum to the biological timestep. Sparse
    resistance and overlap weight are calculated once from post-growth root
    and hyphal density before the loop. Each iteration then recovers solution
    concentration after diffusion, recalculates concentration-dependent
    uptake, and accumulates accepted P while geometry remains fixed.
    ``n_substeps`` is precomputed from the exact CFL ceiling at environment
    construction and is always at least one.
    """
    substep_days = jnp.asarray(dt_days) / jnp.asarray(n_substeps)
    (
        root_sparse_resistance,
        fungus_sparse_resistance,
        continuous_weight,
    ) = uptake_geometry_coefficients(
        state,
        species,
        diffusion_coefficient_cm2_s,
        theta_water,
        impedance_factor,
        b_p,
        reference_time_days,
        transition_exponent,
    )

    def apply_substep(_, current_state):
        """Advance one loop iteration using the shared stable duration."""
        return soil_diffusion_uptake_substep(
            current_state,
            substep_days,
            species,
            cell_volumes_cm3,
            theta_water,
            b_p,
            radial_conductance,
            vertical_conductance,
            root_sparse_resistance,
            fungus_sparse_resistance,
            continuous_weight,
        )

    return jax.lax.fori_loop(
        0,
        n_substeps,
        apply_substep,
        state,
    )


def evolve_soil_p_with_diagnostics(
    state: State,
    dt_days: float,
    species: SpeciesParams,
    cell_volumes_cm3: chex.Array,
    theta_water: float,
    b_p: float,
    radial_conductance: chex.Array,
    vertical_conductance: chex.Array,
    n_substeps: int,
    diffusion_coefficient_cm2_s: float,
    impedance_factor: float,
    reference_time_days: float,
    transition_exponent: float,
) -> tuple[State, tuple[PhosphateUptakeDiagnostics, ...]]:
    """Run the production soil step while retaining each substep diagnostic.

    This qualification-only wrapper uses the same coefficient builder and
    substep transaction as ``evolve_soil_p``. Its Python tuple of diagnostics
    is intended for deterministic offline studies, not the JIT-compiled MARL
    environment path.
    """
    substep_days = jnp.asarray(dt_days) / jnp.asarray(n_substeps)
    coefficients = uptake_geometry_coefficients(
        state,
        species,
        diffusion_coefficient_cm2_s,
        theta_water,
        impedance_factor,
        b_p,
        reference_time_days,
        transition_exponent,
    )
    current_state = state
    diagnostics = []
    for _ in range(n_substeps):
        current_state, substep_diagnostics = (
            soil_diffusion_uptake_substep_with_diagnostics(
                current_state,
                substep_days,
                species,
                cell_volumes_cm3,
                theta_water,
                b_p,
                radial_conductance,
                vertical_conductance,
                *coefficients,
            )
        )
        diagnostics.append(substep_diagnostics)
    return current_state, tuple(diagnostics)
