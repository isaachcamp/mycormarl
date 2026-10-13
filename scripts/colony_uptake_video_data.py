"""Mass-conserving soil experiment for the colony-scale animation phase.

Prescribed hemispherical growth; production partial-volume geometry, buffered
finite-volume diffusion and blended uptake. Closed boundaries, no roots and
no imposed P decay. Semicircular equal-area analysis rings represent a
meridional cutaway; their corresponding 3-D hemispherical shells have unequal
volumes. Raw uptake is volume-rebinned into those shells, not sampled at z=0.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from mycormarl.fungus.mycelium import axisymmetric_hemisphere_density, axisymmetric_hemisphere_cell_fractions
from mycormarl.fungus.traits import FungusTraits
from mycormarl.plant.traits import PlantTraits
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.soil.phosphate_grid import (axisymmetric_cylindrical_cell_volumes,
    axisymmetric_radial_face_areas, axisymmetric_vertical_face_areas,
    labile_amount_to_solution_concentration, solution_concentration_to_labile_amount)
from mycormarl.soil.phosphate_diffusion import (axisymmetric_diffusion_conductances,
    diffuse_labile_amount, explicit_diffusion_cfl_seconds)
from mycormarl.soil.phosphate_uptake import blended_uptake_transaction
from mycormarl.soil.soil import uptake_geometry_coefficients

P_NG_PER_MICROMOL = 30973.761998


def simulate_colony(*, horizon_days=24., dt_days=0.005, output_interval_days=0.1,
                    domain_extent_cm=0.6, n_rings=48, grid_cells=64,
                    initial_radius_cm=0.05, radial_growth_cm_day=0.02):
    """Run the full soil kernels and return inventories plus interval fluxes.

    ``time_days`` are output interval ends. Uptake rates average accepted P
    over the preceding interval. End-of-interval geometry is saved separately.
    No biomass policy is run: constant radial expansion is prescribed, whereas
    P transport, uptake limits and reservoir depletion are calculated.
    """
    scalars = [horizon_days, dt_days, output_interval_days, domain_extent_cm, radial_growth_cm_day]
    if not all(math.isfinite(v) and v > 0 for v in scalars):
        raise ValueError("time, extent and growth parameters must be finite and positive")
    if not math.isfinite(initial_radius_cm) or initial_radius_cm < 0:
        raise ValueError("initial radius must be finite and nonnegative")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 2 for v in [n_rings, grid_cells]):
        raise ValueError("grid and ring counts must be integers at least two")
    final_radius = initial_radius_cm + radial_growth_cm_day*horizon_days
    # The entire support of every uptake-active cell must lie inside the outer
    # analysis shell, so rebinning cannot lose corner portions of accepted P.
    if final_radius + math.sqrt(2)*domain_extent_cm/grid_cells >= domain_extent_cm:
        raise ValueError("growing colony must remain inside analysis domain with a cell margin")
    n_blocks = round(horizon_days/output_interval_days)
    substeps = round(output_interval_days/dt_days)
    if n_blocks < 1 or substeps < 1 or not math.isclose(n_blocks*output_interval_days, horizon_days, rel_tol=1e-10) or not math.isclose(substeps*dt_days, output_interval_days, rel_tol=1e-10):
        raise ValueError("output interval must divide horizon and contain an integer number of timesteps")
    config = EnvConfig()
    species = SpeciesParams(plant=PlantTraits(), fungus=FungusTraits())
    edges = np.linspace(0, domain_extent_cm, grid_cells+1)
    r, z = jnp.asarray(edges), jnp.asarray(edges)
    volumes = axisymmetric_cylindrical_cell_volumes(r, z)
    gr, gz = axisymmetric_diffusion_conductances(r, z,
        axisymmetric_radial_face_areas(r,z), axisymmetric_vertical_face_areas(r,z),
        config.phosphate_diffusion_coefficient_cm2_s, config.theta_water, config.phosphate_impedance_factor)
    safe_cfl = float(explicit_diffusion_cfl_seconds(volumes, config.theta_water, config.b_p, gr, gz))*config.diffusion_cfl_safety
    if dt_days*86400 > safe_cfl:
        raise ValueError("dt_days exceeds safe diffusion CFL")
    initial = solution_concentration_to_labile_amount(jnp.full_like(volumes, config.initial_solution_p_um*1e-3), volumes, config.theta_water, config.b_p)

    def geometry(time):
        return axisymmetric_hemisphere_density(r,z,initial_radius_cm+radial_growth_cm_day*time,species.fungus.saturation_density)

    def one_step(carry, step_index):
        amount, interval_uptake = carry
        amount = diffuse_labile_amount(amount, volumes, config.theta_water, config.b_p, gr, gz, dt_days*86400)
        concentration = labile_amount_to_solution_concentration(amount, volumes, config.theta_water, config.b_p)
        density = geometry((step_index+0.5)*dt_days)
        g = SimpleNamespace(root_length_density=jnp.zeros_like(density), hyphae_length_density=density)
        kr, kf, weight = uptake_geometry_coefficients(g,species,
            config.phosphate_diffusion_coefficient_cm2_s, config.theta_water,
            config.phosphate_impedance_factor, config.b_p,
            config.uptake_reference_time_days, config.uptake_transition_exponent)
        remaining, _, accepted, _ = blended_uptake_transaction(amount, concentration,
            jnp.zeros_like(density), density, volumes, species, dt_days, kr,kf,weight)
        return (remaining, interval_uptake+accepted), None

    def block(amount, block_index):
        indices = block_index*substeps + jnp.arange(substeps)
        (remaining, uptake), _ = jax.lax.scan(one_step, (amount,jnp.zeros_like(amount)), indices)
        time = (block_index+1)*output_interval_days
        return remaining, (remaining,uptake,geometry(time))

    _, fields = jax.jit(lambda amount:jax.lax.scan(block,amount,jnp.arange(n_blocks)))(initial)
    remaining, accepted, density = (np.asarray(field) for field in fields)
    time = np.arange(1,n_blocks+1)*output_interval_days
    ring_edges = domain_extent_cm*np.sqrt(np.arange(n_rings+1)/n_rings)
    # Exact sphere/cylinder intersection primitive in float64, matching the
    # production geometry formula but avoiding cancellation in shell differences.
    def sphere_volume_under(radial_limit, sphere_radius):
        z0, z1 = edges[:-1][None,:], edges[1:][None,:]
        rr = radial_limit[:,None]
        full_depth = np.sqrt(np.maximum(sphere_radius**2-rr**2,0))
        full = rr**2*np.maximum(np.minimum(z1,full_depth)-z0,0)
        p0, p1 = np.maximum(z0,full_depth), np.minimum(z1,sphere_radius)
        primitive = lambda zz:sphere_radius**2*zz-zz**3/3
        partial = np.maximum(primitive(p1)-primitive(p0),0)
        return np.pi*(full+partial)
    exact_volumes = np.pi*np.diff(edges**2)[:,None]*np.diff(edges)[None,:]
    enclosed = np.array([np.where(edges[:-1,None]**2+edges[None,:-1]**2 >= s*s,0,
        sphere_volume_under(edges[1:],s)-sphere_volume_under(edges[:-1],s)) for s in ring_edges])
    enclosed = np.clip(enclosed/exact_volumes,0,1)
    weights = np.maximum(np.diff(enclosed,axis=0),0)
    shell_volumes = 2*np.pi/3*np.diff(ring_edges**3)
    shell_uptake = np.einsum("trz,krz->tk",accepted,weights,dtype=float,optimize=True)
    total_rate = accepted.sum(axis=(1,2),dtype=float)*1000/output_interval_days
    radial_flux = shell_uptake/shell_volumes/output_interval_days*(P_NG_PER_MICROMOL/1000)
    return {
        "format_version":1,
        "time_days":time, "dt_days":dt_days, "output_interval_days":output_interval_days,
        "r_edges_cm":edges, "z_edges_cm":edges,
        "ring_edges_cm":ring_edges, "ring_radius_mm":(ring_edges[1:]+ring_edges[:-1])*5,
        "semiring_area_cm2":np.pi/2*np.diff(ring_edges**2),
        "ring_volume_cm3":shell_volumes, "shell_cell_weights":weights,
        "cell_volume_cm3":np.asarray(volumes),
        "soil_amount_micromol":remaining, "accepted_uptake_micromol":accepted,
        "hyphal_density_field_cm_cm3":density,
        "cumulative_uptake_micromol":np.cumsum(accepted.sum(axis=(1,2),dtype=float)),
        "total_uptake_nmol_day":total_rate,
        "radial_flux_ng_mm3_day":radial_flux,
        "initial_total_micromol":float(np.asarray(initial).sum(dtype=float)),
        "initial_soil_amount_micromol":np.asarray(initial),
        "colony_radius_cm":initial_radius_cm+radial_growth_cm_day*time,
        "initial_colony_radius_cm":initial_radius_cm,
        "radial_growth_cm_day":radial_growth_cm_day,
        "hyphal_density_cm_cm3":species.fungus.saturation_density,
        "safe_cfl_seconds":safe_cfl, "far_field_half_life_days":None,
        "parameters":{
            "initial_solution_p_um":config.initial_solution_p_um,
            "theta_water":config.theta_water,"buffer_power":config.b_p,
            "diffusion_coefficient_cm2_s":config.phosphate_diffusion_coefficient_cm2_s,
            "impedance_factor":config.phosphate_impedance_factor,
            "reference_time_days":config.uptake_reference_time_days,
            "hyphal_radius_cm":species.fungus.hyphal_radius,
            "jmax_micromol_cm2_s":species.fungus.jmax,"km_micromol_cm3":species.fungus.km,
            "grid_cells":grid_cells,"n_analysis_rings":n_rings,"domain_extent_cm":domain_extent_cm,
        },
        "assumptions":"Independent colony-scale experiment, restarted clock; closed finite-inventory buffered soil, production diffusion and blended uptake, no imposed P decay, no plant competitor. Prescribed constant-speed hemispherical geometry, not a policy rollout or BARE reproduction. Equal-area meridional semicircular rings correspond to unequal-volume 3-D shells; flux is shell-volume averaged, total rate sums all accepted fungal P. Branch topology is illustrative, not the simulated density field. Rates average preceding output interval; inventories/geometry are end-of-interval.",
        "reference":"Visual inspiration: Oyarte Galvez et al. (2025), doi:10.1038/s41586-025-08614-x, Figs. 2a and 3e; not fitted to that paper.",
    }
