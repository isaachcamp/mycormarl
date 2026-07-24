from __future__ import annotations

import chex
import jax.numpy as jnp


def hyphal_length_from_fungal_biomass(
        biomass_g: chex.Array,
        gamma_c_g_c_per_g: float,
        tissue_carbon_density_g_c_cm3: float,
        hyphal_radius_cm: float,
    ) -> chex.Array:
    """Convert fungal dry biomass to cylindrical external-hyphal length.

    Dry biomass becomes structural carbon through ``gamma_c``, carbon becomes
    living tissue volume through its tissue carbon density, and cylindrical
    volume becomes length through ``pi * radius²``. Spores and intraradical
    fungal structures are intentionally omitted in this first model.
    """
    biomass_g = jnp.maximum(jnp.asarray(biomass_g), 0.0)
    cross_section_cm2 = jnp.pi * hyphal_radius_cm ** 2
    return (
        biomass_g
        * gamma_c_g_c_per_g
        / (tissue_carbon_density_g_c_cm3 * cross_section_cm2)
    )

def colony_radius_from_length_axisymmetric(total_length, saturation_density):
    """Return the radius of a saturated hemisphere containing total length."""
    return jnp.cbrt((3 * total_length) / (2 * jnp.pi * saturation_density))

def _cylinder_sphere_intersection_primitive(z, radius):
    """Evaluate the axial primitive used for sphere–cylinder intersection."""
    return radius ** 2 * z - (z ** 3) / 3.0

def _volume_under_sphere_within_radius(
        radial_limit: chex.Array,
        z_inner: chex.Array,
        z_outer: chex.Array,
        colony_radius: chex.Array,
    ) -> chex.Array:
    """Volume inside a hemisphere, cylinder radius, and z interval.

    Used to compute the fraction of a cylindrical annular cell that is
    occupied by a hemisphere. The hemisphere is centered at r=0, z=0
    and extends into positive depth.

    Returned array has shape (n_r, n_z).

    ### Only necessary currently for hemispheric geometry assumption.
    """

    radial_limit = jnp.asarray(radial_limit, dtype=jnp.float32)
    z_inner = jnp.asarray(z_inner, dtype=jnp.float32)
    z_outer = jnp.asarray(z_outer, dtype=jnp.float32)
    colony_radius = jnp.asarray(colony_radius, dtype=jnp.float32)

    radial_limit = jnp.maximum(radial_limit, 0.0)
    z_inner = jnp.maximum(z_inner, 0.0)
    z_outer = jnp.maximum(z_outer, z_inner)

    full_radius_depth = jnp.sqrt(jnp.maximum(colony_radius ** 2 - radial_limit ** 2, 0.0))
    sphere_depth = jnp.maximum(colony_radius, 0.0)

    full_z0 = z_inner
    full_z1 = jnp.minimum(z_outer, full_radius_depth)
    full_volume = radial_limit ** 2 * jnp.maximum(full_z1 - full_z0, 0.0)

    partial_z0 = jnp.maximum(z_inner, full_radius_depth)
    partial_z1 = jnp.minimum(z_outer, sphere_depth)
    partial_volume = jnp.maximum(
        _cylinder_sphere_intersection_primitive(partial_z1, colony_radius)
        - _cylinder_sphere_intersection_primitive(partial_z0, colony_radius),
        0.0,
    )
    return jnp.pi * (full_volume + partial_volume)

def axisymmetric_hemisphere_cell_fractions(
        r_edges: chex.Array,
        z_edges: chex.Array,
        colony_radius: chex.Array,
    ) -> chex.Array:
    """Fraction of each cylindrical annular cell occupied by a hemisphere.

    The hemisphere is centred at ``r=0, z=0`` and extends into positive depth.
    The returned array has shape ``(n_r, n_z)``.
    """
    r_edges = jnp.asarray(r_edges, dtype=jnp.float32)
    z_edges = jnp.asarray(z_edges, dtype=jnp.float32)

    r_inner = r_edges[:-1, None]
    r_outer = r_edges[1:, None]
    z_inner = z_edges[:-1][None, :]
    z_outer = z_edges[1:][None, :]

    outer_volume = _volume_under_sphere_within_radius(
        r_outer, z_inner, z_outer, colony_radius
    )
    inner_volume = _volume_under_sphere_within_radius(
        r_inner, z_inner, z_outer, colony_radius
    )
    occupied_volume = outer_volume - inner_volume

    cell_volume = jnp.pi * (r_outer ** 2 - r_inner ** 2) * (z_outer - z_inner)
    fraction = occupied_volume / jnp.maximum(cell_volume, 1e-12)
    return jnp.clip(fraction, 0.0, 1.0)

def axisymmetric_hemisphere_density(
        r_edges: chex.Array,
        z_edges: chex.Array,
        colony_radius: chex.Array,
        saturation_density: float,
    ) -> chex.Array:
    """Hyphal length density for a saturated hemispherical colony in r-z."""
    return saturation_density * axisymmetric_hemisphere_cell_fractions(
        r_edges, z_edges, colony_radius
    )

def axisymmetric_density_from_biomass(
        biomass: chex.Array,
        traits,
        r_edges: chex.Array,
        z_edges: chex.Array,
    ) -> chex.Array:
    """Construct axisymmetric hyphal density from fungal biomass.

    Biomass is first converted to total hyphal length, then to the radius of a
    saturated hemisphere, with partially crossed cells volume-averaged.
    """
    total_length = hyphal_length_from_fungal_biomass(
        biomass,
        traits.gamma_c,
        traits.hyphal_tissue_carbon_density,
        traits.hyphal_radius,
    )
    colony_radius = colony_radius_from_length_axisymmetric(
        total_length, traits.saturation_density
    )
    return axisymmetric_hemisphere_density(
        r_edges, z_edges, colony_radius, traits.saturation_density
    )

def density_field_from_biomass(
        biomass: chex.Array,
        traits,
        r_edges: chex.Array,
        z_edges: chex.Array,
    ) -> chex.Array:
    """Return the saturated in-domain hyphal density for fungal biomass."""
    return axisymmetric_density_from_biomass(
        biomass,
        traits,
        r_edges,
        z_edges,
    )
