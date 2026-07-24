from __future__ import annotations

import chex
import jax.numpy as jnp

from mycormarl.plant.traits import PlantTraits


def root_length_from_plant_biomass(
        biomass_g: chex.Array,
        root_mass_fraction: float,
        specific_root_length_cm_g: float,
    ) -> chex.Array:
    """Convert plant dry biomass to total root length in centimetres.

    ``root_mass_fraction`` converts plant dry mass to root dry mass and is
    deliberately applied exactly once. Specific root length then converts
    root dry mass to length.
    """
    biomass_g = jnp.maximum(jnp.asarray(biomass_g), 0.0)
    return biomass_g * root_mass_fraction * specific_root_length_cm_g

def _root_depth_cdf(depth: chex.Array, beta: float) -> chex.Array:
    """Evaluate the provisional cumulative root-depth distribution."""
    return 1 - beta ** depth

def _depth_weights_from_edges(
        beta: float,
        z_edges: chex.Array,
        max_rooting_depth_cm: float,
    ) -> chex.Array:
    """Return full-profile root fractions represented by bounded soil layers.

    The beta depth distribution is normalised over the intended maximum rooting
    depth, not the simulated soil depth. Edges beyond that rooting horizon are
    clipped before differencing, so deeper soil layers contain no roots.
    """
    z_edges = jnp.asarray(z_edges, dtype=jnp.float32)
    bounded_edges = jnp.minimum(z_edges, max_rooting_depth_cm)
    cdf = _root_depth_cdf(bounded_edges, beta)
    weights = jnp.diff(cdf)
    full_profile_fraction = _root_depth_cdf(max_rooting_depth_cm, beta)
    return weights / jnp.maximum(full_profile_fraction, 1e-12)

def axisymmetric_disc_overlap_fractions(
        r_edges: chex.Array,
        disc_radius: chex.Array,
    ) -> chex.Array:
    """Return annular overlap for one disc radius or one radius per layer."""
    r_edges = jnp.asarray(r_edges, dtype=jnp.float32)
    disc_radius = jnp.asarray(disc_radius, dtype=jnp.float32)
    scalar_radius = disc_radius.ndim == 0
    disc_radii = jnp.atleast_1d(disc_radius)[None, :]

    r_inner = r_edges[:-1, None]
    r_outer = r_edges[1:, None]
    occupied_area = (
        jnp.minimum(r_outer, disc_radii) ** 2
        - jnp.minimum(r_inner, disc_radii) ** 2
    )
    occupied_area = jnp.maximum(occupied_area, 0.0)
    annular_area = r_outer ** 2 - r_inner ** 2
    fractions = jnp.clip(
        occupied_area / jnp.maximum(annular_area, 1e-12),
        0.0,
        1.0,
    )
    return fractions[:, 0] if scalar_radius else fractions


def root_disc_radii_from_biomass(
        biomass: chex.Array,
        traits: PlantTraits,
        z_edges: chex.Array,
    ) -> chex.Array:
    """Derive one root-disc radius per depth layer from uniform density.

    Total biomass-implied root length is partitioned with the beta depth
    distribution. Each layer radius then follows
    ``sqrt(L_layer / (pi * lambda_root * dz))``.
    """
    z_edges = jnp.asarray(z_edges, dtype=jnp.float32)
    total_root_length = jnp.squeeze(
        root_length_from_plant_biomass(
            biomass,
            traits.kroot,
            traits.specific_root_length,
        )
    )
    depth_weights = _depth_weights_from_edges(
        traits.beta_root_distribution,
        z_edges,
        traits.max_rooting_depth_cm,
    )
    layer_lengths = total_root_length * depth_weights
    dz = z_edges[1:] - z_edges[:-1]
    return jnp.sqrt(
        layer_lengths
        / (jnp.pi * traits.root_length_density * dz)
    )

def axisymmetric_stacked_disc_root_density(
        biomass: chex.Array,
        traits: PlantTraits,
        r_edges: chex.Array,
        z_edges: chex.Array,
    ) -> chex.Array:
    """Construct root length density on an axisymmetric cylindrical r-z grid.

    Beta-weighted layer lengths determine one disc radius per depth while the
    within-disc length density remains uniformly ``traits.root_length_density``.
    Annular front cells are volume averaged and radii beyond the soil boundary
    are clipped by the overlap calculation rather than redistributed.
    """
    r_edges = jnp.asarray(r_edges, dtype=jnp.float32)
    z_edges = jnp.asarray(z_edges, dtype=jnp.float32)
    disc_radii = root_disc_radii_from_biomass(biomass, traits, z_edges)
    radial_overlap = axisymmetric_disc_overlap_fractions(r_edges, disc_radii)
    return traits.root_length_density * radial_overlap

def density_field_from_biomass(
        biomass: chex.Array,
        traits: PlantTraits,
        r_edges: chex.Array,
        z_edges: chex.Array,
    ) -> chex.Array:
    """Return the in-domain root length-density field for plant biomass."""
    return axisymmetric_stacked_disc_root_density(
        biomass,
        traits,
        r_edges,
        z_edges,
    )
