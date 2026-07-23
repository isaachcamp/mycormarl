import jax.numpy as jnp
import pytest

from mycormarl.fungus.mycelium import (
    axisymmetric_hemisphere_cell_fractions,
    axisymmetric_hemisphere_density,
    colony_radius_from_length_axisymmetric,
)
from mycormarl.plant.roots import axisymmetric_stacked_disc_root_density
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_grid import (
    axisymmetric_cylindrical_cell_volumes,
    axisymmetric_radial_face_areas,
    axisymmetric_topsoil_fractions,
    axisymmetric_uniform_p_conc,
    axisymmetric_edges_from_intervals,
    axisymmetric_vertical_face_areas,
    validate_axisymmetric_grid_parameters,
)


def test_axisymmetric_cylindrical_cell_volumes():
    """Checks annular cell volumes used for conserved amount and length."""
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0, 3.0])

    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)

    expected = jnp.array([
        [jnp.pi, 2.0 * jnp.pi],
        [3.0 * jnp.pi, 6.0 * jnp.pi],
    ])
    assert volumes.shape == (2, 2)
    assert jnp.allclose(volumes, expected)


def test_axisymmetric_edges_preserve_explicit_uniform_intervals_and_maxima():
    """Uses an accepted interval exactly and ends at the requested extents."""
    r_edges, z_edges = axisymmetric_edges_from_intervals(
        radius_cm=2.0,
        depth_cm=3.0,
        radial_interval_cm=1.0,
        depth_interval_cm=1.0,
    )

    assert jnp.allclose(r_edges, jnp.array([0.0, 1.0, 2.0]))
    assert jnp.allclose(z_edges, jnp.array([0.0, 1.0, 2.0, 3.0]))


def test_axisymmetric_edges_reject_nonuniform_remainders_with_nearest_suggestions():
    """Makes every implicit interval adjustment explicit to the user."""
    with pytest.raises(ValueError) as error:
        axisymmetric_edges_from_intervals(
            radius_cm=2.5,
            depth_cm=3.0,
            radial_interval_cm=1.0,
            depth_interval_cm=2.0,
        )

    message = str(error.value)
    assert "radial_interval_cm=1" in message
    assert "0.833333" in message
    assert "3 cells" in message
    assert "depth_interval_cm=2" in message
    assert "1.5" in message
    assert "2 cells" in message


@pytest.mark.parametrize(
    ("extent_cm", "requested_cm", "suggested_cm", "n_cells"),
    [
        (2.5, 1.0, 2.5 / 3.0, 3),  # nearest valid interval is smaller
        (1.0, 0.45, 0.5, 2),       # nearest valid interval is larger
        (1.0, 1e13, 1.0, 1),       # never accepts a zero-cell grid by tolerance
    ],
)
def test_nearest_uniform_interval_suggestion_is_nearest_by_spacing(
    extent_cm, requested_cm, suggested_cm, n_cells
):
    """Chooses the closest L/n spacing, rather than always refining the grid."""
    with pytest.raises(ValueError) as error:
        axisymmetric_edges_from_intervals(
            radius_cm=extent_cm,
            depth_cm=1.0,
            radial_interval_cm=requested_cm,
            depth_interval_cm=1.0,
        )

    message = str(error.value)
    assert f"{suggested_cm:.12g}" in message
    assert f"{n_cells} cells" in message


def test_axisymmetric_face_areas_have_expected_geometry_and_axis_face_is_zero():
    """Locks face shapes/areas needed by later conservative diffusion."""
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0, 3.0])

    radial = axisymmetric_radial_face_areas(r_edges, z_edges)
    vertical = axisymmetric_vertical_face_areas(r_edges, z_edges)

    assert radial.shape == (3, 2)
    assert jnp.allclose(
        radial,
        jnp.array([
            [0.0, 0.0],
            [2.0 * jnp.pi, 4.0 * jnp.pi],
            [4.0 * jnp.pi, 8.0 * jnp.pi],
        ]),
    )
    assert vertical.shape == (2, 3)
    assert jnp.allclose(
        vertical,
        jnp.array([
            [jnp.pi, jnp.pi, jnp.pi],
            [3.0 * jnp.pi, 3.0 * jnp.pi, 3.0 * jnp.pi],
        ]),
    )


def test_topsoil_fractions_volume_average_a_crossed_layer():
    """Ensures a non-aligned topsoil boundary preserves physical inventory."""
    fractions = axisymmetric_topsoil_fractions(
        z_edges=jnp.array([0.0, 10.0, 30.0, 50.0]),
        topsoil_depth_cm=25.0,
    )

    assert jnp.allclose(fractions, jnp.array([1.0, 0.75, 0.0]))


@pytest.mark.parametrize(
    ("radius_cm", "depth_cm", "radial_interval_cm", "depth_interval_cm"),
    [
        (0.0, 100.0, 1.0, 1.0),
        (10.0, -1.0, 1.0, 1.0),
        (10.0, 100.0, 0.0, 1.0),
        (10.0, 100.0, 1.0, float("nan")),
    ],
)
def test_invalid_axisymmetric_grid_parameters_fail_fast(
    radius_cm, depth_cm, radial_interval_cm, depth_interval_cm
):
    """Rejects invalid geometry before large JAX arrays are allocated."""
    with pytest.raises(ValueError):
        validate_axisymmetric_grid_parameters(
            radius_cm,
            depth_cm,
            radial_interval_cm,
            depth_interval_cm,
        )


def test_axisymmetric_uniform_p_conc_with_partial_topsoil_layer():
    """Checks the transient reset concentration field is volume averaged."""
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0, 3.0])

    soil_p = axisymmetric_uniform_p_conc(
        r_edges, z_edges, conc=2.0, topsoil_depth=1.5
    )

    expected = jnp.array([
        [2.0, 0.5],
        [2.0, 0.5],
    ])
    assert soil_p.shape == (2, 2)
    assert jnp.allclose(soil_p, expected)


def test_hemisphere_cell_fractions_use_annular_volume_fraction():
    """Locks partial-cell volume geometry needed by fungal occupancy."""
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0, 2.0])

    fractions = axisymmetric_hemisphere_cell_fractions(
        r_edges, z_edges, colony_radius=1.0
    )

    assert fractions.shape == (2, 2)
    assert fractions[0, 0] == pytest.approx(2.0 / 3.0)
    assert fractions[1, 0] == pytest.approx(0.0)
    assert fractions[0, 1] == pytest.approx(0.0)
    assert fractions[1, 1] == pytest.approx(0.0)


def test_hemisphere_density_scales_fraction_by_saturation_density():
    """Checks fungal occupancy converts to volume-averaged length density."""
    r_edges = jnp.array([0.0, 1.0])
    z_edges = jnp.array([0.0, 1.0])

    density = axisymmetric_hemisphere_density(
        r_edges, z_edges, colony_radius=1.0, saturation_density=9.0
    )

    assert density.shape == (1, 1)
    assert density[0, 0] == pytest.approx(6.0)


def test_colony_radius_from_length_axisymmetric_inverts_hemisphere_volume():
    """Checks the length-to-colony-radius analytical inverse."""
    radius = colony_radius_from_length_axisymmetric(
        total_length=(2.0 / 3.0) * jnp.pi * 8.0,
        saturation_density=1.0,
    )

    assert radius == pytest.approx(2.0)


def test_axisymmetric_stacked_disc_root_density_conserves_total_length():
    """Checks the provisional root field integrates to represented length."""
    traits = PlantTraits(
        kroot=0.25,
        specific_root_length=100.0,
        root_length_density=100.0,
        beta_root_distribution=0.5,
        max_rooting_depth_cm=2.0,
    )
    r_edges = jnp.array([0.0, 1.0, 2.0])
    z_edges = jnp.array([0.0, 1.0, 2.0])

    density = axisymmetric_stacked_disc_root_density(
        biomass=jnp.array([10.0]),
        traits=traits,
        r_edges=r_edges,
        z_edges=z_edges,
    )
    volumes = axisymmetric_cylindrical_cell_volumes(r_edges, z_edges)

    assert density.shape == (2, 2)
    assert jnp.sum(density * volumes) == pytest.approx(250.0)
    assert density[1, 0] == pytest.approx(0.0)
    assert density[1, 1] == pytest.approx(0.0)
