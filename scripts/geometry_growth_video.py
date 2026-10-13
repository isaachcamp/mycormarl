"""Render side-by-side root and mycelial geometry growth as an animated GIF.

The animation deliberately uses the production growth-geometry functions. Each
frame target is a successive radial grid edge, so the radial interval advances
the fungal radius and the largest root-disc radius by exactly that amount.
Grid construction requires that the interval divide the maximum radius
exactly. Deeper root discs retain their beta-derived smaller radii.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

# Configure writable caches before project imports can transitively load
# plotting libraries on restricted or headless systems.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mycormarl-matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "mycormarl-cache"),
)

import chex
import jax.numpy as jnp

from mycormarl.fungus.mycelium import (
    axisymmetric_density_from_biomass,
    fungal_biomass_for_colony_radius,
)
from mycormarl.fungus.traits import FungusTraits
from mycormarl.plant.roots import axisymmetric_stacked_disc_root_density
from mycormarl.plant.roots import root_disc_radii_from_biomass
from mycormarl.plant.traits import PlantTraits
from mycormarl.soil.phosphate_grid import axisymmetric_edges_from_intervals


MAX_VIDEO_FRAMES = 1_000


def growth_radii_from_grid(
        max_radius_cm: float,
        radial_interval_cm: float,
    ) -> chex.Array:
    """Return successive radial grid edges, enforcing the 1,000-frame cap."""
    r_edges, _ = axisymmetric_edges_from_intervals(
        radius_cm=max_radius_cm,
        depth_cm=1.0,
        radial_interval_cm=radial_interval_cm,
        depth_interval_cm=1.0,
    )
    n_frames = r_edges.shape[0] - 1
    if n_frames > MAX_VIDEO_FRAMES:
        raise ValueError(
            f"The requested grid requires {n_frames} frames; videos are limited "
            f"to {MAX_VIDEO_FRAMES:,} frames."
        )
    return r_edges[1:]


def root_biomass_for_max_disc_radius(
        target_radius_cm: float,
        traits: PlantTraits,
        z_edges: chex.Array,
    ) -> chex.Array:
    """Invert root geometry so its largest layer reaches a target radius.

    Disc radii scale with the square root of biomass at fixed beta weights and
    ``lambda_root``. Evaluating the production function at one gram therefore
    provides an exact algebraic inverse without duplicating its geometry.
    """
    one_gram_radii = root_disc_radii_from_biomass(
        biomass=jnp.array([1.0]),
        traits=traits,
        z_edges=z_edges,
    )
    one_gram_max_radius = jnp.max(one_gram_radii)
    return (jnp.asarray(target_radius_cm) / one_gram_max_radius) ** 2


def render_growth_video(
        output_path: str | Path,
        max_radius_cm: float = 10.0,
        max_depth_cm: float = 25.0,
        radial_interval_cm: float = 0.1,
    depth_interval_cm: float = 0.5,
        fps: int = 12,
        dpi: int = 120,
    ) -> Path:
    """Render the production root and fungal geometries side-by-side to a GIF."""
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output_path must end in .gif because Pillow is the available writer")

    radii = growth_radii_from_grid(max_radius_cm, radial_interval_cm)
    r_edges, z_edges = axisymmetric_edges_from_intervals(
        radius_cm=max_radius_cm,
        depth_cm=max_depth_cm,
        radial_interval_cm=radial_interval_cm,
        depth_interval_cm=depth_interval_cm,
    )
    plant_traits = PlantTraits()
    fungus_traits = FungusTraits()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    r_edges_np = np.asarray(r_edges)
    z_edges_np = np.asarray(z_edges)
    radii_np = np.asarray(radii)

    def geometry_at_radius(radius_cm: float):
        """Evaluate both production density fields for one animation frame."""
        plant_biomass = root_biomass_for_max_disc_radius(
            target_radius_cm=radius_cm,
            traits=plant_traits,
            z_edges=z_edges,
        )
        fungal_biomass = fungal_biomass_for_colony_radius(radius_cm, fungus_traits)
        root_density = axisymmetric_stacked_disc_root_density(
            biomass=jnp.atleast_1d(plant_biomass),
            traits=plant_traits,
            r_edges=r_edges,
            z_edges=z_edges,
        )
        fungal_density = axisymmetric_density_from_biomass(
            biomass=jnp.atleast_1d(fungal_biomass),
            traits=fungus_traits,
            r_edges=r_edges,
            z_edges=z_edges,
        )
        return (
            np.asarray(root_density).T,
            np.asarray(fungal_density).T,
            float(plant_biomass),
            float(fungal_biomass),
        )

    _, _, first_plant_mass, first_fungus_mass = geometry_at_radius(
        float(radii_np[0])
    )

    # Render the axisymmetric primitives themselves, so the animation shows
    # the hemisphere and the beta-weighted stack of discs in 3-D.
    fig = plt.figure(figsize=(12, 6.5), constrained_layout=True)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    root_colour = "#4c9a57"
    root_edge = "#286336"
    fungus_colour = "#8e63b6"
    fungus_edge = "#513276"

    def style_axis(axis, title):
        axis.set_title(title, pad=12, fontsize=12, fontweight="medium")
        axis.set_xlabel("x (cm)", labelpad=5)
        axis.set_ylabel("y (cm)", labelpad=5)
        axis.set_zlabel("depth z (cm)", labelpad=5)
        axis.set_xlim(-max_radius_cm, max_radius_cm)
        axis.set_ylim(-max_radius_cm, max_radius_cm)
        axis.set_zlim(max_depth_cm, 0.0)
        axis.set_box_aspect((2.0 * max_radius_cm, 2.0 * max_radius_cm, max_depth_cm))
        axis.view_init(elev=25, azim=-55)
        axis.grid(True, alpha=0.22)

    def add_root_disc(axis, radius, depth, thickness):
        if radius <= 0.0:
            return
        radial = np.linspace(0.0, radius, 6)
        rr, tt = np.meshgrid(radial, theta)
        x = rr * np.cos(tt)
        y = rr * np.sin(tt)
        z_top = np.full_like(x, max(0.0, depth - thickness / 2.0))
        z_bottom = np.full_like(x, depth + thickness / 2.0)
        axis.plot_surface(x, y, z_top, color=root_colour, alpha=0.22,
                          linewidth=0, shade=True)
        axis.plot_surface(x, y, z_bottom, color=root_colour, alpha=0.10,
                          linewidth=0, shade=True)
        x_side = radius * np.cos(theta)
        y_side = radius * np.sin(theta)
        z_side = np.vstack((
            np.full(theta.shape, max(0.0, depth - thickness / 2.0)),
            np.full(theta.shape, depth + thickness / 2.0),
        ))
        axis.add_collection3d(Poly3DCollection(
            [list(zip(x_side, y_side, z_side[0])),
             list(zip(x_side, y_side, z_side[1]))],
            facecolors=root_colour, edgecolors=root_edge, linewidths=0.15,
            alpha=0.18,
        ))

    def add_hemisphere(axis, radius):
        if radius <= 0.0:
            return
        radial = np.linspace(0.0, radius, 28)
        rr, tt = np.meshgrid(radial, theta)
        x = rr * np.cos(tt)
        y = rr * np.sin(tt)
        z = np.sqrt(np.maximum(radius ** 2 - rr ** 2, 0.0))
        axis.plot_surface(x, y, z, color=fungus_colour, alpha=0.58,
                          edgecolor=fungus_edge, linewidth=0.15, shade=True)

    def draw_frame(frame_index: int):
        radius_cm = float(radii_np[frame_index])
        root_radii = np.asarray(root_disc_radii_from_biomass(
            root_biomass_for_max_disc_radius(radius_cm, plant_traits, z_edges),
            plant_traits,
            z_edges,
        ))
        for axis in axes:
            axis.cla()
        style_axis(axes[0], "Plant: beta-weighted stacked root discs")
        style_axis(axes[1], "Fungus: saturated hemispherical mycelium")
        layer_thickness = float(depth_interval_cm) * 0.82
        layer_depths = 0.5 * (z_edges_np[:-1] + z_edges_np[1:])
        for depth, root_radius in zip(layer_depths, root_radii):
            add_root_disc(axes[0], float(root_radius), float(depth), layer_thickness)
        add_hemisphere(axes[1], radius_cm)

    draw_frame(0)
    status = fig.suptitle(
        f"Geometric growth | radius {radii_np[0] * 10:.1f} mm | "
        f"plant {first_plant_mass:.3g} g | fungus {first_fungus_mass:.3g} g",
        fontsize=13,
    )

    def update(frame_index: int):
        """Update both panels at the next radial grid edge."""
        radius_cm = float(radii_np[frame_index])
        _, _, plant_mass, fungus_mass = geometry_at_radius(radius_cm)
        draw_frame(frame_index)
        status.set_text(
            f"Geometric growth | radius {radius_cm * 10:.1f} mm | "
            f"plant {plant_mass:.3g} g | fungus {fungus_mass:.3g} g"
        )
        return (status,)

    video = animation.FuncAnimation(
        fig,
        update,
        frames=len(radii_np),
        interval=1_000 / fps,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    """Parse command-line controls for the illustrative growth video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/growing_geometries_3d.gif"),
    )
    parser.add_argument("--max-radius-cm", type=float, default=10.0)
    parser.add_argument("--max-depth-cm", type=float, default=25.0)
    parser.add_argument("--radial-interval-cm", type=float, default=0.2)
    parser.add_argument("--depth-interval-cm", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    """Generate the requested side-by-side geometry video."""
    args = _parse_args()
    output = render_growth_video(
        output_path=args.output,
        max_radius_cm=args.max_radius_cm,
        max_depth_cm=args.max_depth_cm,
        radial_interval_cm=args.radial_interval_cm,
        depth_interval_cm=args.depth_interval_cm,
        fps=args.fps,
        dpi=args.dpi,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
