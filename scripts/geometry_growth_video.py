"""Render side-by-side root and mycelial geometry growth as an animated GIF.

The animation deliberately uses the production growth-geometry functions. Each
frame target is a successive radial grid edge, so a 0.1 cm radial interval
advances the fungal radius and the largest root-disc radius by exactly 1 mm
per frame (apart from a potentially shortened final boundary cell). Deeper
root discs retain their beta-derived smaller radii.
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

from mycormarl.fungus.mycelium import axisymmetric_density_from_biomass
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


def fungal_biomass_for_colony_radius(
        colony_radius_cm: float,
        traits: FungusTraits,
    ) -> chex.Array:
    """Invert the fungal geometry conversion for a target colony radius."""
    total_length_cm = (
        traits.saturation_density
        * (2.0 / 3.0)
        * jnp.pi
        * colony_radius_cm**3
    )
    tissue_volume_cm3 = total_length_cm * jnp.pi * traits.hyphal_radius**2
    structural_carbon_g = tissue_volume_cm3 * traits.hyphal_tissue_carbon_density
    return structural_carbon_g / traits.gamma_c


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
        depth_interval_cm: float = 0.1,
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

    final_root, _, _, _ = geometry_at_radius(float(radii_np[-1]))
    root_vmax = max(float(np.max(final_root)), 1e-12)
    first_root, first_fungus, first_plant_mass, first_fungus_mass = (
        geometry_at_radius(float(radii_np[0]))
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    root_mesh = axes[0].pcolormesh(
        r_edges_np,
        z_edges_np,
        first_root,
        shading="flat",
        cmap="YlGn",
        vmin=0.0,
        vmax=root_vmax,
    )
    fungus_mesh = axes[1].pcolormesh(
        r_edges_np,
        z_edges_np,
        first_fungus,
        shading="flat",
        cmap="Purples",
        vmin=0.0,
        vmax=fungus_traits.saturation_density,
    )
    fig.colorbar(root_mesh, ax=axes[0], label=r"Root length density (cm cm$^{-3}$)")
    fig.colorbar(
        fungus_mesh,
        ax=axes[1],
        label=r"Hyphal length density (cm cm$^{-3}$)",
    )
    axes[0].set_title("Plant roots: stacked discs")
    axes[1].set_title("Mycelium: saturated hemisphere")
    for axis in axes:
        axis.set_xlabel("Radial distance (cm)")
        axis.set_ylabel("Depth (cm)")
        axis.set_xlim(0.0, max_radius_cm)
        axis.set_ylim(max_depth_cm, 0.0)
        axis.set_aspect("equal")

    status = fig.suptitle(
        f"Radius {radii_np[0] * 10:.1f} mm | "
        f"plant {first_plant_mass:.3g} g | fungus {first_fungus_mass:.3g} g"
    )

    def update(frame_index: int):
        """Update both panels at the next radial grid edge."""
        radius_cm = float(radii_np[frame_index])
        root, fungus, plant_mass, fungus_mass = geometry_at_radius(radius_cm)
        root_mesh.set_array(root.ravel())
        fungus_mesh.set_array(fungus.ravel())
        status.set_text(
            f"Radius {radius_cm * 10:.1f} mm | "
            f"plant {plant_mass:.3g} g | fungus {fungus_mass:.3g} g"
        )
        return root_mesh, fungus_mesh, status

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
        default=Path("plots/growing_geometries.gif"),
    )
    parser.add_argument("--max-radius-cm", type=float, default=10.0)
    parser.add_argument("--max-depth-cm", type=float, default=25.0)
    parser.add_argument("--radial-interval-cm", type=float, default=0.1)
    parser.add_argument("--depth-interval-cm", type=float, default=0.1)
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
