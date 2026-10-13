"""Manim scene for the model's soil, root, and external-hyphal geometry.

The twelve fine-root disc radii are evaluated by the production function at
2 g plant biomass over 25 cm depth, using one isotropic display scale. Root
and hyphal branch topology and stroke widths are illustrative; the model
represents density fields, rather than individual branching architectures.
The external-hyphal branches are contained within the colony hemisphere.

Render a preview with::

    uv run --with manim manim -pql scripts/manim_soil_root_mycelium.py SoilRootMycelium
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mycormarl-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "mycormarl-cache"))

import jax.numpy as jnp
from manim import *

from mycormarl.fungus.mycelium import fungal_biomass_for_colony_radius
from mycormarl.fungus.traits import FungusTraits
from mycormarl.plant.roots import root_disc_radii_from_biomass
from mycormarl.plant.traits import PlantTraits


SOIL_TOP = 0.0
SOIL_BOTTOM = -5.8
SOIL_WIDTH = 6.6
MODEL_SCALE = abs(SOIL_BOTTOM) / 25.0
COLONY_RADIUS = 2.65


def _curve(start, end, bend=0.0, width=0.045, colour="#E0B86F"):
    """Create a gently bending 3-D biological filament."""
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    perpendicular = np.array([-end[1] + start[1], end[0] - start[0], 0.0])
    norm = np.linalg.norm(perpendicular[:2]) or 1.0
    perpendicular /= norm
    midpoint = (start + end) / 2 + perpendicular * bend
    curve = CubicBezier(start, midpoint, midpoint, end)
    curve.set_stroke(colour, width=width, opacity=0.95)
    return curve


class SoilRootMycelium(ThreeDScene):
    """A cinematic cutaway of the coupled axisymmetric geometry."""

    def construct(self):
        self.camera.background_color = "#071018"
        self.set_camera_orientation(
            phi=66 * DEGREES, theta=-55 * DEGREES, zoom=0.78,
            frame_center=np.array([0, 0, -2.75]),
        )

        title = Text("Geometry beneath the soil", font_size=32, color="#F3E8D0")
        title.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title, shift=0.25 * DOWN), run_time=1.2)

        colony = self._mycelium_envelope()
        discs = self._root_discs()
        roots = self._root_system()
        hyphae = self._hyphal_network()
        labels = self._labels()
        root_label, fungi_label = labels
        self.add_fixed_in_frame_mobjects(root_label, fungi_label)
        self.remove(root_label, fungi_label)

        axis = Line(
            [0, 0, 0.15], [0, 0, SOIL_BOTTOM],
            color="#9AB5B0", stroke_width=0.8, stroke_opacity=0.35,
        )
        self.play(
            FadeIn(axis), GrowFromPoint(colony, ORIGIN),
            FadeIn(fungi_label, shift=0.15 * UP),
            run_time=1.2,
        )
        # Every depth layer shares the same growth clock, preserving the
        # biomass-derived profile rather than implying a descending front.
        self.play(
            *[GrowFromPoint(disc, disc.get_center()) for disc in discs],
            FadeIn(root_label, shift=0.15 * UP),
            run_time=2.0,
        )
        branching_note = Text("illustrative branching only", font_size=12, color="#9AB5B0")
        branching_note.to_edge(DOWN, buff=0.35).set_opacity(0.65)
        self.add_fixed_in_frame_mobjects(branching_note)
        self.remove(branching_note)
        self.play(
            LaggedStart(*[Create(root) for root in roots], lag_ratio=0.06),
            FadeIn(branching_note),
            run_time=2.6,
        )
        self.play(LaggedStart(*[Create(hypha) for hypha in hyphae], lag_ratio=0.035), run_time=3.0)
        self.wait(2.0)

    def _soil_block(self):
        block = Cylinder(radius=SOIL_WIDTH / 2, height=abs(SOIL_BOTTOM), resolution=(24, 2))
        block.move_to([0, 0, (SOIL_TOP + SOIL_BOTTOM) / 2])
        block.set_fill("#8A5F44", opacity=0.025)
        block.set_stroke("#C9966A", width=0.35, opacity=0.18)
        return block

    def _soil_layers(self):
        colours = ["#A56D4D", "#8C6047", "#714B3B", "#55392F", "#3E2D2A"]
        layers = VGroup()
        thickness = abs(SOIL_BOTTOM) / len(colours)
        for index, colour in enumerate(colours):
            layer = Circle(radius=SOIL_WIDTH / 2, color=colour)
            layer.move_to([0, 0, SOIL_TOP - index * thickness])
            layer.set_fill(colour, opacity=0.015)
            layer.set_stroke("#C9966A", width=0.6, opacity=0.3)
            layers.add(layer)
        return layers

    def _soil_particles(self):
        rng = np.random.default_rng(19)
        particles = VGroup()
        for _ in range(28):
            angle = rng.uniform(0, TAU)
            radius = np.sqrt(rng.uniform(0, 1)) * SOIL_WIDTH / 2 * 0.95
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            z = rng.uniform(SOIL_BOTTOM + 0.25, -0.15)
            dot = Dot3D([x, y, z], radius=rng.uniform(0.018, 0.045), color="#D4A57C")
            dot.set_opacity(rng.uniform(0.22, 0.58))
            particles.add(dot)
        return particles

    def _mycelium_envelope(self):
        # Convert the display radius back to cm before evaluating biomass.
        fungus_traits = FungusTraits()
        colony_radius = COLONY_RADIUS
        self.fungal_biomass = float(fungal_biomass_for_colony_radius(colony_radius / MODEL_SCALE, fungus_traits))

        envelope = Surface(
            lambda u, v: np.array([
                colony_radius * np.sin(v) * np.cos(u),
                colony_radius * np.sin(v) * np.sin(u),
                -colony_radius * np.cos(v),
            ]),
            u_range=[0, TAU],
            v_range=[0, PI / 2],
            resolution=(28, 12),
        )
        envelope.set_fill("#75D9C1", opacity=0.18)
        envelope.set_stroke("#69D6C0", width=0.65, opacity=0.48)
        return envelope

    def _root_profile(self):
        """Production layer radii and depths, with one isotropic cm scale."""
        edges = jnp.linspace(0.0, 25.0, 13)
        radii = np.asarray(root_disc_radii_from_biomass(2.0, PlantTraits(), edges))
        return radii * MODEL_SCALE, -np.asarray((edges[:-1] + edges[1:]) / 2) * MODEL_SCALE

    def _root_discs(self):
        discs = VGroup()
        radii, depths = self._root_profile()
        for radius, depth in zip(radii, depths):
            disc = Circle(radius=float(radius), color="#D6A65D")
            disc.move_to([0, 0, float(depth)])
            disc.set_fill("#E0B86F", opacity=0.065)
            disc.set_stroke("#EDC77E", width=1.25, opacity=0.7)
            discs.add(disc)
        return discs

    def _root_system(self):
        radii, depths = self._root_profile()
        roots = VGroup()
        main = _curve([0, 0, 0.25], [0, 0, -5.55], bend=0.035, width=5, colour="#E0B86F")
        roots.add(main)
        for index, (depth, radius) in enumerate(zip(depths, radii)):
            for angle in np.arange(3) * TAU / 3 + index * 0.73:
                def branch(start, heading, length, order):
                    end = start + length * np.array([np.cos(heading), np.sin(heading), 0.0])
                    curve = _curve(start, end, bend=0.025 * length, width=2.6 * 0.55 ** order, colour="#E0B86F")
                    roots.add(curve)
                    if order < 2:
                        node = curve.point_from_proportion(0.72)
                        for sign in [-1, 1]:
                            branch(node, heading + sign * 0.65, length * 0.43, order + 1)
                branch(np.array([0.0, 0.0, depth]), angle, radius * 0.60, 0)
        return roots

    def _hyphal_network(self):
        rng = np.random.default_rng(8)
        hyphae = VGroup()
        def branch(start, direction, length, order):
            direction = direction / np.linalg.norm(direction)
            end = start + direction * length
            end[2] = min(end[2], -0.055)
            if np.linalg.norm(end) > COLONY_RADIUS * 0.96:
                end *= COLONY_RADIUS * 0.96 / np.linalg.norm(end)
            curve = _curve(start, end, bend=0.0, width=1.8 * 0.73 ** order, colour="#7AE6CF")
            hyphae.add(curve)
            if order < 4:
                node = curve.point_from_proportion(0.87)
                side = np.cross(direction, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(side) < 0.01:
                    side = np.array([1.0, 0.0, 0.0])
                side /= np.linalg.norm(side)
                for sign in [-1, 1]:
                    child_direction = direction + sign * side * 0.75 + rng.normal(0, 0.15, 3)
                    branch(node, child_direction, length * 0.64, order + 1)
        for family in range(10):
            angle = family * TAU / 10
            start = np.array([0.18 * np.cos(angle), 0.18 * np.sin(angle), -0.45])
            direction = np.array([np.cos(angle), np.sin(angle), -0.25 - (family % 3) * 0.65])
            branch(start, direction, 0.95, 0)
        return hyphae

    def _labels(self):
        root_label = Text("Fine-root discs", font_size=15, color="#E0B86F")
        root_label.to_corner(DL, buff=0.35)
        fungi_label = Text("Extraradical mycelium", font_size=15, color="#86E7D0")
        fungi_label.to_corner(DR, buff=0.35)
        return VGroup(root_label, fungi_label)


if __name__ == "__main__":
    # Manim owns command-line parsing; keeping this guard makes the file
    # importable for lightweight syntax checks and scene discovery.
    pass
