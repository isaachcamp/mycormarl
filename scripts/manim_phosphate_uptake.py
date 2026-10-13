"""Three-stage phosphate uptake story driven by an isolated sparse cylinder.

Render: uv run --with manim manim -qm scripts/manim_phosphate_uptake.py PhosphateUptake
No TeX or external image assets are required. Branches and advance speed are
illustrative; curves show the repository sparse closure at local contact age,
with prescribed 2-day far-field P half-life and uptake scaled from 1 mm to 1 cm.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import weakref

import numpy as np
from manim import *

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phosphate_uptake_video_data import simulate_uptake_front
from manim_colony_uptake import add_colony_phase, make_colony_title
from hyphal_branching import organic_filament, side_branches, growth_timings, LATERAL_COUNT_RANGE


BACKGROUND = "#071018"
TEAL = "#7AE6CF"
GOLD = "#EDC77E"
MUTED = "#8FAAAE"
SITE_COLOURS = ["#EDC77E", "#78B8EC", "#C7A0E8", "#7AE6CF"]
HYPHA_Y = -1.5
LEADING_TIP_X = 0.82
TIP_ADVANCE_PER_DAY = 0.28  # Illustrative screen distance, not a physical growth rate.
HYPHA_RADIUS = 0.025
MIDDLE_TIME_SCALE = 25.0 / 40.4


def middle_seconds(seconds):
    """Compress the A–D story and return transition to about 25 seconds."""
    return seconds * MIDDLE_TIME_SCALE


def sample(data, name, day):
    """Index one common simulation-time axis, including exact contact times."""
    index = min(int(max(day, 0) / data["dt_days"] + 1e-6), len(data["time_days"]) - 1)
    return data[name][index]


def hud(mobject, scene, location):
    """Keep an overlay at the same screen size while the camera moves."""
    original_width = mobject.width

    def update(mob):
        scale = scene.camera.frame.width / 14.2222222222
        mob.scale_to_fit_width(original_width * scale)
        mob.move_to(scene.camera.frame.get_center() + np.array(location) * scale)
    mobject.add_updater(update)
    update(mobject)
    return mobject


class UptakeCard(VGroup):
    """A live uptake plot for one independent, time-shifted segment."""

    def __init__(self, scene, data, site, letter, clock, colour, x=0, width=4.1, y=1.32):
        super().__init__()
        self.scene_ref = weakref.ref(scene)
        self.data, self.site, self.clock = data, site, clock
        self.screen_x = ValueTracker(x)
        self.screen_y = ValueTracker(y)
        self.screen_width = ValueTracker(width)
        self.x_max = ValueTracker(6.0 if site == 0 else 20.0)
        self.axis_transition = ValueTracker(0.0 if site == 0 else 1.0)
        self.base_width = 3.1
        self.axes = Axes(
            x_range=[0, 20, 10], y_range=[0, 0.15, 0.05],
            x_length=2.48, y_length=0.98,
            axis_config={"include_tip": False, "stroke_width": 0.65, "color": MUTED},
            x_axis_config={"include_ticks": False},
        ).move_to([0.12, -0.03, 0])
        background = RoundedRectangle(width=3.1, height=2.16, corner_radius=0.12,
                                      fill_color="#0E1C27", fill_opacity=0.98,
                                      stroke_color=colour, stroke_opacity=0.45, stroke_width=0.9)
        self.background = background
        contact = float(data["arrival_days"][site])
        contact_label = f"{contact:.1f}".removesuffix(".0")
        title = Text(f"{letter}  •  contact at day {contact_label}",
                     font_size=14, color=colour).move_to([0, 0.82, 0])
        unit = Text("nmol P d⁻¹ • 1 cm segment", font_size=11, color=MUTED).move_to([0, 0.57, 0])
        self.zero = Text("0", font_size=10, color=MUTED).next_to(self.axes.c2p(0, 0), DOWN, buff=0.08)
        self.middle = Text("3", font_size=10, color=MUTED)
        self.end = Text("6", font_size=10, color=MUTED)
        self.global_middle = Text("10", font_size=10, color=MUTED)
        self.global_end = Text("20", font_size=10, color=MUTED)
        self.time_ticks = [
            (value, label, Line(DOWN * 0.025, UP * 0.025, color=MUTED))
            for value, label in [(3, self.middle), (6, self.end),
                                 (10, self.global_middle), (20, self.global_end)]
        ]
        upper = Text("0.15", font_size=10, color=MUTED).next_to(self.axes.c2p(0, 0.15), LEFT, buff=0.04)
        lower = Text("0", font_size=10, color=MUTED).next_to(self.axes.c2p(0, 0), LEFT, buff=0.04)
        xlabel = Text("simulation time (days)", font_size=11, color=MUTED).move_to([0.1, -0.87, 0])
        self.curve = VMobject(stroke_color=colour, stroke_width=2)
        self.dot = Dot(radius=0.038, color=colour)
        self.cursor = Line(ORIGIN, UP, stroke_color=colour, stroke_width=0.5, stroke_opacity=0.35)
        self.add(background, self.axes, title, unit, self.zero, self.middle, self.end,
                 self.global_middle, self.global_end, *(tick for _, _, tick in self.time_ticks), upper, lower,
                 xlabel, self.cursor, self.curve, self.dot)
        self.refresh()

    def refresh(self):
        scene = self.scene_ref()
        scale = scene.camera.frame.width / 14.2222222222
        self.scale_to_fit_width(self.screen_width.get_value() * scale)
        self.move_to(scene.camera.frame.get_center() + np.array([self.screen_x.get_value(), self.screen_y.get_value(), 0]) * scale)
        # Cairo magnifies strokes on zoom; keep graph lines screen-constant.
        self.background.set_stroke(width=0.9 * scale)
        self.axes.set_stroke(width=0.65 * scale)
        self.curve.set_stroke(width=2 * scale)
        self.cursor.set_stroke(width=0.5 * scale)
        xmax = self.x_max.get_value()
        # Axes retain 0..20 internally; map the displayed time window to it.
        def point(t, rate):
            return self.axes.c2p(t * 20 / xmax, rate)
        now = self.clock.get_value()
        arrival = self.data["arrival_days"][self.site]
        idx = np.flatnonzero((self.data["time_days"] >= arrival) &
                             (self.data["time_days"] <= min(now, xmax)))
        if len(idx):
            # Keep plotting light without changing the source numerical data.
            idx = idx[::max(1, len(idx) // 180)]
            last = min(int(now / self.data["dt_days"] + 1e-6), len(self.data["time_days"]) - 1)
            idx = np.unique(np.r_[idx, last])
            points = [point(float(self.data["time_days"][i]), float(self.data["rate_nmol_day"][i, self.site])) for i in idx]
            if len(points) == 1:
                points.append(points[0] + RIGHT * 1e-6)
            self.curve.set_points_as_corners(points)
            self.dot.move_to(point(now, float(sample(self.data, "rate_nmol_day", now)[self.site])))
        else:
            self.dot.move_to(point(arrival, 0))
        self.cursor.put_start_and_end_on(point(now, 0), point(now, 0.15))
        # Labels retain their actual day values while the visible range grows.
        # Hidden out-of-range labels stay at the edge so they cannot expand
        # the card's bounding box and disturb its screen-anchored layout.
        progress = self.axis_transition.get_value()
        for value, label, tick in self.time_ticks:
            position = point(min(value, xmax), 0)
            if value in (3, 6):
                opacity = 1 - progress
            elif value == 10:
                opacity = smooth(float(np.clip((xmax - 10) / 3, 0, 1)))
            else:
                # The range reaches 20 before the camera finishes moving;
                # only then does the 20-day label fade in at its true position.
                opacity = smooth(float(np.clip((progress - 0.8) / 0.2, 0, 1)))
            label.next_to(position, DOWN, buff=0.07 * scale).set_opacity(opacity)
            tick.put_start_and_end_on(position + DOWN * 0.025 * scale,
                                      position + UP * 0.025 * scale)
            tick.set_stroke(width=0.65 * scale, opacity=opacity)

    def use_global_time_axis(self):
        self.x_max.set_value(20)
        self.axis_transition.set_value(1)
        self.refresh()


class PhosphateUptake(MovingCameraScene):
    """Growth → one local uptake curve → multiple sites behind a moving front."""

    def construct(self):
        self.camera.background_color = BACKGROUND
        self.data = simulate_uptake_front()
        self.clock = ValueTracker(0.0)
        self.camera.frame.set(width=8.5).move_to([0, HYPHA_Y, 0])
        self._save_data()

        note = hud(Text("Illustrative branching only • prescribed far-field decay (2-day half-life)", font_size=12, color=MUTED), self, [0, -3.65, 0])
        title = hud(make_colony_title(), self, [0, 3.45, 0])
        self.add(note, title)
        seed, seed_growth = self._seed_branches()
        self.wait(0.8)
        self.add(seed)
        self.play(seed_growth.animate.set_value(1.0), run_time=6.0, rate_func=linear)
        seed.update(0)
        seed.clear_updaters(recursive=True)
        self.wait(0.7)

        # Keep the actual focal branch on screen throughout the camera move.
        # Only the surrounding colony fades; the branch itself straightens
        # and gains a resolved diameter as the camera approaches its apex.
        focal = self.focal_hypha
        seed.remove(focal)
        return_seed = seed.copy()
        self.add(focal)
        focal_points = np.array([focal.point_from_proportion(t) for t in np.linspace(0, 1, 60)])
        self.play(
            self.camera.frame.animate.set(width=1.2).move_to([LEADING_TIP_X, HYPHA_Y, 0]),
            UpdateFromAlphaFunc(focal, lambda mob, alpha: self._straighten_focal_hypha(mob, focal_points, alpha)),
            FadeOut(seed),
            FadeOut(title),
            run_time=2.4,
        )
        title.clear_updaters()
        live_structure = focal
        live_structure.add_updater(lambda mob: self._update_single_hypha(mob))
        site_a_index = int(self.data["plot_sites"][0])
        site_a = self._site_marker(site_a_index, "A", SITE_COLOURS[0], close_up=True)
        self.play(Create(site_a[0]), Write(site_a[1]), run_time=middle_seconds(0.4))
        self.add(site_a)
        site_markers = [site_a]
        self.wait(middle_seconds(0.3))
        card_a = UptakeCard(self, self.data, site_a_index, "A", self.clock, SITE_COLOURS[0], x=-3.7, width=3.5, y=1.6)
        self.play(FadeIn(card_a), run_time=middle_seconds(0.6))
        card_a.add_updater(lambda mob: mob.refresh())
        # Each graph starts the same isolated depletion experiment at contact.
        self.play(self.clock.animate.set_value(3.5), run_time=middle_seconds(8.0), rate_func=linear)
        self.wait(middle_seconds(0.8))

        self.play(
            self.camera.frame.animate.set(width=6.0).move_to([LEADING_TIP_X + 1.1, HYPHA_Y + 0.44, 0]),
            card_a.screen_x.animate.set_value(-5.1),
            card_a.screen_y.animate.set_value(1.32),
            card_a.screen_width.animate.set_value(3.1),
            card_a.x_max.animate(rate_func=lambda t: smooth(min(t / 0.8, 1))).set_value(20),
            card_a.axis_transition.animate(rate_func=linear).set_value(1),
            self.clock.animate(rate_func=linear).set_value(4.0),
            Transform(site_a, self._site_marker(site_a_index, "A", SITE_COLOURS[0])),
            run_time=middle_seconds(2.0),
        )
        follow_start_day = self.clock.get_value()
        self.camera.frame.add_updater(lambda frame: frame.move_to([
            LEADING_TIP_X + 1.1 + 0.70 * TIP_ADVANCE_PER_DAY * (self.clock.get_value() - follow_start_day),
            HYPHA_Y + 0.44, 0,
        ]))
        cards = [card_a]
        for site, letter, colour, xpos in zip(self.data["plot_sites"][1:], "BCD", SITE_COLOURS[1:], [-1.7, 1.7, 5.1]):
            arrival = float(self.data["arrival_days"][site])
            seconds = (arrival - self.clock.get_value()) * 1.25
            self.play(self.clock.animate.set_value(arrival), run_time=middle_seconds(seconds), rate_func=linear)
            card = UptakeCard(self, self.data, site, letter, self.clock, colour, x=xpos, width=3.1)
            card.use_global_time_axis()
            cards.append(card)
            marker = self._site_marker(site, letter, colour)
            site_markers.append(marker)
            self.play(FadeIn(card), FadeIn(marker), run_time=middle_seconds(0.6))
            card.add_updater(lambda mob: mob.refresh())
        self.play(self.clock.animate.set_value(19.5), run_time=middle_seconds(2.5), rate_func=linear)
        self.camera.frame.clear_updaters()
        self.wait(middle_seconds(3.5))
        add_colony_phase(self,cards,site_markers,live_structure,return_seed,focal_points,note,
            return_time_scale=MIDDLE_TIME_SCALE)

    def _seed_branches(self):
        """Fan of advancing runners inspired by the supplied R. irregularis clip.

        Growth advances along each curve. A lateral starts only after its
        parent's tip passes its attachment point; older tissue never stretches.
        """
        rng = np.random.default_rng(73)
        bounds = (np.array([-0.9, HYPHA_Y - 0.71]), np.array([0.9, HYPHA_Y + 0.71]))
        trunk = organic_filament([-0.05, HYPHA_Y - 0.68, 0], [-0.02, HYPHA_Y + 0.65, 0], rng, width=1.65)
        records = [(trunk, None, 0.0, 0)]
        endpoints = [(-0.83, -0.33), (-0.74, 0.12), (-0.58, 0.53),
                     (-0.28, 0.67), (0.29, 0.57), (0.63, 0.33), (LEADING_TIP_X, 0.0)]
        for index, (x, y) in enumerate(endpoints):
            fraction = float(rng.uniform(0.06, 0.30))
            start = trunk.point_from_proportion(fraction)
            runner = organic_filament(start, [x, HYPHA_Y + y, 0], rng, width=1.45)
            records.append((runner, trunk, fraction, 0))
            if index == len(endpoints) - 1:
                self.focal_hypha = runner
            records.extend(side_branches(runner, rng, int(rng.integers(*LATERAL_COUNT_RANGE)), 0.27, bounds))
        records.extend(side_branches(trunk, rng, 12, 0.22, bounds))
        growth = ValueTracker(0.0)
        timings = growth_timings(records, rng)
        finish = max(birth + duration for birth, duration in timings.values())
        branches = VGroup()
        for curve, _, _, order in records:
            birth, duration = timings[id(curve)]
            template = curve.copy()
            def update(mob, birth=birth, duration=duration, template=template, order=order):
                fraction = float(np.clip((growth.get_value() * finish - birth) / duration, 0, 1))
                mob.pointwise_become_partial(template, 0, max(fraction, 1e-6))
                mob.set_stroke(opacity=0 if fraction == 0 else (0.95 if order == 0 else 0.8))
            curve.add_updater(update)
            curve.update(0)
            branches.add(curve)
        return branches, growth

    def _growing_structure(self):
        """One continuous tube with a semicircular growing head, no laterals."""
        tube = VMobject()
        self._update_single_hypha(tube)
        return tube

    def _straighten_focal_hypha(self, tube, original_points, alpha):
        """Resolve a curved filament into a tube without an open/closed morph.

        Corresponding centreline samples stay ordered, avoiding a folded
        polygon when a line is transformed directly into a closed outline.
        The apex stays fixed while the shaft rotates into the horizontal.
        """
        fractions = np.linspace(0, 1, len(original_points))
        horizontal = np.array([
            [x, HYPHA_Y, 0]
            for x in np.linspace(original_points[0, 0], LEADING_TIP_X, len(original_points))
        ])
        centres = (1 - alpha) * original_points + alpha * horizontal
        tangents = np.gradient(centres, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0], np.zeros(len(centres))])
        radius = (1 - alpha) * 0.001 + alpha * HYPHA_RADIUS
        centres -= radius * fractions[:, None] * tangents
        bottom = centres - radius * normals
        top = centres + radius * normals
        cap = [centres[-1] + radius * (np.cos(angle) * tangents[-1] + np.sin(angle) * normals[-1])
               for angle in np.linspace(-PI / 2, PI / 2, 25)]
        tube.set_points_as_corners(np.vstack([bottom, cap, top[::-1], bottom[:1]]))
        tube.set_fill(TEAL, opacity=0.35 * alpha)
        tube.set_stroke(TEAL, width=2 * self.camera.frame.width / 14.2222222222, opacity=0.95)

    def _update_single_hypha(self, tube):
        apex = LEADING_TIP_X + TIP_ADVANCE_PER_DAY * self.clock.get_value()
        left = LEADING_TIP_X - 4.0
        shape = RoundedRectangle(
            width=apex - left, height=2 * HYPHA_RADIUS,
            corner_radius=HYPHA_RADIUS,
            stroke_color=TEAL, stroke_width=2.0 * self.camera.frame.width / 14.2222222222,
            fill_color=TEAL, fill_opacity=0.35,
        ).move_to([(left + apex) / 2, HYPHA_Y, 0])
        tube.become(shape)

    def _site_marker(self, site, letter, colour, close_up=False):
        size = 0.5 if close_up else 1.0
        view_width = 1.2 if close_up else 6.0
        x = LEADING_TIP_X + float(self.data["arrival_days"][site]) * TIP_ADVANCE_PER_DAY
        marker = VGroup(
            Circle(radius=0.07 * size, color=colour, stroke_width=1.2 * view_width / 14.2222222222).move_to([x, HYPHA_Y, 0]),
            Text(letter, font_size=6.8 * size, color=colour).move_to([x, HYPHA_Y - 0.18 * size, 0]),
        )
        return marker

    def _save_data(self):
        path = Path(config.media_dir) / "phosphate-uptake" / "data.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        serialised = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.data.items()}
        path.write_text(json.dumps(serialised, indent=2) + "\n")
