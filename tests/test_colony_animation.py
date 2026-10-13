"""Render-object regressions; run with `uv run --with manim python -m pytest ...`."""
import sys
from pathlib import Path
import numpy as np
import pytest

manim = pytest.importorskip("manim")
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from manim_colony_uptake import (ColonyPlots, recorded_index, make_network,
    ORIGIN_VIEW, VIEW_RADIUS, returning_colony_target, colony_view_radius, COLONY_ENTRY_DAY,
    make_evolving_branches, visible_colony_radius, make_colony_front)
from colony_uptake_video_data import simulate_colony


@pytest.fixture(scope="module")
def data():
    return simulate_colony()


def test_live_sampling_never_uses_a_future_interval():
    d = {"time_days":np.array([.1,.2,.3])}
    assert recorded_index(d,0) == -1
    assert recorded_index(d,.099) == -1
    assert recorded_index(d,.1) == 0
    assert recorded_index(d,.199) == 0
    assert recorded_index(d,.3) == 2


def test_live_and_snapshot_graphs_remain_unfilled(data):
    clock = manim.ValueTracker(0.)
    plot = ColonyPlots(data,clock)
    for day in [0,6,10,24]:
        clock.set_value(day)
        plot.refresh()
        plot.update(0)
        for curve in [plot.live,plot.total,*plot.snapshots]:
            assert curve.get_fill_opacity() == 0
        for snapshot_day,curve in zip([6,10,14,18,22],plot.snapshots):
            assert curve.get_stroke_opacity() == pytest.approx(.9 if day>=snapshot_day else 0)


def test_colony_network_determinism_and_no_closed_filled_filaments(data):
    clock = manim.ValueTracker(24.)
    a = make_network(None,data,clock)
    b = make_network(None,data,clock)
    assert len(a) > 200
    for first,second in zip(a,b):
        np.testing.assert_array_equal(first.points,second.points)
        assert first.get_fill_opacity() == 0


def test_compact_graphs_clear_the_centred_colony_and_refresh_in_place(data):
    clock = manim.ValueTracker(24.)
    plot = ColonyPlots(data,clock).compact()
    plot.refresh()
    assert plot.width < 4
    assert plot.get_left()[0] > ORIGIN_VIEW[0] + VIEW_RADIUS + .3
    assert plot.get_right()[0] < 14.2222222222 / 2
    assert abs(ORIGIN_VIEW[1] + VIEW_RADIUS / 2) < .01
    np.testing.assert_allclose(plot.total_dot.get_center(),
        plot.total_axes.c2p(24,data["total_uptake_nmol_day"][-1]))


def test_initial_colony_is_visible_and_entire_growth_stays_inside_frame(data):
    clock = manim.ValueTracker(0.)
    network = make_network(None,data,clock)
    assert network.width > .5
    for day in [0,6,12,24]:
        clock.set_value(day)
        network.update(0)
        assert network.get_left()[0] > -14.2222222222/2
        assert network.get_right()[0] < 2.3
        assert network.get_top()[1] < 4
        assert network.get_bottom()[1] > -4


def test_returned_colony_retains_branch_geometry_and_grows_inside_rings(data):
    from manim_phosphate_uptake import PhosphateUptake
    scene = PhosphateUptake()
    branches,growth = scene._seed_branches()
    growth.set_value(1.)
    branches.update(0)
    branches.clear_updaters(recursive=True)
    focal = scene.focal_hypha
    branches.remove(focal)
    original = manim.VGroup(branches,focal)
    target = returning_colony_target(original,data)
    assert len(target[0]) == len(branches)
    assert target.width > 1
    # The move preserves all pairwise branch geometry up to one uniform scale.
    ratio = target.width/original.width
    np.testing.assert_allclose(np.diff(target.get_all_points(),axis=0),
        np.diff(original.get_all_points(),axis=0)*ratio,atol=1e-12)
    clock = manim.ValueTracker(24.)
    extensions = make_evolving_branches(target,data,clock)
    assert 350 < len(extensions) <= 900
    final = manim.VGroup(target,extensions)
    assert final.get_left()[0] >= ORIGIN_VIEW[0]-VIEW_RADIUS
    assert final.get_right()[0] <= ORIGIN_VIEW[0]+VIEW_RADIUS
    assert final.get_top()[1] < 4
    assert final.get_bottom()[1] > -4


def test_evolving_branches_preserve_old_tissue_and_respect_parent_birth(data):
    seed = manim.VGroup(manim.VGroup(*[
        manim.Line(ORIGIN_VIEW,ORIGIN_VIEW+colony_view_radius(data,6)*.8*
            np.array([np.cos(a),np.sin(a),0])) for a in np.linspace(.2,2.9,7)]))
    old_points = seed.get_all_points().copy()
    clock = manim.ValueTracker(6.)
    branches = make_evolving_branches(seed,data,clock)
    assert len(branches) > 100
    assert all(curve.get_stroke_opacity() == 0 for curve in branches)
    for day in [6,8,12,18,24]:
        clock.set_value(day)
        branches.update(0)
        np.testing.assert_array_equal(seed.get_all_points(),old_points)
        for curve,record in zip(branches,branches.growth_records):
            assert curve.get_fill_opacity() == 0
            if day <= record["birth"]:
                assert curve.get_stroke_opacity() == 0
            if record["parent"] is not None:
                parent = branches.growth_records[record["parent"]]
                arrival = np.interp(record["contact_parameter"],parent["fractions"],parent["arrival_times"])
                assert record["birth"] > arrival
                np.testing.assert_allclose(record["template"].get_start(),
                    parent["template"].point_from_proportion(record["fraction"]),atol=1e-12)
            if curve.get_stroke_opacity() > 0:
                points = np.array([curve.get_nth_curve_function(i)(f)
                    for i in range(curve.get_num_curves()) for f in np.linspace(0,1,11)])
                radii = np.linalg.norm(points[:,:2]-ORIGIN_VIEW[:2],axis=1)
                assert radii.max() <= colony_view_radius(data,day)+.005
    # Once a piece of a filament is grown, it remains in exactly the same place.
    clock.set_value(12.)
    branches.update(0)
    prefix = branches[0].get_all_points()[:-4].copy()
    clock.set_value(24.)
    branches.update(0)
    np.testing.assert_array_equal(branches[0].get_all_points()[:len(prefix)],prefix)
    duplicate = make_evolving_branches(seed,data,clock)
    for a,b in zip(branches,duplicate):
        np.testing.assert_array_equal(a.points,b.points)


def test_both_sections_reuse_the_same_organic_branching_procedure():
    import hyphal_branching as shared
    import manim_phosphate_uptake as opening
    import manim_colony_uptake as ending
    for name in ["organic_filament","side_branches","growth_timings"]:
        assert getattr(opening,name) is getattr(shared,name)
        assert getattr(ending,name) is getattr(shared,name)


def test_front_uses_visible_tissue_not_hidden_branches_or_control_handles():
    old = manim.VGroup(manim.Line(ORIGIN_VIEW,ORIGIN_VIEW+manim.UP*.7))
    visible = manim.Line(ORIGIN_VIEW,ORIGIN_VIEW+manim.UP*1.2)
    hidden = manim.Line(ORIGIN_VIEW,ORIGIN_VIEW+manim.UP*3).set_stroke(opacity=0)
    bowed = manim.VMobject().set_points(np.array([ORIGIN_VIEW,
        ORIGIN_VIEW+manim.UP*4,ORIGIN_VIEW+manim.UP*4,ORIGIN_VIEW+manim.UP*.9]))
    extensions = manim.VGroup(visible,hidden,bowed)
    assert visible_colony_radius(old,extensions) == pytest.approx(1.2)
    front = make_colony_front(old,extensions)
    for length in [1.2,1.4,2.]:
        visible.put_start_and_end_on(ORIGIN_VIEW,ORIGIN_VIEW+manim.UP*length)
        front.update(0)
        radius = np.linalg.norm(front.get_start()-ORIGIN_VIEW)
        assert length-.02 <= radius <= length


def test_front_never_exceeds_furthest_visible_hypha_during_evolution(data):
    from manim_phosphate_uptake import PhosphateUptake
    scene = PhosphateUptake()
    originals,growth = scene._seed_branches()
    growth.set_value(1.)
    originals.update(0)
    originals.clear_updaters(recursive=True)
    focal = scene.focal_hypha
    originals.remove(focal)
    seed = returning_colony_target(manim.VGroup(originals,focal),data)
    clock = manim.ValueTracker(6.)
    branches = make_evolving_branches(seed,data,clock)
    front = make_colony_front(seed,branches)
    for day in np.linspace(6,24,325):
        clock.set_value(day)
        branches.update(0)
        front.update(0)
        anchors = np.concatenate([curve.get_anchors() for group in [seed,branches]
            for curve in group.family_members_with_points() if curve.get_stroke_opacity()>0])
        furthest = np.max(np.linalg.norm(anchors[:,:2]-ORIGIN_VIEW[:2],axis=1))
        radius = np.linalg.norm(front.get_start()-ORIGIN_VIEW)
        assert furthest-.02 <= radius <= furthest


def test_middle_section_duration_is_25_seconds():
    from manim_phosphate_uptake import middle_seconds
    # A marker/card, local evolution, continuous pullback, B–D arrivals,
    # final local hold, then return to the colony.
    durations = [.4,.3,.6,8.,.8,2.,2.5,.6,7.5,.6,7.5,.6,2.5,3.5,2.4,.6]
    assert sum(durations) == pytest.approx(40.4)
    assert sum(middle_seconds(t) for t in durations) == pytest.approx(25.)


def test_both_colony_sections_have_abundant_laterals_and_fine_twigs(data):
    from manim_phosphate_uptake import PhosphateUptake
    scene = PhosphateUptake()
    seed,growth = scene._seed_branches()
    growth.set_value(1.)
    seed.update(0)
    seed.clear_updaters(recursive=True)
    assert len(seed) > 220
    focal = scene.focal_hypha
    seed.remove(focal)
    target = returning_colony_target(manim.VGroup(seed,focal),data)
    extensions = make_evolving_branches(target,data,manim.ValueTracker(24.))
    orders = [record["order"] for record in extensions.growth_records]
    assert len(extensions) > 350
    assert orders.count(1) >= 8 * orders.count(0)
    assert orders.count(2) > orders.count(1)


def test_final_clock_has_one_uninterrupted_growth_animation():
    import ast
    import inspect
    from manim_colony_uptake import add_colony_phase
    tree = ast.parse(inspect.getsource(add_colony_phase))
    clock_advances = [node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute) and node.func.attr == "set_value"
        and ast.unparse(node.func.value) == "clock.animate"]
    assert len(clock_advances) == 1
    assert ast.literal_eval(clock_advances[0].args[0]) == 24.


def test_colony_title_is_centred_above_geometry_and_graphs(data):
    from manim_colony_uptake import make_colony_title
    title = make_colony_title()
    plots = ColonyPlots(data,manim.ValueTracker(24.)).compact()
    assert title.original_text == "Colony phosphate uptake"
    assert title.get_center()[0] == pytest.approx(0.)
    assert title.get_bottom()[1] > max(ORIGIN_VIEW[1]+VIEW_RADIUS,plots.get_top()[1])+.3
    assert title.get_top()[1] < 4.
