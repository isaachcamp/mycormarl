"""Colony-scale appendix for the phosphate animation; no standalone scene."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from manim import *
from colony_uptake_video_data import simulate_colony
from hyphal_branching import organic_filament, side_branches, growth_timings, LATERAL_COUNT_RANGE

TEAL, GOLD, MUTED = "#7AE6CF", "#EDC77E", "#8FAAAE"
ORIGIN_VIEW = np.array([-1.55, -1.72, 0.])
VIEW_RADIUS = 3.45
SNAPSHOT_DAYS = [6,10,14,18,22]
SNAPSHOT_COLOURS = ["#258D76","#4DA76E","#82BD62","#B4CF57","#E4DC50"]
COLONY_ENTRY_DAY = 6.


def make_colony_title():
    """A quiet heading above the colony and its compact graph column."""
    return Text("Colony phosphate uptake",font_size=32,color="#F3E8D0").to_edge(UP,buff=.35)


def colony_view_radius(data, day):
    return VIEW_RADIUS * (data["initial_colony_radius_cm"] +
        data["radial_growth_cm_day"] * day) / data["parameters"]["domain_extent_cm"]


def returning_colony_target(colony, data):
    """An affine move of the existing colony, without replacing its branches."""
    target = colony.copy().clear_updaters(recursive=True)
    base = target[0][0].get_start().copy()
    points = target.get_all_points()
    radius = float(np.max(np.linalg.norm(points[:, :2]-base[:2],axis=1)))
    target.scale(colony_view_radius(data,COLONY_ENTRY_DAY)/radius,about_point=base)
    target.shift(ORIGIN_VIEW-base)
    return target


def make_evolving_branches(colony, data, clock):
    """Reuse the opening's curved runners, lateral generator and tip timing.

    The only additional constraint is the independent soil experiment's front.
    The existing colony stays fixed; new tissue follows fixed organic curves.
    """
    rng = np.random.default_rng(20261008)
    final_radius = colony_view_radius(data,24.)
    paths = []
    candidates = []
    for curve in colony[0]:
        start = curve.get_end().copy()
        relative = start-ORIGIN_VIEW
        if curve.get_stroke_width() < 1.:
            continue
        if np.linalg.norm(relative[:2]) < colony_view_radius(data,COLONY_ENTRY_DAY)*.45:
            continue
        if relative[1] <= .015:
            continue
        candidates.append(start)
    # Not every small terminal twig becomes a long exploratory runner. Keep a
    # sparse angular spread so the evolving topology and rings stay legible.
    candidates.sort(key=lambda p:np.arctan2(p[1]-ORIGIN_VIEW[1],p[0]-ORIGIN_VIEW[0]))
    indices = np.linspace(0,len(candidates)-1,min(14,len(candidates)),dtype=int)
    for index in indices:
        start = candidates[index]
        angle = np.arctan2(start[1]-ORIGIN_VIEW[1],start[0]-ORIGIN_VIEW[0])+rng.uniform(-.08,.08)
        end = ORIGIN_VIEW+final_radius*rng.uniform(.94,.985)*np.array([np.cos(angle),np.sin(angle),0])
        runner = organic_filament(start,end,rng,width=1.45)
        paths.append((runner,None,0.,0))
        # Same lateral angles, relative lengths, recursion and count as the
        # opening, adjusted only for the larger drawing's segment length.
        scale = float(np.linalg.norm(end-start))
        bounds = (ORIGIN_VIEW[:2]+[-final_radius,.01],ORIGIN_VIEW[:2]+[final_radius,final_radius])
        paths.extend(side_branches(runner,rng,int(rng.integers(*LATERAL_COUNT_RANGE)),.27*scale,bounds))
    if not paths:
        empty = VGroup()
        empty.growth_records = []
        return empty
    timings = growth_timings(paths,rng)
    finish = max(birth+duration for birth,duration in timings.values())
    time_scale = (24-COLONY_ENTRY_DAY)/finish
    records, lookup = [], {}
    front_speed = VIEW_RADIUS*data["radial_growth_cm_day"]/data["parameters"]["domain_extent_cm"]
    initial_radius = colony_view_radius(data,0.)
    for template,parent,fraction,order in paths:
        raw_birth,raw_duration = timings[id(template)]
        birth = COLONY_ENTRY_DAY+raw_birth*time_scale
        contact_parameter = None
        parent_index = None
        if parent is not None:
            parent_index = lookup[id(parent)]
            parent_record = records[parent_index]
            # point_from_proportion uses length-weighted curve segments,
            # whereas partial rendering uses equally weighted Bezier segments.
            lengths = np.array([length for _,length in parent.get_curve_functions_with_lengths()])
            cumulative = np.cumsum(lengths)
            target = fraction*cumulative[-1]
            segment = min(int(np.searchsorted(cumulative,target)),len(lengths)-1)
            previous = 0 if segment == 0 else cumulative[segment-1]
            contact_parameter = (segment+(target-previous)/lengths[segment])/len(lengths)
            arrival = np.interp(contact_parameter,parent_record["fractions"],parent_record["arrival_times"])
            pb,pd = timings[id(parent)]
            birth = arrival+(raw_birth-pb-pd*fraction)*time_scale
        duration = raw_duration*time_scale
        fractions = np.linspace(0,1,513)
        functions = [template.get_nth_curve_function(i) for i in range(template.get_num_curves())]
        points = []
        for f in fractions:
            i = min(int(f*len(functions)),len(functions)-1)
            points.append(functions[i](f*len(functions)-i))
        radii = np.maximum.accumulate(np.linalg.norm(np.array(points)[:,:2]-ORIGIN_VIEW[:2],axis=1))
        front_times = (radii-initial_radius)/front_speed
        # Waiting at the front delays subsequent growth too; never catch up
        # by instantaneously filling a curved segment behind the front.
        delay = np.maximum.accumulate(np.maximum(birth,front_times-duration*fractions))
        arrival_times = duration*fractions+delay
        lookup[id(template)] = len(records)
        records.append(dict(template=template,birth=float(birth),duration=float(duration),
            order=order,parent=parent_index,fraction=fraction,contact_parameter=contact_parameter,
            fractions=fractions,arrival_times=arrival_times))
    network = VGroup()
    network.growth_records = records
    for record in records:
        template = record["template"]
        curve = template.copy().set_fill(opacity=0)
        def update(mob,record=record,template=template):
            fraction = float(np.interp(clock.get_value(),record["arrival_times"],record["fractions"]))
            mob.pointwise_become_partial(template,0,max(fraction,1e-8))
            mob.set_fill(opacity=0).set_stroke(opacity=0 if fraction == 0 else (.95 if record["order"] == 0 else .8))
        curve.add_updater(update)
        curve.update(0)
        network.add(curve)
    return network


def recorded_index(data, day):
    """Use only completed output intervals, never future uptake."""
    return int(np.searchsorted(data["time_days"],day+1e-8,side="right")-1)


def visible_colony_radius(*groups):
    """Conservative radial extent from actual visible curve anchors/tips.

    Bezier handles are not tissue, and unrevealed branches must not enlarge
    the front. Every anchor lies on a visible filament, so this radius cannot
    exceed the furthest drawn hypha, even for a strongly bowed curve.
    """
    radius = 0.
    for group in groups:
        for curve in group.family_members_with_points():
            if np.max(curve.get_stroke_opacity()) <= 0:
                continue
            anchors = np.asarray(curve.get_anchors())
            if len(anchors):
                radius = max(radius,float(np.max(np.linalg.norm(anchors[:,:2]-ORIGIN_VIEW[:2],axis=1))))
    return radius


def make_colony_front(*groups):
    """Keep the front just inside the furthest currently visible tissue."""
    front = Arc(radius=1e-6,start_angle=0,angle=PI,arc_center=ORIGIN_VIEW,
        color=TEAL,stroke_width=1.5,stroke_opacity=.3)
    def update(mob):
        # Small inset also keeps the visible stroke behind the growing tips.
        radius = max(1e-6,visible_colony_radius(*groups)-.01)
        mob.become(Arc(radius=radius,start_angle=0,angle=PI,arc_center=ORIGIN_VIEW,
            color=TEAL,stroke_width=1.5,stroke_opacity=.3))
    front.add_updater(update)
    front.update(0)
    return front


def make_network(scene, data, clock):
    """Deterministic illustrative filaments revealed by the physical front.

    Curves have monotonically increasing distance from the shared origin.
    Their visible lengths follow the hemisphere radius, but this sparse
    drawing does not claim to resolve the model's volume-averaged density.
    """
    rng = np.random.default_rng(20261008)
    extent = data["parameters"]["domain_extent_cm"]
    records = []
    for angle in np.linspace(.06,PI-.06,13):
        radii = np.linspace(0,extent*rng.uniform(.93,1.),48)
        angles = angle + rng.uniform(-.07,.07)*np.sin(np.linspace(0,PI,48))
        records.append((radii,angles,0))
        for fraction in np.sort(rng.uniform(.12,.84,9)):
            start_r = radii[-1]*fraction
            start_angle = np.interp(start_r,radii,angles)
            end_r = min(start_r+extent*rng.uniform(.07,.24),extent*.99)
            child_r = np.linspace(start_r,end_r,18)
            sign = rng.choice([-1,1])
            child_angle = np.clip(start_angle+sign*rng.uniform(.1,.35)*np.linspace(0,1,18),.02,PI-.02)
            records.append((child_r,child_angle,1))
            for twig_fraction in rng.uniform(.25,.65,int(rng.integers(1,4))):
                twig_r0 = start_r+(end_r-start_r)*twig_fraction
                twig_a0 = np.interp(twig_r0,child_r,child_angle)
                twig_r = np.linspace(twig_r0,min(twig_r0+extent*rng.uniform(.02,.07),extent*.99),10)
                twig_angle = np.clip(twig_a0-sign*rng.uniform(.08,.24)*np.linspace(0,1,10),.01,PI-.01)
                records.append((twig_r,twig_angle,2))
    network = VGroup()
    maximum_flux = float(np.max(data["radial_flux_ng_mm3_day"]))
    for radii,angles,order in records:
        points = ORIGIN_VIEW + VIEW_RADIUS/extent*np.column_stack([radii*np.cos(angles),radii*np.sin(angles),np.zeros_like(radii)])
        curve = VMobject().set_points_as_corners(points)
        template = curve.copy()
        def update(mob,radii=radii,template=template,order=order):
            day = clock.get_value()
            radius = data["initial_colony_radius_cm"]+data["radial_growth_cm_day"]*day
            visible = float(np.interp(radius,radii,np.linspace(0,1,len(radii))))
            mob.pointwise_become_partial(template,0,max(visible,1e-7))
            idx = recorded_index(data,day)
            midpoint = (radii[0]+min(radius,radii[-1]))/2
            shell = int(np.clip(np.searchsorted(data["ring_edges_cm"],midpoint)-1,0,len(data["ring_radius_mm"])-1))
            flux = 0 if idx < 0 else data["radial_flux_ng_mm3_day"][idx,shell]
            activity = np.clip(flux/maximum_flux,0,1)
            colour = interpolate_color(ManimColor("#789699"),ManimColor(TEAL),float(activity))
            mob.set_stroke(colour,width=[1.35,.85,.6][order],opacity=0 if radius<=radii[0] else .9)
        curve.add_updater(update)
        curve.update(0)
        network.add(curve)
    return network


class ColonyPlots(VGroup):
    def __init__(self,data,clock):
        super().__init__()
        self.data, self.clock = data,clock
        self.flux_axes = Axes(x_range=[0,6,2],y_range=[0,5,1],x_length=4.45,y_length=1.65,
            axis_config={"include_tip":False,"color":MUTED,"stroke_width":.8}).move_to([3.25,1.28,0])
        ymax = np.ceil(float(np.max(data["total_uptake_nmol_day"])))
        self.total_axes = Axes(x_range=[0,24,6],y_range=[0,ymax,ymax/2],x_length=4.45,y_length=1.0,
            axis_config={"include_tip":False,"color":MUTED,"stroke_width":.8}).move_to([3.25,-1.75,0])
        self.add(RoundedRectangle(width=5.65,height=3.25,corner_radius=.14,
            fill_color="#0E1C27",fill_opacity=.8,stroke_color="#385060",stroke_width=.7).move_to([3.25,1.45,0]),
            RoundedRectangle(width=5.65,height=2.45,corner_radius=.14,
            fill_color="#0E1C27",fill_opacity=.8,stroke_color="#385060",stroke_width=.7).move_to([3.25,-1.68,0]),
            self.flux_axes,self.total_axes)
        self.add(Text("Radial P uptake",font_size=19,color=TEAL).move_to([3.25,2.79,0]),
            Text("ng P mm⁻³ day⁻¹ • shell-volume average",font_size=12,color=MUTED).move_to([3.25,2.48,0]),
            Text("distance from colony origin (mm)",font_size=12,color=MUTED).move_to([3.25,.04,0]),
            Text("Total network uptake • nmol P/day",font_size=16,color=TEAL).move_to([3.25,-.73,0]),
            Text("colony simulation time (days)",font_size=12,color=MUTED).move_to([3.25,-2.73,0]))
        for axes,xvalues,yvalues in [(self.flux_axes,[0,2,4,6],[0,2,4]),(self.total_axes,[0,6,12,18,24],[0,ymax/2,ymax])]:
            for value in xvalues:
                self.add(Text(f"{value:g}",font_size=11,color=MUTED).next_to(axes.c2p(value,0),DOWN,buff=.07))
            for value in yvalues:
                self.add(Text(f"{value:g}",font_size=11,color=MUTED).next_to(axes.c2p(0,value),LEFT,buff=.08))
        self.snapshots = VGroup()
        for day,colour in zip(SNAPSHOT_DAYS,SNAPSHOT_COLOURS):
            i = recorded_index(data,day)
            points = [self.flux_axes.c2p(0,data["radial_flux_ng_mm3_day"][i,0])]
            points += [self.flux_axes.c2p(r,q) for r,q in zip(data["ring_radius_mm"],data["radial_flux_ng_mm3_day"][i])]
            points.append(self.flux_axes.c2p(6,0))
            line = VMobject(stroke_color=colour,stroke_width=2).set_points_as_corners(points)
            line.add_updater(lambda mob,day=day:mob.set_fill(opacity=0).set_stroke(opacity=.9 if clock.get_value()+1e-8>=day else 0))
            self.snapshots.add(line)
        self.live = VMobject(stroke_color=TEAL,stroke_width=2)
        self.total = VMobject(stroke_color=TEAL,stroke_width=2.2)
        self.total_dot = Dot(radius=.035,color=TEAL)
        self.add(self.snapshots,self.live,self.total,self.total_dot)
        for j,(day,colour) in enumerate(zip(SNAPSHOT_DAYS,SNAPSHOT_COLOURS)):
            self.add(Text(f"{day} d",font_size=11,color=colour).move_to([1.5+j*.86,2.23,0]))
        self.refresh()

    def compact(self):
        """Keep both graphs beside, rather than across, the centred colony."""
        return self.scale(.67).move_to([4.82, 0., 0.])

    def refresh(self):
        i = recorded_index(self.data,self.clock.get_value())
        if i < 0:
            self.live.set_fill(opacity=0).set_stroke(opacity=0)
            self.total.set_fill(opacity=0).set_stroke(opacity=0)
            self.total_dot.set_opacity(0)
            return
        q = self.data["radial_flux_ng_mm3_day"][i]
        points = [self.flux_axes.c2p(0,q[0])]
        points += [self.flux_axes.c2p(r,rate) for r,rate in zip(self.data["ring_radius_mm"],q)]
        points.append(self.flux_axes.c2p(6,0))
        self.live.set_points_as_corners(points).set_fill(opacity=0).set_stroke(opacity=.7)
        colour = interpolate_color(ManimColor(TEAL),ManimColor(GOLD),min(self.clock.get_value()/24,1))
        self.live.set_stroke(colour)
        history = [self.total_axes.c2p(0,0)] + [self.total_axes.c2p(t,rate) for t,rate in zip(self.data["time_days"][:i+1],self.data["total_uptake_nmol_day"][:i+1])]
        self.total.set_points_as_corners(history).set_fill(opacity=0).set_stroke(opacity=1)
        self.total_dot.move_to(history[-1]).set_opacity(1)


def add_colony_phase(scene,cards,markers,focal,surroundings,original_points,note,return_time_scale=1.0):
    """Return to the original colony, then introduce an independent soil run."""
    for mob in [*cards,*markers,focal,note]:
        mob.clear_updaters(recursive=True)
    start_apex = float(focal.get_right()[0])
    start_left = float(focal.get_left()[0])
    start_centres = np.column_stack([np.linspace(start_left,start_apex,len(original_points)),
        np.full(len(original_points),original_points[-1,1]),np.zeros(len(original_points))])
    def restore(mob,alpha):
        centres = (1-alpha)*start_centres+alpha*original_points
        tangent = np.gradient(centres,axis=0)
        tangent /= np.linalg.norm(tangent,axis=1)[:,None]
        normal = np.column_stack([-tangent[:,1],tangent[:,0],np.zeros(len(centres))])
        radius = .025*(1-alpha)+.001*alpha
        bottom,top = centres-radius*normal,centres+radius*normal
        cap = np.array([centres[-1]+radius*(np.cos(a)*tangent[-1]+np.sin(a)*normal[-1]) for a in np.linspace(-PI/2,PI/2,25)])
        mob.set_points_as_corners(np.vstack([bottom,cap,top[::-1],bottom[:1]]))
        mob.set_fill(TEAL,opacity=.35*(1-alpha)).set_stroke(TEAL,width=2*scene.camera.frame.width/14.2222222222)
    scene.play(scene.camera.frame.animate.set(width=8.5).move_to([0,-1.5,0]),
        UpdateFromAlphaFunc(focal,restore),FadeIn(surroundings),
        *(FadeOut(card) for card in cards),*(FadeOut(marker) for marker in markers),FadeOut(note),run_time=2.4*return_time_scale)
    scene.wait(.6*return_time_scale)
    data = simulate_colony()
    scene.colony_data = data
    out = Path(config.media_dir)/"phosphate-uptake"/"colony"
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"rendered-data.npz",**{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
    (out/"rendered-metadata.json").write_text(json.dumps({k:v for k,v in data.items() if not isinstance(v,np.ndarray)},indent=2)+"\n")
    # Enter at an established colony state, retaining the complete earlier
    # simulation history in the graphs rather than visually restarting at zero.
    clock = ValueTracker(COLONY_ENTRY_DAY)
    returning_colony = VGroup(surroundings, focal)
    target = returning_colony_target(returning_colony,data)
    scene.play(scene.camera.frame.animate.set(width=14.2222222222).move_to(ORIGIN),
        Transform(returning_colony,target),run_time=2.)
    # The returned colony is now stationary; only newly constructed tips grow.
    extensions = make_evolving_branches(returning_colony,data,clock)
    scene.add(extensions)
    ring_edges = data["ring_edges_cm"]
    # Display every third fine analysis edge: 16 equal-area visible bands.
    grid = VGroup(*(Arc(radius=VIEW_RADIUS*r/ring_edges[-1],start_angle=0,angle=PI,
        arc_center=ORIGIN_VIEW,color="#61C5D9",stroke_width=.65,stroke_opacity=.38)
        for r in ring_edges[3::3]))
    grid.add(Line(ORIGIN_VIEW+LEFT*VIEW_RADIUS,ORIGIN_VIEW+RIGHT*VIEW_RADIUS,
        color="#61C5D9",stroke_width=.65,stroke_opacity=.5))
    title = make_colony_title()
    scene.play(Create(grid),FadeIn(title),run_time=1.5)
    plots = ColonyPlots(data,clock).compact()
    scene.play(FadeIn(plots),run_time=.8)
    plots.add_updater(lambda mob:mob.refresh())
    halo = make_colony_front(returning_colony,extensions)
    scene.add(halo)
    # Snapshot curves reveal themselves at their clock thresholds without holds.
    scene.play(clock.animate.set_value(24.),run_time=(24.-COLONY_ENTRY_DAY)*.6,rate_func=linear)
    scene.wait(3.)
