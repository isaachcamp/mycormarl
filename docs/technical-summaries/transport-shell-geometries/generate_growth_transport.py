"""Two-panel growth-shell carbon transport illustration, not production code.

From repository root:
MPLCONFIGDIR=/tmp/mycormarl-mpl .venv/bin/python \
    docs/technical-summaries/transport-shell-geometries/generate_growth_transport.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ROOT_DENSITY = 1.0
ELL = 2.0 / (3.0 * np.sqrt(np.pi * ROOT_DENSITY))
Z_EDGES = np.linspace(0., 30., 31)
BETA = .96
weights = BETA**Z_EDGES[:-1] - BETA**Z_EDGES[1:]
R_ROOT = 18. * np.sqrt(weights / weights[0])
CASES = [dict(label='A', before=11., after=12., title='Growth front entirely within roots'),
         dict(label='B', before=23., after=24., title='Growth front extends beyond roots')]


def distance_to_roots(r, z):
    """Distance to the union of root-containing finite-thickness discs."""
    d2 = np.full(np.broadcast_shapes(r.shape, z.shape), np.inf)
    for lo, hi, radius in zip(Z_EDGES[:-1], Z_EDGES[1:], R_ROOT):
        dr = np.maximum(r-radius, 0.)
        dz = np.maximum(np.maximum(lo-z, z-hi), 0.)
        d2 = np.minimum(d2, dr*dr + dz*dz)
    return np.sqrt(d2)


def nearest_root(r, z):
    candidates=[]
    for lo, hi, radius in zip(Z_EDGES[:-1], Z_EDGES[1:], R_ROOT):
        dest_r=min(r, radius)
        dest_z=np.clip(z, lo, hi)
        candidates.append(((r-dest_r)**2+(z-dest_z)**2, dest_r, dest_z))
    _, dest_r, dest_z=min(candidates)
    return float(dest_r), float(dest_z)


def shell_statistics(case, n):
    a,b=case['before'],case['after']
    mu=(np.arange(n)+.5)/n
    s=a+(np.arange(n)+.5)*(b-a)/n
    rr=s[:,None]*np.sqrt(1-mu[None,:]**2)
    zz=s[:,None]*mu[None,:]
    gap=distance_to_roots(rr,zz)
    w=np.broadcast_to(s[:,None]**2,gap.shape)
    mean_gap=float(np.sum(w*gap)/np.sum(w))
    return dict(case=case['label'], old_front_cm=a, new_front_cm=b,
                growth_width_cm=b-a, mean_footprint_gap_cm=mean_gap,
                mean_root_distance_cm=ELL+mean_gap,
                shell_fraction_inside_roots=float(np.sum(w*(gap==0))/np.sum(w)),
                new_shell_volume_cm3=float(2*np.pi/3*(b**3-a**3)))


coarse=[shell_statistics(c,320) for c in CASES]
fine=[shell_statistics(c,640) for c in CASES]
errors=[abs(c['mean_root_distance_cm']-f['mean_root_distance_cm']) for c,f in zip(coarse,fine)]
assert fine[0]['mean_footprint_gap_cm']==0.
assert abs(fine[0]['mean_root_distance_cm']-ELL)<1e-14
assert 0 < fine[1]['shell_fraction_inside_roots'] < 1
assert fine[1]['mean_root_distance_cm'] > ELL
assert max(errors)<2e-4
# Independent radial shell check: exact volume and its mean radial position.
checks=[]
for a,b in [(11.,12.),(23.,24.)]:
    s=np.linspace(a,b,100001)
    denom=(b**3-a**3)/3
    exact=3*(b**4-a**4)/(4*(b**3-a**3))
    numeric=np.trapezoid(s**3,s)/denom
    assert abs(exact-numeric)<1e-8
    checks.append(dict(inner=a,outer=b,mean_radius_exact=exact,mean_radius_numeric=float(numeric)))

BG='#f6f8fa'; INK='#183444'; MUTED='#516573'
ROOT='#d0e5d9'; OLD='#e3e8eb'; OLD_ROOT='#bfcec6'
NEW_ROOT='#97b785'; NEW_OUT='#efb15d'; ROOT_EDGE='#397b5b'; FRONT_EDGE='#b66a19'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                     'mathtext.fontset':'dejavusans','axes.titleweight':'bold'})
fig=plt.figure(figsize=(19,10.3),facecolor=BG)
fig.text(.05,.95,'Fungal carbon transport to the growth front',fontsize=25,weight='bold',color=INK)
fig.text(.05,.913,'Two root-overlap scenarios  |  only newly constructed hyphae set the transport-distance average',fontsize=12,color=MUTED)
fig.text(.05,.855,r'$\Delta C_{\rm tr}=\epsilon_C\,\Delta C_G\,\overline{d}_t,\qquad d(x)=\delta(x)+\ell_\lambda,\qquad \ell_\lambda=\frac{2}{3\sqrt{\pi\lambda_{\rm root}}}$',fontsize=21,color=INK)
fig.text(.05,.813,r'$\Delta C_G$: carbon delivered for growth; $\epsilon_C$: extra transport C per delivered C per cm (unfitted). Polyphosphate transport cost = 0.',fontsize=11,color=MUTED)
gs=fig.add_gridspec(1,2,left=.05,right=.96,top=.735,bottom=.225,wspace=.10)
for idx,(case,stat) in enumerate(zip(CASES,fine)):
    sub=gs[0,idx].subgridspec(1,2,width_ratios=[.46,.54],wspace=.09)
    ax=fig.add_subplot(sub[0,0]);tx=fig.add_subplot(sub[0,1]);tx.axis('off')
    x0=gs[0,idx].get_position(fig).x0
    fig.text(x0,.762,f"{case['label']}  {case['title']}",fontsize=16,weight='bold',color=INK)
    a,b=case['before'],case['after']
    r=(np.arange(550)+.5)*27/550
    z=(np.arange(620)+.5)*31/620
    rr,zz=np.meshgrid(r,z)
    root=np.zeros(rr.shape,bool)
    for lo,hi,radius in zip(Z_EDGES[:-1],Z_EDGES[1:],R_ROOT):
        root|=(zz>=lo)&(zz<hi)&(rr<radius)
    ss=np.sqrt(rr*rr+zz*zz)
    old=ss<a
    new=(ss>=a)&(ss<=b)
    rgb=np.ones((*rr.shape,3))
    for mask,colour in [(root,ROOT),(old,OLD),(root&old,OLD_ROOT),
                        (new,NEW_OUT),(root&new,NEW_ROOT)]:
        rgb[mask]=matplotlib.colors.to_rgb(colour)
    ax.imshow(rgb,extent=(0,27,31,0),aspect='equal',interpolation='nearest')
    ax.plot(np.repeat(R_ROOT,2),np.column_stack((Z_EDGES[:-1],Z_EDGES[1:])).ravel(),
            color=ROOT_EDGE,lw=1.8)
    theta=np.linspace(0,np.pi/2,600)
    ax.plot(a*np.sin(theta),a*np.cos(theta),color=FRONT_EDGE,lw=1.5,ls=(0,(4,3)))
    ax.plot(b*np.sin(theta),b*np.cos(theta),color=FRONT_EDGE,lw=2)
    ax.scatter([0],[0],s=23,color=INK,zorder=8,clip_on=False)
    # Shell callout uses a leader, not a fictitious resolved root spacing.
    angle=.96
    shell_pt=((a+b)/2*np.sin(angle),(a+b)/2*np.cos(angle))
    if idx==0:
        ax.annotate('New growth',xy=shell_pt,xytext=(12.4,12.0),fontsize=10,color=INK,
                    arrowprops=dict(arrowstyle='-',color=INK,lw=1),ha='left')
    else:
        sr,sz=23.1,5.1
        dr,dz=nearest_root(sr,sz)
        ax.annotate('',xy=(sr,sz),xytext=(dr,dz),
                    arrowprops=dict(arrowstyle='->',color=INK,lw=2))
        ax.plot(sr,sz,'o',color=INK,ms=4)
        ax.plot(dr,dz,'o',color=ROOT_EDGE,ms=4)
        ax.text((dr+sr)/2-.4, min(dz,sz)-1.0,r'C: $\delta(x)$',fontsize=11,color=INK,ha='center')
        ax.annotate('New growth',xy=(21.7,9.1),xytext=(17.0,15.7),fontsize=10,color=INK,
                    arrowprops=dict(arrowstyle='-',color=INK,lw=1),ha='left')
    ax.set_xlim(0,27);ax.set_ylim(31,0)
    ax.set_xlabel(r'Radius $r$ (cm)',fontsize=10)
    ax.set_ylabel(r'Depth $z$ (cm)',fontsize=10)
    ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20,30]);ax.tick_params(labelsize=9)
    ax.spines[['top','right']].set_visible(False)
    tx.text(0,.97,r'$\mathcal{A}_t=\{x:R_t\leq\|x\|\leq R_{t+\Delta t},\ z\geq0\}$',fontsize=12,color=INK,va='top')
    tx.text(0,.865,rf'$R_t={a:g},\quad R_{{t+\Delta t}}={b:g}$ cm',fontsize=12,color=MUTED)
    tx.text(0,.775,r'$w_t=R_{t+\Delta t}-R_t$',fontsize=14,color=INK)
    if idx==0:
        tx.text(0,.63,r'$\mathcal{A}_t\subseteq\mathcal{R}\ \Rightarrow\ \delta(x)=0$',fontsize=15,color=INK)
        tx.text(0,.485,r'$\overline{d}_t=\ell_\lambda$',fontsize=21,color=INK)
        tx.text(0,.34,r'$\Delta C_{\rm tr}=\epsilon_C\,\Delta C_G\,\ell_\lambda$',fontsize=17,color=INK)
        detail='Only unresolved distance to a root\ncontributes to carbon-delivery cost.'
    else:
        tx.text(0,.63,r'$\delta(x)=\operatorname{dist}(x,\mathcal{R})$',fontsize=15,color=INK)
        tx.text(0,.48,r'$\overline{d}_t=\ell_\lambda+\frac{\int_{\mathcal{A}_t\setminus\mathcal{R}}\lambda_{\rm new}\delta\,dV}{\int_{\mathcal{A}_t}\lambda_{\rm new}\,dV}$',fontsize=18,color=INK)
        tx.text(0,.32,r'$\Delta C_{\rm tr}=\epsilon_C\,\Delta C_G\,\overline{d}_t$',fontsize=17,color=INK)
        detail='Outside growth adds a root-zone gap.\nAverage over the entire new shell,\nincluding its portion within roots.'
    tx.text(0,.21,detail,fontsize=11,color=MUTED,va='top',linespacing=1.45)
    tx.text(0,.035,rf'Illustrative $\overline{{d}}_t={stat["mean_root_distance_cm"]:.3f}$ cm',fontsize=13,weight='bold',color=INK)
legend=[Patch(facecolor=ROOT,edgecolor=ROOT_EDGE,label='Root footprint'),
        Patch(facecolor=OLD,label='Existing mycelium'),
        Patch(facecolor=NEW_ROOT,label='New growth within roots'),
        Patch(facecolor=NEW_OUT,label='New growth outside roots')]
fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(.045,.133),ncol=4,frameon=False,fontsize=11)
fig.text(.05,.106,'Dashed arc: previous front. Solid arc: new front. The highlighted band is the current step\'s radial expansion.',fontsize=11,color=INK)
fig.text(.05,.077,r'Axisymmetric $r$-$z$ sections; rotate about the depth axis. $dV=2\pi r\,dr\,dz$. Existing mycelium remains uptake-active.',fontsize=10.5,color=MUTED)
fig.text(.05,.049,r'Uniform carbon delivery per unit new hyphal length assumed. $\lambda_{\rm root}=1\ \mathrm{cm}^{-2}$ is the reference-density proxy; dimensions are illustrative.',fontsize=10.5,color=MUTED)
fig.text(.05,.022,'No growth => no construction-transport charge. Ordinary maintenance and uptake continue. No transport-capacity penalty is modelled.',fontsize=10.5,color=MUTED)
for ext in ['png','pdf','svg']:
    kwargs={'dpi':190} if ext=='png' else {}
    fig.savefig(OUT/f'fungal-growth-transport-costs.{ext}',facecolor=BG,**kwargs)
plt.close(fig)
result=dict(status='Illustrative mathematical figure, not a fitted transport model',
            assumed_transport_stream='Carbon delivered for growth',
            extra_P_translocation_cost=0,
            carbon_cost_law='Delta_C_tr = epsilon_C * Delta_C_G * mean_distance',
            epsilon_C_units='g C spent / (g C delivered * cm)',
            epsilon_C_value=None,
            reference_root_density_cm_cm3=ROOT_DENSITY,
            spacing_closure='2/(3*sqrt(pi*lambda_root))',spacing_cm=float(ELL),
            geometry='30 stacked root discs, beta .96, first-layer radius18cm',
            coarse_resolution=[320,320],fine_resolution=[640,640],
            refinement_errors_cm=errors,cases=fine,mean_radius_checks=checks)
(OUT/'growth-transport-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'cases':fine,'refinement_errors_cm':errors},indent=2))
