"""Illustrative geometry/transport-cost figure; does not modify the model.

Run from repository root with MPLCONFIGDIR=/tmp/mycormarl-mpl .venv/bin/python
    docs/technical-summaries/transport-shell-geometries/generate.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle

OUT = Path(__file__).resolve().parent
ROOT_DENSITY = 1.0  # cm root / cm^3 soil; existing reference-density proxy
ELL = 2 / (3 * np.sqrt(np.pi * ROOT_DENSITY))
Z_EDGES = np.linspace(0., 30., 31)
BETA = .96
# Same depth dependence as the production stack-of-discs closure, with a
# deliberately illustrative surface-disc radius rather than a fitted biomass.
weights = BETA**Z_EDGES[:-1] - BETA**Z_EDGES[1:]
R_ROOT = 18. * np.sqrt(weights / weights[0])
CASES = [('A', 0., 12., 'Filled hemisphere', 'Entirely within roots'),
         ('B', 0., 24., 'Filled hemisphere', 'Partly outside roots'),
         ('C', 9., 12., 'Empty centre', 'Entirely within roots'),
         ('D', 9., 24., 'Empty centre', 'Partly outside roots')]


def distance_to_roots(r, z):
    """Exact distance to union of finite-thickness root discs at sampled points."""
    d2 = np.full(np.broadcast_shapes(r.shape, z.shape), np.inf)
    for lo, hi, radius in zip(Z_EDGES[:-1], Z_EDGES[1:], R_ROOT):
        dr = np.maximum(r-radius, 0.)
        dz = np.maximum(np.maximum(lo-z, z-hi), 0.)
        d2 = np.minimum(d2, dr*dr+dz*dz)
    return np.sqrt(d2)


def quadrature(n):
    # Midpoint quadrature in spherical coordinates: dV=2*pi*s^2 ds dmu,
    # mu=cos(theta) in [0,1]. This handles shell limits exactly and includes
    # the axisymmetric volume weighting, not a uniform average of r-z pixels.
    mu = (np.arange(n)+.5)/n
    stats = []
    for label, a, b, _, _ in CASES:
        s = a+(np.arange(n)+.5)*(b-a)/n
        rr = s[:, None]*np.sqrt(1-mu[None, :]**2)
        zz = s[:, None]*mu[None, :]
        delta = distance_to_roots(rr, zz)
        w = np.broadcast_to(s[:, None]**2, delta.shape)
        gap = np.sum(w*delta)/np.sum(w)
        stats.append({'case':label, 'inner_radius_cm':a, 'outer_radius_cm':b,
                      'mean_intercell_envelope_gap_cm':float(gap),
                      'mean_root_distance_cm':float(ELL+gap),
                      'contact_volume_fraction':float(np.sum(w*(delta==0))/np.sum(w)),
                      'hyphal_volume_cm3':float(2*np.pi/3*(b**3-a**3))})
    return stats


coarse = quadrature(320)
fine = quadrature(640)
assert fine[0]['mean_intercell_envelope_gap_cm'] == 0.
assert fine[2]['mean_intercell_envelope_gap_cm'] == 0.
assert fine[3]['mean_root_distance_cm'] > fine[1]['mean_root_distance_cm'] > ELL
assert 0 < fine[1]['contact_volume_fraction'] < 1
assert 0 < fine[3]['contact_volume_fraction'] < 1
# Since the removed inner ball is entirely root-containing, the unnormalised
# excess-distance integral is unchanged, even though its network mean rises.
expected = fine[1]['mean_intercell_envelope_gap_cm'] * 24**3/(24**3-9**3)
assert abs(fine[3]['mean_intercell_envelope_gap_cm']-expected) < 2e-4
errors=[abs(c['mean_root_distance_cm']-f['mean_root_distance_cm']) for c,f in zip(coarse,fine)]
assert max(errors) < 2e-4
# Independent 1D check of the concentric-hemisphere closed form, including
# the no-contact situation (its geometric distance is not authorised trade).
checks=[]
for a,b,R in [(0.,24.,16.),(9.,24.,16.),(18.,24.,16.),(0.,12.,16.),(11.9,12.,16.)]:
    s=np.linspace(a,b,100001)
    numerical=np.trapezoid(s*s*np.maximum(s-R,0),s)/((b**3-a**3)/3)
    if b<=R:
        exact=0.
    elif a<R:
        exact=(3*b**4-4*R*b**3+R**4)/(4*(b**3-a**3))
    else:
        exact=3*(b**4-a**4)/(4*(b**3-a**3))-R
    assert abs(numerical-exact)<1e-7
    checks.append({'a':a,'b':b,'R':R,'mean_gap_closed_form':exact,'quadrature':float(numerical)})

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                     'mathtext.fontset':'dejavusans','axes.titleweight':'bold'})
BG='#f6f8fa'; GREEN='#80b99c'; ORANGE='#e4a34e'; INK='#183444'; MUTED='#516573'
fig=plt.figure(figsize=(19,14),facecolor=BG)
fig.text(.055,.955,'Transport cost across a living hyphal network',fontsize=25,weight='bold',color=INK)
fig.text(.055,.925,'Four geometries  |  persistent root footprint + living hyphal volume  |  illustrative dimensions in cm',fontsize=12,color=MUTED)
fig.text(.055,.882,r'$d(x)=\delta(x)+\ell_\lambda,\quad \ell_\lambda=2/(3\sqrt{\pi\lambda_{\rm root}}),\qquad c_{\rm tr}=\epsilon_d\,\overline{d},\qquad \dot C_{\rm tr}=\epsilon_d\,\dot Q\,\overline{d}$',fontsize=18,color=INK)
fig.text(.055,.853,'Cost per unit transported resource; uniform transported amount per living hyphal length assumed. No capacity penalty.',fontsize=11,color=MUTED)
gs=fig.add_gridspec(2,2,left=.055,right=.96,top=.817,bottom=.16,wspace=.11,hspace=.18)
for index, (case, stat) in enumerate(zip(CASES,fine)):
    label,a,b,rowtitle,coltitle=case
    inner=gs[index//2,index%2].subgridspec(1,2,width_ratios=[.48,.52],wspace=.08)
    ax=fig.add_subplot(inner[0,0]); tx=fig.add_subplot(inner[0,1]); tx.axis('off')
    ax.set_facecolor('white')
    ax.set_title(f'{label}  {rowtitle}',loc='left',fontsize=14,color=INK,pad=14)
    r=(np.arange(420)+.5)*27/420
    z=(np.arange(480)+.5)*31/480
    rr,zz=np.meshgrid(r,z)
    root=np.zeros(rr.shape,bool)
    for lo,hi,radius in zip(Z_EDGES[:-1],Z_EDGES[1:],R_ROOT):
        root|=(zz>=lo)&(zz<hi)&(rr<radius)
    ss=np.sqrt(rr*rr+zz*zz)
    h=(ss>=a)&(ss<=b)
    # Single RGB image avoids misleading transparency at occupied overlaps.
    rgb=np.ones((*rr.shape,3))
    rgb[root]=matplotlib.colors.to_rgb('#cfe4d9')
    rgb[h]=matplotlib.colors.to_rgb('#f3c98f')
    rgb[root & h]=matplotlib.colors.to_rgb('#a7bf91')
    ax.imshow(rgb,extent=(0,27,31,0),aspect='equal',interpolation='nearest')
    stairs_r=np.repeat(R_ROOT,2)
    stairs_z=np.column_stack((Z_EDGES[:-1],Z_EDGES[1:])).ravel()
    ax.plot(stairs_r,stairs_z,color='#397b5b',lw=1.8)
    theta=np.linspace(0,np.pi/2,500)
    ax.plot(b*np.sin(theta),b*np.cos(theta),color='#b96b17',lw=2)
    if a:
        ax.plot(a*np.sin(theta),a*np.cos(theta),color='#b96b17',lw=1.6,ls='--')
        ax.text(1.1,3.7,'No living\nhyphae',fontsize=10,color=INK)
    if index%2:
        source_r,source_z=22.,5.
        options=[]
        for lo,hi,radius in zip(Z_EDGES[:-1],Z_EDGES[1:],R_ROOT):
            dest_r=min(source_r,radius); dest_z=np.clip(source_z,lo,hi)
            options.append(((source_r-dest_r)**2+(source_z-dest_z)**2,dest_r,dest_z))
        _,dest_r,dest_z=min(options)
        ax.annotate('',xy=(source_r,source_z),xytext=(dest_r,dest_z),
                    arrowprops={'arrowstyle':'<->','lw':1.8,'color':INK})
        ax.plot(source_r,source_z,'o',color=INK,ms=4)
        ax.text((source_r+dest_r)/2, min(source_z,dest_z)-1.1,r'$\delta$',color=INK,fontsize=13)
    ax.scatter([0],[0],s=22,color=INK,zorder=8,clip_on=False)
    ax.set_xlim(0,27);ax.set_ylim(31,0)
    ax.set_xlabel(r'Radius $r$ (cm)',fontsize=10)
    ax.set_ylabel(r'Depth $z$ (cm)',fontsize=10)
    ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20,30]);ax.tick_params(labelsize=9)
    ax.spines[['top','right']].set_visible(False)
    tx.text(0,.97,coltitle,fontsize=14,weight='bold',color=INK,va='top')
    tx.text(0,.85,r'$\mathcal{H}_{a,b}=\{x:a\leq\|x\|\leq b,\ z\geq0\}$',fontsize=12,color=INK)
    tx.text(0,.75,rf'$a={a:g},\quad b={b:g}$',fontsize=13,color=INK)
    if index%2==0:
        tx.text(0,.61,r'$\mathcal{H}_{a,b}\subseteq\mathcal{R}\ \Rightarrow\ \delta(x)=0$',fontsize=13,color=INK)
        tx.text(0,.47,r'$\overline{d}=\ell_\lambda$',fontsize=19,color=INK)
        tx.text(0,.34,r'$c_{\rm tr}=\epsilon_d\,\ell_\lambda$',fontsize=17,color=INK)
        explanation='An empty centre changes living length,\nbut not cost per unit transported\nwhen all surviving hyphae remain inside roots.' if a else 'All hyphae pay the same unresolved\nroot-spacing distance.'
    else:
        tx.text(0,.61,r'$\delta(x)=\operatorname{dist}(x,\mathcal{R})$',fontsize=13,color=INK)
        tx.text(0,.46,r'$\overline{d}=\ell_\lambda+\frac{\int_{\mathcal{H}\setminus\mathcal{R}}\lambda_h\,\delta\,dV}{\int_{\mathcal{H}}\lambda_h\,dV}$',fontsize=17,color=INK)
        tx.text(0,.32,r'$c_{\rm tr}=\epsilon_d\,\overline{d}$',fontsize=17,color=INK)
        explanation='Only outside hyphae add a gap term.\nRemoving inner, near-root hyphae can\nraise the mean distance of survivors.' if a else 'Average the outside gap over the\nentire living hyphal length, including\ninside hyphae with zero gap.'
    tx.text(0,.19,explanation,fontsize=10.5,color=MUTED,va='top',linespacing=1.5)
    tx.text(0,-.005,rf'Illustration: $\overline{{d}}={stat["mean_root_distance_cm"]:.3f}$ cm',fontsize=12,color=INK,weight='bold')
legend=[Patch(facecolor='#cfe4d9',edgecolor='#397b5b',label='Root footprint only'),
        Patch(facecolor='#f3c98f',edgecolor='#b96b17',label='Hyphae outside roots'),
        Patch(facecolor='#a7bf91',label='Living hyphae + root footprint')]
fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(.05,.103),ncol=3,frameon=False,fontsize=11)
fig.text(.055,.082,r'Axisymmetric $r$-$z$ sections: rotate about the depth axis; $dV=2\pi r\,dr\,dz$. Root footprint uses the model\'s stacked discs.',fontsize=10.5,color=MUTED)
fig.text(.055,.057,r'$\overline{d}$ is length-weighted. With heterogeneous resource flux, replace $\lambda_h$ by that flux density. $\lambda_{\rm root}=1\ \mathrm{cm}^{-2}$ here.',fontsize=10.5,color=MUTED)
fig.text(.055,.032,'All four illustrated networks retain root contact. No root contact => no bilateral trade, even if geometric distance is finite.',fontsize=10.5,color=INK)
fig.savefig(OUT/'transport-shell-costs.png',dpi=180,facecolor=BG)
fig.savefig(OUT/'transport-shell-costs.pdf',facecolor=BG)
fig.savefig(OUT/'transport-shell-costs.svg',facecolor=BG)
plt.close(fig)
result={'status':'illustrative quadrature, not physiological calibration',
        'root_spacing_cm':float(ELL),'reference_root_length_density_cm_cm3':ROOT_DENSITY,
        'quadrature':'midpoints in s and mu=cos(theta), weights s^2',
        'coarse_resolution':[320,320],'fine_resolution':[640,640],
        'absolute_distance_refinement_differences_cm':errors,'cases':fine,
        'concentric_closed_form_checks':checks}
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'cases':fine,'refinement_errors_cm':errors},indent=2))
