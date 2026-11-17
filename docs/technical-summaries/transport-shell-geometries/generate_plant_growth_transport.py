"""Single-panel plant growth-front-to-stem transport-cost illustration.

Illustrative geometry and unfitted cost law; no production model changes.
Run with MPLCONFIGDIR=/tmp/mycormarl-mpl .venv/bin/python <this file>.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT=Path(__file__).resolve().parent
Z_EDGES=np.arange(0.,31.,1.)
BETA=.96
LAMBDA_NEW=1.0
depth_weights=BETA**Z_EDGES[:-1]-BETA**Z_EDGES[1:]
R_OLD=18.*np.sqrt(depth_weights/depth_weights[0])
R_NEW=R_OLD*(19./18.)


def mean_distance(n, independent=False):
    """Integrate the actual new annular bands with 3D volume weights."""
    nodes,weights=np.polynomial.legendre.leggauss(n)
    numerator=0.
    volume=0.
    for lo,hi,a,b in zip(Z_EDGES[:-1],Z_EDGES[1:],R_OLD,R_NEW):
        z=(lo+hi)/2+(hi-lo)/2*nodes
        wz=(hi-lo)/2*weights
        volume+=np.pi*(b*b-a*a)*(hi-lo)
        if independent:
            r=(a+b)/2+(b-a)/2*nodes
            wr=(b-a)/2*weights
            values=2*np.pi*r[:,None]*np.sqrt(r[:,None]**2+z[None,:]**2)
            numerator+=np.sum(wr[:,None]*wz[None,:]*values)
        else:
            # Analytically integrate 2*pi*r*sqrt(r^2+z^2) over radius.
            radial=(2*np.pi/3)*((b*b+z*z)**1.5-(a*a+z*z)**1.5)
            numerator+=np.sum(wz*radial)
    return numerator/volume,volume


coarse,volume=mean_distance(32)
fine,_=mean_distance(64)
independent,_=mean_distance(24,independent=True)
assert abs(coarse-fine)<1e-10
assert abs(independent-fine)<1e-10
expected_layer_volumes=np.pi*(R_NEW**2-R_OLD**2)*np.diff(Z_EDGES)
assert np.allclose(expected_layer_volumes/np.sum(expected_layer_volumes),
                   depth_weights/np.sum(depth_weights),rtol=1e-12,atol=1e-14)
assert np.all(R_NEW>R_OLD)
# A coarse positive-weight sampling gives an independent bounds check.
min_d=min(np.sqrt(R_OLD**2+Z_EDGES[:-1]**2))
max_d=max(np.sqrt(R_NEW**2+Z_EDGES[1:]**2))
assert min_d<fine<max_d

BG='#f6f8fa';INK='#183444';MUTED='#516573'
OLD='#c9d7ce';NEW='#97b785';EDGE='#397b5b';OLD_EDGE='#718679'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                     'mathtext.fontset':'dejavusans','axes.titleweight':'bold'})
fig=plt.figure(figsize=(13.6,10.5),facecolor=BG)
fig.text(.055,.95,'Plant transport from the growth front to the stem',fontsize=24,weight='bold',color=INK)
fig.text(.055,.911,'New fine-root growth bands  |  one geometry  |  illustrative distance-based carbon expenditure',fontsize=11.5,color=MUTED)
fig.text(.055,.853,r'$\Delta C_{\rm tr,p}=\epsilon_p\,\Delta C_{G,p}\,\overline{d}_{p,t}$',fontsize=25,color=INK)
fig.text(.055,.812,r'$\Delta C_{G,p}$: carbon delivered for fine-root construction; $\epsilon_p$: extra transport C per delivered C per cm (unfitted).',fontsize=10.5,color=MUTED)
gs=fig.add_gridspec(1,2,left=.055,right=.96,top=.745,bottom=.205,width_ratios=[.44,.56],wspace=.15)
ax=fig.add_subplot(gs[0,0]);tx=fig.add_subplot(gs[0,1]);tx.axis('off')
r=(np.arange(500)+.5)*24/500
z=(np.arange(650)+.5)*31/650
rr,zz=np.meshgrid(r,z)
old=np.zeros(rr.shape,bool);new=np.zeros_like(old)
for lo,hi,a,b in zip(Z_EDGES[:-1],Z_EDGES[1:],R_OLD,R_NEW):
    layer=(zz>=lo)&(zz<hi)
    old|=layer&(rr<a)
    new|=layer&(rr>=a)&(rr<b)
rgb=np.ones((*rr.shape,3))
rgb[old]=matplotlib.colors.to_rgb(OLD)
rgb[new]=matplotlib.colors.to_rgb(NEW)
ax.imshow(rgb,extent=(0,24,31,0),aspect='equal',interpolation='nearest')
zz_stair=np.column_stack((Z_EDGES[:-1],Z_EDGES[1:])).ravel()
ax.plot(np.repeat(R_OLD,2),zz_stair,color=OLD_EDGE,lw=1.8,ls=(0,(4,3)))
ax.plot(np.repeat(R_NEW,2),zz_stair,color=EDGE,lw=2)
# Stem glyph above soil; target is specifically its base/collar at (0,0).
ax.plot([0,0],[-3.2,0],color=EDGE,lw=4,clip_on=False)
ax.plot([0,1.8],[-2.,-3.],color=EDGE,lw=2,clip_on=False)
ax.scatter([0],[0],s=40,color=INK,zorder=10,clip_on=False)
ax.annotate('Central stem base',xy=(0,0),xytext=(3.8,-2.2),fontsize=10.5,color=INK,
            arrowprops=dict(arrowstyle='-',color=INK,lw=1),ha='left')
ax.axhline(0,color='#9aa8a5',lw=.8,zorder=0)
# Bidirectional dimension line: distance rather than a nutrient-flow direction.
k=15
sample_r=float((R_OLD[k]+R_NEW[k])/2);sample_z=15.5
ax.annotate('',xy=(sample_r,sample_z),xytext=(0,0),
            arrowprops=dict(arrowstyle='<->',color=INK,lw=1.7))
ax.plot(sample_r,sample_z,'o',color=INK,ms=4)
ax.text(sample_r*.40+1.0,sample_z*.43,r'$d_p(r,z)$',fontsize=13,color=INK,
        rotation=-48,rotation_mode='anchor')
ax.annotate('New fine-root\ngrowth',xy=(float((R_OLD[23]+R_NEW[23])/2),23.5),
            xytext=(16.4,25.0),fontsize=10.5,color=INK,
            arrowprops=dict(arrowstyle='-',color=INK,lw=1),ha='left')
ax.set_xlim(0,24);ax.set_ylim(31,-4)
ax.set_xlabel(r'Radius $r$ (cm)',fontsize=11);ax.set_ylabel(r'Depth $z$ (cm)',fontsize=11)
ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20,30]);ax.tick_params(labelsize=10)
ax.spines[['top','right']].set_visible(False)
tx.text(0,.97,'New growth in each depth layer',fontsize=16,weight='bold',color=INK,va='top')
tx.text(0,.858,r'$\mathcal{A}_{p,t}=\bigcup_k\{(r,z):$',fontsize=16,color=INK)
tx.text(.045,.772,r'$R_{k,t}\leq r\leq R_{k,t+\Delta t},\ z_k\leq z<z_{k+1}\}$',fontsize=13.5,color=INK)
tx.text(0,.672,r'$w_{k,t}=R_{k,t+\Delta t}-R_{k,t}$',fontsize=16,color=INK)
tx.text(0,.54,r'$d_p(r,z)=\sqrt{r^2+z^2}$',fontsize=23,color=INK)
tx.text(0,.462,'Straight-line distance to the stem base\nused as an explicit geometric proxy.',fontsize=11,color=MUTED,linespacing=1.5)
tx.text(0,.315,r'$\overline{d}_{p,t}=\frac{\int_{\mathcal{A}_{p,t}}\lambda_{\rm new}\,d_p\,dV}{\int_{\mathcal{A}_{p,t}}\lambda_{\rm new}\,dV}$',fontsize=22,color=INK)
tx.text(0,.166,'Average over all newly constructed root length,\nincluding the depth dependence of transport.',fontsize=11,color=MUTED,linespacing=1.5)
tx.text(0,.035,rf'Illustrative $\overline{{d}}_{{p,t}}={fine:.3f}$ cm',fontsize=15,weight='bold',color=INK)
fig.legend(handles=[Patch(facecolor=OLD,label='Established root region / implicit backbone'),
                    Patch(facecolor=NEW,label='New fine-root growth')],
           loc='lower left',bbox_to_anchor=(.05,.127),ncol=2,frameon=False,fontsize=11)
fig.text(.055,.105,'Dashed boundary: previous fine-root front. Solid boundary: new front. Each layer has its own radial growth width.',fontsize=10.5,color=INK)
fig.text(.055,.077,r'Axisymmetric $r$-$z$ section; rotate about the depth axis. $dV=2\pi r\,dr\,dz$. Dimensions are illustrative, not a fitted plant.',fontsize=10.5,color=MUTED)
fig.text(.055,.049,'The front-to-stem distance also prices outward carbon delivery for construction. Actual branched root paths are unresolved.',fontsize=10.5,color=MUTED)
fig.text(.055,.022,'Uniform delivery per unit new fine-root length assumed. Existing uptake and maintenance continue; no growth gives no construction-transport charge.',fontsize=10,color=MUTED)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'plant-growth-transport-costs.{ext}',facecolor=BG,**({'dpi':190} if ext=='png' else {}))
plt.close(fig)
result=dict(status='Illustrative plant growth transport geometry; no production implementation',
            root_geometry='30 depth layers, beta .96; top radius18 to19cm; proportional radii at depth',
            stem_base_cm=[0.,0.],distance_proxy='sqrt(r^2+z^2)',
            old_layer_radii_cm=R_OLD.tolist(),new_layer_radii_cm=R_NEW.tolist(),
            new_root_region_volume_cm3=float(volume),reference_new_root_density_cm_cm3=LAMBDA_NEW,
            mean_distance_cm=float(fine),coarse_GL_order=32,fine_GL_order=64,
            refinement_difference_cm=float(abs(coarse-fine)),
            independent_tensor_GL_order=24,independent_difference_cm=float(abs(independent-fine)),
            sampled_distance_bounds_cm=[float(min_d),float(max_d)],
            coefficient_value=None,coefficient_units='g C transport / (g C delivered * cm)',
            funded_stream='C delivered to fine-root construction, not whole-plant growth C')
(OUT/'plant-growth-transport-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['mean_distance_cm','refinement_difference_cm','independent_difference_cm','new_root_region_volume_cm3']},indent=2))
