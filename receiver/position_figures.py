"""The measured path to XYZ: distinguish clock, corrections, geometry and evidence."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import MaxNLocator
from .common import TEAL, GOLD, RED, MUTED, heading, finish
from .position_models import C, enu_matrix


def position_figures(summary, ranges, output):
    good=[e for e in summary['epochs'] if e.get('accepted')]
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    prns=sorted({r['prn'] for r in ranges})
    colours={prn:plt.get_cmap('tab20')(i%20) for i,prn in enumerate(prns)}
    for prn in prns:
        rr=[r for r in ranges if r['prn']==prn]
        axes[0,0].plot([r['receive_time_s'] for r in rr],[r['raw_pseudorange_m']/1e6 for r in rr],'.-',ms=3,label=f'E{prn:02d}',color=colours[prn])
    axes[0,0].set(title='01  One receive epoch, a different transmit time for each satellite',
        xlabel='Time from recording start [s]',ylabel='Raw pseudorange [thousand km]')
    if prns: axes[0,0].legend(ncol=4,fontsize=8)
    if good:
        last=good[-1]; at=[r for r in ranges if r['receive_time_s']==last['receive_time_s'] and r['used_in_accepted_solution']]
        terms=[('Clock + relativity',lambda r:C*(r['clock_polynomial_s']+r['relativity_s'])),
               ('−scaled BGD',lambda r:-C*r.get('bgd_factor',1.)*r.get('bgd_s',r.get('bgd_e1_e5b_s',0.))),('−hardware',lambda r:-r['hardware_delay_m']),
               ('−troposphere',lambda r:-r['tropo_m']),('−ionosphere',lambda r:-r['iono_m']),
               ('−Shapiro',lambda r:-r['shapiro_m'])]
        values=np.array([[fun(r) for r in at] for _,fun in terms])
        limit=max(1.,float(np.max(np.abs(values))))
        im=axes[0,1].imshow(values,aspect='auto',cmap='RdBu_r',norm=SymLogNorm(1.,vmin=-limit,vmax=limit))
        for i in range(len(terms)):
            for j in range(len(at)):
                value=values[i,j]
                label=f'{value:+.2f}' if abs(value)<100 else f'{value:+.0f}'
                axes[0,1].text(j,i,label,ha='center',va='center',fontsize=8,
                    color='white' if abs(float(im.norm(value))-.5)>.3 else '#183247')
        axes[0,1].set(xticks=np.arange(len(at)),xticklabels=[f"E{r['prn']:02d}" for r in at],
            yticks=np.arange(len(terms)),yticklabels=[label for label,_ in terms],
            title='02  Additive range corrections [m] · final accepted epoch')
        fig.colorbar(im,ax=axes[0,1],label='Signed logarithmic colour scale; linear within ±1 m',shrink=.8)
        tt=[e['receive_time_s'] for e in good]
        axes[1,0].errorbar(tt,[e['clock_bias_s']*1e6 for e in good],
            yerr=[e['clock_sigma_s']*1e6 for e in good],fmt='.-',capsize=2)
        axes[1,0].set(title='03  The fourth unknown is the receiver clock',xlabel='Time from recording start [s]',
            ylabel='Clock offset from chosen origin [µs; formal ±1σ]')
        axes[1,0].ticklabel_format(axis='y',style='plain',useOffset=False)
        axes[1,1].plot(tt,[e['pdop'] for e in good],label='PDOP')
        axes[1,1].set_ylim(0,max(6.,1.1*max(e['pdop'] for e in good)))
        axes[1,1].set(title='04  Geometry controls how range errors become position errors',xlabel='Time from recording start [s]',ylabel='PDOP')
        right=axes[1,1].twinx()
        right.step(tt,[len(e['observations']) for e in good],where='mid',color=GOLD)
        right.set_ylabel('Satellites used',color=GOLD)
        right.set_ylim(0,max(8,1+max(len(e['observations']) for e in good)))
        right.yaxis.set_major_locator(MaxNLocator(integer=True))
        result=f"{len(good)}/{summary['attempted_epochs']} epochs accepted · {summary['corrections']['ionosphere']}"
    else:
        for ax in (axes[0,1],axes[1,0],axes[1,1]): ax.set_axis_off()
        reasons=[r['reason'] for r in summary['channel_rejections']]
        reasons += [e.get('reason','Residual consistency check failed') for e in summary['epochs'] if not e.get('accepted')]
        reasons=list(dict.fromkeys(reasons)) or ['Need four healthy, time-anchored channels from one recording.']
        import textwrap
        text='\n\n'.join(textwrap.fill(r,65) for r in reasons[:6])
        axes[0,1].text(0,1,'Why a position is not available\n\n'+text,va='top',fontsize=11,transform=axes[0,1].transAxes)
        axes[1,0].text(0,1,'No nominal satellite ranges, missing time anchors,\nor fabricated coordinates are substituted.',va='top',transform=axes[1,0].transAxes)
        result='Insufficient consistent evidence · available range observations are retained'
    for ax in axes.flat: ax.grid(ax is not axes[0,1])
    heading(fig,'05','From code epochs to pseudoranges',result,
        'Page TOW fixes the integer code epochs; fractional boundaries align all channels at common receive samples.\nThe navigation-derived clock origin is arbitrary and solved out. Device timestamps are not assumed to be GST. Earth rotation acts on satellite coordinates.')
    finish(fig,output/'05_pseudoranges')
    if not good: return
    last=good[-1]; xyz=np.array([e['state'][:3] for e in good]); center=xyz[0]
    fig=plt.figure(figsize=(16,11))
    globe=fig.add_subplot(221,projection='3d')
    u,v=np.linspace(0,2*np.pi,60),np.linspace(0,np.pi,30)
    globe.plot_wireframe(np.outer(np.cos(u),np.sin(v)),np.outer(np.sin(u),np.sin(v)),
        np.outer(np.ones_like(u),np.cos(v)),color='#b3c9d2',lw=.35)
    normalized=xyz[-1]/np.linalg.norm(xyz[-1]); globe.scatter(*normalized,color=GOLD,s=100)
    globe.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),title='01  The estimated antenna position on Earth')
    globe.set_box_aspect((1,1,1)); globe.set_axis_off()
    globe.view_init(elev=last['latitude_deg'],azim=last['longitude_deg'])
    sky=fig.add_subplot(222,projection='polar')
    sky.set_theta_zero_location('N'); sky.set_theta_direction(-1); sky.set_ylim(0,90)
    sky.set_yticks([0,30,60,90]); sky.set_yticklabels(['90°','60°','30°','0°'])
    sky.set_title('02  Use geometry, not just acquisition strength',pad=25)
    for row in last['observations']:
        angle=math.radians(row['azimuth_deg']); radius=90-row['elevation_deg']
        sky.scatter(angle,radius,s=65,color=TEAL)
        sky.annotate(f"E{row['prn']:02d}",(angle,radius),xytext=(6,4),textcoords='offset points',fontsize=8)
    ax=fig.add_subplot(223)
    enu=(enu_matrix(center)@(xyz-center).T).T
    sc=ax.scatter(enu[:,0],enu[:,1],c=[e['receive_time_s'] for e in good],s=24,cmap='viridis')
    # Last-epoch horizontal ellipse is conditional covariance, not repeatability or truth accuracy.
    rotation=enu_matrix(center)
    covariance=rotation@np.asarray(last['covariance_xyz_clock_m2'])[:3,:3]@rotation.T
    values,vectors=np.linalg.eigh(covariance[:2,:2]); values=np.maximum(values,0)
    ellipse=Ellipse(enu[-1,:2],2*math.sqrt(5.991*values[1]),2*math.sqrt(5.991*values[0]),
        angle=math.degrees(math.atan2(vectors[1,1],vectors[0,1])),fill=False,color=GOLD,lw=1.7)
    ax.add_patch(ellipse); ax.autoscale_view(); ax.set_aspect('equal',adjustable='datalim')
    ax.set(title='03  Position scatter and a conditional 95% horizontal ellipse',xlabel='East from first accepted estimate [m]',ylabel='North from first accepted estimate [m]')
    ax.grid(True); fig.colorbar(sc,ax=ax,label='Recording time [s]',shrink=.75)
    ax=fig.add_subplot(224)
    for prn in prns:
        rr=[(e['receive_time_s'],d['residual_m']) for e in summary['epochs'] for d in e.get('observations',[])+e.get('excluded_observations',[]) if d['prn']==prn]
        if rr: ax.plot(*np.array(rr).T,'.-',ms=3,label=f'E{prn:02d}',color=colours[prn])
    for e in summary['epochs']:
        if not e.get('accepted'): ax.axvline(e['receive_time_s'],color=RED,alpha=.2)
    ax.axhline(0,color=MUTED,lw=.8); ax.grid(True); ax.legend(ncol=3,fontsize=8)
    ax.set(title='04  Keep the residuals, including rejected epochs',xlabel='Time from recording start [s]',ylabel='Post-fit code residual [m]')
    heading(fig,'05','A receiver position emerges from the ranges',
        f"{last['latitude_deg']:.7f}°  ·  {last['longitude_deg']:.7f}°  ·  h {last['ellipsoidal_height_m']:.1f} m  ·  PDOP {last['pdop']:.2f}",
        'Height is ellipsoidal, not height above sea level. Scatter is relative to an estimate, not surveyed truth. Ellipse assumes the stated range-error model.\nExactly four satellites have no residual redundancy. Multipath, ionosphere-model bias and uncalibrated differential hardware delays can dominate formal uncertainty.')
    finish(fig,output/'05_position_geometry')
