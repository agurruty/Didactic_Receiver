"""Show the modelled delay and its effect on the solution without claiming measured accuracy."""
import numpy as np
import matplotlib.pyplot as plt
from .common import TEAL, GOLD, MUTED, heading, finish


def ionosphere_figure(summary, ranges, output):
    rows=[r for r in ranges if r.get('iono_model')=='NTCM-G v1.0' and r.get('used_in_accepted_solution')]
    if not rows: return
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    prns=sorted({r['prn'] for r in rows})
    colours={prn:plt.get_cmap('tab10')(i%10) for i,prn in enumerate(prns)}
    for prn in prns:
        rr=[r for r in rows if r['prn']==prn]
        axes[0,0].plot([r['receive_time_s'] for r in rr],[r['iono_m'] for r in rr],color=colours[prn],label=f'E{prn:02d}')
    axes[0,0].set(title='01  Estimate a delay along each satellite path',xlabel='Time from recording start [s]',ylabel='E1 ionospheric code delay [m; subtracted]')
    axes[0,0].set_ylim(bottom=0); axes[0,0].legend(ncol=4,fontsize=9)
    last=max(r['receive_time_s'] for r in rows)
    at=[r for r in rows if r['receive_time_s']==last]
    x=np.arange(len(at))
    axes[0,1].bar(x-.19,[r['iono_vtec_tecu'] for r in at],width=.38,color=TEAL,label='Vertical TEC at pierce point')
    axes[0,1].bar(x+.19,[r['iono_stec_tecu'] for r in at],width=.38,color=GOLD,label='Slant TEC along signal path')
    axes[0,1].set(xticks=x,xticklabels=[f"E{r['prn']:02d}" for r in at],title='02  Geometry turns vertical TEC into slant TEC',ylabel='Modelled electron content [TECU]')
    axes[0,1].legend(fontsize=9)
    compared=[e for e in summary['epochs'] if e.get('accepted') and e.get('ionosphere_comparison',{}).get('available')]
    if compared:
        values=np.array([e['ionosphere_comparison']['delta_enu_m'] for e in compared])
        for i,(label,colour) in enumerate(zip(('East','North','Up'),(TEAL,GOLD,'#6574ad'))):
            axes[1,0].plot([e['receive_time_s'] for e in compared],values[:,i],label=label,color=colour)
        axes[1,0].axhline(0,color=MUTED,lw=.8); axes[1,0].legend(ncol=3)
        axes[1,0].set(title='03  How applying the correction moves the solution',xlabel='Time from recording start [s]',ylabel='Corrected − uncorrected position [m]')
    else:
        axes[1,0].text(.05,.5,'No comparable uncorrected solution available.',transform=axes[1,0].transAxes)
    final=at[0]; context=final['broadcast_iono']; coeff=context['coefficients']
    axes[1,1].set_axis_off()
    axes[1,1].text(0,1,'04  From decoded broadcast data to a model estimate',weight='bold',fontsize=12,va='top',transform=axes[1,1].transAxes)
    text=[f"WT5 coefficients:  ai0 = {coeff[0]:g}   ai1 = {coeff[1]:g}   ai2 = {coeff[2]:g}",
          'One shared coefficient set; each satellite has a different pierce point.',
          '',f"Model UTC: {final['iono_utc'][:19].replace('T',' ')}",'Calendar era resolved separately; GST−UTC comes from decoded WT6.',
          '',f"Assumed residual zenith sigma: {summary['configuration']['iono_model_sigma_m']:g} m",
          'Uncertainty is retained. The model does not measure the actual delay.']
    if compared:
        c=compared[-1]['ionosphere_comparison']
        text+=['',f"Final shift: {c['horizontal_shift_m']:.2f} m horizontally; {c['delta_enu_m'][2]:+.2f} m up.",
               'This is a change in the estimate, not a measured improvement.']
    axes[1,1].text(0,.86,'\n'.join(text),fontsize=10,linespacing=1.55,va='top',transform=axes[1,1].transAxes)
    for ax in (axes[0,0],axes[0,1],axes[1,0]): ax.grid(True,axis='y')
    delays=[r['iono_m'] for r in rows]
    heading(fig,'05','Account for the ionosphere',f'NTCM-G · modelled E1 delay {min(delays):.2f}–{max(delays):.2f} m',
        'Broadcast coefficients + calendar time + path geometry → modelled slant electron content → E1 code-delay correction.\nBefore/after fits use the same satellites and ionosphere sigmas. Residual model bias, multipath and hardware delays remain; no surveyed truth is implied.')
    finish(fig,output/'05_ionosphere')
