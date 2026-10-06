"""Page TOW + counted 4 ms epochs + fractional boundaries → transmit timestamps.
No absolute device timestamp is interpreted. Times are near a reference GST week,
not large epoch-second floats; this avoids losing metres to subtraction roundoff.
"""
import csv
import json
import math
from pathlib import Path
import numpy as np
from .common import FS
from .navigation import validate_ephemeris, decode_sisa, wrap_week_time
from .words import assemble_navigation
from .position_models import C, satellite_state

WEEK = 604800.


def week_delta(week, reference):
    return (int(week)-int(reference)+2048)%4096-2048


def load_channel(directory, config):
    directory=Path(directory)
    meta=json.loads((directory/'02_tracking/metadata.json').read_text())
    decoded=json.loads((directory/'03_telemetry/decoded_inav.json').read_text())
    if not meta.get('recording') or decoded.get('recording')!=meta['recording']:
        raise ValueError('Missing or mismatched recording identity in tracking and telemetry')
    if meta.get('boundary_sample_convention')!='start of each integration in recording sample coordinates':
        raise ValueError('Tracking must declare fractional START boundaries; rerun stage 02')
    if int(meta['acquisition']['prn'])!=int(decoded['satellite_prn']):
        raise ValueError('Tracking and telemetry PRNs differ')
    with np.load(directory/'02_tracking/continuous_tracking.npz',allow_pickle=False) as data:
        arrays={key:np.array(data[key]) for key in ('boundary_sample','raw_start_sample','carrier_lock','cn0_dbhz')}
    b=arrays['boundary_sample']
    if b.ndim!=1 or len(b)<2 or any(v.shape!=b.shape for v in arrays.values()):
        raise ValueError('Tracking arrays must be matching nonempty vectors')
    if not np.all(np.isfinite(b)) or np.any(np.diff(b)<=0):
        raise ValueError('Fractional boundaries must be finite and strictly increasing')
    if np.any(np.abs(b-arrays['raw_start_sample'])>.501):
        raise ValueError('Fractional boundaries disagree with raw integration starts')
    if abs(float(meta['recording']['sample_rate_sps'])-FS)>.001:
        raise ValueError('Unsupported recording sample rate')
    nav=assemble_navigation(decoded)
    ephems=nav.get('complete_ephemerides',[])
    if not ephems and nav['latest_complete_ephemeris']:
        ephems=[nav['latest_complete_ephemeris']]
    ephems=[e for e in ephems if all(c['pass'] for c in validate_ephemeris(e,decoded['satellite_prn']))]
    if not ephems: raise ValueError('No physically valid coherent CRC-passing ephemeris')
    messages=nav['messages']
    anchors=[]
    week_pages=[m for m in messages if m['word_type']==5 and 0<=m['decoded'].get('GST_TOW_s',-1)<WEEK and m.get('boundary_sample') is not None]
    if not week_pages: raise ValueError('No CRC-valid WT5 page with GST week and sample timing')
    reference=week_pages[0]['decoded']['GST_week']
    for m in messages:
        wt=m['word_type']; d=m['decoded']
        if wt not in (5,6) or m.get('boundary_sample') is None: continue
        tow=d.get('GST_TOW_s',d.get('TOW_s',-1))
        if not 0<=tow<WEEK: continue
        index=int(np.argmin(np.abs(b-float(m['boundary_sample']))))
        if abs(b[index]-m['boundary_sample'])>.001:
            raise ValueError('Decoded page boundary is not a saved tracking boundary')
        if wt==5:
            tx=week_delta(d['GST_week'],reference)*WEEK+tow
        else:
            nearest=min(week_pages,key=lambda p:abs(p['boundary_sample']-b[index]))
            nd=nearest['decoded']
            near=week_delta(nd['GST_week'],reference)*WEEK+nd['GST_TOW_s']
            predicted=near+(b[index]-nearest['boundary_sample'])/FS
            tx=tow+round((predicted-tow)/WEEK)*WEEK
        anchors.append(dict(index=index,sample=float(b[index]),tow_s=tow,tx_s=float(tx),word_type=wt))
    good=np.isfinite(arrays['carrier_lock']) & (arrays['carrier_lock']>=config.get('min_lock',.7))
    good &= np.isfinite(arrays['cn0_dbhz']) & (arrays['cn0_dbhz']>=config.get('min_cn0_dbhz',25.))
    contiguous=np.r_[False,np.abs(np.diff(b)/FS-.004)<.00004]
    segments=[]; rejected=[]; start=None
    for i in range(len(b)+1):
        valid=i<len(b) and good[i]
        if start is not None and (not valid or not contiguous[i]):
            stop=i
            aa=[a for a in anchors if start<=a['index']<stop]
            if aa:
                origins=np.array([a['tx_s']-.004*a['index'] for a in aa])
                residual=float(np.ptp(origins))
                if residual<=1e-6 and stop-start>=2:
                    segments.append(dict(start=start,stop=stop,origin_s=float(np.median(origins)),anchors=aa,anchor_spread_s=residual))
                else: rejected.append(dict(start=start,stop=stop,reason='page TOW / counted code epochs disagree',anchor_spread_s=residual))
            else: rejected.append(dict(start=start,stop=stop,reason='no CRC-valid time anchor in uninterrupted lock segment'))
            start=None
        if valid and start is None: start=i
    if not segments: raise ValueError('No uninterrupted, time-anchored tracking segment passes the quality gates')
    return dict(prn=int(decoded['satellite_prn']),directory=str(directory),recording=meta['recording'],
        capture_metadata=meta.get('capture_metadata',{}),boundaries=b,arrays=arrays,week=reference,
        segments=segments,rejected_segments=rejected,ephemerides=ephems,health_pages=week_pages,
        utc_pages=[m for m in messages if m['word_type']==6 and m.get('boundary_sample') is not None])


def observation(channel, sample, reference_week, receiver_origin_s, config, ionosphere=None):
    b=channel['boundaries']
    segment=next((s for s in channel['segments'] if b[s['start']]<=sample<=b[s['stop']-1]),None)
    if segment is None: raise ValueError('No time anchor reachable without crossing a tracking gap')
    left=int(np.searchsorted(b,sample,side='right')-1)
    left=min(left,segment['stop']-2)
    frac=(sample-b[left])/(b[left+1]-b[left])
    # Navigation TOW fixes integer code periods. Only the fraction is interpolated.
    epoch=left+frac
    tx=segment['origin_s']+.004*epoch+week_delta(channel['week'],reference_week)*WEEK
    origin=receiver_origin_s
    receive=origin+sample/FS
    raw=C*((origin-(segment['origin_s']+week_delta(channel['week'],reference_week)*WEEK)) + (sample/FS-.004*epoch))
    health=min(channel['health_pages'],key=lambda p:abs(p['boundary_sample']-sample))
    if abs(health['boundary_sample']-sample)/FS>config.get('max_ephemeris_age_s',14400.):
        raise ValueError('Signal health/BGD page too old')
    d=health['decoded']
    if d['E1B_HS']!=0 or d['E1B_DVS']!=0:
        raise ValueError('E1 signal health or data validity flag rejects this satellite')
    ephem=min(channel['ephemerides'],key=lambda e:abs(wrap_week_time(tx%WEEK-e['toe_s'])))
    age=max(abs(wrap_week_time(tx%WEEK-ephem['toe_s'])),abs(wrap_week_time(tx%WEEK-ephem['toc_s'])))
    if age>config.get('max_ephemeris_age_s',14400.): raise ValueError('Ephemeris or clock is stale')
    sisa=decode_sisa(int(ephem['SISA_index']))
    if sisa['kind']!='value': raise ValueError('SISA unavailable, reserved or NAPA')
    sat,clock,state,tx_gst=satellite_state(ephem,tx%WEEK,d['BGD_E1_E5b_s'])
    delay=config.get('common_delay_ns',0.)+config.get('differential_delay_ns',{}).get(str(channel['prn']),0.)
    result=dict(prn=channel['prn'],receive_sample=float(sample),receive_time_s=sample/FS,
        receiver_tow_unwrapped_s=receive,transmit_sv_s=tx,transmit_gst_tow_s=tx_gst%WEEK,
        raw_pseudorange_m=float(raw),satellite_ecef_m=sat,satellite_clock_e1_s=clock,
        clock_polynomial_s=state['clock_polynomial_s'],relativity_s=state['relativistic_correction_s'],
        bgd_e1_e5b_s=d['BGD_E1_E5b_s'],hardware_delay_m=delay*1e-9*C,
        differential_delay_sigma_m=config.get('differential_delay_sigma_ns',{}).get(str(channel['prn']),0.)*1e-9*C,
        sisa_m=sisa['metres'],iodnav=ephem['IODnav'],ephemeris_age_s=age,
        cn0_dbhz=float(np.interp(sample,b,channel['arrays']['cn0_dbhz'])),
        carrier_lock=float(np.interp(sample,b,channel['arrays']['carrier_lock'])),
        anchor_count=len(segment['anchors']),anchor_spread_s=segment['anchor_spread_s'],
        epoch_index=float(epoch),integer_code_periods=int(math.floor(epoch-segment['anchors'][0]['index'])))
    if ionosphere is not None:
        delay,sigma=ionosphere.at(channel['prn'],sample/FS)
        result.update(external_iono_m=delay,external_iono_sigma_m=sigma)
    return result


class SlantIonosphere:
    """Explicit E1 slant-code corrections; no extrapolation and no substitute for missing PRNs."""
    def __init__(self,path,max_gap_s=120.):
        self.path=str(Path(path).resolve()); self.max_gap=max_gap_s; self.rows={}
        with Path(path).open() as f:
            for row in csv.DictReader(f):
                prn=int(row['prn'])
                values=tuple(float(row[k]) for k in ('receive_time_s','delay_m','sigma_m'))
                if not all(math.isfinite(v) for v in values) or values[1]<0 or values[2]<=0:
                    raise ValueError('Ionosphere CSV needs finite times, nonnegative delay and positive sigma')
                self.rows.setdefault(prn,[]).append(values)
        for prn,rows in self.rows.items():
            data=np.asarray(sorted(rows)); self.rows[prn]=data
            if np.any(np.diff(data[:,0])<=0): raise ValueError('Duplicate ionosphere timestamps')
    def at(self,prn,time):
        if prn not in self.rows: raise ValueError('Missing E1 slant ionosphere correction for this PRN')
        data=self.rows[prn]; i=int(np.searchsorted(data[:,0],time))
        if i<len(data) and abs(data[i,0]-time)<1e-9: return tuple(data[i,1:])
        if i==0 or i==len(data) or data[i,0]-data[i-1,0]>self.max_gap:
            raise ValueError('Ionosphere correction cannot be extrapolated or bridged across a large gap')
        return tuple(float(np.interp(time,data[:,0],data[:,j])) for j in (1,2))
