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


def load_channel(directory, config, additional_navigation=()):
    directory=Path(directory)
    meta=json.loads((directory/'02_tracking/metadata.json').read_text())
    band=meta.get('band','e1')
    from .signals import SIGNALS
    signal=SIGNALS[band]
    FS=float(meta['recording']['sample_rate_sps'])
    signal.validate_rate(FS)
    period=signal.period
    filename='decoded_fnav.json' if signal.navigation=='F/NAV' else 'decoded_inav.json'
    decoded=json.loads((directory/'03_telemetry'/filename).read_text())
    if decoded.get('band',band)!=band: raise ValueError('Tracking and telemetry bands differ')
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
    if additional_navigation:
        if band not in ('e1','e5b'): raise ValueError('Only E1/E5b share I/NAV ephemeris assembly.')
        merged=[]
        for source in (decoded,*additional_navigation):
            if source.get('band','e1') not in ('e1','e5b') or source['satellite_prn']!=decoded['satellite_prn']:
                raise ValueError('Cannot merge different satellites or navigation formats.')
            if source is not decoded:
                from .multiband_position import validate_session
                validate_session([{'capture_metadata':meta['capture_metadata']},{'capture_metadata':source.get('capture_metadata',{})}])
            source_rate=float(source['recording']['sample_rate_sps'])
            for page in source.get('pages',[]):
                # Only orbit/clock pages are combined. Local time anchors and
                # signal health remain tied to their independently decoded band.
                if page.get('word_type') in (1,2,3,4) and page.get('boundary_sample') is not None:
                    merged.append(dict(page,time_start_s=page['boundary_sample']/source_rate,
                                       source_band=source.get('band','e1')))
        fused=assemble_navigation(dict(decoded,pages=merged))
        nav['complete_ephemerides']=fused['complete_ephemerides']
        nav['latest_complete_ephemeris']=fused['latest_complete_ephemeris']
    ephems=nav.get('complete_ephemerides',[])
    if not ephems and nav['latest_complete_ephemeris']:
        ephems=[nav['latest_complete_ephemeris']]
    ephems=[e for e in ephems if all(c['pass'] for c in validate_ephemeris(e,decoded['satellite_prn']))]
    if not ephems: raise ValueError('No physically valid coherent CRC-passing ephemeris')
    messages=nav['messages']
    anchors=[]
    week_pages=[m for m in messages if 'GST_week' in m['decoded'] and 0<=m['decoded'].get('GST_TOW_s',-1)<WEEK and m.get('boundary_sample') is not None]
    if not week_pages: raise ValueError('No CRC-valid page with GST week and sample timing')
    reference=week_pages[0]['decoded']['GST_week']
    for m in messages:
        wt=m['word_type']; d=m['decoded']
        if not any(k in d for k in ('GST_TOW_s','TOW_s')) or m.get('boundary_sample') is None: continue
        tow=d.get('GST_TOW_s',d.get('TOW_s',-1))
        if not 0<=tow<WEEK: continue
        index=int(np.argmin(np.abs(b-float(m['boundary_sample']))))
        if abs(b[index]-m['boundary_sample'])>.001:
            raise ValueError('Decoded page boundary is not a saved tracking boundary')
        if 'GST_week' in d:
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
    contiguous=np.r_[False,np.abs(np.diff(b)/FS-period)<period*.01]
    segments=[]; rejected=[]; start=None
    for i in range(len(b)+1):
        valid=i<len(b) and good[i]
        if start is not None and (not valid or not contiguous[i]):
            stop=i
            aa=[a for a in anchors if start<=a['index']<stop]
            if aa:
                origins=np.array([a['tx_s']-period*a['index'] for a in aa])
                residual=float(np.ptp(origins))
                if residual<=1e-6 and stop-start>=2:
                    segments.append(dict(start=start,stop=stop,origin_s=float(np.median(origins)),anchors=aa,anchor_spread_s=residual))
                else: rejected.append(dict(start=start,stop=stop,reason='page TOW / counted code epochs disagree',anchor_spread_s=residual))
            else: rejected.append(dict(start=start,stop=stop,reason='no CRC-valid time anchor in uninterrupted lock segment'))
            start=None
        if valid and start is None: start=i
    if not segments: raise ValueError('No uninterrupted, time-anchored tracking segment passes the quality gates')
    return dict(band=band,sample_rate_sps=FS,primary_period_s=period,prn=int(decoded['satellite_prn']),directory=str(directory),recording=meta['recording'],
        capture_metadata=meta.get('capture_metadata',{}),boundaries=b,arrays=arrays,week=reference,
        segments=segments,rejected_segments=rejected,ephemerides=ephems,health_pages=[m for m in messages if m['word_type']==(1 if band=='e5a' else 5) and m.get('boundary_sample') is not None],
        utc_pages=[m for m in messages if m['word_type']==(4 if band=='e5a' else 6) and m.get('boundary_sample') is not None])


def observation(channel, sample, reference_week, receiver_origin_s, config, ionosphere=None):
    FS=channel.get('sample_rate_sps',12_500_000.)
    period=channel.get('primary_period_s',.004)
    band=channel.get('band','e1')
    from .signals import SIGNALS
    frequency=SIGNALS[band].frequency_hz
    b=channel['boundaries']
    segment=next((s for s in channel['segments'] if b[s['start']]<=sample<=b[s['stop']-1]),None)
    if segment is None: raise ValueError('No time anchor reachable without crossing a tracking gap')
    left=int(np.searchsorted(b,sample,side='right')-1)
    left=min(left,segment['stop']-2)
    frac=(sample-b[left])/(b[left+1]-b[left])
    # Navigation TOW fixes integer code periods. Only the fraction is interpolated.
    epoch=left+frac
    tx=segment['origin_s']+period*epoch+week_delta(channel['week'],reference_week)*WEEK
    origin=receiver_origin_s
    receive=origin+sample/FS
    raw=C*((origin-(segment['origin_s']+week_delta(channel['week'],reference_week)*WEEK)) + (sample/FS-period*epoch))
    health=min(channel['health_pages'],key=lambda p:abs(p['boundary_sample']-sample))
    if abs(health['boundary_sample']-sample)/FS>config.get('max_ephemeris_age_s',14400.):
        raise ValueError('Signal health/BGD page too old')
    d=health['decoded']
    health_prefix={'e1':'E1B','e5a':'E5a','e5b':'E5b'}[band]
    if d[health_prefix+'_HS']!=0 or d[health_prefix+'_DVS']!=0:
        raise ValueError(f'{band} signal health or data validity rejects this satellite')
    ephem=min(channel['ephemerides'],key=lambda e:abs(wrap_week_time(tx%WEEK-e['toe_s'])))
    age=max(abs(wrap_week_time(tx%WEEK-ephem['toe_s'])),abs(wrap_week_time(tx%WEEK-ephem['toc_s'])))
    if age>config.get('max_ephemeris_age_s',14400.): raise ValueError('Ephemeris or clock is stale')
    sisa=decode_sisa(int(ephem['SISA_index']))
    if sisa['kind']!='value': raise ValueError('SISA unavailable, reserved or NAPA')
    bgd=d['BGD_E1_E5a_s' if band=='e5a' else 'BGD_E1_E5b_s']
    factor=1. if band=='e1' else (1575.42e6/frequency)**2
    sat,clock,state,tx_gst=satellite_state(ephem,tx%WEEK,factor*bgd)
    delay=config.get('common_delay_ns',0.)+config.get('differential_delay_ns',{}).get(str(channel['prn']),0.)
    result=dict(band=band,frequency_hz=frequency,prn=channel['prn'],receive_sample=float(sample),receive_time_s=sample/FS,
        receiver_tow_unwrapped_s=receive,transmit_sv_s=tx,transmit_gst_tow_s=tx_gst%WEEK,
        raw_pseudorange_m=float(raw),satellite_ecef_m=sat,satellite_clock_e1_s=clock,
        clock_polynomial_s=state['clock_polynomial_s'],relativity_s=state['relativistic_correction_s'],
        bgd_s=bgd,bgd_e1_e5b_s=d.get('BGD_E1_E5b_s'),bgd_factor=factor,hardware_delay_m=delay*1e-9*C,
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
