"""Form E1 pseudoranges at common receive samples and solve XYZ + receiver clock."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from .common import FS, write_json
from .observables import load_channel, observation, week_delta, WEEK, SlantIonosphere
from .position_models import C, screened_solution, RejectedSolution, uncorrected_comparison


def serial(value):
    if isinstance(value,np.ndarray): return serial(value.tolist())
    if isinstance(value,dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    if isinstance(value,(float,np.floating)): return float(value) if math.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    return value


def csv_write(path,rows,columns):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def read_calibration(path):
    if path is None: return {}
    data=json.loads(Path(path).read_text())
    if not data.get('source'):
        raise ValueError('Calibration must identify its source; use a measured calibration, not an assumed UTC offset')
    allowed={'source','first_sample_gst','common_delay_ns','common_delay_sigma_ns','differential_delay_ns','differential_delay_sigma_ns','recording'}
    if set(data)-allowed: raise ValueError('Unknown calibration fields: '+str(sorted(set(data)-allowed)))
    for key in ('common_delay_ns','common_delay_sigma_ns'):
        if key in data and (not math.isfinite(float(data[key])) or (key.endswith('sigma_ns') and data[key]<0)):
            raise ValueError('Invalid hardware delay calibration')
    for key in ('differential_delay_ns','differential_delay_sigma_ns'):
        for prn,value in data.get(key,{}).items():
            if not 1<=int(prn)<=50 or not math.isfinite(float(value)) or (key.endswith('sigma_ns') and value<0):
                raise ValueError('Invalid per-PRN delay calibration')
    if 'first_sample_gst' in data:
        anchor=data['first_sample_gst']
        if not 0<=anchor['week']<4096 or int(anchor['week'])!=anchor['week'] or not 0<=anchor['tow_s']<WEEK:
            raise ValueError('GST anchor requires week modulo 4096 and seconds inside that week')
        if not math.isfinite(anchor['uncertainty_s']) or anchor['uncertainty_s']<0:
            raise ValueError('GST anchor must declare finite nonnegative uncertainty_s')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir',type=Path,help='Run containing E*/02_tracking and E*/03_telemetry artifacts.')
    p.add_argument('--output',type=Path,help='New output directory; default <run>/05_position.')
    p.add_argument('--interval',type=float,default=1.,help='Common receive-epoch spacing in seconds.')
    p.add_argument('--start',type=float,help='First recording-relative receive time [s].')
    p.add_argument('--stop',type=float,help='Last recording-relative receive time [s].')
    p.add_argument('--calibration',type=Path,help='Optional sourced GST time and hardware-delay calibration JSON.')
    p.add_argument('--elevation-mask',type=float,default=10.,help='Minimum satellite elevation [degrees].')
    p.add_argument('--max-pdop',type=float,default=20.,help='Reject poorly conditioned position geometry.')
    p.add_argument('--code-sigma',type=float,default=10.,help='Assumed zenith code noise [m], not a measured accuracy.')
    p.add_argument('--min-cn0',type=float,default=25.,help='Minimum finite pilot C/N0 [dB-Hz] along a continuous segment.')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--ionosphere',choices=('ntcm-g','none'),help='Default: broadcast NTCM-G. none reproduces uncorrected processing.')
    p.add_argument('--gst-era',type=int,help='Explicit 4096-week GST era; otherwise resolved from capture calendar metadata, never device time.')
    p.add_argument('--iono-model-sigma',type=float,default=10.,help='Assumed residual zenith ionosphere sigma [m] after NTCM-G; not model-certified accuracy.')
    group.add_argument('--ionosphere-csv',type=Path,help='Sourced E1 slant corrections: prn,receive_time_s,delay_m,sigma_m.')
    group.add_argument('--vtec',type=float,help='Externally supplied VTEC [TECU] for a 450 km thin-shell model.')
    p.add_argument('--vtec-sigma',type=float,default=5.,help='VTEC model uncertainty [TECU]; used only with --vtec.')
    p.add_argument('--ionosphere-source',help='Provenance of supplied VTEC or slant corrections; required when supplying either.')
    from .position_map import add_map_arguments, validate_map_arguments, render_map
    p.add_argument('--map-only', action='store_true', help='Render a geographic map from <run_dir>/position_solution.json without rerunning the solver.')
    add_map_arguments(p)
    args=p.parse_args()
    validate_map_arguments(args)
    if args.map_only:
        summary=json.loads((args.run_dir/'position_solution.json').read_text())
        out=args.output or args.run_dir/'map-view'
        out.mkdir(parents=True,exist_ok=False)
        return 0 if render_map(summary,out,args) else 3
    requested_iono=args.ionosphere or ('external' if args.ionosphere_csv or args.vtec is not None else 'ntcm-g')
    if args.gst_era is not None and not 0<=args.gst_era<=10: raise ValueError('--gst-era must be 0..10')
    numeric=[args.interval,args.max_pdop,args.code_sigma,args.vtec_sigma,args.iono_model_sigma]
    if not all(math.isfinite(v) and v>0 for v in numeric): raise ValueError('Interval, PDOP and sigmas must be finite and positive')
    if not math.isfinite(args.elevation_mask) or not 5<=args.elevation_mask<90: raise ValueError('Elevation mask must be 5..<90 degrees')
    if not math.isfinite(args.min_cn0): raise ValueError('C/N0 gate must be finite')
    if args.vtec is not None and (not math.isfinite(args.vtec) or args.vtec<0): raise ValueError('VTEC must be nonnegative')
    if (args.vtec is not None or args.ionosphere_csv) and not args.ionosphere_source:
        raise ValueError('Supply --ionosphere-source with external ionospheric corrections')
    for value in (args.start,args.stop):
        if value is not None and (not math.isfinite(value) or value<0): raise ValueError('Start/stop must be finite nonnegative recording times')
    calibration=read_calibration(args.calibration)
    config=dict(interval_s=args.interval,elevation_mask_deg=args.elevation_mask,max_pdop=args.max_pdop,
        code_sigma_m=args.code_sigma,min_cn0_dbhz=args.min_cn0,min_lock=.7,
        max_ephemeris_age_s=14400.,vtec_tecu=args.vtec,vtec_sigma_tecu=args.vtec_sigma,
        uncorrected_iono_sigma_m=10.,iono_model_sigma_m=args.iono_model_sigma,ionosphere_mode=requested_iono,**calibration)
    channels=[]; rejections=[]; identity=None
    for directory in sorted(args.run_dir.glob('E*')):
        if not directory.is_dir(): continue
        try:
            channel=load_channel(directory,config)
            if any(c['prn']==channel['prn'] for c in channels):
                raise ValueError('Duplicate satellite channel would duplicate a range observation')
            if identity is not None and channel['recording']!=identity:
                raise ValueError('Channel comes from a different recording')
            if calibration.get('recording') is not None and calibration['recording']!=channel['recording']:
                raise ValueError('Calibration belongs to a different recording')
            identity=channel['recording']; channels.append(channel)
        except (ValueError,KeyError,OSError) as exc:
            rejections.append(dict(channel=directory.name,reason=str(exc)))
    out=args.output or args.run_dir/'05_position'
    out.mkdir(parents=True,exist_ok=False)
    reference_week=channels[0]['week'] if channels else None
    clock=dict(mode='navigation-aligned receiver clock; offset solved at each epoch',
        device_timestamp_interpreted=False,reference_gst_week=reference_week,
        week_ambiguity='broadcast GST week modulo 4096; no absolute 4096-week era or UTC inferred',
        common_delay_calibrated='common_delay_ns' in calibration,
        common_delay_note='Unknown common antenna/cable/receiver delay is inseparable from receiver clock offset',
        calibration=calibration or None)
    epochs=[]; ranges=[]; origin=None
    if channels:
        origins=[s['origin_s']+week_delta(c['week'],reference_week)*WEEK-c['boundaries'][s['start']]/FS+.004*s['start']
                 for c in channels for s in c['segments']]
        if np.ptp(origins)>.5:
            rejections.append(dict(channel='all',reason='Decoded transmit epochs disagree by more than 0.5 s across channels'))
            channels=[]
        else:
            # This 80 ms is ONLY a coordinate origin for the freely estimated receiver clock.
            # Changing it adds a common constant to all raw pseudoranges and the solved clock, not XYZ.
            origin=float(np.median(origins)+.080)
            if 'first_sample_gst' in calibration:
                anchor=calibration['first_sample_gst']
                calibrated=week_delta(anchor['week'],reference_week)*WEEK+anchor['tow_s']
                if abs(calibrated-origin)>1.:
                    raise ValueError('Calibrated GST and navigation epochs disagree by >1 s; check time scale/week')
                origin=calibrated
                clock['mode']='explicit calibrated first-sample GST; residual receiver clock still estimated'
            clock['receiver_clock_at_sample_zero_s']=origin
            clock['arbitrary_initial_flight_time_s']=.080 if 'first_sample_gst' not in calibration else None
            clock['first_sample_device_time_s']=channels[0]['capture_metadata'].get('first_sample_device_time_s')
    ionosphere=SlantIonosphere(args.ionosphere_csv) if args.ionosphere_csv else None
    from .ionosphere import BroadcastIonosphere, MODEL_URL
    broadcast=None; broadcast_error=None
    if channels and requested_iono=='ntcm-g':
        try:
            broadcast=BroadcastIonosphere(channels,reference_week,args.gst_era)
            clock['ionosphere_calendar']=dict(full_reference_week=broadcast.week,era_sources=broadcast.era_sources,
                utc_source='CRC-valid WT6 GST–UTC parameters; evaluated at solved receive GST')
            clock['week_ambiguity']='GST era resolved for NTCM-G calendar; provenance retained in ionosphere_calendar'
        except ValueError as exc: broadcast_error=str(exc)
    if channels:
        first=min(c['boundaries'][s['start']]/FS for c in channels for s in c['segments'])
        last=max(c['boundaries'][s['stop']-1]/FS for c in channels for s in c['segments'])
        start=math.ceil(first/args.interval)*args.interval if args.start is None else args.start
        stop=last if args.stop is None else min(args.stop,last)
        if stop<start: rejections.append(dict(channel='all',reason='No requested common receive epochs inside tracking coverage'))
        initial=None
        count=max(0,int(math.floor((stop-start)/args.interval))+1)
        if count>1_000_000: raise ValueError('Requested more than one million solution epochs; increase --interval')
        for index in range(count):
            time=start+index*args.interval
            sample=float(round(time*FS)); time=sample/FS
            obs=[]; excluded=[]
            for c in channels:
                try: obs.append(observation(c,sample,reference_week,origin,config,ionosphere))
                except ValueError as exc: excluded.append(dict(prn=c['prn'],reason=str(exc)))
            row=dict(receive_time_s=time,receive_sample=sample,available_prns=[o['prn'] for o in obs],input_rejections=excluded)
            try:
                if broadcast_error: raise ValueError(broadcast_error)
                if broadcast is not None:
                    context=broadcast.at(sample,FS)
                    row['broadcast_iono']=context
                    for o in obs: o['broadcast_iono']=context
                result=screened_solution(obs,config,initial)
                row.update(result)
                if row['accepted']:
                    initial=row['state']
                    if broadcast is not None: row['ionosphere_comparison']=uncorrected_comparison(obs,row,config)
                    true_receive=origin+time-row['clock_bias_s']
                    row['receive_gst_week']=(reference_week+math.floor(true_receive/WEEK))%4096
                    row['receive_gst_tow_s']=true_receive%WEEK
                    # This is ADC epoch only if common path delay is calibrated; otherwise antenna-arrival epoch.
                    row['receive_time_reference']='ADC sample epoch' if clock['common_delay_calibrated'] else 'antenna-arrival epoch; unknown common path delay absorbed in clock'
                    if 'first_sample_gst' not in calibration:
                        row['inferred_first_sample_gst_tow_s']=(true_receive-time)%WEEK
                    row['common_calibration_sigma_s']=math.hypot(calibration.get('common_delay_sigma_ns',0.)*1e-9,calibration.get('first_sample_gst',{}).get('uncertainty_s',0.)) if clock['common_delay_calibrated'] and 'common_delay_sigma_ns' in calibration else None
            except RejectedSolution as exc:
                row.update(exc.result)
            except (ValueError,np.linalg.LinAlgError) as exc:
                row.update(accepted=False,reason=str(exc))
            diagnostics={d['prn']:d for d in row.get('observations',[])+row.get('excluded_observations',[])}
            used={d['prn'] for d in row.get('observations',[])} if row['accepted'] else set()
            reasons={d['prn']:d['reason'] for d in row.get('excluded',[])}
            for o in obs:
                ranges.append({**o,**diagnostics.get(o['prn'],{}),
                    'used_in_accepted_solution':o['prn'] in used,'solution_accepted':row['accepted'],'exclusion_reason':reasons.get(o['prn'])})
            epochs.append(row)
    accepted=[e for e in epochs if e['accepted']]
    mode='NTCM-G broadcast model' if requested_iono=='ntcm-g' else ('external E1 slant corrections' if ionosphere else ('external VTEC thin-shell approximation' if args.vtec is not None else 'uncorrected; 10 m zenith ionosphere sigma assumed'))
    summary=dict(schema_version=1,status='position estimates available' if accepted else 'insufficient consistent evidence',
        recording=identity,clock=clock,configuration=config,
        corrections=dict(satellite_clock='I/NAV polynomial + relativity - BGD(E1,E5b)',earth_rotation=True,
            shapiro=True,troposphere='standard-atmosphere Saastamoinen + Black–Eisner mapping; no local weather measurement',
            ionosphere=mode,ionosphere_source=MODEL_URL if requested_iono=='ntcm-g' else args.ionosphere_source,
            ionosphere_noise_note='Residual sigma is an independent-error assumption, not a broadcast accuracy guarantee; spatially correlated errors remain',ionosphere_csv=str(args.ionosphere_csv) if args.ionosphere_csv else None),
        positioning_quality='provisional: ionosphere uncorrected' if requested_iono=='none' else 'conditional on supplied correction and noise models',
        accepted_epochs=len(accepted),attempted_epochs=len(epochs),channel_rejections=rejections,
        channels=[dict(prn=c['prn'],time_anchors=[a for s in c['segments'] for a in s['anchors']],
            rejected_segments=c['rejected_segments']) for c in channels],
        limitations=['Formal covariance is conditional on assumed independent range errors; it is not an accuracy guarantee.',
            'Four satellites give no residual redundancy; coherent code biases and multipath can remain undetected.',
            'Broadcast data are CRC-checked, not OSNMA-authenticated. E1-C BOC side-peak bias is not eliminated by CRC.',
            'Common delay is inseparable from receiver clock unless calibrated; differential antenna/receiver delays can bias position.',
            'Ionospheric correction is a broadcast/external model estimate, not a dual-frequency measurement; residual correlated errors remain.'],
        epochs=epochs)
    write_json(out/'position_solution.json',serial(summary))
    write_json(out/'pseudoranges.json',serial(ranges))
    columns=['receive_time_s','receive_sample','prn','transmit_sv_s','transmit_gst_tow_s','raw_pseudorange_m',
        'satellite_clock_e1_s','bgd_e1_e5b_s','hardware_delay_m','earth_rotation_range_m','tropo_m','iono_m','shapiro_m',
        'iono_sigma_m','iono_vtec_tecu','iono_stec_tecu','iono_mapping','iono_ipp_latitude_deg','iono_ipp_longitude_deg','iono_utc',
        'corrected_pseudorange_m','geometric_range_m','residual_m','standardized_residual','sigma_m','elevation_deg',
        'azimuth_deg','cn0_dbhz','anchor_count','anchor_spread_s','integer_code_periods','used_in_accepted_solution','exclusion_reason']
    csv_write(out/'pseudoranges.csv',serial(ranges),columns)
    csv_write(out/'position.csv',serial(epochs),['receive_time_s','accepted','reason','receive_gst_week','receive_gst_tow_s',
        'x_m','y_m','z_m','latitude_deg','longitude_deg','ellipsoidal_height_m','clock_bias_s','clock_sigma_s','pdop','hdop','vdop',
        'residual_rms_m','residual_dof','chi_square','chi_square_limit','integrity'])
    from .position_figures import position_figures
    position_figures(summary,ranges,out)
    if requested_iono=='ntcm-g':
        from .ionosphere_figures import ionosphere_figure
        ionosphere_figure(summary,ranges,out)
    render_map(summary,out,args)
    print(f"Position: {len(accepted)}/{len(epochs)} epochs accepted. Ionosphere: {mode}.")
    if accepted:
        last=accepted[-1]
        print(f"Last estimate: {last['latitude_deg']:.8f}°, {last['longitude_deg']:.8f}°, ellipsoidal height {last['ellipsoidal_height_m']:.2f} m")
        print(f"PDOP {last['pdop']:.2f}; {last['integrity']}")
    return 0 if accepted else 3
