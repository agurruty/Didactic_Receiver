"""E1 single-frequency observation model and weighted XYZ/clock solution.
Galileo OS SIS ICD v2.2 §5.1.4–5.1.5; ESA Navipedia satellite coordinates / troposphere.
All clock unknowns are metres. No UTC or Unix timestamp conversions occur here.
"""
import math
import numpy as np
from scipy.stats import chi2
from .navigation import propagate_ecef, OMEGA_E, MU

C = 299792458.0
A = 6378137.0
E2 = 6.6943799901413165e-3


class RejectedSolution(ValueError):
    """A computed fit failed a gate; keep its diagnostics for the teaching record."""
    def __init__(self, reason, result):
        super().__init__(reason)
        self.result = {**result, "accepted": False, "reason": reason}


def geodetic(xyz):
    x, y, z = xyz
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    if p < 1e-8:
        return math.copysign(math.pi/2, z), lon, abs(z)-A*math.sqrt(1-E2)
    lat = math.atan2(z, p*(1-E2))
    for _ in range(12):
        n = A/math.sqrt(1-E2*math.sin(lat)**2)
        new = math.atan2(z+E2*n*math.sin(lat), p)
        if abs(new-lat) < 1e-14:
            lat = new
            break
        lat = new
    n = A/math.sqrt(1-E2*math.sin(lat)**2)
    height = p*math.cos(lat)+z*math.sin(lat)-n*(1-E2*math.sin(lat)**2)
    return lat, lon, height


def ecef(lat, lon, height):
    n = A/math.sqrt(1-E2*math.sin(lat)**2)
    return np.array([(n+height)*math.cos(lat)*math.cos(lon),
                     (n+height)*math.cos(lat)*math.sin(lon), (n*(1-E2)+height)*math.sin(lat)])


def enu_matrix(xyz):
    lat, lon, _ = geodetic(xyz)
    s, c, sl, cl = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return np.array([[-sl, cl, 0], [-s*cl, -s*sl, c], [c*cl, c*sl, s]])


def earth_rotation(satellite, receiver):
    """Rotate emission ECEF to reception ECEF using geometric, not clock-biased, flight time."""
    rotated = np.asarray(satellite, dtype=float)
    for _ in range(3):
        angle = OMEGA_E*np.linalg.norm(rotated-receiver)/C
        c, s = math.cos(angle), math.sin(angle)
        rotated = np.array([c*satellite[0]+s*satellite[1], -s*satellite[0]+c*satellite[1], satellite[2]])
    return rotated


def satellite_state(ephem, transmit_sv_tow, bgd_s):
    """I/NAV E1: delta_tSV(E1) = polynomial + relativity - BGD(E1,E5b)."""
    tx = transmit_sv_tow
    for _ in range(4):
        state = propagate_ecef(ephem, tx % 604800)
        clock = state['clock_total_s']-bgd_s
        tx = transmit_sv_tow-clock
    state = propagate_ecef(ephem, tx % 604800)
    return np.array([state[k] for k in ('x_m','y_m','z_m')]), clock, state, tx


def elevation_azimuth(receiver, satellite):
    east, north, up = enu_matrix(receiver) @ (satellite-receiver)
    return math.atan2(up,math.hypot(east,north)), math.atan2(east,north) % (2*math.pi)


def atmosphere(receiver, elevation, config):
    """Standard-atmosphere Saastamoinen zenith terms, Black–Eisner slant mapping.
    VTEC is optional external input, never fabricated from Galileo ai coefficients.
    Return tropo, iono and their assumed standard deviations [m].
    """
    lat, _, height = geodetic(receiver)
    h = float(np.clip(height, -500, 10000))
    el = max(elevation, math.radians(5))
    mapping = 1.001/math.sqrt(.002001+math.sin(el)**2)
    pressure = 1013.25*(1-2.2557e-5*h)**5.2568
    temp = 288.15-.0065*h
    vapour = .5*6.108*math.exp(17.15*(temp-273.15)/(234.7+temp-273.15))
    dry = .0022768*pressure/(1-.00266*math.cos(2*lat)-.00028*h/1000)
    wet = .002277*(1255/temp+.05)*vapour
    tropo = (dry+wet)*mapping
    iono_mapping = 1/math.sqrt(1-(6371000/(6371000+450000)*math.cos(el))**2)
    if config.get('vtec_tecu') is None:
        iono, iono_sigma = 0., config.get('uncorrected_iono_sigma_m',10.)*iono_mapping
    else:
        factor = 40.3e16/(1575.42e6**2)*iono_mapping
        iono = config['vtec_tecu']*factor
        iono_sigma = config['vtec_sigma_tecu']*factor
    return tropo, iono, .3*mapping, iono_sigma


def design(observations, state, config, corrections=True):
    rows, residuals, sigmas, details = [], [], [], []
    receiver, bias = state[:3], state[3]
    near_earth = np.linalg.norm(receiver) > 5e6
    for o in observations:
        sat = earth_rotation(o['satellite_ecef_m'], receiver)
        rho = float(np.linalg.norm(sat-receiver))
        el, az = elevation_azimuth(receiver,sat) if near_earth else (math.pi/2,0.)
        tropo=iono=tsigma=isigma=gravity=0.
        iono_details={}
        if corrections and near_earth:
            tropo, iono, tsigma, isigma = atmosphere(receiver,el,config)
            if o.get('broadcast_iono') is not None:
                from .ionosphere import broadcast_delay
                lat,lon,_=geodetic(receiver)
                iono_details=broadcast_delay(lat,lon,el,az,o['receiver_tow_unwrapped_s']-bias/C,o['broadcast_iono'],config)
                iono,isigma=iono_details['iono_m'],iono_details['iono_sigma_m']
            frequency_scale=(1575.42e6/o.get('frequency_hz',1575.42e6))**2
            iono*=frequency_scale; isigma*=frequency_scale
            if iono_details:
                iono_details.update(iono_m=iono,iono_sigma_m=isigma)
            if o.get('ionosphere_free'):
                iono=0.; isigma=config.get('higher_order_iono_sigma_m',.1)
            if o.get('comparison_iono_sigma_m') is not None: isigma=o['comparison_iono_sigma_m']
            if o.get('external_iono_m') is not None:
                iono, isigma = o['external_iono_m'], o['external_iono_sigma_m']
            rr, rs = np.linalg.norm(receiver), np.linalg.norm(sat)
            gravity = 2*MU/C**2*math.log((rr+rs+rho)/(rr+rs-rho))
        corrected = o['raw_pseudorange_m'] + C*o['satellite_clock_e1_s'] - o['hardware_delay_m'] - tropo - iono - gravity
        residual = corrected-rho-bias
        # A stochastic model, not a calibrated tracking accuracy claim.
        scale = max(math.sin(max(el,math.radians(5))), .2)
        sigma = math.sqrt((o.get('code_sigma_m',config.get('code_sigma_m',10.))/scale)**2 + o['sisa_m']**2 + tsigma**2 + isigma**2 + o.get('differential_delay_sigma_m',0.)**2)
        rows.append(np.r_[(receiver-sat)/rho,1.])
        residuals.append(residual); sigmas.append(sigma)
        details.append(dict(prn=o['prn'], geometric_range_m=rho, residual_m=residual,
            sigma_m=sigma, elevation_deg=math.degrees(el), azimuth_deg=math.degrees(az),
            tropo_m=tropo, iono_m=iono, shapiro_m=gravity,
            earth_rotation_range_m=rho-float(np.linalg.norm(o['satellite_ecef_m']-receiver)),
            satellite_reception_ecef_m=sat.tolist(), corrected_pseudorange_m=corrected))
        details[-1].update(iono_details)
        details[-1]['iono_sigma_m']=isigma
    return np.asarray(rows),np.asarray(residuals),np.asarray(sigmas),details


def solve_epoch(observations, config, initial=None):
    if len(observations)<4:
        raise ValueError('Fewer than four usable satellites')
    state = np.zeros(4) if initial is None else np.asarray(initial,dtype=float).copy()
    # Bootstrap without atmosphere, then recompute corrections and the same four unknowns.
    for corrections in (False,True):
        for iteration in range(20):
            h, v, sigma, details = design(observations,state,config,corrections)
            hw = h/sigma[:,None]
            step,_,rank,_ = np.linalg.lstsq(hw,v/sigma,rcond=None)
            if rank<4 or np.linalg.cond(hw)>1e8:
                raise ValueError('Rank-deficient or nearly singular satellite geometry')
            state += step
            if not np.all(np.isfinite(state)) or np.linalg.norm(state[:3])>1e8:
                raise ValueError('Position iteration diverged')
            if np.linalg.norm(step)<1e-4: break
        else:
            raise ValueError('Position iteration did not converge')
    h,v,sigma,details=design(observations,state,config)
    _,_,height=geodetic(state[:3])
    covariance=np.linalg.inv((h/sigma[:,None]).T@(h/sigma[:,None]))
    geometry=np.linalg.inv(h.T@h)
    pdop=float(np.sqrt(np.trace(geometry[:3,:3])))
    dof=len(v)-4
    chi=float(np.sum((v/sigma)**2))
    limit=float(chi2.ppf(.999,dof)) if dof else None
    leverage=np.sum((h@covariance)*h,axis=1)/sigma**2
    standardized=v/(sigma*np.sqrt(np.maximum(1-leverage,1e-12))) if dof else None
    passed=dof==0 or (chi<=limit and np.max(np.abs(standardized))<=5.)
    inflation=max(1.,chi/dof) if dof else 1.
    covariance*=inflation
    rot=enu_matrix(state[:3]); local_cov=rot@covariance[:3,:3]@rot.T
    local_geom=rot@geometry[:3,:3]@rot.T
    for i,d in enumerate(details):
        d['standardized_residual']=float(standardized[i]) if dof else None
    lat,lon,height=geodetic(state[:3])
    result = dict(state=state.tolist(),x_m=float(state[0]),y_m=float(state[1]),z_m=float(state[2]),latitude_deg=math.degrees(lat),longitude_deg=math.degrees(lon),
        ellipsoidal_height_m=height, clock_bias_s=float(state[3]/C),
        covariance_xyz_clock_m2=covariance.tolist(),covariance_enu_m2=local_cov.tolist(),
        model_sigma_enu_m=np.sqrt(np.diag(local_cov)).tolist(),clock_sigma_s=float(np.sqrt(covariance[3,3])/C),
        pdop=pdop,hdop=float(np.sqrt(local_geom[0,0]+local_geom[1,1])),vdop=float(np.sqrt(local_geom[2,2])),
        gdop=float(np.sqrt(np.trace(geometry))), residual_rms_m=float(np.sqrt(np.mean(v*v))),
        residual_dof=dof,chi_square=chi,chi_square_limit=limit,residual_test_passed=bool(passed),
        integrity='residual consistency only; not certified RAIM' if dof else 'unavailable: exactly four satellites, no residual redundancy',
        uncertainty_scope='conditional model covariance; correlated multipath, unmodelled biases and common time/delay calibration excluded',
        observations=details)
    if not -1000 < height < 20000:
        raise RejectedSolution("Solution outside the supported terrestrial height interval", result)
    if pdop>config.get("max_pdop",20.):
        raise RejectedSolution(f"PDOP {pdop:.1f} exceeds the geometry gate", result)
    return result


def screened_solution(observations, config, initial=None):
    """Use all visible healthy channels; do not silently choose the strongest four."""
    active=list(observations); excluded=[]
    for _ in range(len(active)):
        result=solve_epoch(active,config,initial)
        low=[d['prn'] for d in result['observations'] if d['elevation_deg']<config.get('elevation_mask_deg',10.)]
        if not low: break
        excluded.extend(dict(prn=p,reason='below elevation mask') for p in low)
        active=[o for o in active if o['prn'] not in low]
        if len(active)<4:
            result['excluded']=excluded
            raise RejectedSolution('Fewer than four satellites above the elevation mask',result)
        initial=result['state']
    if not result['residual_test_passed']:
        # >=6 allows a leave-one-out fit with remaining redundancy. Ambiguous cases stay rejected.
        alternatives=[]
        if len(active)>=6:
            for o in active:
                try:
                    trial=solve_epoch([q for q in active if q['prn']!=o['prn']],config,result['state'])
                    if trial['residual_test_passed'] and all(d['elevation_deg']>=config.get('elevation_mask_deg',10.) for d in trial['observations']):
                        alternatives.append((o['prn'],trial))
                except ValueError: pass
        if len(alternatives)==1:
            prn,result=alternatives[0]
            excluded.append(dict(prn=prn,reason='unique leave-one-out residual-consistent solution'))
        else:
            result['accepted']=False
            result['reason']='Residual inconsistency; no unique redundant exclusion'
    result.setdefault('accepted',True)
    result['excluded']=excluded
    removed=[o for o in observations if o['prn'] in {e['prn'] for e in excluded}]
    result['excluded_observations']=design(removed,np.asarray(result['state']),config)[3] if removed else []
    return result


def uncorrected_comparison(observations, corrected, config):
    """Hold the used satellite set and ionosphere noise assumptions; switch off the delay only."""
    diagnostics={d['prn']:d for d in corrected['observations']}
    obs=[{**{k:v for k,v in o.items() if k!='broadcast_iono'},
          'comparison_iono_sigma_m':diagnostics[o['prn']]['iono_sigma_m']} for o in observations if o['prn'] in diagnostics]
    try:
        baseline=solve_epoch(obs,config,corrected['state'])
        delta=np.asarray(corrected['state'][:3])-np.asarray(baseline['state'][:3])
        enu=enu_matrix(baseline['state'][:3])@delta
        return dict(available=True,uncorrected_state=baseline['state'],delta_enu_m=enu.tolist(),
            horizontal_shift_m=float(np.linalg.norm(enu[:2])),same_prns=sorted(diagnostics),
            baseline_residual_test_passed=baseline['residual_test_passed'],
            meaning='Correction-induced shift, not measured improvement; same PRNs and ionosphere sigmas')
    except (ValueError,np.linalg.LinAlgError) as exc:
        return dict(available=False,reason=str(exc))
