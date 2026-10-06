"""NTCM-G v1.0, independently implemented from EU/GSC equations 1–34 (May 2022).
https://www.gsc-europa.eu/sites/default/files/NTCM-G_Ionospheric_Model_Description_-_v1.0.pdf
UTC: Galileo OS SIS ICD v2.2 §5.1.7. No device timestamp or fixed leap offset is used.
"""
import math
from datetime import datetime, timedelta, timezone
from .common import FS

WEEK = 604800.
GST_CALENDAR_ORIGIN = datetime(1999, 8, 22, tzinfo=timezone.utc)
E1_FACTOR = 40.3e16/(1575.42e6**2)
MODEL_URL = 'https://www.gsc-europa.eu/sites/default/files/NTCM-G_Ionospheric_Model_Description_-_v1.0.pdf'


def asin(x):
    return math.asin(max(-1., min(1., x)))


def ntcm_g(latitude, longitude, elevation, azimuth, doy, ut_hours, coefficients):
    """Angles in radians; UTC day-of-year and hours; output STEC [TECU] / E1 delay [m]."""
    a0, a1, a2 = coefficients
    if not all(math.isfinite(v) for v in (latitude, longitude, elevation, azimuth, doy, ut_hours, a0, a1, a2)):
        raise ValueError('NTCM-G requires finite inputs')
    if not 1<=doy<=366 or not 0<=ut_hours<24 or abs(latitude)>math.pi/2 or not 0<=elevation<=math.pi/2+1e-12:
        raise ValueError('NTCM-G input outside supported time/angle range')
    azpar = math.sqrt(max(0., a0*a0+1633.33*a1*a1+4802000*a2*a2+3266.67*a0*a2))
    psi = math.pi/2-elevation-asin(6371/6821*math.cos(elevation))
    ipp_lat = asin(math.sin(latitude)*math.cos(psi)+math.cos(latitude)*math.sin(psi)*math.cos(azimuth))
    # Eq. 26 as specified by NTCM-G (including its principal asin branch).
    cos_ipp = math.cos(ipp_lat)
    ipp_lon = longitude+asin(math.sin(psi)*math.sin(azimuth)/cos_ipp) if abs(cos_ipp)>1e-14 else longitude
    lt = ut_hours+math.degrees(ipp_lon)/15
    dec = math.radians(23.44*math.sin(math.radians(.9856*(doy-80.7))))
    cc = math.cos(ipp_lat-dec)
    c2, c3 = cc-2/math.pi*ipp_lat*math.sin(dec), cc+.4
    vd, vsd, vtd = 2*math.pi*(lt-14)/24, 2*math.pi*lt/12, 2*math.pi*lt/8
    f1 = c3+c2*(.92519*math.cos(vd)+.16951*math.cos(vsd)+.00443*math.sin(vsd)
               +.06626*math.cos(vtd)+.00899*math.sin(vtd))
    f2 = 1+.21289*math.cos(2*math.pi*(doy-18)/365.25)-.15414*math.cos(4*math.pi*(doy-6)/365.25)
    magnetic = asin(math.sin(ipp_lat)*math.sin(math.radians(79.74))
        +math.cos(ipp_lat)*math.cos(math.radians(79.74))*math.cos(ipp_lon-math.radians(-71.78)))
    f3 = 1-.38439*math.cos(magnetic)
    magnetic_deg = math.degrees(magnetic)
    f4 = 1+1.14023*math.exp(-(magnetic_deg-16)**2/(2*12**2))+1.20556*math.exp(-(magnetic_deg+10)**2/(2*13**2))
    f5 = 1.41808+.13985*azpar
    vtec = f1*f2*f3*f4*f5
    mapping = 1/math.sqrt(1-(6371/6821*math.sin(.9782*(math.pi/2-elevation)))**2)
    stec = vtec*mapping
    return dict(iono_m=E1_FACTOR*stec, iono_vtec_tecu=vtec, iono_stec_tecu=stec,
        iono_mapping=mapping, iono_azpar=azpar, iono_ipp_latitude_deg=math.degrees(ipp_lat),
        iono_ipp_longitude_deg=(math.degrees(ipp_lon)+180)%360-180)


def unfold_week(value, reference, modulus):
    delta = (int(value)-int(reference)+modulus//2)%modulus-modulus//2
    if abs(delta)==modulus//2: raise ValueError('Ambiguous truncated week number')
    return int(reference)+delta


def resolve_week(week, capture_metadata, era=None):
    """Host calendar date resolves only the era, never seconds-of-week or clock offset."""
    if era is not None:
        if not isinstance(era,int) or not 0<=era<=10: raise ValueError('GST era must be an integer in 0..10')
        return int(week)+4096*era, dict(source='explicit --gst-era', era=era)
    text = capture_metadata.get('created_utc')
    if not text: raise ValueError('Broadcast ionosphere requires capture created_utc to resolve GST era, or explicit --gst-era')
    date = datetime.fromisoformat(text.replace('Z','+00:00'))
    if date.tzinfo is None: raise ValueError('Capture created_utc must include a timezone')
    approximate_week = int((date-GST_CALENDAR_ORIGIN).total_seconds()//WEEK)
    full_week = unfold_week(week, approximate_week, 4096)
    # A capture date more than a week away is not trustworthy recording context.
    if abs(full_week-approximate_week)>1:
        raise ValueError('Capture calendar date disagrees with broadcast GST week; supply verified --gst-era')
    return full_week, dict(source='capture created_utc used ONLY for calendar-era resolution', created_utc=text, era=full_week//4096)


def gst_to_utc(full_week, tow, parameters):
    """ICD cases a/c; reject ±6 h of an actual leap change (Python has no second 60).
    The bounded unsupported interval is explicit, not silently flattened into POSIX time.
    """
    p = parameters
    if not math.isfinite(tow) or not all(math.isfinite(p[k]) for k in ('A0_s','A1_s_s','tot_s','delta_tLS_s','delta_tLSF_s')):
        raise ValueError('Nonfinite GST–UTC parameters')
    if not any(p[k] for k in ('A0_s','A1_s_s','tot_s','WNot','delta_tLS_s','delta_tLSF_s','WNLSF','DN')):
        raise ValueError('All-zero GST–UTC parameters cannot establish a calendar time')
    if not 0<=p['tot_s']<WEEK: raise ValueError('Invalid GST–UTC reference time of week')
    full_week += math.floor(tow/WEEK); tow %= WEEK
    ref_week = unfold_week(p['WNot'],full_week,256)
    dt = tow-p['tot_s']+(full_week-ref_week)*WEEK
    if abs(dt)>14*86400: raise ValueError('GST–UTC polynomial reference older than the 14-day processing gate')
    leap = p['delta_tLS_s']
    if p['delta_tLSF_s']!=leap:
        if p['DN'] not in range(1,8) or abs(p['delta_tLSF_s']-leap)!=1:
            raise ValueError('Invalid broadcast leap-second transition')
        leap_week = unfold_week(p['WNLSF'],full_week,256)
        delta = (full_week-leap_week)*WEEK+tow-p['DN']*86400-leap
        if abs(delta)<=21601:
            raise ValueError('Broadcast ionosphere UTC conversion unavailable within ±6 h of a leap-second transition')
        if delta>0: leap=p['delta_tLSF_s']
    offset = leap+p['A0_s']+p['A1_s_s']*dt
    utc = GST_CALENDAR_ORIGIN+timedelta(weeks=full_week,seconds=tow-offset)
    return utc, offset


class BroadcastIonosphere:
    """One coherent WT5 coefficient set for all lines of sight at a common epoch."""
    def __init__(self, channels, reference_week, era=None):
        self.week = None; self.era_sources=[]; self.pages={5:[],6:[]}
        for c in channels:
            # Use the common reference week; rollover relative to this week stays unwrapped.
            week, provenance = resolve_week(reference_week,c['capture_metadata'],era)
            if self.week is not None and self.week!=week: raise ValueError('Channels disagree on GST calendar era')
            self.week=week; self.era_sources.append(dict(prn=c['prn'],**provenance))
            for wt, pages in ((5,c['health_pages']), (6,c['utc_pages'])):
                for p in pages:
                    health=p if wt==5 else min(c['health_pages'],key=lambda h:abs(h['boundary_sample']-p['boundary_sample']))
                    if health['decoded']['E1B_HS']!=0 or health['decoded']['E1B_DVS']!=0: continue
                    if abs(health['boundary_sample']-p['boundary_sample'])/FS>14400: continue
                    self.pages[wt].append(dict(prn=c['prn'],sample=p['boundary_sample'],decoded=p['decoded']))
        for wt in (5,6):
            self.pages[wt].sort(key=lambda p:(p['sample'],p['prn']))
            if not self.pages[wt]: raise ValueError(f'Broadcast ionosphere needs CRC-valid WT{wt} pages with sample timing')

    def at(self,sample,fs):
        selected=[]
        for wt in (5,6):
            before=[p for p in self.pages[wt] if p['sample']<=sample]
            # Offline recordings may use their earliest decoded page to cover the initial interval.
            p=before[-1] if before else self.pages[wt][0]
            age=(sample-p['sample'])/fs
            if abs(age)>14400: raise ValueError(f'Broadcast ionosphere WT{wt} older than four hours')
            selected.append((p,age))
        (five,age5),(six,age6)=selected
        coeff=[five['decoded'][k] for k in ('ai0','ai1','ai2')]
        if not all(math.isfinite(c) for c in coeff) or not any(coeff):
            raise ValueError('Broadcast ionosphere coefficients missing/all zero; no nominal substitute is applied')
        return dict(full_reference_week=self.week,coefficients=coeff,utc_parameters=six['decoded'],
            coefficient_prn=five['prn'],coefficient_sample=five['sample'],coefficient_age_s=age5,
            utc_prn=six['prn'],utc_sample=six['sample'],utc_age_s=age6,
            offline_backfill=age5<0 or age6<0,iono_disturbance_flags=five['decoded']['iono_disturbance_flags'])


def broadcast_delay(latitude,longitude,elevation,azimuth,receiver_gst_s,context,config):
    utc, offset = gst_to_utc(context['full_reference_week'],receiver_gst_s,context['utc_parameters'])
    hours = utc.hour+utc.minute/60+(utc.second+utc.microsecond/1e6)/3600
    result = ntcm_g(latitude,longitude,max(0.,elevation),azimuth,utc.timetuple().tm_yday,hours,context['coefficients'])
    # Deliberately retain the previous 10 m zenith error assumption by default; a model is not proof of better accuracy.
    result.update(iono_sigma_m=config.get('iono_model_sigma_m',10.)*result['iono_mapping'],
        iono_model='NTCM-G v1.0',iono_utc=utc.isoformat(),gst_minus_utc_s=offset)
    return result
