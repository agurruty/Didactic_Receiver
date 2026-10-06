#!/usr/bin/env python3
"""
Galileo I/NAV navigation-data interpreter + broadcast-orbit propagator.
INPUT
-----
decoded_inav.json produced by 03_galileo_e1_decode.py; parses CRC-valid WT1–WT6 here.
This stage turns decoded navigation bits into physical navigation products:
  * validates a complete WT1-WT4 full-precision CED ephemeris
  * parses/labels additional I/NAV words (0, 7, 8, 9, 10, 16-20)
  * decodes SISA into metres where defined
  * assembles partial Galileo almanac information from WT7-WT10
  * identifies RedCED (WT16) and RS-FEC2 (WT17-WT20) availability
  * anchors Galileo Time-of-Week from WT5 if available, otherwise WT6
  * propagates the broadcast ephemeris to ECEF X/Y/Z
  * estimates ECEF velocity numerically
  * computes broadcast satellite clock polynomial + relativistic correction
  * generates a didactic "navigation bits -> satellite in space" dashboard
RedCED and RS-FEC2 are explicitly recognized but are not reimplemented here.
Once a CRC-valid coherent WT1-WT4 set exists, that full-precision CED is the
authoritative orbit/clock source for this stage.
Example:
    python3 04_galileo_e1_message.py \
        /path/to/decoded_inav.json \
        --output orbit_dashboard
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from .common import finish, write_json
from .words import assemble_navigation

WEEK_S = 604800.0
MU = 3.986004418e14
OMEGA_E = 7.2921151467e-5
RELATIVISTIC_F = -4.442807309e-10
NOMINAL_GALILEO_A_M = 29_600_000.0
NOMINAL_GALILEO_SQRTA = math.sqrt(NOMINAL_GALILEO_A_M)
NOMINAL_ALMANAC_INCLINATION_RAD = math.radians(56.0)

HEALTH_TEXT = {0: "OK", 1: "out of service", 2: "will be out of service", 3: "component in test"}


def bits_from_string(s: str) -> np.ndarray:
    if len(s) != 128 or any(c not in "01" for c in s):
        raise ValueError("Expected exactly 128 bits")
    return np.fromiter((int(c) for c in s), dtype=np.uint8)


def u(bits: np.ndarray, start: int, length: int) -> int:
    value = 0
    for b in bits[start - 1 : start - 1 + length]:
        value = (value << 1) | int(b)
    return value


def s(bits: np.ndarray, start: int, length: int) -> int:
    value = u(bits, start, length)
    if value & (1 << (length - 1)):
        value -= 1 << length
    return value


def wrap_week_time(dt: float) -> float:
    if dt > WEEK_S / 2:
        dt -= WEEK_S
    elif dt < -WEEK_S / 2:
        dt += WEEK_S
    return dt


def decode_sisa(index: int):
    if 0 <= index <= 49:
        return {"kind": "value", "metres": index * 0.01}
    if 50 <= index <= 74:
        return {"kind": "value", "metres": 0.50 + (index - 50) * 0.02}
    if 75 <= index <= 99:
        return {"kind": "value", "metres": 1.00 + (index - 75) * 0.04}
    if 100 <= index <= 125:
        return {"kind": "value", "metres": 2.00 + (index - 100) * 0.16}
    if 126 <= index <= 254:
        return {"kind": "spare", "metres": None}
    if index == 255:
        return {"kind": "NAPA", "metres": None}
    return {"kind": "invalid", "metres": None}


def parse_word_0(bits):
    time_flag = u(bits, 7, 2)
    return {
        "word_type": 0,
        "name": "I/NAV spare word",
        "time_flag": time_flag,
        "time_valid": time_flag == 0b10,
        "GST_week": u(bits, 97, 12),
        "GST_TOW_s": u(bits, 109, 20),
    }


def parse_word_7(bits):
    return {
        "word_type": 7,
        "name": "Almanac SVID1 (1/2)",
        "IODa": u(bits, 7, 4),
        "WNa_mod4": u(bits, 11, 2),
        "t0a_s": u(bits, 13, 10) * 600.0,
        "SVID1": u(bits, 23, 6),
        "delta_sqrtA_m_sqrt": s(bits, 29, 13) * 2.0**-9,
        "eccentricity": u(bits, 42, 11) * 2.0**-16,
        "omega_rad": s(bits, 53, 16) * math.pi * 2.0**-15,
        "delta_i_rad": s(bits, 69, 11) * math.pi * 2.0**-14,
        "Omega0_rad": s(bits, 80, 16) * math.pi * 2.0**-15,
        "OmegaDot_rad_s": s(bits, 96, 11) * math.pi * 2.0**-33,
        "M0_rad": s(bits, 107, 16) * math.pi * 2.0**-15,
    }


def parse_word_8(bits):
    return {
        "word_type": 8,
        "name": "Almanac SVID1 (2/2) + SVID2 (1/2)",
        "IODa": u(bits, 7, 4),
        "SVID1_af0_s": s(bits, 11, 16) * 2.0**-19,
        "SVID1_af1_s_s": s(bits, 27, 13) * 2.0**-38,
        "SVID1_E5b_health_raw": u(bits, 40, 2),
        "SVID1_E1B_health_raw": u(bits, 42, 2),
        "SVID2": u(bits, 44, 6),
        "SVID2_delta_sqrtA_m_sqrt": s(bits, 50, 13) * 2.0**-9,
        "SVID2_eccentricity": u(bits, 63, 11) * 2.0**-16,
        "SVID2_omega_rad": s(bits, 74, 16) * math.pi * 2.0**-15,
        "SVID2_delta_i_rad": s(bits, 90, 11) * math.pi * 2.0**-14,
        "SVID2_Omega0_rad": s(bits, 101, 16) * math.pi * 2.0**-15,
        "SVID2_OmegaDot_rad_s": s(bits, 117, 11) * math.pi * 2.0**-33,
    }


def parse_word_9(bits):
    return {
        "word_type": 9,
        "name": "Almanac SVID2 (2/2) + SVID3 (1/2)",
        "IODa": u(bits, 7, 4),
        "WNa_mod4": u(bits, 11, 2),
        "t0a_s": u(bits, 13, 10) * 600.0,
        "SVID2_M0_rad": s(bits, 23, 16) * math.pi * 2.0**-15,
        "SVID2_af0_s": s(bits, 39, 16) * 2.0**-19,
        "SVID2_af1_s_s": s(bits, 55, 13) * 2.0**-38,
        "SVID2_E5b_health_raw": u(bits, 68, 2),
        "SVID2_E1B_health_raw": u(bits, 70, 2),
        "SVID3": u(bits, 72, 6),
        "SVID3_delta_sqrtA_m_sqrt": s(bits, 78, 13) * 2.0**-9,
        "SVID3_eccentricity": u(bits, 91, 11) * 2.0**-16,
        "SVID3_omega_rad": s(bits, 102, 16) * math.pi * 2.0**-15,
        "SVID3_delta_i_rad": s(bits, 118, 11) * math.pi * 2.0**-14,
    }


def parse_word_10(bits):
    return {
        "word_type": 10,
        "name": "Almanac SVID3 (2/2) + GST-GPS conversion",
        "IODa": u(bits, 7, 4),
        "SVID3_Omega0_rad": s(bits, 11, 16) * math.pi * 2.0**-15,
        "SVID3_OmegaDot_rad_s": s(bits, 27, 11) * math.pi * 2.0**-33,
        "SVID3_M0_rad": s(bits, 38, 16) * math.pi * 2.0**-15,
        "SVID3_af0_s": s(bits, 54, 16) * 2.0**-19,
        "SVID3_af1_s_s": s(bits, 70, 13) * 2.0**-38,
        "SVID3_E5b_health_raw": u(bits, 83, 2),
        "SVID3_E1B_health_raw": u(bits, 85, 2),
        "A0G_s": s(bits, 87, 16) * 2.0**-35,
        "A1G_s_s": s(bits, 103, 12) * 2.0**-51,
        "t0G_s": u(bits, 115, 8) * 3600.0,
        "WN0G_mod64": u(bits, 123, 6),
    }


def classify_modern_word(word_type: int):
    if word_type == 16:
        return {
            "word_type": 16,
            "name": "Reduced CED (RedCED)",
            "role": "single-word coarse clock/ephemeris for fast initial positioning",
        }
    if word_type in (17, 18, 19, 20):
        return {
            "word_type": word_type,
            "name": f"RS-FEC2 redundancy word {word_type}",
            "role": "Reed-Solomon redundancy protecting/recovering WT1-WT4 CED",
        }
    if word_type == 22:
        return {"word_type": 22, "name": "ARAIM Integrity Support Message", "role": "integrity-support data"}
    if word_type == 63:
        return {"word_type": 63, "name": "Dummy word", "role": "no navigation payload"}
    return {"word_type": word_type, "name": f"Word Type {word_type}", "role": "not parsed by this stage"}


AUX_PARSERS = {0: parse_word_0, 7: parse_word_7, 8: parse_word_8, 9: parse_word_9, 10: parse_word_10}


def validate_ephemeris(ephemeris: dict, tracked_prn: int):
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    iod = ephemeris.get("IODnav")
    svid = ephemeris.get("SVID")
    toe = ephemeris.get("toe_s")
    toc = ephemeris.get("toc_s")
    sqrt_a = ephemeris.get("sqrtA_m_sqrt")
    a = ephemeris.get("semi_major_axis_m")
    ecc = ephemeris.get("eccentricity")
    inc = ephemeris.get("i0_rad")
    add("SVID matches tracked PRN", svid == tracked_prn, f"decoded SVID={svid}, tracked PRN={tracked_prn}")
    add("IODnav present", isinstance(iod, int) and 0 <= iod <= 1023, f"IODnav={iod}")
    add("toe inside Galileo week", toe is not None and 0 <= toe < WEEK_S, f"toe={toe}")
    add("toc inside Galileo week", toc is not None and 0 <= toc < WEEK_S, f"toc={toc}")
    add(
        "toe/toc coherent",
        toe is not None and toc is not None and abs(wrap_week_time(toe - toc)) <= 14400,
        f"toe-toc={wrap_week_time(toe - toc) if toe is not None and toc is not None else None} s",
    )
    add(
        "sqrt(A) plausible for Galileo MEO",
        sqrt_a is not None and 5200.0 < sqrt_a < 5700.0,
        f"sqrt(A)={sqrt_a}",
    )
    add(
        "semi-major axis plausible",
        a is not None and 28_000_000.0 < a < 31_500_000.0,
        f"A={a/1000.0 if a is not None else None} km",
    )
    add("eccentricity physically plausible", ecc is not None and 0.0 <= ecc < 0.05, f"e={ecc}")
    inc_deg = math.degrees(inc) if inc is not None else None
    add(
        "inclination plausible for Galileo MEO",
        inc_deg is not None and 45.0 < inc_deg < 65.0,
        f"i0={inc_deg} deg",
    )
    required = [
        "M0_rad",
        "Omega0_rad",
        "omega_rad",
        "OmegaDot_rad_s",
        "delta_n_rad_s",
        "iDot_rad_s",
        "Cuc_rad",
        "Cus_rad",
        "Crc_m",
        "Crs_m",
        "Cic_rad",
        "Cis_rad",
        "af0_s",
        "af1_s_s",
        "af2_s_s2",
    ]
    missing = [k for k in required if k not in ephemeris]
    add(
        "all orbit/clock fields present",
        not missing,
        "missing=" + (", ".join(missing) if missing else "none"),
    )
    nonfinite = [k for k, v in ephemeris.items() if isinstance(v, (int, float)) and not math.isfinite(v)]
    add("finite numeric fields", not nonfinite, "nonfinite=" + str(nonfinite))
    return checks


def derive_time_anchor(messages, explicit_tow=None, explicit_week=None):
    if explicit_tow is not None:
        return {
            "tow_at_decoder_zero_s": None,
            "evaluation_tow_s": float(explicit_tow) % WEEK_S,
            "gst_week": int(explicit_week) if explicit_week is not None else None,
            "source": "user evaluation epoch; decoder-to-GST mapping unavailable",
            "absolute_week_known": explicit_week is not None,
        }
    wt5 = [
        m for m in messages if m["word_type"] == 5 and 0 <= m.get("decoded", {}).get("GST_TOW_s", -1) < WEEK_S
    ]
    if wt5:
        m = wt5[-1]
        tow = float(m["decoded"]["GST_TOW_s"])
        rel = float(m["time_start_s"])
        week = m["decoded"].get("GST_week")
        return {
            "tow_at_decoder_zero_s": (tow - rel) % WEEK_S,
            "gst_week": (
                int(explicit_week)
                if explicit_week is not None
                else (int(week) + math.floor((tow - rel) / WEEK_S)) % 4096
            ),
            "broadcast_page_week": int(week),
            "source": "WT5 GST week/TOW",
            "absolute_week_known": True,
            "week_reference": "GST week modulo 4096 at decoder zero (user week overrides this)",
        }
    wt6 = [m for m in messages if m["word_type"] == 6 and 0 <= m.get("decoded", {}).get("TOW_s", -1) < WEEK_S]
    if wt6:
        m = wt6[-1]
        tow = float(m["decoded"]["TOW_s"])
        rel = float(m["time_start_s"])
        return {
            "tow_at_decoder_zero_s": (tow - rel) % WEEK_S,
            "gst_week": int(explicit_week) if explicit_week is not None else None,
            "source": "WT6 TOW (week not contained in WT6)",
            "absolute_week_known": explicit_week is not None,
        }
    return {
        "tow_at_decoder_zero_s": None,
        "gst_week": int(explicit_week) if explicit_week is not None else None,
        "source": "no WT5/WT6 time anchor available",
        "absolute_week_known": False,
    }


def solve_kepler(mean_anomaly: float, ecc: float) -> float:
    M = math.remainder(mean_anomaly, 2.0 * math.pi)
    E = M
    for _ in range(20):
        f = E - ecc * math.sin(E) - M
        fp = 1.0 - ecc * math.cos(E)
        step = f / fp
        E -= step
        if abs(step) < 1e-13:
            break
    return E


def propagate_ecef(ephem: dict, tow_s: float):
    sqrtA = ephem["sqrtA_m_sqrt"]
    A = sqrtA * sqrtA
    ecc = ephem["eccentricity"]
    tk = wrap_week_time(tow_s - ephem["toe_s"])
    n0 = math.sqrt(MU / A**3)
    n = n0 + ephem["delta_n_rad_s"]
    M = ephem["M0_rad"] + n * tk
    E = solve_kepler(M, ecc)
    sinE = math.sin(E)
    cosE = math.cos(E)
    v = math.atan2(math.sqrt(1.0 - ecc * ecc) * sinE, cosE - ecc)
    phi = v + ephem["omega_rad"]
    sin2phi = math.sin(2.0 * phi)
    cos2phi = math.cos(2.0 * phi)
    du = ephem["Cus_rad"] * sin2phi + ephem["Cuc_rad"] * cos2phi
    dr = ephem["Crs_m"] * sin2phi + ephem["Crc_m"] * cos2phi
    di = ephem["Cis_rad"] * sin2phi + ephem["Cic_rad"] * cos2phi
    u_corr = phi + du
    r = A * (1.0 - ecc * cosE) + dr
    inc = ephem["i0_rad"] + ephem["iDot_rad_s"] * tk + di
    x_orb = r * math.cos(u_corr)
    y_orb = r * math.sin(u_corr)
    Omega = ephem["Omega0_rad"] + (ephem["OmegaDot_rad_s"] - OMEGA_E) * tk - OMEGA_E * ephem["toe_s"]
    cosO = math.cos(Omega)
    sinO = math.sin(Omega)
    cosi = math.cos(inc)
    sini = math.sin(inc)
    x = x_orb * cosO - y_orb * cosi * sinO
    y = x_orb * sinO + y_orb * cosi * cosO
    z = y_orb * sini
    relativity_s = RELATIVISTIC_F * ecc * sqrtA * sinE
    tc = wrap_week_time(tow_s - ephem["toc_s"])
    clock_poly_s = ephem["af0_s"] + ephem["af1_s_s"] * tc + ephem["af2_s_s2"] * tc * tc
    clock_total_s = clock_poly_s + relativity_s
    return {
        "tow_s": tow_s % WEEK_S,
        "tk_s": tk,
        "E_rad": E,
        "true_anomaly_rad": v,
        "corrected_argument_latitude_rad": u_corr,
        "radius_m": r,
        "inclination_rad": inc,
        "Omega_rad": Omega,
        "x_m": x,
        "y_m": y,
        "z_m": z,
        "clock_polynomial_s": clock_poly_s,
        "relativistic_correction_s": relativity_s,
        "clock_total_s": clock_total_s,
    }


def propagate_with_velocity(ephem: dict, tow_s: float):
    center = propagate_ecef(ephem, tow_s)
    h = 0.5
    before = propagate_ecef(ephem, (tow_s - h) % WEEK_S)
    after = propagate_ecef(ephem, (tow_s + h) % WEEK_S)
    center["vx_m_s"] = (after["x_m"] - before["x_m"]) / (2.0 * h)
    center["vy_m_s"] = (after["y_m"] - before["y_m"]) / (2.0 * h)
    center["vz_m_s"] = (after["z_m"] - before["z_m"]) / (2.0 * h)
    center["speed_m_s"] = math.sqrt(center["vx_m_s"] ** 2 + center["vy_m_s"] ** 2 + center["vz_m_s"] ** 2)
    center["ecef_radius_m"] = math.sqrt(center["x_m"] ** 2 + center["y_m"] ** 2 + center["z_m"] ** 2)
    return center


def build_partial_almanac(aux_words):
    satellites = defaultdict(dict)
    batches = set()
    for w in aux_words:
        wt = w["word_type"]
        if wt in (7, 8, 9, 10) and "IODa" in w:
            batches.add(w["IODa"])
        if wt == 7:
            sv = w["SVID1"]
            satellites[sv].update(
                {
                    "IODa": w["IODa"],
                    "WNa_mod4": w["WNa_mod4"],
                    "t0a_s": w["t0a_s"],
                    "delta_sqrtA_m_sqrt": w["delta_sqrtA_m_sqrt"],
                    "eccentricity": w["eccentricity"],
                    "omega_rad": w["omega_rad"],
                    "delta_i_rad": w["delta_i_rad"],
                    "Omega0_rad": w["Omega0_rad"],
                    "OmegaDot_rad_s": w["OmegaDot_rad_s"],
                    "M0_rad": w["M0_rad"],
                }
            )
        elif wt == 8:
            sv2 = w["SVID2"]
            satellites[sv2].update(
                {
                    "IODa": w["IODa"],
                    "delta_sqrtA_m_sqrt": w["SVID2_delta_sqrtA_m_sqrt"],
                    "eccentricity": w["SVID2_eccentricity"],
                    "omega_rad": w["SVID2_omega_rad"],
                    "delta_i_rad": w["SVID2_delta_i_rad"],
                    "Omega0_rad": w["SVID2_Omega0_rad"],
                    "OmegaDot_rad_s": w["SVID2_OmegaDot_rad_s"],
                }
            )
        elif wt == 9:
            sv3 = w["SVID3"]
            satellites[sv3].update(
                {
                    "IODa": w["IODa"],
                    "WNa_mod4": w["WNa_mod4"],
                    "t0a_s": w["t0a_s"],
                    "delta_sqrtA_m_sqrt": w["SVID3_delta_sqrtA_m_sqrt"],
                    "eccentricity": w["SVID3_eccentricity"],
                    "omega_rad": w["SVID3_omega_rad"],
                    "delta_i_rad": w["SVID3_delta_i_rad"],
                }
            )
        elif wt == 10:
            candidates = [
                sv for sv, d in satellites.items() if d.get("IODa") == w["IODa"] and "Omega0_rad" not in d
            ]
            if len(candidates) == 1:
                satellites[candidates[0]].update(
                    {
                        "Omega0_rad": w["SVID3_Omega0_rad"],
                        "OmegaDot_rad_s": w["SVID3_OmegaDot_rad_s"],
                        "M0_rad": w["SVID3_M0_rad"],
                        "af0_s": w["SVID3_af0_s"],
                        "af1_s_s": w["SVID3_af1_s_s"],
                        "E5b_health_raw": w["SVID3_E5b_health_raw"],
                        "E1B_health_raw": w["SVID3_E1B_health_raw"],
                    }
                )
    required_orbit = {
        "delta_sqrtA_m_sqrt",
        "eccentricity",
        "omega_rad",
        "delta_i_rad",
        "Omega0_rad",
        "OmegaDot_rad_s",
        "M0_rad",
    }
    result = []
    for sv, data in sorted(satellites.items()):
        data = dict(data)
        data["SVID"] = sv
        data["sqrtA_m_sqrt"] = (
            NOMINAL_GALILEO_SQRTA + data["delta_sqrtA_m_sqrt"] if "delta_sqrtA_m_sqrt" in data else None
        )
        data["inclination_rad"] = (
            NOMINAL_ALMANAC_INCLINATION_RAD + data["delta_i_rad"] if "delta_i_rad" in data else None
        )
        data["orbit_complete"] = required_orbit.issubset(data.keys())
        if "E1B_health_raw" in data:
            data["E1B_health"] = HEALTH_TEXT.get(data["E1B_health_raw"], "unknown")
        result.append(data)
    return {"IODa_batches_seen": sorted(batches), "satellites": result}


def print_status_report(prn, raw_pages, ephem, validation, time_anchor, aux_words, almanac, propagated):
    word_types = [int(p["word_type"]) for p in raw_pages]
    redced_count = word_types.count(16)
    rs_types = sorted({w for w in word_types if w in (17, 18, 19, 20)})
    print("=" * 78)
    print(f"GALILEO E{prn:02d} — PROFESSIONAL NAVIGATION DATA STATUS")
    print("=" * 78)
    print("\nTelemetry / message layer")
    print(f"  CRC-valid pages                 {len(raw_pages)}")
    print(f"  Word types seen                 {sorted(set(word_types))}")
    print(f"  Reduced CED (WT16)              {'YES' if redced_count else 'NO'}")
    print(f"  RS-FEC2 words                   {', '.join(map(str, rs_types)) if rs_types else 'none'}")
    print("\nFull-precision broadcast ephemeris")
    print(f"  IODnav                          {ephem['IODnav']}")
    print(f"  SVID                            {ephem['SVID']}")
    print("  WT1-WT4 coherent CED            YES")
    print(f"  Semi-major axis                 {ephem['semi_major_axis_m']/1000:.3f} km")
    print(f"  Eccentricity                    {ephem['eccentricity']:.9f}")
    print(f"  Inclination                     {math.degrees(ephem['i0_rad']):.4f} deg")
    sisa = decode_sisa(int(ephem["SISA_index"]))
    if sisa["kind"] == "value":
        print(f"  SISA                            {sisa['metres']:.2f} m (index {ephem['SISA_index']})")
    else:
        print(f"  SISA                            {sisa['kind']} (index {ephem['SISA_index']})")
    print("\nSemantic validation")
    for check in validation:
        print(f"  {'PASS' if check['pass'] else 'FAIL':4s}  {check['check']:<34s} {check['detail']}")
    print("\nTime")
    print(f"  Time anchor source              {time_anchor['source']}")
    print(f"  GST TOW at decoder t=0          {time_anchor['tow_at_decoder_zero_s']}")
    print(f"  GST week                        {time_anchor['gst_week']}")
    if not time_anchor["absolute_week_known"]:
        print("  Note                            week unknown; ECEF propagation still valid")
    print("\nPropagated satellite state")
    print(f"  Evaluation TOW                  {propagated['tow_s']:.3f} s")
    print(f"  X                               {propagated['x_m']/1000:.3f} km")
    print(f"  Y                               {propagated['y_m']/1000:.3f} km")
    print(f"  Z                               {propagated['z_m']/1000:.3f} km")
    print(f"  ECEF speed                      {propagated['speed_m_s']:.3f} m/s")
    print(f"  Satellite clock correction      {propagated['clock_total_s']*1e6:+.3f} us")
    print(f"    polynomial                    {propagated['clock_polynomial_s']*1e6:+.3f} us")
    print(f"    relativity                    {propagated['relativistic_correction_s']*1e9:+.3f} ns")
    print("\nReceiver readiness")
    print(f"  Orbit propagation               {'READY' if all(c['pass'] for c in validation) else 'REVIEW'}")
    print("  Single-satellite clock/orbit     READY")
    print("  Absolute receiver PVT            NOT YET")
    print("  Next requirement                 >=4 simultaneous tracked satellites + pseudoranges")
    print("=" * 78)


def write_orbit_csv(path, samples):
    fields = [
        "tow_s",
        "x_m",
        "y_m",
        "z_m",
        "vx_m_s",
        "vy_m_s",
        "vz_m_s",
        "speed_m_s",
        "ecef_radius_m",
        "clock_polynomial_s",
        "relativistic_correction_s",
        "clock_total_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({k: sample[k] for k in fields})


from .navigation_figures import navigation_dashboard as make_dashboard


def main():
    p = argparse.ArgumentParser(description="Interpret Galileo I/NAV and propagate the broadcast orbit.")
    p.add_argument("navigation_json", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--tow", type=float, default=None, help="Optional Galileo Time-of-Week [s]")
    p.add_argument(
        "--gst-week", type=int, default=None, help="GST week context (modulo 4096), not UTC or GPS week."
    )
    p.add_argument(
        "--span-minutes",
        type=float,
        default=90.0,
        help="Orbit illustration half-span; predictions are not measured positions.",
    )
    args = p.parse_args()
    if not math.isfinite(args.span_minutes) or args.span_minutes <= 0:
        raise ValueError("--span-minutes must be finite and positive")
    if args.tow is not None and (not math.isfinite(args.tow) or not 0 <= args.tow < WEEK_S):
        raise ValueError("--tow must be inside a Galileo week")
    if args.gst_week is not None and not 0 <= args.gst_week < 4096:
        raise ValueError("--gst-week must be 0..4095")
    src = args.navigation_json.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    raw = json.loads(src.read_text(encoding="utf-8"))
    if "pages" in raw:
        raw = assemble_navigation(raw)
    prn = int(raw["satellite_prn"])
    messages = raw.get("messages", [])
    ephem = raw.get("latest_complete_ephemeris")
    outdir = args.output.expanduser().resolve() if args.output else src.parent / "orbit_dashboard"
    outdir.mkdir(parents=True, exist_ok=False)
    write_json(outdir / "navigation.json", raw)
    if ephem is None:
        from .navigation_figures import incomplete_navigation

        incomplete_navigation(raw, outdir, "Missing a coherent WT1–WT4 ephemeris")
        write_json(
            outdir / "receiver_navigation_status.json",
            {
                "satellite_prn": prn,
                "readiness": {"orbit_propagation": False, "receiver_pvt": False},
                "reason": "No complete CRC-valid WT1–WT4 set with one IODnav",
                "ephemeris_word_coverage": raw.get("ephemeris_word_coverage", {}),
            },
        )
        return 3
    raw_pages = [
        {
            "time_start_s": float(m["time_start_s"]),
            "word_type": int(m["word_type"]),
            "raw_bits": m["raw_bits"],
            "decoded": m.get("decoded", {}),
        }
        for m in messages
    ]
    aux_words = []
    for page in raw_pages:
        wt = page["word_type"]
        bits = bits_from_string(page["raw_bits"])
        if wt in AUX_PARSERS:
            parsed = AUX_PARSERS[wt](bits)
        elif wt in (16, 17, 18, 19, 20, 22, 63):
            parsed = classify_modern_word(wt)
        else:
            continue
        parsed["time_start_s"] = page["time_start_s"]
        parsed["raw_bits"] = page["raw_bits"]
        aux_words.append(parsed)
    almanac = build_partial_almanac(aux_words)
    validation = validate_ephemeris(ephem, prn)
    time_anchor = derive_time_anchor(raw_pages, args.tow, args.gst_week)
    if not all(c["pass"] for c in validation):
        from .navigation_figures import incomplete_navigation

        incomplete_navigation(raw, outdir, "Ephemeris failed physical consistency checks")
        write_json(
            outdir / "receiver_navigation_status.json",
            {
                "satellite_prn": prn,
                "validation": validation,
                "all_semantic_checks_pass": False,
                "readiness": {"orbit_propagation": False, "receiver_pvt": False},
            },
        )
        return 3
    if args.tow is not None:
        evaluation_tow = args.tow % WEEK_S
    elif time_anchor["tow_at_decoder_zero_s"] is not None:
        mid_rel = (
            0.5 * (min(p["time_start_s"] for p in raw_pages) + max(p["time_start_s"] for p in raw_pages))
            if raw_pages
            else 0.0
        )
        evaluation_tow = (time_anchor["tow_at_decoder_zero_s"] + mid_rel) % WEEK_S
    else:
        evaluation_tow = float(ephem["toe_s"])
    time_anchor["evaluation_epoch_source"] = (
        "user override"
        if args.tow is not None
        else (
            "broadcast TOW"
            if time_anchor["tow_at_decoder_zero_s"] is not None
            else "toe fallback; capture epoch unknown"
        )
    )
    evaluation = propagate_with_velocity(ephem, evaluation_tow)
    span_s = args.span_minutes * 60.0
    sample_offsets = np.linspace(-span_s, +span_s, 361)
    orbit_samples = [propagate_with_velocity(ephem, (evaluation_tow + dt) % WEEK_S) for dt in sample_offsets]
    print_status_report(prn, raw_pages, ephem, validation, time_anchor, aux_words, almanac, evaluation)
    (outdir / "parsed_auxiliary_words.json").write_text(
        json.dumps({"satellite_prn": prn, "auxiliary_words": aux_words, "partial_almanac": almanac}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    write_orbit_csv(outdir / "orbit_samples.csv", orbit_samples)
    status = {
        "satellite_prn": prn,
        "IODnav": ephem["IODnav"],
        "ephemeris": ephem,
        "sisa": decode_sisa(int(ephem["SISA_index"])),
        "validation": validation,
        "all_semantic_checks_pass": all(c["pass"] for c in validation),
        "time_anchor": time_anchor,
        "evaluation_state": evaluation,
        "modern_inav": {
            "redced_word_count": sum(int(p["word_type"] == 16) for p in raw_pages),
            "rs_fec2_types_seen": sorted(
                {int(p["word_type"]) for p in raw_pages if int(p["word_type"]) in (17, 18, 19, 20)}
            ),
            "note": (
                "WT16 RedCED and WT17-WT20 RS-FEC2 are recognized but are not "
                "used to replace a complete full-precision WT1-WT4 CED."
            ),
        },
        "almanac": almanac,
        "readiness": {
            "orbit_propagation": True,
            "satellite_clock_model": True,
            "clock_model_scope": "broadcast polynomial plus relativity; E1 BGD not applied",
            "single_satellite_navigation_data": True,
            "pseudorange_observables": False,
            "receiver_pvt": False,
            "next_requirement": (
                "simultaneously track at least four satellites and form "
                "absolute transmit-time/pseudorange observables"
            ),
        },
    }
    (outdir / "receiver_navigation_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nJSON status : {outdir / 'receiver_navigation_status.json'}")
    print(f"Orbit CSV   : {outdir / 'orbit_samples.csv'}")
    print(f"Aux words   : {outdir / 'parsed_auxiliary_words.json'}")
    make_dashboard(raw_pages, ephem, validation, time_anchor, evaluation, orbit_samples, outdir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
