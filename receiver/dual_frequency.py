"""Frequency combinations with explicit noise and hardware-bias accounting.
ESA Navipedia: Ionosphere-free Combination for Dual Frequency Receivers.
"""

import math
import numpy as np
from .position_models import C


def coefficients(f1, f2):
    if not all(math.isfinite(f) and f > 0 for f in (f1, f2)) or f1 == f2:
        raise ValueError("Two distinct positive frequencies are required.")
    return f1 * f1 / (f1 * f1 - f2 * f2), -f2 * f2 / (f1 * f1 - f2 * f2)


def combine(e1, e5, sigma1=10.0, sigma5=3.0):
    if (
        e1["prn"] != e5["prn"]
        or e1.get("band") != "e1"
        or e5.get("band") not in ("e5a", "e5b")
    ):
        raise ValueError(
            "An E1/E5 combination requires the same satellite on two supported bands."
        )
    if abs(e1["receive_time_s"] - e5["receive_time_s"]) > 1e-9:
        raise ValueError(
            "Dual-frequency observations must refer to the same receive epoch."
        )
    if e1["iodnav"] != e5["iodnav"]:
        raise ValueError(
            "E1/E5 broadcast ephemerides have different IODnav; do not mix them."
        )
    if np.linalg.norm(e1["satellite_ecef_m"] - e5["satellite_ecef_m"]) > 1.0:
        raise ValueError("E1/E5 satellite states disagree by more than 1 m.")
    a, b = coefficients(e1["frequency_hz"], e5["frequency_hz"])
    p1 = (
        e1["raw_pseudorange_m"]
        + C * e1["satellite_clock_e1_s"]
        - e1["hardware_delay_m"]
    )
    p5 = (
        e5["raw_pseudorange_m"]
        + C * e5["satellite_clock_e1_s"]
        - e5["hardware_delay_m"]
    )
    # Single-band clocks include their BGD factors. Weighted together they
    # cancel the matched-pair BGD, rather than applying BGD a second time.
    out = {
        **e1,
        "band": "e1+" + e5["band"],
        "raw_pseudorange_m": a * e1["raw_pseudorange_m"] + b * e5["raw_pseudorange_m"],
        "satellite_clock_e1_s": a * e1["satellite_clock_e1_s"]
        + b * e5["satellite_clock_e1_s"],
        "clock_polynomial_s": a * e1["clock_polynomial_s"]
        + b * e5["clock_polynomial_s"],
        "relativity_s": a * e1["relativity_s"] + b * e5["relativity_s"],
        "bgd_s": 0.0,
        "bgd_factor": 0.0,
        "bgd_e1_e5b_s": 0.0,
        "hardware_delay_m": a * e1["hardware_delay_m"] + b * e5["hardware_delay_m"],
        "ionosphere_free": True,
        "satellite_ecef_m": a * e1["satellite_ecef_m"] + b * e5["satellite_ecef_m"],
        "sisa_m": max(e1["sisa_m"], e5["sisa_m"]),
        "code_sigma_m": math.hypot(a * sigma1, b * sigma5),
        "differential_delay_sigma_m": math.hypot(
            a * e1.get("differential_delay_sigma_m", 0),
            b * e5.get("differential_delay_sigma_m", 0),
        ),
    }
    out.pop("broadcast_iono", None)
    gamma = (e1["frequency_hz"] / e5["frequency_hz"]) ** 2
    diagnostic = dict(
        prn=e1["prn"],
        receive_time_s=e1["receive_time_s"],
        pair=out["band"],
        alpha=a,
        beta=b,
        e1_clock_corrected_range_m=p1,
        e5_clock_corrected_range_m=p5,
        ionosphere_free_range_m=a * p1 + b * p5,
        geometry_free_code_m=p5 - p1,
        apparent_e1_iono_m=(p5 - p1) / (gamma - 1),
        equal_noise_amplification=math.hypot(a, b),
        combined_code_sigma_m=out["code_sigma_m"],
        interpretation="Code-derived apparent ionosphere includes receiver inter-frequency delay, multipath and noise; not calibrated TEC.",
    )
    return out, diagnostic
