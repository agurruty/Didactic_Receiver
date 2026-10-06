"""Galileo E5a F/NAV: ICD 2.2 §4.2, tables 30–33, parameter scales §5.1.
Raw pages and CRC results remain visible; only coherent IODnav sets form an orbit.
"""

import math
import numpy as np
from .telemetry import viterbi_soft, crc24_galileo, bits_to_uint

PREAMBLE = 1 - 2 * np.array([int(x) for x in "101101110000"], dtype=float)


def decode_stream(soft, times, boundaries):
    scale = np.median(abs(soft))
    x = soft / max(scale, 1e-30)
    scores = np.zeros(500)
    for offset in range(500):
        values = []
        for i in range(offset, len(x) - 11, 500):
            z = x[i : i + 12]
            norm = np.linalg.norm(z) * np.sqrt(12)
            if norm:
                values.append(float(z @ PREAMBLE / norm))
        scores[offset] = np.mean(values) if values else 0
    start = int(np.argmax(abs(scores)))
    polarity = 1 if scores[start] >= 0 else -1
    alternatives = []
    for reverse in (False, True):
        pages = []
        for i in range(start, len(x) - 499, 500):
            coded = (x[i + 12 : i + 500] * polarity).reshape(8, 61).T.reshape(-1).copy()
            coded[1::2] *= -1
            bits, metric = viterbi_soft(coded, reverse, expected_symbols=488)
            crc = crc24_galileo(bits[:214])
            ok = bool(np.array_equal(crc, bits[214:238]))
            pages.append(
                dict(
                    crc_ok=ok,
                    word_type=bits_to_uint(bits[:6]),
                    page_bits_244="".join(map(str, bits)),
                    time_start_s=float(times[i]),
                    boundary_sample=float(boundaries[i]),
                    symbol_start=i,
                    path_metric=metric,
                    polynomial_reversed=reverse,
                )
            )
        alternatives.append(pages)
    pages = max(
        alternatives,
        key=lambda pp: (
            sum(p["crc_ok"] for p in pp),
            -sum(p["path_metric"] for p in pp),
        ),
    )
    return pages, scores.tolist()


def parse_page(text):
    if len(text) != 244 or set(text) - {"0", "1"}:
        raise ValueError("F/NAV requires exactly 244 decoded bits.")
    offset = 0

    def take(width, scale=1, signed=False):
        nonlocal offset
        value = int(text[offset : offset + width], 2)
        offset += width
        if signed and value & (1 << (width - 1)):
            value -= 1 << width
        return value * scale

    page = int(take(6))
    d = {"word_type": page}

    def fields(spec):
        for name, width, scale, signed in spec:
            d[name] = take(width, scale, signed)

    if page == 1:
        fields(
            [
                ("SVID", 6, 1, False),
                ("IODnav", 10, 1, False),
                ("toc_s", 14, 60, False),
                ("af0_s", 31, 2**-34, True),
                ("af1_s_s", 21, 2**-46, True),
                ("af2_s_s2", 6, 2**-59, True),
                ("SISA_index", 8, 1, False),
                ("ai0", 11, 2**-2, False),
                ("ai1", 11, 2**-8, True),
                ("ai2", 14, 2**-15, True),
            ]
        )
        d["iono_disturbance_flags"] = [take(1) for _ in range(5)]
        fields(
            [
                ("BGD_E1_E5a_s", 10, 2**-32, True),
                ("E5a_HS", 2, 1, False),
                ("GST_week", 12, 1, False),
                ("GST_TOW_s", 20, 1, False),
                ("E5a_DVS", 1, 1, False),
            ]
        )
    elif page == 2:
        fields(
            [
                ("IODnav", 10, 1, False),
                ("M0_rad", 32, math.pi * 2**-31, True),
                ("OmegaDot_rad_s", 24, math.pi * 2**-43, True),
                ("eccentricity", 32, 2**-33, False),
                ("sqrtA_m_sqrt", 32, 2**-19, False),
                ("Omega0_rad", 32, math.pi * 2**-31, True),
                ("iDot_rad_s", 14, math.pi * 2**-43, True),
                ("GST_week", 12, 1, False),
                ("GST_TOW_s", 20, 1, False),
            ]
        )
        d["semi_major_axis_m"] = d["sqrtA_m_sqrt"] ** 2
    elif page == 3:
        fields(
            [
                ("IODnav", 10, 1, False),
                ("i0_rad", 32, math.pi * 2**-31, True),
                ("omega_rad", 32, math.pi * 2**-31, True),
                ("delta_n_rad_s", 16, math.pi * 2**-43, True),
                ("Cuc_rad", 16, 2**-29, True),
                ("Cus_rad", 16, 2**-29, True),
                ("Crc_m", 16, 2**-5, True),
                ("Crs_m", 16, 2**-5, True),
                ("toe_s", 14, 60, False),
                ("GST_week", 12, 1, False),
                ("GST_TOW_s", 20, 1, False),
            ]
        )
    elif page == 4:
        fields(
            [
                ("IODnav", 10, 1, False),
                ("Cic_rad", 16, 2**-29, True),
                ("Cis_rad", 16, 2**-29, True),
                ("A0_s", 32, 2**-30, True),
                ("A1_s_s", 24, 2**-50, True),
                ("delta_tLS_s", 8, 1, True),
                ("tot_s", 8, 3600, False),
                ("WNot", 8, 1, False),
                ("WNLSF", 8, 1, False),
                ("DN", 3, 1, False),
                ("delta_tLSF_s", 8, 1, True),
                ("t0G_s", 8, 3600, False),
                ("A0G_s", 16, 2**-35, True),
                ("A1G_s_s", 12, 2**-51, True),
                ("WN0G", 6, 1, False),
                ("TOW_s", 20, 1, False),
            ]
        )
    return d


def assemble(raw):
    messages = []
    groups = {}
    complete = []
    for page in sorted(raw.get("pages", []), key=lambda p: p["time_start_s"]):
        if not page.get("crc_ok"):
            continue
        decoded = parse_page(page["page_bits_244"])
        if decoded["word_type"] != page["word_type"]:
            raise ValueError("F/NAV page type disagrees with raw bits.")
        message = {**page, "decoded": decoded}
        messages.append(message)
        wt = decoded["word_type"]
        if wt not in (1, 2, 3, 4):
            continue
        iod = decoded["IODnav"]
        group = groups.setdefault(iod, {})
        group = {
            k: v
            for k, v in group.items()
            if page["time_start_s"] - v["time_start_s"] <= 14400
        }
        group[wt] = message
        groups[iod] = group
        if set(group) == {1, 2, 3, 4}:
            ephem = {
                k: v
                for w in range(1, 5)
                for k, v in group[w]["decoded"].items()
                if k != "word_type"
            }
            ephem["source_page_times_s"] = {
                str(w): group[w]["time_start_s"] for w in range(1, 5)
            }
            complete.append(ephem)
    return dict(
        satellite_prn=raw["satellite_prn"],
        messages=messages,
        complete_ephemerides=complete,
        latest_complete_ephemeris=complete[-1] if complete else None,
        recording=raw.get("recording"),
        capture_metadata=raw.get("capture_metadata"),
        ephemeris_word_coverage={str(k): sorted(v) for k, v in groups.items()},
        rejected_crc_pages=sum(not p.get("crc_ok") for p in raw.get("pages", [])),
    )
