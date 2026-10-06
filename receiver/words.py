"""I/NAV WT1–6: OS SIS ICD v2.2, tables 42–47, 67, 69–70.
Source: https://www.gsc-europa.eu/sites/default/files/sites/all/files/Galileo_OS_SIS_ICD_v2.2.pdf
Bit offsets below are zero-based, MSB first. Only CRC-passing words enter assembly.
"""

import math


def parse_word(text):
    if len(text) != 128 or set(text) - {"0", "1"}:
        raise ValueError("An I/NAV word must contain exactly 128 binary digits.")

    def u(start, width):
        return int(text[start : start + width], 2)

    def s(start, width):
        v = u(start, width)
        return v - (1 << width) if v & (1 << (width - 1)) else v

    wt = u(0, 6)
    d = {"word_type": wt}
    if wt in (1, 2, 3, 4):
        d["IODnav"] = u(6, 10)
    if wt == 1:
        d.update(
            toe_s=u(16, 14) * 60,
            M0_rad=s(30, 32) * 2**-31 * math.pi,
            eccentricity=u(62, 32) * 2**-33,
            sqrtA_m_sqrt=u(94, 32) * 2**-19,
        )
        d["semi_major_axis_m"] = d["sqrtA_m_sqrt"] ** 2
    elif wt == 2:
        d.update(
            Omega0_rad=s(16, 32) * 2**-31 * math.pi,
            i0_rad=s(48, 32) * 2**-31 * math.pi,
            omega_rad=s(80, 32) * 2**-31 * math.pi,
            iDot_rad_s=s(112, 14) * 2**-43 * math.pi,
        )
    elif wt == 3:
        d.update(
            OmegaDot_rad_s=s(16, 24) * 2**-43 * math.pi,
            delta_n_rad_s=s(40, 16) * 2**-43 * math.pi,
            Cuc_rad=s(56, 16) * 2**-29,
            Cus_rad=s(72, 16) * 2**-29,
            Crc_m=s(88, 16) * 2**-5,
            Crs_m=s(104, 16) * 2**-5,
            SISA_index=u(120, 8),
        )
    elif wt == 4:
        d.update(
            SVID=u(16, 6),
            Cic_rad=s(22, 16) * 2**-29,
            Cis_rad=s(38, 16) * 2**-29,
            toc_s=u(54, 14) * 60,
            af0_s=s(68, 31) * 2**-34,
            af1_s_s=s(99, 21) * 2**-46,
            af2_s_s2=s(120, 6) * 2**-59,
        )
    elif wt == 5:
        d.update(
            ai0=u(6, 11) * 2**-2,
            ai1=s(17, 11) * 2**-8,
            ai2=s(28, 14) * 2**-15,
            iono_disturbance_flags=[u(42 + i, 1) for i in range(5)],
            BGD_E1_E5a_s=s(47, 10) * 2**-32,
            BGD_E1_E5b_s=s(57, 10) * 2**-32,
            E5b_HS=u(67, 2),
            E1B_HS=u(69, 2),
            E5b_DVS=u(71, 1),
            E1B_DVS=u(72, 1),
            GST_week=u(73, 12),
            GST_TOW_s=u(85, 20),
        )
    elif wt == 6:
        d.update(
            A0_s=s(6, 32) * 2**-30,
            A1_s_s=s(38, 24) * 2**-50,
            delta_tLS_s=s(62, 8),
            tot_s=u(70, 8) * 3600,
            WNot=u(78, 8),
            WNLSF=u(86, 8),
            DN=u(94, 3),
            delta_tLSF_s=s(97, 8),
            TOW_s=u(105, 20),
        )
    return d


def assemble_navigation(raw):
    messages, groups, complete = [], {}, []
    for page in sorted(raw.get("pages", []), key=lambda p: p["time_start_s"]):
        if page.get("crc_ok") is not True:
            continue
        bits = page["word_bits_128"]
        decoded = parse_word(bits)
        wt = decoded["word_type"]
        if wt != page["word_type"]:
            raise ValueError("Word type disagrees with the preserved navigation bits.")
        message = {**page, "raw_bits": bits, "decoded": decoded}
        messages.append(message)
        if wt not in (1, 2, 3, 4):
            continue
        iod = decoded["IODnav"]
        group = groups.setdefault(iod, {})
        # Bound assembly in receive time; never combine distant reuses of a 10-bit IOD.
        group = {k: v for k, v in group.items() if message["time_start_s"] - v["time_start_s"] <= 14400}
        group[wt] = message
        groups[iod] = group
        if set(group) == {1, 2, 3, 4}:
            ephem = {k: v for w in range(1, 5) for k, v in group[w]["decoded"].items() if k != "word_type"}
            ephem["source_page_times_s"] = {str(w): group[w]["time_start_s"] for w in range(1, 5)}
            complete.append(ephem)
    return dict(
        satellite_prn=raw["satellite_prn"],
        messages=messages,
        latest_complete_ephemeris=complete[-1] if complete else None,
        complete_ephemerides=complete,
        ephemeris_word_coverage={str(iod): sorted(group) for iod, group in groups.items()},
        rejected_crc_pages=sum(p.get("crc_ok") is not True for p in raw.get("pages", [])),
        recording=raw.get("recording"),
        capture_metadata=raw.get("capture_metadata"),
    )
