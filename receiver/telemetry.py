#!/usr/bin/env python3
"""
Decode Galileo E1-B I/NAV page parts from the Prompt correlator output of
02_galileo_e1_track.py.

INPUT
-----
The script expects the previous tracking stage output:

    <tracking_dir>/tracking_validation.npz
    <tracking_dir>/metadata.json

Each Prompt correlation corresponds to one Galileo E1-B code period:
    4 ms = one transmitted I/NAV coded symbol.

THIS SCRIPT PERFORMS
--------------------
1. Convert tracked Prompt I values into soft navigation symbols.
2. Resolve the Costas-loop 180 degree ambiguity automatically.
3. Find the unencoded 10-symbol I/NAV synchronization word:
       0101100000
4. Align the stream to 1-second I/NAV page parts:
       10 sync symbols + 240 encoded/interleaved symbols.
5. Undo the Galileo 8-row x 30-column block interleaver.
6. Undo the inversion on convolutional encoder branch G2.
7. Soft-decision Viterbi decode:
       rate 1/2, K=7, G1=171(oct), G2=133(oct)
8. Pair E1-B even + odd page parts.
9. Verify the 24-bit Galileo CRC.
10. Extract the 128-bit nominal I/NAV word and its 6-bit Word Type.

The goal is a decisive first telemetry milestone:
    CRC PASS = the recovered Galileo navigation page passes its integrity check.

This script intentionally stops at the word level. Parsing the individual
ephemeris / clock / GST fields is the next stage, after we have proven that
telemetry synchronization, FEC and CRC are all correct.

Reference structure (Galileo OS SIS ICD):
    page part = 250 symbols = 10 sync + 240 FEC/interleaved symbols
    decoded page part = 120 bits
    sync pattern = 0101100000
    convolutional code = rate 1/2, K=7, G1=171o, G2=133o
    interleaver = 30 columns x 8 rows
    nominal E1-B vertical page = even part + odd part
    CRC = 24 bits, polynomial 0x1864CFB

Example:
    python3 03_galileo_e1_decode.py \
        /path/to/tracking

For a useful demonstration, the tracking file should contain several seconds.
Five seconds is enough for a couple of complete pages; 15-30 seconds is much
nicer because you will see multiple I/NAV word types.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from .common import finish

SYNC_BITS = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
SYNC_SIGNS = 1.0 - 2.0 * SYNC_BITS.astype(np.float64)  # 0 -> +1, 1 -> -1

SYMBOL_PERIOD_S = 0.004
SYMBOLS_PER_PART = 250
SYNC_SYMBOLS = 10
CODED_SYMBOLS = 240
DECODED_BITS = 120

INTERLEAVER_ROWS = 8
INTERLEAVER_COLS = 30

G1 = 0o171
G2 = 0o133
K = 7
NSTATES = 1 << (K - 1)

CRC_POLY = 0x1864CFB  # degree 24, includes x^24 coefficient

WORD_TYPE_NAMES = {
    0: "Spare",
    1: "Ephemeris 1/4",
    2: "Ephemeris 2/4",
    3: "Ephemeris 3/4",
    4: "Ephemeris 4/4 + SVID",
    5: "Iono + BGD + GST + health",
    6: "GST-UTC + TOW",
    7: "Almanac",
    8: "Almanac",
    9: "Almanac",
    10: "Almanac",
    16: "Reduced CED",
    17: "RS FEC2",
    18: "RS FEC2",
    19: "RS FEC2",
    20: "RS FEC2",
    22: "ARAIM ISM",
    63: "Dummy",
}


def parse_args():
    p = argparse.ArgumentParser(description="Decode Galileo E1-B I/NAV from tracking_validation.npz.")
    p.add_argument("tracking_dir", type=Path, help="Directory produced by 02_galileo_e1_track.py")
    p.add_argument(
        "--output", type=Path, default=None, help="Output directory. Default: <tracking_dir>/inav_decode"
    )
    p.add_argument(
        "--skip-seconds",
        type=float,
        default=0.4,
        help="Ignore initial tracking settling interval. Default: 0.4 s.",
    )
    return p.parse_args()


def parity(x: int) -> int:
    return x.bit_count() & 1


def reverse_n_bits(x: int, n: int) -> int:
    y = 0
    for _ in range(n):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def build_trellis(poly_reverse: bool = False):
    """
    Build a 64-state K=7 trellis.

    We support both polynomial bit orientations and select the one that gives
    the most physically valid Galileo page parts / CRCs. This makes the script
    robust to convention differences in polynomial diagrams without hiding
    that convention issue from the student.
    """
    g1 = reverse_n_bits(G1, K) if poly_reverse else G1
    g2 = reverse_n_bits(G2, K) if poly_reverse else G2
    next_state = np.zeros((NSTATES, 2), dtype=np.int16)
    outputs = np.zeros((NSTATES, 2, 2), dtype=np.uint8)
    for state in range(NSTATES):
        for bit in (0, 1):
            reg = ((state << 1) | bit) & 0x7F
            ns = reg & 0x3F
            outputs[state, bit, 0] = parity(reg & g1)
            outputs[state, bit, 1] = parity(reg & g2)
            next_state[state, bit] = ns
    return next_state, outputs


def viterbi_soft(symbols: np.ndarray, poly_reverse: bool = False, expected_symbols: int = 240):
    """
    Soft Viterbi decoder for 240 input soft symbols -> 120 decoded bits.

    Convention:
        positive soft value -> bit 0 is more likely
        negative soft value -> bit 1 is more likely

    Galileo's second convolutional output branch is inverted in the transmitted
    waveform. The caller must undo that inversion before entering this decoder.

    The 6 zero tail bits terminate the encoder, therefore state 0 is used as
    the final traceback state.
    """
    if len(symbols) != expected_symbols or len(symbols) % 2:
        raise ValueError(f"Viterbi expects exactly {expected_symbols} soft symbols")
    ns, out = build_trellis(poly_reverse)
    pairs = symbols.reshape(-1, 2)
    steps = len(pairs)
    metric = np.full(NSTATES, np.inf, dtype=np.float64)
    metric[0] = 0.0
    prev_state = np.zeros((steps, NSTATES), dtype=np.int16)
    prev_bit = np.zeros((steps, NSTATES), dtype=np.uint8)
    for k, (s0, s1) in enumerate(pairs):
        new_metric = np.full(NSTATES, np.inf, dtype=np.float64)
        for state in range(NSTATES):
            m0 = metric[state]
            if not np.isfinite(m0):
                continue
            for bit in (0, 1):
                state2 = int(ns[state, bit])
                b0 = int(out[state, bit, 0])
                b1 = int(out[state, bit, 1])
                # Expected BPSK signs: 0 -> +1, 1 -> -1.
                e0 = 1.0 if b0 == 0 else -1.0
                e1 = 1.0 if b1 == 0 else -1.0
                # Squared Euclidean soft metric.
                bm = (s0 - e0) ** 2 + (s1 - e1) ** 2
                cand = m0 + bm
                if cand < new_metric[state2]:
                    new_metric[state2] = cand
                    prev_state[k, state2] = state
                    prev_bit[k, state2] = bit
        metric = new_metric
    # Tail bits force state zero.
    state = 0
    bits = np.zeros(steps, dtype=np.uint8)
    for k in range(steps - 1, -1, -1):
        bits[k] = prev_bit[k, state]
        state = int(prev_state[k, state])
    return bits, float(metric[0])


def deinterleave(soft: np.ndarray):
    """
    Galileo I/NAV deinterleaver matching:
        out[c * rows + r] = in[r * cols + c]

    8 rows x 30 columns = 240 symbols.
    """
    if len(soft) != 240:
        raise ValueError("Expected 240 interleaved symbols")
    out = np.empty_like(soft)
    for r in range(INTERLEAVER_ROWS):
        for c in range(INTERLEAVER_COLS):
            out[c * INTERLEAVER_ROWS + r] = soft[r * INTERLEAVER_COLS + c]
    return out


def decode_page_part(coded_soft: np.ndarray, poly_reverse: bool):
    soft = deinterleave(coded_soft)
    # Galileo ICD convolutional encoder has an inverter on G2.
    # Symbol order is G1, G2, G1, G2, ...
    soft = soft.copy()
    soft[1::2] *= -1.0
    # Normalize so Euclidean metrics are well conditioned but keep soft weights.
    scale = np.median(np.abs(soft))
    if scale > 0:
        soft /= scale
    bits, path_metric = viterbi_soft(soft, poly_reverse=poly_reverse)
    tail_weight = int(np.sum(bits[-6:]))
    even_odd = int(bits[0])
    page_type = int(bits[1])
    structure_score = 0
    structure_score += 2 if tail_weight == 0 else -tail_weight
    structure_score += 1 if even_odd in (0, 1) else -10
    structure_score += 1 if page_type in (0, 1) else -10
    return {
        "bits": bits,
        "path_metric": path_metric,
        "tail_weight": tail_weight,
        "even_odd": even_odd,
        "page_type": page_type,
        "structure_score": structure_score,
        "soft_deinterleaved": soft,
    }


def crc24_galileo(message_bits):
    """
    Direct MSB-first polynomial division per Galileo OS SIS ICD.

    Remainder of m(x)*x^24 divided by G(x), with:
        G = 0x1864CFB
    """
    poly_bits = np.array([(CRC_POLY >> i) & 1 for i in range(24, -1, -1)], dtype=np.uint8)
    work = np.concatenate([np.asarray(message_bits, dtype=np.uint8), np.zeros(24, dtype=np.uint8)])
    for i in range(len(message_bits)):
        if work[i]:
            work[i : i + 25] ^= poly_bits
    return work[-24:]


def bits_to_uint(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def nominal_vertical_page(even_bits, odd_bits):
    """
    Assemble and CRC-check a nominal E1-B vertical page.

    Current E1-B nominal layout:
      EVEN:
        [0]      even/odd = 0
        [1]      page type = 0
        [2:114]  Data k (1/2), 112 bits
        [114:120] tail

      ODD:
        [0]      even/odd = 1
        [1]      page type = 0
        [2:18]   Data j (2/2), 16 bits
        [18:58]  OSNMA, 40 bits
        [58:80]  SAR, 22 bits
        [80:82]  Spare, 2 bits
        [82:106] CRC, 24 bits
        [106:114] SSP, 8 bits
        [114:120] tail
    """
    if int(even_bits[0]) != 0 or int(odd_bits[0]) != 1:
        return None
    if int(even_bits[1]) != 0 or int(odd_bits[1]) != 0:
        return None
    data_1 = even_bits[2:114]
    data_2 = odd_bits[2:18]
    osnma = odd_bits[18:58]
    sar = odd_bits[58:80]
    spare = odd_bits[80:82]
    received_crc = odd_bits[82:106]
    ssp = odd_bits[106:114]
    protected = np.concatenate(
        [even_bits[:114], odd_bits[:82]]  # EO + type + 112 data  # EO + type + 16 data + OSNMA + SAR + spare
    )
    calculated_crc = crc24_galileo(protected)
    crc_ok = bool(np.array_equal(calculated_crc, received_crc))
    word = np.concatenate([data_1, data_2])
    word_type = bits_to_uint(word[:6])
    return {
        "crc_ok": crc_ok,
        "word": word,
        "word_type": word_type,
        "word_name": WORD_TYPE_NAMES.get(word_type, "Other / reserved"),
        "received_crc": received_crc,
        "calculated_crc": calculated_crc,
        "osnma": osnma,
        "sar": sar,
        "ssp": ssp,
    }


def sync_offset_search(soft_symbols):
    """
    Search for the 10-symbol sync pattern repeating every 250 symbols.

    Costas tracking has a +/-180 degree ambiguity. We therefore test both
    stream polarities and return the one whose repeated sync correlations are
    strongest.
    """
    max_offset = SYMBOLS_PER_PART
    results = []
    # Normalize locally only for sync scoring.
    scale = np.median(np.abs(soft_symbols))
    x = soft_symbols / scale if scale > 0 else soft_symbols.copy()
    for polarity in (+1.0, -1.0):
        xp = polarity * x
        score_by_offset = np.full(max_offset, np.nan)
        count_by_offset = np.zeros(max_offset, dtype=int)
        for off in range(max_offset):
            vals = []
            pos = off
            while pos + SYNC_SYMBOLS <= len(xp):
                window = xp[pos : pos + SYNC_SYMBOLS]
                # Normalized dot product with known sync sequence.
                den = np.linalg.norm(window) * np.linalg.norm(SYNC_SIGNS)
                if den > 0:
                    vals.append(float(np.dot(window, SYNC_SIGNS) / den))
                pos += SYMBOLS_PER_PART
            if vals:
                # Favor both high mean score and consistency across repetitions.
                score_by_offset[off] = float(np.mean(vals))
                count_by_offset[off] = len(vals)
        if not np.any(np.isfinite(score_by_offset)):
            raise ValueError("No nonzero finite symbols for synchronization")
        best_off = int(np.nanargmax(score_by_offset))
        results.append(
            {
                "polarity": polarity,
                "scores": score_by_offset,
                "counts": count_by_offset,
                "best_offset": best_off,
                "best_score": float(score_by_offset[best_off]),
            }
        )
    return max(results, key=lambda d: d["best_score"]), results


def decode_all_parts(soft_symbols, sync_result, poly_reverse):
    polarity = sync_result["polarity"]
    start = sync_result["best_offset"]
    x = polarity * soft_symbols
    parts = []
    part_index = 0
    pos = start
    while pos + SYMBOLS_PER_PART <= len(x):
        sync_soft = x[pos : pos + 10]
        coded_soft = x[pos + 10 : pos + 250]
        den = np.linalg.norm(sync_soft) * np.linalg.norm(SYNC_SIGNS)
        sync_corr = float(np.dot(sync_soft, SYNC_SIGNS) / den) if den > 0 else 0.0
        dec = decode_page_part(coded_soft, poly_reverse=poly_reverse)
        parts.append(
            {
                "part_index": part_index,
                "symbol_start": pos,
                "time_start_s": pos * SYMBOL_PERIOD_S,
                "sync_corr": sync_corr,
                **dec,
            }
        )
        pos += SYMBOLS_PER_PART
        part_index += 1
    return parts


def pair_pages(parts):
    """
    Pair adjacent nominal even/odd parts:
        even part (EO=0) followed by odd part (EO=1)
    """
    pages = []
    for i in range(len(parts) - 1):
        a = parts[i]
        b = parts[i + 1]
        if a["even_odd"] == 0 and b["even_odd"] == 1 and a["page_type"] == 0 and b["page_type"] == 0:
            page = nominal_vertical_page(a["bits"], b["bits"])
            if page is not None:
                pages.append(
                    {"even_part_index": i, "odd_part_index": i + 1, "time_start_s": a["time_start_s"], **page}
                )
    return pages


def choose_viterbi_orientation(soft_symbols, sync_result):
    """
    Try both polynomial bit conventions. Prefer:
      1. most CRC-passing full pages
      2. most correctly terminated page parts
      3. lower aggregate Viterbi path metric
    """
    candidates = []
    for reverse in (False, True):
        parts = decode_all_parts(soft_symbols, sync_result, reverse)
        pages = pair_pages(parts)
        crc_pass = sum(int(p["crc_ok"]) for p in pages)
        tails_ok = sum(int(p["tail_weight"] == 0) for p in parts)
        metric_sum = sum(p["path_metric"] for p in parts)
        candidates.append(
            {
                "poly_reverse": reverse,
                "parts": parts,
                "pages": pages,
                "crc_pass": crc_pass,
                "tails_ok": tails_ok,
                "metric_sum": metric_sum,
            }
        )
    candidates.sort(key=lambda c: (c["crc_pass"], c["tails_ok"], -c["metric_sum"]), reverse=True)
    return candidates[0], candidates


from .figures import telemetry_figure as make_plot


def main():
    args = parse_args()
    if not np.isfinite(args.skip_seconds) or args.skip_seconds < 0:
        raise ValueError("--skip-seconds must be finite and nonnegative")
    tracking_dir = args.tracking_dir.expanduser().resolve()
    npz_path = tracking_dir / "tracking_validation.npz"
    metadata_path = tracking_dir / "metadata.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")
    outdir = args.output.expanduser().resolve() if args.output is not None else tracking_dir / "inav_decode"
    outdir.mkdir(parents=True, exist_ok=False)
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    prn = int(meta["acquisition"]["prn"])
    d = np.load(npz_path)
    prompt = np.asarray(d["prompt"], dtype=np.complex128)
    time_arr = np.asarray(d["time"], dtype=np.float64)
    if (
        prompt.ndim != 1
        or time_arr.shape != prompt.shape
        or not np.all(np.isfinite(prompt))
        or not np.all(np.isfinite(time_arr))
    ):
        raise ValueError("Tracking symbols and times must be finite, matching one-dimensional arrays.")
    if np.any(np.diff(time_arr) <= 0):
        raise ValueError("Tracking times must increase strictly.")
    # Insufficient time is an observed outcome, with artifacts the next stage can inspect.
    skip = int(round(args.skip_seconds / SYMBOL_PERIOD_S))
    if len(prompt) - skip < 500:
        from .common import write_json, heading, finish

        reason = "Fewer than 500 coded symbols remain after settling; no complete 2-second page is guaranteed"
        write_json(
            outdir / "decoded_inav.json",
            {
                "satellite_prn": prn,
                "pages": [],
                "page_parts": [],
                "reason": reason,
                "recording": meta.get("recording"),
                "symbols_available": max(0, len(prompt) - skip),
            },
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_arr, prompt.real, label="Prompt I")
        ax.plot(time_arr, prompt.imag, label="Prompt Q", alpha=0.6)
        ax.set(xlabel="Tracking time [s]", ylabel="Prompt correlation [counts]")
        ax.legend()
        ax.grid(True)
        heading(
            fig,
            "03",
            "More symbols are needed",
            f"E{prn:02d} · {max(0,len(prompt)-skip)} symbols after settling",
            reason,
        )
        finish(fig, outdir / "03_telemetry_incomplete")
        print(reason)
        return 3
    prompt = prompt[skip:]
    time_arr = time_arr[skip:]
    # Prompt I is the natural BPSK soft-decision statistic after Costas lock.
    soft = np.real(prompt).astype(np.float64)
    # Remove slow amplitude scale only, never hard slice before FEC.
    med = np.median(np.abs(soft))
    if med > 0:
        soft /= med
    print("=" * 76)
    print("03 / GALILEO E1-B I/NAV TELEMETRY DECODER")
    print("=" * 76)
    print(f"Satellite             E{prn:02d}")
    print(f"Prompt symbols        {len(soft)}")
    print(f"Symbol interval       {SYMBOL_PERIOD_S*1e3:.1f} ms")
    print(f"Available telemetry   {len(soft)*SYMBOL_PERIOD_S:.3f} s")
    print(f"Skipped settling      {args.skip_seconds:.3f} s")
    # ---- synchronization ----
    sync_best, sync_all = sync_offset_search(soft)
    print("\nPAGE SYNCHRONIZATION")
    print("--------------------")
    print(f"Best offset modulo 250 symbols : {sync_best['best_offset']}")
    print(f"Costas polarity                : {sync_best['polarity']:+.0f}")
    print(f"Mean sync correlation          : {sync_best['best_score']:.4f}")
    print("Known sync word                : 0101100000")
    # ---- Viterbi convention selection ----
    orientation, alternatives = choose_viterbi_orientation(soft, sync_best)
    print("\nFEC DECODING")
    print("------------")
    print("Polynomial bit convention      : " + ("reversed" if orientation["poly_reverse"] else "standard"))
    print(f"Decoded page parts             : {len(orientation['parts'])}")
    print(f"Zero-tail parts (forced end state): {orientation['tails_ok']}")
    print(f"Complete nominal pages         : {len(orientation['pages'])}")
    print(f"CRC-passing pages              : {orientation['crc_pass']}")
    # Keep transmit symbol timing and receive sample timing separately.
    # The tracking time axis is relative to the first tracked boundary; it is not GST.
    raw_samples = (
        np.asarray(d["raw_start_sample"], dtype=np.int64)[skip:] if "raw_start_sample" in d else None
    )
    boundaries = np.asarray(d["boundary_sample"], dtype=float)[skip:] if "boundary_sample" in d else None
    for part in orientation["parts"]:
        idx = part["symbol_start"]
        part["time_start_s"] = float(time_arr[idx] - time_arr[0])
        part["tracking_time_s"] = float(time_arr[idx])
        part["raw_start_sample"] = int(raw_samples[idx]) if raw_samples is not None else None
        part["boundary_sample"] = float(boundaries[idx]) if boundaries is not None else None
    for page in orientation["pages"]:
        part = orientation["parts"][page["even_part_index"]]
        for key in ("time_start_s", "tracking_time_s", "raw_start_sample", "boundary_sample", "symbol_start"):
            page[key] = part[key]
    # ---- print page parts ----
    print("\nPAGE PARTS")
    print("----------")
    print(" idx   t[s]    sync    E/O  type  tail errors")
    for p in orientation["parts"]:
        print(
            f" {p['part_index']:3d}  {p['time_start_s']:6.2f}  "
            f"{p['sync_corr']:6.3f}    {p['even_odd']}     "
            f"{p['page_type']}        {p['tail_weight']}"
        )
    print("\nDECODED COMPLETE I/NAV PAGES")
    print("----------------------------")
    if not orientation["pages"]:
        print("No complete nominal even+odd pages were available.")
    else:
        print(" page   t[s]   CRC       word type   meaning")
        for i, p in enumerate(orientation["pages"]):
            status = "PASS" if p["crc_ok"] else "FAIL"
            print(
                f" {i:3d}   {p['time_start_s']:5.1f}   {status:4s}      "
                f"{p['word_type']:2d}        {p['word_name']}"
            )
    valid_pages = [p for p in orientation["pages"] if p["crc_ok"]]
    if valid_pages:
        print("\nSUCCESS: CRC-VALID GALILEO I/NAV WORDS RECOVERED")
        print("----------------------------------------")
        print(f"{len(valid_pages)} complete page(s) passed the Galileo 24-bit CRC.")
        print(
            "This supports consistent page synchronization, deinterleaving, convolutional "
            "FEC decoding and bit polarity."
        )
    else:
        print("\nNO CRC-PASSING COMPLETE PAGE YET")
        print("--------------------------------")
        print(
            "Synchronization/FEC may still be close, but do not trust navigation "
            "fields until at least one complete page passes CRC."
        )
    # Save machine-readable result.
    result_json = {
        "satellite_prn": prn,
        "recording": meta.get("recording"),
        "capture_metadata": meta.get("capture_metadata"),
        "decoder_start_tracking_s": float(time_arr[0]),
        "time_convention": "time_start_s is relative to first retained decoder symbol; raw indices refer to recording start",
        "symbol_period_s": SYMBOL_PERIOD_S,
        "symbols_used": int(len(soft)),
        "sync": {
            "pattern": "0101100000",
            "offset_mod_250": int(sync_best["best_offset"]),
            "polarity": float(sync_best["polarity"]),
            "mean_correlation": float(sync_best["best_score"]),
        },
        "fec": {
            "interleaver_rows": 8,
            "interleaver_columns": 30,
            "convolutional_rate": "1/2",
            "constraint_length": 7,
            "g1_octal": "171",
            "g2_octal": "133",
            "g2_transmitted_inverted": True,
            "polynomial_bit_convention": ("reversed" if orientation["poly_reverse"] else "standard"),
        },
        "page_parts": [
            {
                "index": int(p["part_index"]),
                "time_start_s": float(p["time_start_s"]),
                "tracking_time_s": p["tracking_time_s"],
                "raw_start_sample": p["raw_start_sample"],
                "boundary_sample": p["boundary_sample"],
                "symbol_start": p["symbol_start"],
                "sync_correlation": float(p["sync_corr"]),
                "even_odd": int(p["even_odd"]),
                "page_type": int(p["page_type"]),
                "tail_weight": int(p["tail_weight"]),
            }
            for p in orientation["parts"]
        ],
        "pages": [
            {
                "index": i,
                "time_start_s": float(p["time_start_s"]),
                "tracking_time_s": p["tracking_time_s"],
                "raw_start_sample": p["raw_start_sample"],
                "boundary_sample": p["boundary_sample"],
                "symbol_start": p["symbol_start"],
                "crc_ok": bool(p["crc_ok"]),
                "word_type": int(p["word_type"]),
                "word_name": p["word_name"],
                "word_bits_128": "".join(str(int(b)) for b in p["word"]),
                "ssp_bits": "".join(str(int(b)) for b in p["ssp"]),
                "received_crc_bits": "".join(str(int(b)) for b in p["received_crc"]),
                "calculated_crc_bits": "".join(str(int(b)) for b in p["calculated_crc"]),
                "osnma_bits": "".join(str(int(b)) for b in p["osnma"]),
                "sar_bits": "".join(str(int(b)) for b in p["sar"]),
            }
            for i, p in enumerate(orientation["pages"])
        ],
    }
    (outdir / "decoded_inav.json").write_text(json.dumps(result_json, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        outdir / "telemetry_diagnostics.npz",
        soft_symbols=soft,
        sync_scores=sync_best["scores"],
        tracking_time_s=time_arr,
    )
    make_plot(prompt=prompt, soft=soft, sync=sync_best, result=orientation, prn=prn, output=outdir)
    return 0 if valid_pages else 3


if __name__ == "__main__":
    raise SystemExit(main())
