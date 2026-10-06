"""Search E1-C once and persist the exact hypotheses used by tracking."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .common import FS, recording, fingerprint, write_json
from .tracking import load_codes, open_sc16, first_nominal_blocks, acquire_prn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture_dir", type=Path)
    p.add_argument(
        "--output", type=Path, required=True, help="New directory for ranking, maps and tracking seeds."
    )
    p.add_argument(
        "--prns", type=int, nargs="+", default=list(range(1, 51)), help="PRNs to search; default 1–50."
    )
    p.add_argument(
        "--top", type=int, default=4, help="Number of above-threshold candidates to refine and save."
    )
    p.add_argument("--dwells", type=int, default=20, help="Noncoherent sum of this many 4 ms code periods.")
    p.add_argument("--doppler-max", type=float, default=7500, help="Symmetric search limit in Hz.")
    p.add_argument("--doppler-step", type=float, default=250, help="Coarse frequency spacing in Hz.")
    p.add_argument("--fine-step", type=float, default=25, help="Refinement frequency spacing in Hz.")
    p.add_argument(
        "--threshold-db",
        type=float,
        default=7,
        help="Heuristic peak/median-floor gate in dB, not C/N0 or a calibrated false-alarm probability.",
    )
    p.add_argument("--gnss-tools", type=Path)
    a = p.parse_args()
    if a.top < 1 or a.dwells < 1 or any(v < 1 or v > 50 for v in a.prns):
        raise ValueError("PRNs must be 1–50; top and dwells must be positive.")
    if not all(
        np.isfinite(v) and v > 0 for v in (a.doppler_max, a.doppler_step, a.fine_step)
    ) or not np.isfinite(a.threshold_db):
        raise ValueError("Search steps and limit must be finite and positive; threshold must be finite.")
    raw, meta = recording(a.capture_dir)
    _, e1c, _ = load_codes(a)
    blocks = first_nominal_blocks(open_sc16(raw), a.dwells)
    a.output.mkdir(parents=True, exist_ok=False)
    dopplers = np.arange(-a.doppler_max, a.doppler_max + 0.5 * a.doppler_step, a.doppler_step)
    rows, maps = [], []
    for prn in sorted(set(a.prns)):
        metric, di, ci, power = acquire_prn(blocks, prn, e1c, dopplers)
        row = dict(
            prn=prn,
            metric_db=metric,
            doppler_hz=float(dopplers[di]),
            code_index=ci,
            detected=bool(metric >= a.threshold_db),
        )
        rows.append(row)
        maps.append((metric, row, power))
        maps.sort(key=lambda item: item[0], reverse=True)
        maps = maps[: a.top]
        print(f"E{prn:02d}: {metric:5.2f} dB peak/floor, {dopplers[di]:+.0f} Hz", flush=True)
    rows.sort(key=lambda row: row["metric_db"], reverse=True)
    selected = []
    from .figures import acquisition_figure

    # Plot even a subthreshold strongest candidate, without marking it as a detection.
    for rank, (_, row, coarse_map) in enumerate(maps):
        if not row["detected"] and rank > 0:
            continue
        fine = np.arange(
            row["doppler_hz"] - a.doppler_step,
            row["doppler_hz"] + a.doppler_step + 0.5 * a.fine_step,
            a.fine_step,
        )
        metric, di, ci, power = acquire_prn(blocks, row["prn"], e1c, fine)
        acq = dict(
            prn=row["prn"],
            metric_db=metric,
            doppler_hz=float(fine[di]),
            code_index=ci,
            fine_map=power,
            fine_dopplers=fine,
            coarse_map=coarse_map,
            coarse_dopplers=dopplers,
        )
        if row["detected"]:
            selected.append(row["prn"])
            np.savez_compressed(a.output / f"E{row['prn']:02d}_seed.npz", **acq)
        else:
            np.savez_compressed(a.output / f"E{row['prn']:02d}_below_gate.npz", **acq)
        acquisition_figure(rows, acq, a, a.output)
    summary = dict(
        schema_version=1,
        component="Galileo E1-C",
        replica="sinBOC(1,1)",
        recording=fingerprint(raw),
        analysis_start_sample=0,
        coherent_ms=4,
        noncoherent_dwells=a.dwells,
        threshold_db=a.threshold_db,
        metric="peak / guarded median acquisition-map floor [dB]; not C/N0",
        ranked_candidates=rows,
        selected_prns=selected,
    )
    write_json(a.output / "acquisition.json", summary)
    return 0 if selected else 3


def load_seed(directory, raw, prn=None):
    import json

    directory = Path(directory)
    summary = json.loads((directory / "acquisition.json").read_text())
    if summary["recording"] != fingerprint(raw):
        raise ValueError(
            "Acquisition seed belongs to a different or modified recording. Run acquisition again."
        )
    selected = summary["selected_prns"]
    if not selected:
        raise ValueError("No candidates passed the acquisition gate.")
    prn = selected[0] if prn is None else prn
    if prn not in selected:
        raise ValueError(
            f"E{prn:02d} has no saved above-threshold seed; rerun acquisition with --top or --prns."
        )
    with np.load(directory / f"E{prn:02d}_seed.npz", allow_pickle=False) as data:
        return {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data.files}
