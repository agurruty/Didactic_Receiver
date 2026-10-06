"""Run milestones sequentially, preserving failures and an offline figure gallery."""

import argparse
import html
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from .common import ROOT, recording, write_json


def gallery(out, manifest):
    panels = []
    for path in sorted(out.rglob("*.png"), key=lambda p: (p.relative_to(out).parts[0] == "05_position", p.name == "05_position_map.png", str(p))):
        rel = path.relative_to(out).as_posix()
        panels.append(
            f"<section><h2>{html.escape(str(path.parent.relative_to(out)))}</h2>"
            f'<a href="{html.escape(rel)}"><img loading="lazy" src="{html.escape(rel)}"></a>'
            f'<p><a href="{html.escape(path.with_suffix(".pdf").relative_to(out).as_posix())}">Download PDF</a></p></section>'
        )
    channels = []
    for meta_path in sorted(out.glob("E*/02_tracking/metadata.json")):
        meta = json.loads(meta_path.read_text())
        channel = meta_path.parent.parent
        decode_path = channel / "03_telemetry/decoded_inav.json"
        nav_path = channel / "04_navigation/receiver_navigation_status.json"
        decoded = json.loads(decode_path.read_text()) if decode_path.exists() else {}
        nav = json.loads(nav_path.read_text()) if nav_path.exists() else {}
        cn0 = meta["final"].get("cn0_dbhz")
        lock = meta["final"].get("carrier_lock")
        channels.append(
            {
                "prn": meta["acquisition"]["prn"],
                "final_cn0_dbhz": cn0,
                "final_phase_concentration": lock,
                "crc_passing_pages": sum(p["crc_ok"] for p in decoded.get("pages", [])),
                "orbit_available": bool(nav.get("readiness", {}).get("orbit_propagation")),
            }
        )
    write_json(out / "channel_summary.json", channels)
    channel_rows = ""
    for c in channels:
        cn0 = f"{c['final_cn0_dbhz']:.2f}" if c["final_cn0_dbhz"] is not None else "Unavailable"
        lock = (
            f"{c['final_phase_concentration']:.3f}"
            if c["final_phase_concentration"] is not None
            else "Unavailable"
        )
        channel_rows += f"<tr><td>E{c['prn']:02d}</td><td>{cn0}</td><td>{lock}</td><td>{c['crc_passing_pages']}</td><td>{'Available' if c['orbit_available'] else 'Incomplete'}</td></tr>"
    stages = "".join(
        f'<tr><td>{html.escape(s["name"])}</td><td>{ {0:"Completed",3:"Insufficient evidence"}.get(s["exit_code"],"Error") }</td>'
        f'<td><a href="{html.escape(s["log"])}">Log</a></td></tr>'
        for s in manifest["stages"]
    )
    position_path = out / "05_position/position_solution.json"
    position = json.loads(position_path.read_text()) if position_path.exists() else None
    position_text = "Pseudorange/position stage has not run yet."
    if position:
        position_text = f"Position milestone: {position['accepted_epochs']}/{position['attempted_epochs']} accepted epochs. Ionosphere: {position['corrections']['ionosphere']}. Inspect residuals, geometry and uncertainty below."
    (out / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Galileo · A signal becomes knowledge</title><style>
body{font:16px system-ui,sans-serif;color:#183247;background:#f5f7fa;max-width:1250px;margin:auto;padding:40px 24px}
h1{font-size:40px;letter-spacing:-1px}p{line-height:1.6;max-width:85ch}section{margin:36px 0}img{width:100%;border-radius:12px}
a{color:#087f8c}table{border-collapse:collapse;width:100%;background:white}td{padding:12px;border-bottom:1px solid #e5eaf0}
</style><h1>Galileo / A signal becomes knowledge</h1><p>Recorded I/Q → code and Doppler search → tracking loops → CRC-checked navigation → satellite orbit and clock → pseudoranges → receiver position.</p>
<p>These figures show the available evidence, including failed gates. Satellite position is computed from the decoded broadcast model. Position estimates require consistent timing, usable geometry and the stated correction models.</p>
<table>"""
        + "<caption>" + html.escape(position_text) + "</caption>"
        + "<tr><th>Stage</th><th>Outcome</th><th>Details</th></tr>"
        + stages
        + "</table><h2>Tracked channels</h2><p>Final-window pilot estimates and recovered navigation evidence. Acquisition ranking uses a different metric.</p><table><tr><th>PRN</th><th>C/N₀ [dB-Hz]</th><th>Phase concentration</th><th>CRC-valid pages</th><th>Orbit</th></tr>"
        + channel_rows
        + "</table>"
        + "".join(panels)
        + "</html>",
        encoding="utf8",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "capture_dir", type=Path, help="Existing SC16 + metadata capture; never starts hardware implicitly."
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New run directory, containing plots, data, logs and index.html.",
    )
    p.add_argument(
        "--top", type=int, default=8, help="Process up to this many above-gate candidates; position uses all usable channels."
    )
    p.add_argument("--prns", type=int, nargs="+", help="Restrict the PRN search; default 1–50.")
    p.add_argument("--dwells", type=int, default=20, help="Number of 4 ms noncoherent acquisition dwells.")
    p.add_argument(
        "--threshold-db", type=float, default=7, help="Heuristic acquisition gate in peak/floor dB."
    )
    p.add_argument(
        "--skip-seconds", type=float, default=0.4, help="Initial tracking settling discarded by the decoder."
    )
    p.add_argument("--skip-position", action="store_true", help="Stop after per-satellite navigation interpretation.")
    p.add_argument("--position-interval", type=float, default=1., help="Spacing of common receive epochs [s].")
    p.add_argument("--calibration", type=Path, help="Sourced GST timestamp / hardware delay JSON for stage 05.")
    iono = p.add_mutually_exclusive_group()
    iono.add_argument("--ionosphere", choices=("ntcm-g","none"), help="Default: broadcast NTCM-G ionosphere correction.")
    p.add_argument("--gst-era",type=int,help="Explicit GST week era if capture calendar metadata is unavailable.")
    p.add_argument("--iono-model-sigma",type=float,default=10.,help="Assumed residual NTCM-G zenith uncertainty [m].")
    iono.add_argument("--ionosphere-csv", type=Path, help="E1 slant ionosphere corrections for stage 05.")
    iono.add_argument("--vtec", type=float, help="External VTEC [TECU]; used with a thin-shell approximation.")
    p.add_argument("--vtec-sigma", type=float, default=5., help="External VTEC uncertainty [TECU].")
    p.add_argument("--ionosphere-source", help="Provenance of supplied ionosphere data.")
    from .position_map import add_map_arguments, validate_map_arguments
    add_map_arguments(p)
    a = p.parse_args()
    validate_map_arguments(a)
    if (a.ionosphere_csv or a.vtec is not None) and not a.ionosphere_source:
        raise ValueError("External ionosphere inputs require --ionosphere-source")
    if not a.skip_position:
        from .positioning import read_calibration
        from .observables import SlantIonosphere
        read_calibration(a.calibration)
        if a.gst_era is not None and not 0<=a.gst_era<=10: raise ValueError("--gst-era must be 0..10")
        if not math.isfinite(a.iono_model_sigma) or a.iono_model_sigma<=0: raise ValueError("--iono-model-sigma must be positive")
        if a.ionosphere_csv: SlantIonosphere(a.ionosphere_csv)
        if not math.isfinite(a.position_interval) or a.position_interval<=0:
            raise ValueError('--position-interval must be finite and positive')
        if a.vtec is not None and (not math.isfinite(a.vtec) or a.vtec<0):
            raise ValueError('--vtec must be finite and nonnegative')
        if not math.isfinite(a.vtec_sigma) or a.vtec_sigma<=0:
            raise ValueError('--vtec-sigma must be finite and positive')
    recording(a.capture_dir)
    if a.top < 1:
        raise ValueError("--top must be positive")
    out = a.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture": str(a.capture_dir.resolve()),
        "stages": [],
    }
    write_json(out / "run.json", manifest)

    def run(name, script, *args):
        log = Path("logs") / (name + ".log")
        command = [sys.executable, str(ROOT / script), *[str(v) for v in args], "--no-show"]
        print(f"\n{name}", flush=True)
        with (out / log).open("w") as f:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    f.write(line)
                    f.flush()
                code = proc.wait()
            except BaseException:
                proc.terminate()
                proc.wait()
                raise
        manifest["stages"].append(dict(name=name, command=command, exit_code=code, log=str(log)))
        write_json(out / "run.json", manifest)
        gallery(out, manifest)
        return code

    status = run(
        "00_signal", "00_galileo_e1_capture.py", "--input", a.capture_dir, "--output", out / "00_signal"
    )
    if status:
        return status
    search = ["--prns", *a.prns] if a.prns else []
    status = run(
        "01_acquisition",
        "01_galileo_e1_acquire.py",
        a.capture_dir,
        "--output",
        out / "01_acquisition",
        "--top",
        a.top,
        "--dwells",
        a.dwells,
        "--threshold-db",
        a.threshold_db,
        *search,
    )
    if status:
        return status
    prns = json.loads((out / "01_acquisition/acquisition.json").read_text())["selected_prns"]
    final_status = 0
    for prn in prns:
        channel = out / f"E{prn:02d}"
        track = channel / "02_tracking"
        decode = channel / "03_telemetry"
        nav = channel / "04_navigation"
        code = run(
            f"E{prn:02d}_02_tracking",
            "02_galileo_e1_track.py",
            a.capture_dir,
            "--acquisition",
            out / "01_acquisition",
            "--prn",
            prn,
            "--output",
            track,
        )
        if code:
            final_status = 1 if code != 3 or final_status == 1 else 3
            continue
        code = run(
            f"E{prn:02d}_03_telemetry",
            "03_galileo_e1_decode.py",
            track,
            "--output",
            decode,
            "--skip-seconds",
            a.skip_seconds,
        )
        if code not in (0, 3):
            final_status = 1
            continue
        nav_code = run(
            f"E{prn:02d}_04_navigation",
            "04_galileo_e1_message.py",
            decode / "decoded_inav.json",
            "--output",
            nav,
        )
        if nav_code or code:
            final_status = 1 if nav_code not in (0, 3) or final_status == 1 else 3
    if not a.skip_position:
        options = ["--interval", a.position_interval, "--map-source", a.map_source, "--map-width", a.map_width]
        if a.map_cache is not None: options.extend(["--map-cache", a.map_cache])
        if a.map_reference is not None:
            options.extend(["--map-reference", *a.map_reference, "--map-reference-source", a.map_reference_source])
        for option, value in (("--calibration", a.calibration), ("--ionosphere-csv", a.ionosphere_csv),
                              ("--vtec", a.vtec), ("--vtec-sigma", a.vtec_sigma),
                              ("--ionosphere-source", a.ionosphere_source), ("--ionosphere", a.ionosphere),
                              ("--gst-era", a.gst_era), ("--iono-model-sigma", a.iono_model_sigma)):
            if value is not None:
                options.extend([option, value])
        code = run("05_position", "05_galileo_e1_position.py", out, "--output", out / "05_position", *options)
        if code:
            final_status = 1 if code != 3 or final_status == 1 else 3
    print(f'\nOpen {out / "index.html"}', flush=True)
    return final_status
