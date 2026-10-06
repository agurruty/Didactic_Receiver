"""Didactic Galileo receiver: explicit stages and an automatic evidence gallery."""

import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from .common import write_json, heading, finish
from .signals import SIGNALS
from . import multiband_processing as processing


def message(args):
    from .words import assemble_navigation
    from .navigation import validate_ephemeris, propagate_ecef

    raw = json.loads(args.decoded.read_text())
    nav = assemble_navigation(raw)
    args.output.mkdir(parents=True, exist_ok=False)
    ephem = nav["latest_complete_ephemeris"]
    checks = validate_ephemeris(ephem, raw["satellite_prn"]) if ephem else []
    valid = bool(ephem and all(c["pass"] for c in checks))
    write_json(
        args.output / "navigation.json",
        dict(
            nav,
            checks=checks,
            status="coherent ephemeris"
            if valid
            else "insufficient valid navigation evidence",
        ),
    )
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(121, projection="3d")
    clock = fig.add_subplot(122)
    if valid:
        times = np.arange(ephem["toe_s"] - 3600, ephem["toe_s"] + 3601, 60.0)
        states = [propagate_ecef(ephem, float(t) % 604800) for t in times]
        ax.plot(*[[s[k] / 1e6 for s in states] for k in ("x_m", "y_m", "z_m")])
        ax.set(
            xlabel="X [Mm]",
            ylabel="Y [Mm]",
            zlabel="Z [Mm]",
            title="Broadcast orbit in Earth-fixed coordinates",
        )
        clock.plot(
            (times - ephem["toc_s"]) / 60, [s["clock_total_s"] * 1e6 for s in states]
        )
        clock.set(
            xlabel="Minutes from clock reference",
            ylabel="Satellite clock [µs]",
            title="Polynomial + relativistic clock correction",
        )
        write_json(args.output / "orbit_samples.json", states)
    else:
        ax.set_axis_off()
        clock.set_axis_off()
        clock.text(
            0,
            0.9,
            "Complete CRC-valid ephemeris unavailable.\n\n"
            + str(nav["ephemeris_word_coverage"]),
            va="top",
        )
    heading(
        fig,
        "04",
        f"{raw.get('band', 'e1').upper()} navigation → satellite state",
        f"E{raw['satellite_prn']:02} · {nav['rejected_crc_pages']} CRC-rejected pages",
        "This is the satellite orbit, not the antenna position. Signal-specific BGD is applied later when forming receiver observations.",
    )
    finish(fig, args.output / "04_navigation")
    return 0 if valid else 3


def inspect(args):
    from .multiband_capture import recording
    from .figures import raw_signal

    raw, meta, _ = recording(args.capture, args.band)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "signal_quality.json", raw_signal(raw, meta, args.output))
    return 0


def add_acquisition(p):
    p.add_argument("--prns", type=int, nargs="+", default=list(range(1, 51)))
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--dwells", type=int, default=20)
    p.add_argument("--doppler-max", type=float, default=7500.0)
    p.add_argument("--doppler-step", type=float, default=250.0)
    p.add_argument(
        "--threshold-db",
        type=float,
        default=10.0,
        help="Heuristic detection threshold, not calibrated false-alarm probability.",
    )


def add_position(p):
    p.add_argument(
        "--mode",
        choices=["auto", "e1", "e5a", "e5b", "e1+e5a", "e1+e5b"],
        default="auto",
    )
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--min-cn0", type=float, default=25.0)
    p.add_argument("--elevation-mask", type=float, default=10.0)
    p.add_argument("--max-pdop", type=float, default=20.0)
    p.add_argument(
        "--code-sigma-e1",
        type=float,
        default=10.0,
        help="Assumed zenith code sigma [m], not measured accuracy.",
    )
    p.add_argument(
        "--code-sigma-e5",
        type=float,
        default=3.0,
        help="Assumed zenith code sigma [m], not measured accuracy.",
    )
    p.add_argument(
        "--ionosphere",
        choices=["ntcm-g", "none"],
        default="ntcm-g",
        help="Single-band correction; dual-frequency mode uses an IF combination.",
    )
    p.add_argument("--gst-era", type=int)
    from .position_map import add_map_arguments

    add_map_arguments(p)


def run(args):
    from .multiband_position import position

    args.output.mkdir(parents=True, exist_ok=False)
    statuses = []
    bands = args.bands.split("+")
    if len(set(bands)) != len(bands) or any(b not in SIGNALS for b in bands):
        raise ValueError("Choose unique processing bands e1, e5a and/or e5b.")
    base = vars(args)
    for band in bands:
        output = args.output / band
        output.mkdir()
        acq = output / "01_acquisition"
        try:
            status = processing.acquire(
                SimpleNamespace(**{**base, "band": band, "output": acq})
            )
            statuses.append(dict(band=band, stage="acquisition", exit_code=status))
            summary = json.loads((acq / "acquisition.json").read_text())
        except (OSError, ValueError, RuntimeError) as exc:
            statuses.append(
                dict(band=band, stage="acquisition", exit_code=1, error=str(exc))
            )
            continue
        for prn in summary["selected_prns"]:
            channel = output / f"E{prn:02}"
            channel.mkdir()
            try:
                processing.track(
                    SimpleNamespace(
                        capture=args.capture,
                        acquisition=acq,
                        prn=prn,
                        output=channel / "02_tracking",
                    )
                )
                code = processing.decode(
                    SimpleNamespace(
                        tracking=channel / "02_tracking",
                        output=channel / "03_telemetry",
                        skip_seconds=0.4,
                    )
                )
                statuses.append(
                    dict(band=band, prn=prn, stage="telemetry", exit_code=code)
                )
                filename = "decoded_fnav.json" if band == "e5a" else "decoded_inav.json"
                message(
                    SimpleNamespace(
                        decoded=channel / "03_telemetry" / filename,
                        output=channel / "04_navigation",
                    )
                )
            except (OSError, ValueError, RuntimeError) as exc:
                statuses.append(
                    dict(
                        band=band, prn=prn, stage="channel", exit_code=1, error=str(exc)
                    )
                )
    status = 3
    if not args.skip_position:
        status = position(
            SimpleNamespace(
                **{**base, "run": args.output, "output": args.output / "05_position"}
            )
        )
    else:
        status = (
            0
            if any(
                s.get("stage") == "telemetry" and s["exit_code"] == 0 for s in statuses
            )
            else 3
        )
    write_json(
        args.output / "run.json", dict(bands=bands, statuses=statuses, exit_code=status)
    )
    gallery = [
        '<!doctype html><meta charset="utf-8"><title>Galileo receiver evidence</title>',
        "<style>body{font:16px system-ui;max-width:1200px;margin:3rem auto;background:#f5f7fa;color:#183247}img{width:100%}section{background:white;padding:1.5rem;margin:2rem 0}pre{white-space:pre-wrap}</style>",
        "<h1>Galileo: from radio samples to position</h1><p>Measured artifacts and explicit missing evidence. Open images for full resolution.</p>",
        "<pre>" + html.escape(json.dumps(statuses, indent=2)) + "</pre>",
    ]
    for image in sorted(args.output.rglob("*.png")):
        relative = image.relative_to(args.output).as_posix()
        safe = html.escape(relative, quote=True)
        gallery.append(
            f'<section><h2>{html.escape(relative)}</h2><a href="{safe}"><img loading="lazy" src="{safe}" alt="{safe}"></a></section>'
        )
    (args.output / "index.html").write_text("\n".join(gallery))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    capture = commands.add_parser(
        "capture", help="Plan or record synchronized RF streams."
    )
    capture.add_argument(
        "--bands",
        default="e5b",
        help="e1, e5a, e5b, e5 (full-band), or combinations such as e1+e5b.",
    )
    capture.add_argument("--duration", type=float, default=120.0)
    capture.add_argument("--sample-rate", type=float)
    capture.add_argument("--subdev", help='One UHD subdevice per band, e.g. "A:0 A:1".')
    capture.add_argument(
        "--antennas", nargs="+", help="One antenna port or one per band; default RX1."
    )
    capture.add_argument("--gains", nargs="+", type=float)
    capture.add_argument(
        "--args",
        default="type=x300,addr=192.168.40.2,master_clock_rate=200e6,recv_frame_size=8000",
    )
    capture.add_argument("--gps-timeout", type=float, default=120.0)
    capture.add_argument(
        "--plan",
        action="store_true",
        help="Print rates/storage/channel plan without accessing hardware.",
    )
    capture.add_argument("--output", type=Path, required=True)
    from .multiband_capture import capture as capture_main

    capture.set_defaults(handler=capture_main)
    for name, handler in [("inspect", inspect), ("acquire", processing.acquire)]:
        sub = commands.add_parser(name)
        sub.add_argument("capture", type=Path)
        sub.add_argument("--band", choices=SIGNALS, default="e5b")
        sub.add_argument("--output", type=Path, required=True)
        if name == "acquire":
            add_acquisition(sub)
        sub.set_defaults(handler=handler)
    sub = commands.add_parser("track")
    sub.add_argument("capture", type=Path)
    sub.add_argument("--acquisition", type=Path, required=True)
    sub.add_argument("--prn", type=int)
    sub.add_argument("--output", type=Path, required=True)
    sub.set_defaults(handler=processing.track)
    sub = commands.add_parser("decode")
    sub.add_argument("tracking", type=Path)
    sub.add_argument("--skip-seconds", type=float, default=0.4)
    sub.add_argument("--output", type=Path, required=True)
    sub.set_defaults(handler=processing.decode)
    sub = commands.add_parser("message")
    sub.add_argument("decoded", type=Path)
    sub.add_argument("--output", type=Path, required=True)
    sub.set_defaults(handler=message)
    from .multiband_position import position

    sub = commands.add_parser("position")
    sub.add_argument("run", type=Path)
    sub.add_argument("--output", type=Path, required=True)
    add_position(sub)
    sub.set_defaults(handler=position)
    from .e5_channelizer import split

    sub = commands.add_parser(
        "split-e5",
        help="Extract E5a/E5b from full-band E5; independent sideband processing.",
    )
    sub.add_argument("capture", type=Path)
    sub.add_argument("--output", type=Path, required=True)
    sub.set_defaults(handler=split)
    sub = commands.add_parser("run")
    sub.add_argument("capture", type=Path)
    sub.add_argument("--bands", default="e5b")
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--skip-position", action="store_true")
    add_acquisition(sub)
    add_position(sub)
    sub.set_defaults(handler=run)
    args = p.parse_args()
    return args.handler(args)
