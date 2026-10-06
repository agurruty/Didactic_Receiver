"""Capture or inspect a recording, without performing a satellite search."""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .common import FS, recording, write_json
from .capture import capture_single_channel, lock_and_set_gps_time, sensor_bool


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="Inspect an existing capture instead of opening UHD.")
    p.add_argument(
        "--output", type=Path, required=True, help="New directory for capture or inspection artifacts."
    )
    p.add_argument(
        "--duration", type=float, default=30, help="Capture seconds; 30 s uses 1.5 GB at 12.5 MS/s."
    )
    p.add_argument(
        "--gain", type=float, default=90, help="TwinRX gain in dB; check clipping in the raw-signal figure."
    )
    p.add_argument("--subdev", default="A:0", help="One TwinRX receiver: A:0, A:1, B:0 or B:1.")
    p.add_argument("--antenna", default="RX1", help="RF port connected to the GNSS antenna feed.")
    p.add_argument(
        "--args",
        default="type=x300,addr=192.168.40.2,master_clock_rate=200e6,recv_frame_size=8000",
        help="UHD device address and transport settings.",
    )
    p.add_argument("--gps-timeout", type=float, default=120, help="Maximum GPSDO lock wait in seconds.")
    a = p.parse_args()
    if not math.isfinite(a.duration) or a.duration <= 0 or a.gps_timeout <= 0 or not math.isfinite(a.gain):
        raise ValueError("Duration and GPS timeout must be positive; gain must be finite.")
    if a.input:
        raw, meta = recording(a.input)
        a.output.mkdir(parents=True, exist_ok=False)
    else:
        if sys.byteorder != "little":
            raise RuntimeError("SC16 capture requires a little-endian host.")
        import uhd

        a.output.mkdir(parents=True, exist_ok=False)
        meta = {
            "valid": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_sample_rate_sps": FS,
            "sample_format": "little-endian signed int16 I,Q pairs",
            "device_args": a.args,
            "rx_subdevice_spec": a.subdev,
        }
        write_json(a.output / "metadata.json", meta)
        try:
            usrp = uhd.usrp.MultiUSRP(a.args)
            usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(a.subdev), 0)
            gps = lock_and_set_gps_time(usrp, uhd, a.gps_timeout)
            usrp.set_rx_rate(FS, 0)
            usrp.set_rx_freq(1_575_420_000.0, 0)
            usrp.set_rx_gain(a.gain, 0)
            usrp.set_rx_antenna(a.antenna, 0)
            rate = float(usrp.get_rx_rate(0))
            if not math.isclose(rate, FS, rel_tol=0, abs_tol=0.001):
                raise ValueError(f"UHD applied unsupported sample rate {rate}.")
            names = set(usrp.get_rx_sensor_names(0))
            lo = sensor_bool(usrp.get_rx_sensor("lo_locked", 0)) if "lo_locked" in names else None
            if lo is False:
                raise RuntimeError("RX local oscillator is not locked.")
            words, first, scheduled, raw = capture_single_channel(usrp, uhd, np, a, a.output)
            meta.update(
                valid=True,
                captured_samples=len(words),
                capture_duration_s=len(words) / FS,
                first_sample_device_time_s=first,
                scheduled_start_device_time_s=scheduled,
                gpsdo=gps,
                lo_locked=lo,
                antenna=str(usrp.get_rx_antenna(0)),
                gain_db=float(usrp.get_rx_gain(0)),
                actual_center_frequency_hz=float(usrp.get_rx_freq(0)),
            )
            del words
        except Exception as exc:
            meta["errors"] = [str(exc)]
            write_json(a.output / "metadata.json", meta)
            raise
        write_json(a.output / "metadata.json", meta)
    from .figures import raw_signal

    stats = raw_signal(raw, meta, a.output)
    write_json(a.output / "signal_quality.json", stats)
    return 0
