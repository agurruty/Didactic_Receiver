"""Timed X300/TwinRX capture with one clock, one streamer and explicit RF channels."""

import json
import math
import shutil
import sys
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .capture import lock_and_set_gps_time, sensor_bool
from .common import write_json
from .signals import SIGNALS


def capture_plan(bands, duration, rate=None, subdev=None, antennas=None, gains=None):
    names = bands.split("+")
    if (
        len(set(names)) != len(names)
        or not names
        or any(n not in (*SIGNALS, "e5") for n in names)
    ):
        raise ValueError(
            "Choose unique bands from e1, e5a, e5b, e5 (full-band RF capture)."
        )
    if "e5" in names and any(n in names for n in ("e5a", "e5b")):
        raise ValueError("Full-band e5 already contains E5a and E5b.")
    if len(names) > 4 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Use 1–4 channels and a positive finite duration.")
    if rate is None:
        rate = 100e6 if "e5" in names else max(SIGNALS[n].sample_rate for n in names)
    if (
        not math.isfinite(rate)
        or rate <= 0
        or not math.isclose(200e6 / rate, round(200e6 / rate), abs_tol=1e-8)
    ):
        raise ValueError(
            "TwinRX rate must be a positive integer decimation of its 200 MHz clock."
        )
    if rate > 100e6:
        raise ValueError("This workflow supports up to 100 MS/s per channel.")
    for name in names:
        if name == "e5":
            if rate < 80e6:
                raise ValueError("Full-band E5 capture requires 100 MS/s on TwinRX.")
        else:
            SIGNALS[name].validate_rate(rate)
    subdev = subdev or " ".join(("A:0", "A:1", "B:0", "B:1")[: len(names)])
    if len(subdev.split()) != len(names) or len(set(subdev.split())) != len(names):
        raise ValueError("Provide one distinct --subdev entry per band, in band order.")

    def expand(values, default):
        values = values or [default]
        if len(values) == 1:
            return values * len(names)
        if len(values) != len(names):
            raise ValueError("Provide one antenna/gain or one per band.")
        return values

    antennas, gains = expand(antennas, "RX1"), expand(gains, 90.0)
    if any(not math.isfinite(g) for g in gains):
        raise ValueError("Gains must be finite.")
    channels = [
        dict(
            band=n,
            channel=i,
            subdevice=subdev.split()[i],
            antenna=antennas[i],
            gain_db=gains[i],
            frequency_hz=1191.795e6 if n == "e5" else SIGNALS[n].frequency_hz,
            file=f"{n}/{n}.sc16",
        )
        for i, n in enumerate(names)
    ]
    if round(duration * rate) < 1:
        raise ValueError("Capture duration must include at least one sample.")
    throughput = rate * 4 * len(names)
    if throughput > 900e6:
        raise ValueError(
            "Requested SC16 payload exceeds the conservative single-10GbE budget (900 MB/s)."
        )
    return dict(
        schema_version=2,
        bands=names,
        sample_rate_sps=rate,
        duration_s=duration,
        samples_per_channel=round(duration * rate),
        subdev=subdev,
        channels=channels,
        payload_bytes_per_second=throughput,
        expected_bytes=round(duration * rate) * 4 * len(names),
        note="Independent RF LOs; common ADC clock and timed stream. RF path delays are not calibrated.",
    )


def capture(args):
    plan = capture_plan(
        args.bands,
        args.duration,
        args.sample_rate,
        args.subdev,
        args.antennas,
        args.gains,
    )
    print(json.dumps(plan, indent=2))
    if args.plan:
        return 0
    if sys.byteorder != "little":
        raise RuntimeError("SC16 capture requires a little-endian host.")
    if not math.isfinite(args.gps_timeout) or args.gps_timeout <= 0:
        raise ValueError("GPS timeout must be positive.")
    import uhd

    output = args.output
    # Check storage before touching hardware; never overwrite a previous capture.
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < plan["expected_bytes"] * 1.05:
        raise ValueError("Insufficient disk space for this capture plus 5% reserve.")
    meta = {
        **plan,
        "valid": False,
        "session_id": str(uuid.uuid4()),
        "device_args": args.args,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_format": "sc16_le",
        "synchronization": "single timed multi-channel UHD streamer",
    }
    write_json(output / "metadata.json", meta)
    received = 0
    try:
        usrp = uhd.usrp.MultiUSRP(args.args)
        usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(plan["subdev"]), 0)
        meta["gpsdo"] = lock_and_set_gps_time(usrp, uhd, args.gps_timeout)
        rate = plan["sample_rate_sps"]
        for c in plan["channels"]:
            ch = c["channel"]
            usrp.set_rx_rate(rate, ch)
            usrp.set_rx_antenna(c["antenna"], ch)
            usrp.set_rx_freq(c["frequency_hz"], ch)
            usrp.set_rx_gain(c["gain_db"], ch)
        # Read back after ALL channels are configured (shared front-end settings).
        for c in plan["channels"]:
            ch = c["channel"]
            c.update(
                actual_sample_rate_sps=float(usrp.get_rx_rate(ch)),
                actual_center_frequency_hz=float(usrp.get_rx_freq(ch)),
                actual_gain_db=float(usrp.get_rx_gain(ch)),
                actual_antenna=str(usrp.get_rx_antenna(ch)),
            )
            if (
                abs(c["actual_sample_rate_sps"] - rate) > 0.001
                or abs(c["actual_center_frequency_hz"] - c["frequency_hz"]) > 1
            ):
                raise RuntimeError(
                    "Applied RF frequency or sample rate differs from the capture plan."
                )
            names = usrp.get_rx_sensor_names(ch)
            if "lo_locked" in names:
                deadline = time.monotonic() + 5
                while not sensor_bool(usrp.get_rx_sensor("lo_locked", ch)):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"Channel {ch} LO failed to lock.")
                    time.sleep(0.05)
        stream_args = uhd.usrp.StreamArgs("sc16", "sc16")
        stream_args.channels = list(range(len(plan["channels"])))
        streamer = usrp.get_rx_stream(stream_args)
        buffer = np.empty(
            (len(plan["channels"]), streamer.get_max_num_samps()), dtype=np.uint32
        )
        md = uhd.types.RXMetadata()
        start = math.floor(usrp.get_time_now().get_real_secs()) + 2.0
        meta["scheduled_start_device_time_s"] = start
        command = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        command.stream_now = False
        command.time_spec = uhd.types.TimeSpec(start)
        first = None
        with ExitStack() as stack:
            handles = []
            for c in plan["channels"]:
                path = output / c["file"]
                path.parent.mkdir()
                handles.append(stack.enter_context(path.open("xb")))
            try:
                streamer.issue_stream_cmd(command)
                while received < plan["samples_per_channel"]:
                    count = streamer.recv(buffer, md, 3.5 if received == 0 else 1.0)
                    if (
                        md.error_code != uhd.types.RXMetadataErrorCode.none
                        or count <= 0
                    ):
                        raise RuntimeError(
                            f"UHD stream discontinuity: {md.strerror()} ({count} samples)"
                        )
                    if not md.has_time_spec:
                        raise RuntimeError("UHD packet lacks the required timestamp.")
                    if first is None:
                        first = md.time_spec
                        if (
                            abs((first - uhd.types.TimeSpec(start)).get_real_secs())
                            > 1 / rate
                        ):
                            raise RuntimeError(
                                "First sample does not match the scheduled stream start."
                            )
                        meta["first_sample_device_time_s"] = first.get_real_secs()
                    elapsed = (md.time_spec - first).get_real_secs()
                    if abs(elapsed - received / rate) > 1.5 / rate:
                        raise RuntimeError(
                            "Receive timestamps reveal missing or misaligned samples."
                        )
                    keep = min(count, plan["samples_per_channel"] - received)
                    for i, handle in enumerate(handles):
                        buffer[i, :keep].tofile(handle)
                    received += keep
            finally:
                streamer.issue_stream_cmd(
                    uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                )
        meta.update(valid=True, captured_samples=received)
    except Exception as exc:
        meta.update(errors=[str(exc)], captured_samples=received)
        raise
    finally:
        write_json(output / "metadata.json", meta)
        for c in plan["channels"]:
            directory = output / c["band"]
            if directory.exists():
                write_json(
                    directory / "metadata.json",
                    dict(
                        meta,
                        band=c["band"],
                        channels=[c],
                        actual_center_frequency_hz=c.get(
                            "actual_center_frequency_hz", c["frequency_hz"]
                        ),
                        raw_sample_rate_sps=plan["sample_rate_sps"],
                        session_start_offset_s=0.0,
                    ),
                )
    return 0


def recording(directory, band):
    directory = Path(directory).expanduser().resolve()
    if (directory / band / "metadata.json").exists():
        directory = directory / band
    meta = json.loads((directory / "metadata.json").read_text())
    rate = float(meta.get("sample_rate_sps", meta.get("raw_sample_rate_sps", 0)))
    SIGNALS[band].validate_rate(rate)
    if (
        meta.get("band", band) != band
        or meta.get("valid") is False
        or meta.get("errors")
    ):
        raise ValueError("Wrong band or capture metadata reports a stream error.")
    frequency = meta.get("actual_center_frequency_hz")
    if frequency is None and band != "e1":
        raise ValueError("E5 metadata must declare actual_center_frequency_hz.")
    if frequency is not None and (
        not math.isfinite(float(frequency))
        or abs(float(frequency) - SIGNALS[band].frequency_hz) > 1
    ):
        raise ValueError("RF frequency disagrees with the requested band.")
    candidates = [directory / f"{band}.sc16"]
    if band == "e1":
        candidates.append(directory / "galileo_e1.sc16")
    files = [p for p in candidates if p.exists()]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one {band}.sc16 recording.")
    raw = files[0]
    size = raw.stat().st_size
    if not size or size % 4:
        raise ValueError("SC16 must contain complete int16 I/Q pairs.")
    count = meta.get("captured_samples", meta.get("samples_per_channel"))
    if count is not None and int(count) * 4 != size:
        raise ValueError("Metadata sample count disagrees with file size.")
    return raw, meta, rate
