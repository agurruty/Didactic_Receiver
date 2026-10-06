"""UHD capture and SC16 helpers from the live-tested diagnostic."""

import math
import sys
import time
from pathlib import Path

RAW_RATE = 12_500_000.0


def sensor_bool(sensor) -> bool:
    try:
        return bool(sensor.to_bool())
    except (AttributeError, RuntimeError):
        return str(sensor).strip().lower() in {"true", "locked", "yes", "1"}


def wait_for_lock(usrp, name: str, timeout: float) -> None:
    names = set(usrp.get_mboard_sensor_names(0))
    if name not in names:
        raise RuntimeError(f"Motherboard does not expose required sensor {name!r}")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sensor_bool(usrp.get_mboard_sensor(name, 0)):
            return
        print(f"  waiting for {name} ...")
        time.sleep(1.0)

    raise RuntimeError(f"Timed out after {timeout:g} s waiting for {name}")


def lock_and_set_gps_time(usrp, uhd, timeout: float) -> dict:
    usrp.set_clock_source("gpsdo", 0)
    usrp.set_time_source("gpsdo", 0)

    wait_for_lock(usrp, "ref_locked", timeout)
    wait_for_lock(usrp, "gps_locked", timeout)

    gps_time = int(usrp.get_mboard_sensor("gps_time", 0).to_int())
    usrp.set_time_next_pps(uhd.types.TimeSpec(gps_time + 1), 0)
    time.sleep(2.0)

    gps_after = int(usrp.get_mboard_sensor("gps_time", 0).to_int())
    device_last_pps = int(round(usrp.get_time_last_pps(0).get_real_secs()))

    if abs(device_last_pps - gps_after) > 1:
        raise RuntimeError(
            "GPSDO locked but X300 device time did not align to GPS time "
            f"(device={device_last_pps}, GPS={gps_after})"
        )

    return {
        "clock_source": usrp.get_clock_source(0),
        "time_source": usrp.get_time_source(0),
        "ref_locked": True,
        "gps_locked": True,
        "gps_time_after_sync": gps_after,
        "device_time_last_pps": device_last_pps,
    }


def capture_single_channel(usrp, uhd, np, args, output_dir: Path):
    target_samples = int(round(args.duration * RAW_RATE))
    stream_args = uhd.usrp.StreamArgs("sc16", "sc16")
    stream_args.channels = [0]
    streamer = usrp.get_rx_stream(stream_args)

    max_samples = streamer.get_max_num_samps()
    rx_buffer = np.empty((1, max_samples), dtype=np.uint32)
    raw_path = output_dir / "e1.sc16"
    raw_file = raw_path.open("xb")
    md = uhd.types.RXMetadata()

    start_time = math.floor(usrp.get_time_now().get_real_secs()) + 2.0
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = False
    cmd.time_spec = uhd.types.TimeSpec(start_time)

    print(f"\nCapturing {args.duration:.3f} s = {target_samples:,} samples ...")

    received = 0
    first_time = None
    try:
        streamer.issue_stream_cmd(cmd)
        while received < target_samples:
            timeout = 3.5 if received == 0 else 1.0
            count = streamer.recv(rx_buffer, md, timeout)

            if md.error_code != uhd.types.RXMetadataErrorCode.none:
                raise RuntimeError(f"UHD receive error: {md.strerror()}")
            if count <= 0:
                raise RuntimeError("UHD returned zero samples")

            if first_time is None:
                first_time = md.time_spec.get_real_secs()

            keep = min(count, target_samples - received)
            rx_buffer[0, :keep].tofile(raw_file)
            received += keep
    finally:
        stop = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        try:
            streamer.issue_stream_cmd(stop)
        finally:
            raw_file.close()

    raw_path = output_dir / "e1.sc16"
    raw_words = np.memmap(raw_path, dtype=np.uint32, mode="r")

    return raw_words, first_time, start_time, raw_path


def sc16_words_to_complex(raw_words, np):
    if sys.byteorder != "little":
        raise RuntimeError("This script expects a little-endian Linux host")
    iq = raw_words.view("<i2").reshape(-1, 2)
    return iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)


def adc_statistics(x, np) -> dict:
    i = np.real(x)
    q = np.imag(x)
    clip = (i <= -32768) | (i >= 32767) | (q <= -32768) | (q >= 32767)
    return {
        "rms_counts": float(np.sqrt(np.mean(np.abs(x) ** 2))),
        "i_mean_counts": float(np.mean(i)),
        "q_mean_counts": float(np.mean(q)),
        "i_std_counts": float(np.std(i)),
        "q_std_counts": float(np.std(q)),
        "clip_fraction": float(np.mean(clip)),
    }
