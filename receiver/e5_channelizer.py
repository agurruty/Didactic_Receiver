"""Extract E5a/E5b sidebands from a 100 MS/s full-band E5 RF recording.
This is independent sideband processing, not a joint AltBOC tracking receiver.
"""

import json
import numpy as np
from scipy.signal import firwin, fftconvolve
from .common import write_json
from .multiband_processing import identity


def split(args):
    root = args.capture / "e5" if (args.capture / "e5").exists() else args.capture
    meta = json.loads((root / "metadata.json").read_text())
    if meta.get("valid") is not True or meta.get("errors") or meta.get("band") != "e5":
        raise ValueError("Need a valid full-band E5 capture.")
    fs = float(meta["sample_rate_sps"])
    if fs != 100e6 or abs(meta["actual_center_frequency_hz"] - 1191.795e6) > 1:
        raise ValueError("Channelizer expects E5 centered at 1191.795 MHz, 100 MS/s.")
    raw = root / "e5.sc16"
    size = raw.stat().st_size
    if size % 4 or size // 4 != meta["captured_samples"]:
        raise ValueError("Full-band capture sample count mismatch.")
    mm = np.memmap(raw, dtype="<i2", mode="r").reshape(-1, 2)
    if len(mm) < 1024:
        raise ValueError("Full-band capture is too short.")
    taps = firwin(257, 10.8e6, fs=fs, window=("kaiser", 8.6))
    half = len(taps) // 2
    args.output.mkdir(parents=True, exist_ok=False)
    parent = identity(raw, fs)
    for band, offset in [("e5a", -15.345e6), ("e5b", 15.345e6)]:
        directory = args.output / band
        directory.mkdir()
        clipped = 0
        count = 0
        child = dict(
            meta,
            valid=False,
            band=band,
            sample_rate_sps=25e6,
            raw_sample_rate_sps=25e6,
            actual_center_frequency_hz=1191.795e6 + offset,
            source_recording=parent,
            channelizer=dict(
                method="frequency translation + symmetric FIR + decimation by four",
                taps=257,
                cutoff_hz=10.8e6,
                window="Kaiser beta 8.6",
                group_delay_compensated=True,
                edge_transient_samples=32,
                output_amplitude_scale=0.5,
            ),
            channels=[],
        )
        write_json(directory / "metadata.json", child)
        try:
            with (directory / f"{band}.sc16").open("xb") as handle:
                for start in range(0, len(mm), 1_000_000):
                    stop = min(start + 1_000_000, len(mm))
                    left = max(0, start - half)
                    right = min(len(mm), stop + half)
                    iq = mm[left:right].astype(np.float64)
                    # Modular phase avoids precision loss over multi-minute captures.
                    phase = np.remainder(
                        np.arange(left, right, dtype=np.float64) * (offset / fs), 1.0
                    )
                    z = (iq[:, 0] + 1j * iq[:, 1]) * np.exp(-2j * np.pi * phase)
                    z = np.pad(z, (max(0, half - start), max(0, stop + half - len(mm))))
                    filtered = fftconvolve(z, taps, mode="valid")[::4] * 0.5
                    values = np.column_stack((filtered.real, filtered.imag))
                    clipped += int(
                        np.any((values < -32768) | (values > 32767), axis=1).sum()
                    )
                    np.clip(np.rint(values), -32768, 32767).astype("<i2").tofile(handle)
                    count += len(filtered)
            child.update(
                valid=clipped == 0,
                captured_samples=count,
                samples_per_channel=count,
                duration_s=count / 25e6,
                channelizer_clipped_samples=clipped,
            )
            if clipped:
                child["errors"] = ["Channelizer output clipped; reduce input level."]
        except Exception as exc:
            child["errors"] = [str(exc)]
            raise
        finally:
            write_json(directory / "metadata.json", child)
        if not child["valid"]:
            raise ValueError(child["errors"][0])
    write_json(
        args.output / "metadata.json",
        dict(meta, derived_sidebands=["e5a", "e5b"], source_recording=parent),
    )
    return 0
