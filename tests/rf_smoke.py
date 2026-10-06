import json, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from receiver.signals import SIGNALS
from receiver.multiband_processing import acquire, track, decode
from receiver.common import write_json


def main():
    root = Path(tempfile.mkdtemp(prefix="galileo-e5-rf-"))
    print("Synthetic RF test artifacts:", root, flush=True)
    for name in ("e5b", "e5a"):
        s = SIGNALS[name]
        data, pilot = s.codes()
        fs = 25e6
        fd = 862.5
        lag = 1743.25
        rate = s.chip_rate * (1 + fd / s.frequency_hz)
        count = round(fs * 1.6)
        directory = root / name
        directory.mkdir()
        rng = np.random.default_rng(71)
        secondary = s.pilot_secondary(pilot, 1)
        with (directory / f"{name}.sc16").open("wb") as handle:
            for start in range(0, count, 250000):
                n = min(250000, count - start)
                sample = np.arange(start, start + n)
                phase = (sample - lag) * rate / fs
                epoch = np.floor(phase / s.code_length).astype(int)
                init = float(phase[0])
                whole = int(np.floor(init))
                frac = init - whole
                q = pilot.code(1, whole, frac, rate / fs, n) * secondary[epoch % 100]
                i = (
                    data.code(1, whole, frac, rate / fs, n)
                    * data.secondary_code[epoch % s.symbol_epochs]
                )
                carrier = np.exp(2j * np.pi * fd * sample / fs)
                z = 20 * (i + 1j * q) * carrier + 20 * (
                    rng.normal(size=n) + 1j * rng.normal(size=n)
                )
                np.rint(np.column_stack((z.real, z.imag))).astype("<i2").tofile(handle)
        write_json(
            directory / "metadata.json",
            dict(
                band=name,
                valid=True,
                sample_rate_sps=fs,
                captured_samples=count,
                actual_center_frequency_hz=s.frequency_hz,
            ),
        )
        a = SimpleNamespace(
            capture=directory,
            band=name,
            dwells=8,
            top=1,
            prns=[1, 2],
            doppler_max=1000.0,
            doppler_step=250.0,
            threshold_db=14.0,
            output=directory / "acq",
        )
        assert acquire(a) == 0
        seed = json.loads((a.output / "E01_seed.json").read_text())
        assert abs(seed["doppler_hz"] - fd) <= 12.5, seed
        assert abs(seed["code_index"] - lag) < 3, seed
        track(
            SimpleNamespace(
                capture=directory,
                acquisition=a.output,
                prn=1,
                output=directory / "track",
            )
        )
        meta = json.loads((directory / "track/metadata.json").read_text())
        with np.load(directory / "track/continuous_tracking.npz") as result:
            fd_final = float(np.median(result["doppler_hz"][-200:]))
            lock = float(np.nanmedian(result["carrier_lock"][-200:]))
            print(
                name,
                "final Doppler",
                fd_final,
                "lock",
                lock,
                "secondary",
                meta["secondary_sync"]["secondary_shift"],
                flush=True,
            )
            assert abs(fd_final - fd) < 5
            assert lock > 0.9
            # The true received code boundary obeys a known analytic Doppler-scaled lattice.
            phase = (result["boundary_sample"][-200:] - lag) * rate / fs
            errors = (phase + s.code_length / 2) % s.code_length - s.code_length / 2
            print("median boundary error chips", np.median(errors), flush=True)
            assert abs(float(np.median(errors))) < 0.15
        assert (
            decode(
                SimpleNamespace(
                    tracking=directory / "track",
                    output=directory / "decode",
                    skip_seconds=0.4,
                )
            )
            == 3
        )
    print("PASS independent synthesized E5a/E5b RF acquisition and tracking")


if __name__ == "__main__":
    main()
