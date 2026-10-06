"""Physical invariants and independently encoded navigation test vectors; no RF data files."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from receiver.signals import SIGNALS
from receiver.tracking import TrackingEngine, local_replica
from receiver.multiband_capture import capture_plan, capture
from receiver.dual_frequency import combine, coefficients
from receiver.fnav import decode_stream, parse_page
from receiver.telemetry import crc24_galileo, nominal_vertical_page, bits_to_uint
from receiver.multiband_position import validate_session


def bits(value, width):
    return np.array(
        [int(b) for b in format(value % (1 << width), f"0{width}b")], dtype=np.uint8
    )


def encode(word):
    # Independent shift-register encoder, G1=171/G2=133 octal, inverted G2.
    register = 0
    out = []
    for b in word:
        register = ((register << 1) | int(b)) & 127
        out.extend(
            [(register & 0o171).bit_count() % 2, 1 - (register & 0o133).bit_count() % 2]
        )
    return np.array(out, dtype=np.uint8)


class SignalTests(unittest.TestCase):
    def test_profiles_and_isolation(self):
        self.assertEqual(SIGNALS["e5a"].period, 0.001)
        self.assertEqual(SIGNALS["e5b"].symbol_epochs, 4)
        e1 = TrackingEngine(SIGNALS["e1"], 12.5e6)
        e5 = TrackingEngine(SIGNALS["e5b"], 25e6)
        self.assertEqual(e1.NOMINAL_SAMPLES_PER_CODE, 50000)
        self.assertEqual(e5.NOMINAL_SAMPLES_PER_CODE, 25000)
        self.assertEqual(e1.FS, 12.5e6)
        with self.assertRaises(ValueError):
            TrackingEngine(SIGNALS["e5a"], 12.5e6)

    def test_boundary_geometry(self):
        for name in SIGNALS:
            signal = SIGNALS[name]
            engine = TrackingEngine(signal, signal.sample_rate)
            rate = signal.chip_rate * (1 + 1200 / signal.frequency_hz)
            boundary = engine.next_primary_boundary_from_phase(123.4, rate)
            self.assertAlmostEqual(
                boundary / engine.FS, (signal.code_length - 123.4) / rate
            )
            start, stop, phase, next_boundary = engine.epoch_geometry(boundary, rate)
            self.assertAlmostEqual(phase, (start - boundary) * rate / engine.FS)
            self.assertAlmostEqual(
                (next_boundary - boundary) / engine.FS, signal.code_length / rate
            )
            self.assertGreater(stop, start)

    def test_secondary_sync_recovers_shift_and_frequency(self):
        for name in ("e1", "e5a", "e5b"):
            signal = SIGNALS[name]
            engine = TrackingEngine(signal, signal.sample_rate)
            _, pilot = signal.codes()
            secondary = signal.pilot_secondary(pilot, 1)
            n = signal.secondary_length * 3
            k = np.arange(n)
            shift = 17
            residual = 7.5
            samples = k * engine.NOMINAL_SAMPLES_PER_CODE
            z = secondary[(k + shift) % len(secondary)] * np.exp(
                2j * np.pi * residual * k * signal.period + 0.2j
            )
            result = engine.synchronize_secondary_code(
                dict(prompts=z, boundaries=samples), secondary, 1000.0
            )
            self.assertEqual(result["secondary_shift"], shift)
            self.assertAlmostEqual(result["refined_doppler_hz"], 1007.5)

    def test_e1_compatibility_replica(self):
        signal = SIGNALS["e1"]
        _, pilot = signal.codes()
        a = local_replica(1, pilot, 1.3, signal.chip_rate, 100)
        b = TrackingEngine(signal, 12.5e6).local_replica(
            1, pilot, 1.3, signal.chip_rate, 100
        )
        np.testing.assert_array_equal(a, b)


class NavigationTests(unittest.TestCase):
    def test_crc24q_check_value(self):
        message = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
        self.assertEqual(bits_to_uint(crc24_galileo(message)), 0xCDE703)

    def test_fnav_fec_crc_polarity_and_sample_anchor(self):
        payload = np.r_[
            bits(2, 6), np.random.default_rng(8).integers(0, 2, 208, dtype=np.uint8)
        ]
        word = np.r_[payload, crc24_galileo(payload), np.zeros(6, dtype=np.uint8)]
        coded = encode(word).reshape(61, 8).T.reshape(-1)
        preamble = np.array([int(x) for x in "101101110000"])
        frame = 1 - 2 * np.r_[preamble, coded].astype(float)
        soft = np.r_[np.zeros(37), -frame, -frame, np.zeros(12)]
        times = np.arange(len(soft)) * 0.02
        boundaries = 1234.25 + np.arange(len(soft)) * 500000
        pages, _ = decode_stream(soft, times, boundaries)
        self.assertEqual(len(pages), 2)
        self.assertTrue(all(p["crc_ok"] for p in pages))
        self.assertEqual(pages[0]["page_bits_244"], "".join(map(str, word)))
        self.assertEqual(pages[0]["boundary_sample"], boundaries[37])
        self.assertEqual(pages[0]["time_start_s"], times[37])

    def test_fnav_type1_fields(self):
        word = np.zeros(244, dtype=np.uint8)
        # ICD table 30: explicit absolute offsets, including signed clock/BGD.
        for offset, width, value in [
            (0, 6, 1),
            (6, 6, 12),
            (12, 10, 19),
            (22, 14, 100),
            (36, 31, -100),
            (102, 11, 320),
            (143, 10, -9),
            (153, 2, 0),
            (155, 12, 1400),
            (167, 20, 123456),
            (187, 1, 1),
        ]:
            word[offset : offset + width] = bits(value, width)
        parsed = parse_page("".join(map(str, word)))
        self.assertEqual(parsed["SVID"], 12)
        self.assertEqual(parsed["IODnav"], 19)
        self.assertEqual(parsed["toc_s"], 6000)
        self.assertEqual(parsed["af0_s"], -100 * 2**-34)
        self.assertEqual(parsed["BGD_E1_E5a_s"], -9 * 2**-32)
        self.assertEqual(parsed["GST_week"], 1400)
        self.assertEqual(parsed["GST_TOW_s"], 123456)
        self.assertEqual(parsed["E5a_DVS"], 1)

    def test_inav_vertical_crc(self):
        rng = np.random.default_rng(9)
        even = np.zeros(120, dtype=np.uint8)
        odd = np.zeros(120, dtype=np.uint8)
        even[2:114] = rng.integers(0, 2, 112)
        odd[0] = 1
        odd[2:82] = rng.integers(0, 2, 80)
        odd[82:106] = crc24_galileo(np.r_[even[:114], odd[:82]])
        self.assertTrue(nominal_vertical_page(even, odd)["crc_ok"])
        odd[38] ^= 1
        self.assertFalse(nominal_vertical_page(even, odd)["crc_ok"])


class CombinationTests(unittest.TestCase):
    def test_first_order_iono_and_bgd_cancel(self):
        from receiver.position_models import C

        for band in ("e5a", "e5b"):
            f1, f5 = SIGNALS["e1"].frequency_hz, SIGNALS[band].frequency_hz
            gamma = (f1 / f5) ** 2
            distance = 24e6
            iono = 8.0
            clock = 1e-5
            bgd = 4e-9
            common = dict(
                prn=3,
                receive_time_s=12.0,
                iodnav=8,
                satellite_ecef_m=np.array([20e6, 10e6, 15e6]),
                sisa_m=0.5,
                hardware_delay_m=0.0,
                clock_polynomial_s=clock,
                relativity_s=0.0,
            )
            e1 = dict(
                common,
                band="e1",
                frequency_hz=f1,
                raw_pseudorange_m=distance + iono - C * (clock - bgd),
                satellite_clock_e1_s=clock - bgd,
            )
            e5 = dict(
                common,
                band=band,
                frequency_hz=f5,
                raw_pseudorange_m=distance + gamma * iono - C * (clock - gamma * bgd),
                satellite_clock_e1_s=clock - gamma * bgd,
            )
            result, diagnostic = combine(e1, e5, 2.0, 1.0)
            self.assertAlmostEqual(
                result["raw_pseudorange_m"] + C * result["satellite_clock_e1_s"],
                distance,
                places=6,
            )
            self.assertAlmostEqual(diagnostic["apparent_e1_iono_m"], iono, places=6)
            a, b = coefficients(f1, f5)
            self.assertAlmostEqual(a + b, 1.0)
            self.assertAlmostEqual(result["code_sigma_m"] ** 2, 4 * a * a + b * b)
            e5["receive_time_s"] += 0.001
            with self.assertRaises(ValueError):
                combine(e1, e5)

    def test_unrelated_captures_rejected(self):
        with self.assertRaises(ValueError):
            validate_session([dict(capture_metadata={})])
        meta = dict(
            session_id="a",
            scheduled_start_device_time_s=1.0,
            first_sample_device_time_s=1.0,
            valid=True,
            synchronization="single timed multi-channel UHD streamer",
            session_start_offset_s=0.0,
        )
        validate_session([dict(capture_metadata=meta), dict(capture_metadata=meta)])
        with self.assertRaises(ValueError):
            validate_session(
                [
                    dict(capture_metadata=meta),
                    dict(capture_metadata={**meta, "session_id": "b"}),
                ]
            )


class CaptureTests(unittest.TestCase):
    def test_plans_and_transport_budget(self):
        plan = capture_plan("e1+e5b", 120.0)
        self.assertEqual(plan["sample_rate_sps"], 25e6)
        self.assertEqual(plan["expected_bytes"], 24_000_000_000)
        self.assertEqual(plan["subdev"], "A:0 A:1")
        self.assertEqual(capture_plan("e1+e5", 1.0)["payload_bytes_per_second"], 800e6)
        with self.assertRaises(ValueError):
            capture_plan("e1+e5a+e5b", 1.0, rate=100e6)
        with self.assertRaises(ValueError):
            capture_plan("e5b", 1.0, rate=12.5e6)
        with self.assertRaises(ValueError):
            capture_plan("e1+e1", 1.0)

    def test_fake_synchronized_stream_and_timestamp_gap(self):
        class Time:
            def __init__(self, t):
                self.t = t

            def get_real_secs(self):
                return self.t

            def __sub__(self, other):
                return Time(self.t - other.t)

        class Stream:
            def __init__(self, gap):
                self.received = 0
                self.gap = gap
                self.commands = []

            def get_max_num_samps(self):
                return 256

            def issue_stream_cmd(self, cmd):
                self.commands.append(cmd.mode)

            def recv(self, buffer, md, timeout):
                md.error_code = 0
                md.has_time_spec = True
                md.time_spec = Time(
                    2
                    + self.received / 25e6
                    + (1e-4 if self.gap and self.received else 0)
                )
                buffer[0] = 1
                buffer[1] = 2
                self.received += 256
                return 256

        class Device:
            def __init__(self, stream):
                self.stream = stream
                self.freq = {}
                self.gain = {}
                self.antenna = {}

            def set_rx_subdev_spec(self, *a):
                pass

            def set_rx_rate(self, *a):
                pass

            def set_rx_freq(self, v, c):
                self.freq[c] = v

            def set_rx_gain(self, v, c):
                self.gain[c] = v

            def set_rx_antenna(self, v, c):
                self.antenna[c] = v

            def get_rx_rate(self, c):
                return 25e6

            def get_rx_freq(self, c):
                return self.freq[c]

            def get_rx_gain(self, c):
                return self.gain[c]

            def get_rx_antenna(self, c):
                return self.antenna[c]

            def get_rx_sensor_names(self, c):
                return []

            def get_rx_stream(self, a):
                return self.stream

            def get_time_now(self):
                return Time(0)

        for gap in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                stream = Stream(gap)
                device = Device(stream)
                uhd = SimpleNamespace(
                    usrp=SimpleNamespace(
                        MultiUSRP=lambda a: device,
                        SubdevSpec=lambda a: a,
                        StreamArgs=lambda *a: SimpleNamespace(),
                    ),
                    types=SimpleNamespace(
                        TimeSpec=Time,
                        RXMetadata=lambda: SimpleNamespace(),
                        RXMetadataErrorCode=SimpleNamespace(none=0),
                        StreamMode=SimpleNamespace(
                            start_cont="start", stop_cont="stop"
                        ),
                        StreamCMD=lambda mode: SimpleNamespace(mode=mode),
                    ),
                )
                args = SimpleNamespace(
                    bands="e1+e5b",
                    duration=0.0001,
                    sample_rate=None,
                    subdev=None,
                    antennas=None,
                    gains=None,
                    plan=False,
                    gps_timeout=1,
                    output=Path(temp) / "capture",
                    args="fake",
                )
                with (
                    patch.dict("sys.modules", uhd=uhd),
                    patch(
                        "receiver.multiband_capture.lock_and_set_gps_time",
                        return_value={},
                    ),
                ):
                    if gap:
                        with self.assertRaisesRegex(RuntimeError, "timestamps"):
                            capture(args)
                    else:
                        self.assertEqual(capture(args), 0)
                meta = json.loads((args.output / "metadata.json").read_text())
                self.assertEqual(meta["valid"], not gap)
                self.assertEqual(stream.commands[-1], "stop")
                for band in ("e1", "e5b"):
                    child = json.loads(
                        (args.output / band / "metadata.json").read_text()
                    )
                    self.assertEqual(child["valid"], not gap)
                    self.assertEqual(
                        (args.output / band / f"{band}.sc16").stat().st_size,
                        meta["captured_samples"] * 4,
                    )


class ChannelizerTests(unittest.TestCase):
    def test_two_sidebands_timing_and_chunk_continuity(self):
        from receiver.e5_channelizer import split

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture_dir = root / "input"
            capture_dir.mkdir()
            fs = 100e6
            n = 1_020_000
            sample = np.arange(n)
            # Distinct baseband tones in each RF sideband, with known absolute phase.
            z = 1000 * np.exp(2j * np.pi * (-15.345e6 + 0.2e6) * sample / fs + 0.3j)
            z += 700 * np.exp(2j * np.pi * (15.345e6 - 0.3e6) * sample / fs - 0.6j)
            np.rint(np.column_stack((z.real, z.imag))).astype("<i2").tofile(
                capture_dir / "e5.sc16"
            )
            meta = dict(
                valid=True,
                band="e5",
                sample_rate_sps=fs,
                captured_samples=n,
                actual_center_frequency_hz=1191.795e6,
            )
            (capture_dir / "metadata.json").write_text(json.dumps(meta))
            split(SimpleNamespace(capture=capture_dir, output=root / "output"))
            for band, amp, freq, phase in [
                ("e5a", 500, 0.2e6, 0.3),
                ("e5b", 350, -0.3e6, -0.6),
            ]:
                iq = np.fromfile(
                    root / "output" / band / f"{band}.sc16", dtype="<i2"
                ).reshape(-1, 2)
                actual = iq[:, 0] + 1j * iq[:, 1]
                expected = amp * np.exp(
                    2j * np.pi * freq * np.arange(len(actual)) / 25e6 + 1j * phase
                )
                self.assertEqual(len(actual), n // 4)
                self.assertLess(
                    float(np.max(abs(actual[100:-100] - expected[100:-100]))), 2.0
                )
                child = json.loads(
                    (root / "output" / band / "metadata.json").read_text()
                )
                self.assertTrue(child["valid"])
                self.assertTrue(child["channelizer"]["group_delay_compensated"])


class FnavObservationTests(unittest.TestCase):
    def test_complete_navigation_to_fractional_range(self):
        from receiver.observables import load_channel, observation
        from receiver.position_models import C
        from receiver.navigation import propagate_ecef

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracking = root / "02_tracking"
            telemetry = root / "03_telemetry"
            tracking.mkdir()
            telemetry.mkdir()
            recording = dict(path="synthetic", sample_rate_sps=25e6)
            meta = dict(
                band="e5a",
                recording=recording,
                acquisition=dict(prn=1),
                capture_metadata={},
                boundary_sample_convention="start of each integration in recording sample coordinates",
            )
            (tracking / "metadata.json").write_text(json.dumps(meta))
            b = np.arange(40001) * 25000.0
            np.savez(
                tracking / "continuous_tracking.npz",
                boundary_sample=b,
                raw_start_sample=b.astype(int),
                carrier_lock=np.ones(len(b)),
                cn0_dbhz=np.full(len(b), 45.0),
            )
            pages = []

            def page(kind, fields, second):
                word = np.zeros(244, dtype=np.uint8)
                word[:6] = bits(kind, 6)
                for offset, width, value in fields:
                    word[offset : offset + width] = bits(value, width)
                pages.append(
                    dict(
                        word_type=kind,
                        crc_ok=True,
                        page_bits_244="".join(map(str, word)),
                        time_start_s=second,
                        boundary_sample=second * 25e6,
                    )
                )

            page(
                1,
                [
                    (6, 6, 1),
                    (12, 10, 9),
                    (22, 14, 1667),
                    (94, 8, 50),
                    (102, 11, 160),
                    (143, 10, -9),
                    (155, 12, 1400),
                    (167, 20, 100000),
                ],
                0,
            )
            page(
                2,
                [
                    (6, 10, 9),
                    (16, 32, round(0.5 / math.pi * 2**31)),
                    (72, 32, round(0.001 * 2**33)),
                    (104, 32, round(math.sqrt(29600000) * 2**19)),
                    (182, 12, 1400),
                    (194, 20, 100010),
                ],
                10,
            )
            page(
                3,
                [
                    (6, 10, 9),
                    (16, 32, round(56 / 180 * 2**31)),
                    (160, 14, 1667),
                    (174, 12, 1400),
                    (186, 20, 100020),
                ],
                20,
            )
            page(4, [(6, 10, 9), (189, 20, 100030)], 30)
            decoded = dict(
                band="e5a",
                navigation="F/NAV",
                satellite_prn=1,
                recording=recording,
                pages=pages,
            )
            (telemetry / "decoded_fnav.json").write_text(json.dumps(decoded))
            channel = load_channel(root, {})
            self.assertEqual(channel["primary_period_s"], 0.001)
            self.assertEqual(len(channel["segments"][0]["anchors"]), 4)
            self.assertEqual(channel["segments"][0]["origin_s"], 100000.0)
            sample = 123.5 * 25000
            o = observation(channel, sample, 1400, 100000.08, {})
            self.assertAlmostEqual(o["raw_pseudorange_m"], 0.08 * C, delta=0.001)
            self.assertAlmostEqual(o["transmit_sv_s"], 100000.1235, places=8)
            gamma = (1575.42 / 1176.45) ** 2
            self.assertAlmostEqual(o["bgd_factor"], gamma)
            clock = propagate_ecef(channel["ephemerides"][0], o["transmit_gst_tow_s"])[
                "clock_total_s"
            ]
            self.assertAlmostEqual(
                o["satellite_clock_e1_s"], clock - gamma * (-9 * 2**-32), places=14
            )
            channel["health_pages"][0]["decoded"]["E5a_DVS"] = 1
            with self.assertRaisesRegex(ValueError, "health or data validity"):
                observation(channel, sample, 1400, 100000.08, {})


if __name__ == "__main__":
    unittest.main()
