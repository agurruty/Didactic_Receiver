#!/usr/bin/env python3
"""
Galileo E1 continuous pilot-aided tracker.
This version intentionally follows the architecture of a real GNSS tracking
channel more closely than the earlier educational scripts:
  1. Work on the continuous 12.5 MS/s SC16 stream.
  2. Acquire Galileo E1-C pilot code phase + Doppler.
  3. Convert the acquisition code phase into the first TRUE primary-code
     boundary in the raw sample stream.
  4. From then on, every integration interval is bounded by predicted code
     epochs. It is NOT forced to a fixed 4 ms / fixed-sample grid.
  5. Synchronize and wipe the 25-chip E1-C secondary code.
  6. Use an FLL-assisted pilot PLL during pull-in, then a narrow pilot PLL.
  7. Use carrier-aided VEML DLL tracking.
  8. Form E1-B Prompt correlations from exactly the same carrier/code NCOs.
Because the code rate is Doppler shifted, an actual Galileo E1 code period is
not always exactly 50,000 samples at 12.5 MS/s. The predicted boundaries are
stored as floating-point sample positions and the integer block edges follow
them. Therefore integrations naturally become e.g. 49,999 / 50,000 / 50,001
samples when required.
Input directory:
    e1.sc16
Example:
    python3 02_galileo_e1_track.py \
        recordings \
        --acquisition runs/acquisition --prn 26 \
        --gnss-tools /path/to/GNSS-DSP-tools \
        --output tracking
Output:
    continuous_tracking.npz
    tracking_validation.npz      # compatible with 03_galileo_e1_decode.py
    metadata.json
    02_tracking_loops.png + 02_tracking_symbols.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from .common import recording, fingerprint
from .figures import tracking_figures as make_plot


# Galileo E1 / recording constants

FS = 12_500_000.0
E1_HZ = 1_575_420_000.0

CHIP_RATE = 1_023_000.0
CODE_LENGTH = 4092.0
CODE_PERIOD_S = 0.004

NOMINAL_SAMPLES_PER_CODE = FS * CODE_PERIOD_S  # exactly 50,000
SECONDARY_LENGTH = 25

# VEML spacings similar to common Galileo E1 receiver configurations.
EARLY_LATE_CHIPS = 0.15
VERY_EARLY_LATE_CHIPS = 0.60

# Tracking state machine.
FLL_PULLIN_SECONDS = 1.0
NARROW_AFTER_SECONDS = 5.0

FLL_BW_HZ = 10.0
PLL_WIDE_BW_HZ = 15.0
PLL_NARROW_BW_HZ = 5.0

DLL_WIDE_BW_HZ = 2.0
DLL_NARROW_BW_HZ = 0.75

DAMPING = 1.0 / math.sqrt(2.0)


# CLI / code repository


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Continuous-boundary Galileo E1-C pilot tracker with parallel "
            "E1-B navigation-symbol extraction."
        )
    )
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--output", type=Path, default=Path("e1_continuous_tracking"))
    p.add_argument("--gnss-tools", type=Path, default=None)
    p.add_argument(
        "--prn",
        type=int,
        default=None,
        help="Saved candidate to track; default is the strongest.",
    )
    p.add_argument(
        "--acquisition",
        type=Path,
        required=True,
        help="Directory from 01_galileo_e1_acquire.py; avoids reacquisition.",
    )
    p.add_argument(
        "--secondary-sync-epochs",
        type=int,
        default=75,
        help="Aligned E1-C primary epochs used for secondary-code sync.",
    )
    return p.parse_args()


def load_codes(args):
    roots = []
    if args.gnss_tools is not None:
        roots.append(args.gnss_tools)
    here = Path(__file__).resolve().parent.parent
    roots += [
        here / "GNSS-DSP-tools",
        here.parent / "GNSS-DSP-tools",
        Path.cwd() / "GNSS-DSP-tools",
        Path.cwd().parent / "GNSS-DSP-tools",
    ]
    checked = []
    for root in roots:
        root = root.expanduser().resolve()
        checked.append(str(root))
        if (root / "gnsstools" / "galileo" / "e1b.py").exists() and (
            root / "gnsstools" / "galileo" / "e1c.py"
        ).exists():
            sys.path.insert(0, str(root))
            from gnsstools.galileo import e1b, e1c

            return e1b, e1c, root
    raise RuntimeError(
        "GNSS-DSP-tools not found.\n"
        "Clone it once with:\n"
        "  git clone https://github.com/pmonta/GNSS-DSP-tools.git\n"
        "Then pass --gnss-tools /path/to/GNSS-DSP-tools\n\n"
        "Checked:\n  " + "\n  ".join(checked)
    )


# Raw SC16 access


def open_sc16(path: Path):
    raw = np.memmap(path, dtype="<i2", mode="r")
    if raw.size % 2:
        raise RuntimeError("SC16 file has an odd number of int16 values")
    return raw.reshape(-1, 2)


def read_complex(mm, start: int, stop: int):
    """
    Read only the requested complex samples from the memory-mapped SC16 file.
    """
    q = mm[start:stop]
    return q[:, 0].astype(np.float32) + 1j * q[:, 1].astype(np.float32)


# Galileo local replicas


# Acquisition on first fixed nominal blocks


# Continuous code-boundary model


# Aligned pilot Prompts for secondary-code synchronization


# Loop filters


def natural_frequency(bw_hz):
    z = DAMPING
    return 8.0 * z * bw_hz / (4.0 * z * z + 1.0)


def pll_gains(bw_hz):
    """
    PI loop:
      phase error [rad] -> frequency correction [Hz].
    """
    wn = natural_frequency(bw_hz)
    kp = DAMPING * wn / math.pi
    ki = wn * wn / (2.0 * math.pi)
    return kp, ki


def dll_gains(bw_hz):
    """
    PI DLL:
      code phase error [chips] -> code-rate correction [chips/s].
    """
    wn = natural_frequency(bw_hz)
    kp = 2.0 * DAMPING * wn
    ki = wn * wn
    return kp, ki


def fll_alpha(bw_hz, dt):
    return 1.0 - math.exp(-2.0 * math.pi * bw_hz * dt)


# VEML discriminator


def veml_discriminator(very_early, early, late, very_late):
    left = abs(very_early) + abs(early)
    right = abs(late) + abs(very_late)
    return (left - right) / max(left + right, 1e-15)


# Tracking-quality estimators


def carrier_lock_metric(pilot_window):
    """
    Phase concentration of the secondary-wiped pilot.
    1 = one tight phase cluster; ~0 = circular/random phase.
    """
    if len(pilot_window) < 25:
        return np.nan
    z = np.asarray(pilot_window, dtype=np.complex128)
    mag = np.abs(z)
    good = mag > 0
    if not np.any(good):
        return 0.0
    return float(abs(np.mean(z[good] / mag[good])))


def estimate_cn0(pilot_window, integration_times):
    """
    Simple coherent-pilot C/N0 estimate used only AFTER lock.
    It is intentionally suppressed if the phase concentration is poor.
    """
    if len(pilot_window) < 50:
        return np.nan
    lock = carrier_lock_metric(pilot_window)
    if not np.isfinite(lock) or lock < 0.7:
        return np.nan
    z = np.asarray(pilot_window, dtype=np.complex128)
    mean = np.mean(z)
    signal_power = abs(mean) ** 2
    noise_power = np.mean(np.abs(z - mean) ** 2)
    if signal_power <= 0 or noise_power <= 0:
        return np.nan
    tint = float(np.mean(integration_times))
    cn0 = signal_power / noise_power / tint
    return 10.0 * math.log10(cn0)


# Continuous tracker


# Plot


class TrackingEngine:
    """Independent signal/rate state: no process-global retuning between bands."""

    def __init__(self, signal, sample_rate):
        signal.validate_rate(sample_rate)
        self.signal = signal
        self.FS = float(sample_rate)
        self.CARRIER_HZ = signal.frequency_hz
        self.CHIP_RATE = signal.chip_rate
        self.CODE_LENGTH = signal.code_length
        self.CODE_PERIOD_S = signal.period
        self.NOMINAL_SAMPLES_PER_CODE = self.FS * signal.period
        self.SECONDARY_LENGTH = signal.secondary_length
        self.EARLY_LATE_CHIPS = 0.15 if signal.name == "e1" else 0.25
        self.VERY_EARLY_LATE_CHIPS = 0.60 if signal.name == "e1" else 0.75

    def local_replica(
        self,
        prn: int,
        module,
        start_chip: float,
        code_rate: float,
        n_samples: int,
        offset_chips: float = 0.0,
    ):
        """
        Galileo E1 primary code times sinBOC(1,1).
        start_chip is the code phase at the FIRST RAW SAMPLE of this correlation
        interval.
        The code generator accepts arbitrary fractional phase and arbitrary code
        rate, so variable-length epochs are naturally supported.
        """
        start = start_chip + offset_chips
        whole = math.floor(start)
        frac = start - whole
        chip_increment = code_rate / self.FS
        code = module.code(
            prn, int(whole), float(frac), float(chip_increment), int(n_samples)
        ).astype(np.float32)
        if self.signal.name != "e1":
            return code
        chip_phase = start + chip_increment * np.arange(n_samples, dtype=np.float64)
        # sinBOC(1,1): subcarrier changes sign every half chip.
        half_chip = np.floor(2.0 * chip_phase).astype(np.int64)
        boc = np.where((half_chip & 1) == 0, 1.0, -1.0).astype(np.float32)
        return code * boc

    def nominal_epoch_replica(self, prn, module):
        n = int(self.NOMINAL_SAMPLES_PER_CODE)
        return self.local_replica(prn, module, 0.0, self.CHIP_RATE, n)

    def first_nominal_blocks(self, mm, count: int):
        n = int(self.NOMINAL_SAMPLES_PER_CODE)
        required = count * n
        if len(mm) < required:
            raise RuntimeError("Recording is too short for acquisition")
        x = read_complex(mm, 0, required)
        x -= np.mean(x)
        rms = np.sqrt(np.mean(np.abs(x) ** 2))
        if rms <= 0:
            raise RuntimeError("Zero-power recording")
        x /= rms
        return x.reshape(count, n)

    def acquire_prn(self, blocks, prn, e1c, dopplers):
        n = blocks.shape[1]
        code = self.nominal_epoch_replica(prn, e1c)
        code_fft_conj = np.conj(np.fft.fft(code))
        sample_index = np.arange(n, dtype=np.float64)
        acquisition_map = np.zeros((len(dopplers), n), dtype=np.float32)
        for di, fd in enumerate(dopplers):
            carrier = np.exp(-1j * 2.0 * np.pi * fd * sample_index / self.FS).astype(
                np.complex64
            )
            spectrum = np.fft.fft(blocks * carrier[None, :], axis=1)
            corr = np.fft.ifft(spectrum * code_fft_conj[None, :], axis=1)
            # Noncoherent accumulation removes unknown E1-C secondary signs.
            acquisition_map[di] = np.sum(np.abs(corr) ** 2, axis=0).astype(np.float32)
        flat = int(np.argmax(acquisition_map))
        best_di, best_ci = np.unravel_index(flat, acquisition_map.shape)
        peak = float(acquisition_map[best_di, best_ci])
        tmp = acquisition_map.copy()
        doppler_guard = 2
        code_guard = int(round(2.0 * self.FS / self.CHIP_RATE))
        d0 = max(0, best_di - doppler_guard)
        d1 = min(len(dopplers), best_di + doppler_guard + 1)
        indices = (best_ci + np.arange(-code_guard, code_guard + 1)) % n
        tmp[d0:d1, :][:, indices] = np.nan
        floor = float(np.nanmedian(tmp))
        metric_db = 10.0 * math.log10(max(peak, 1e-30) / max(floor, 1e-30))
        return (metric_db, int(best_di), int(best_ci), acquisition_map)

    def resolve_code_phase_at_sample_zero(self, first_block, acq, e1c):
        """
        FFT circular-correlation lag can have either sign convention.
        Test both possible code phases directly on sample zero and retain whichever
        produces the stronger coherent pilot Prompt.
        """
        fd = acq["doppler_hz"]
        code_rate = self.CHIP_RATE * (1.0 + fd / self.CARRIER_HZ)
        lag_chips = acq["code_index"] * self.CHIP_RATE / self.FS
        candidates = [lag_chips % self.CODE_LENGTH, (-lag_chips) % self.CODE_LENGTH]
        n = len(first_block)
        idx = np.arange(n, dtype=np.float64)
        wiped = first_block * np.exp(-1j * 2.0 * np.pi * fd * idx / self.FS)
        values = []
        for phase in candidates:
            r = self.local_replica(acq["prn"], e1c, phase, code_rate, n)
            values.append(abs(np.vdot(r, wiped)))
        selected = int(np.argmax(values))
        return float(candidates[selected])

    def next_primary_boundary_from_phase(
        self, code_phase_at_sample0: float, code_rate: float
    ):
        """
        Return the first primary-code boundary AFTER sample zero as a floating-point
        raw-sample position.
        A boundary is code phase 0 modulo 4092 chips.
        """
        phase = code_phase_at_sample0 % self.CODE_LENGTH
        chips_to_boundary = (
            self.CODE_LENGTH - phase if phase > 1e-12 else self.CODE_LENGTH
        )
        seconds_to_boundary = chips_to_boundary / code_rate
        return seconds_to_boundary * self.FS

    def epoch_geometry(self, boundary_sample_float: float, code_rate: float):
        """
        Build one integration interval from this predicted primary-code boundary
        to the next.
        Returns:
          start_idx
          stop_idx
          code_phase_at_start_sample
          next_boundary_float
        boundary_sample_float itself can lie between ADC samples.
        """
        samples_per_code = self.FS * self.CODE_LENGTH / code_rate
        next_boundary = boundary_sample_float + samples_per_code
        # Nearest ADC sample to each continuous boundary.
        start_idx = int(round(boundary_sample_float))
        stop_idx = int(round(next_boundary))
        if stop_idx <= start_idx:
            raise RuntimeError("Invalid predicted code epoch length")
        # The first ADC sample is generally slightly before/after the exact
        # continuous-time code boundary.
        start_chip = (start_idx - boundary_sample_float) * code_rate / self.FS
        return (start_idx, stop_idx, start_chip, next_boundary)

    def collect_aligned_prompts(
        self, mm, prn, e1c, first_boundary, initial_doppler_hz, count
    ):
        """
        Form pilot Prompt correlations over real predicted code epochs.
        No secondary-code wipeoff is performed yet.
        """
        code_rate = self.CHIP_RATE * (1.0 + initial_doppler_hz / self.CARRIER_HZ)
        boundary = float(first_boundary)
        carrier_phase = 0.0
        prompts = []
        boundaries = []
        lengths = []
        for _ in range(count):
            (start, stop, start_chip, next_boundary) = self.epoch_geometry(
                boundary, code_rate
            )
            if stop > len(mm):
                break
            x = read_complex(mm, start, stop)
            n = len(x)
            sample_index = np.arange(n, dtype=np.float64)
            local_carrier = np.exp(
                -1j
                * (
                    carrier_phase
                    + 2.0 * np.pi * initial_doppler_hz * sample_index / self.FS
                )
            )
            wiped = x * local_carrier
            r = self.local_replica(prn, e1c, start_chip, code_rate, n)
            prompt = np.vdot(r, wiped)
            prompts.append(prompt)
            boundaries.append(boundary)
            lengths.append(n)
            duration = n / self.FS
            carrier_phase = math.remainder(
                carrier_phase + 2.0 * np.pi * initial_doppler_hz * duration, 2.0 * np.pi
            )
            boundary = next_boundary
        return {
            "prompts": np.asarray(prompts, dtype=np.complex128),
            "boundaries": np.asarray(boundaries, dtype=np.float64),
            "lengths": np.asarray(lengths, dtype=np.int32),
            "end_boundary": boundary,
            "carrier_phase_end": carrier_phase,
            "code_rate": code_rate,
        }

    def synchronize_secondary_code(self, aligned, secondary, initial_doppler_hz):
        prompts = aligned["prompts"]
        if len(prompts) < self.SECONDARY_LENGTH:
            raise RuntimeError(
                "Need at least one full pilot secondary period for synchronization"
            )
        # Use exact elapsed time from predicted continuous code boundaries.
        t = (aligned["boundaries"] - aligned["boundaries"][0]) / self.FS
        # Fine residual frequency around the already-refined acquisition value.
        residual_grid = np.arange(-50.0, 50.01, 0.5)
        best = None
        for shift in range(self.SECONDARY_LENGTH):
            signs = np.asarray(
                [
                    secondary[(k + shift) % self.SECONDARY_LENGTH]
                    for k in range(len(prompts))
                ],
                dtype=np.float64,
            )
            z = prompts * signs
            for residual in residual_grid:
                coherent = np.sum(z * np.exp(-1j * 2.0 * np.pi * residual * t))
                power = abs(coherent) ** 2
                if best is None or power > best["power"]:
                    best = {
                        "secondary_shift": shift,
                        "residual_hz": float(residual),
                        "refined_doppler_hz": float(initial_doppler_hz + residual),
                        "coherent_phase_rad": float(np.angle(coherent)),
                        "power": float(power),
                    }
        return best

    def calibrate_veml_slope(self, prn, e1c):
        """
        Numerically calibrate the sign and slope of our sampled discriminator.
        The synthetic truth is code phase = 0. The local replicas are shifted by
        the test error. Therefore the fitted slope directly maps discriminator
        output to LOCAL-minus-TRUE code phase error in chips.
        """
        n = int(self.NOMINAL_SAMPLES_PER_CODE)
        truth = self.local_replica(prn, e1c, 0.0, self.CHIP_RATE, n)
        offsets = np.linspace(-0.08, 0.08, 17)
        d = []
        for error in offsets:
            ve = np.vdot(
                self.local_replica(
                    prn, e1c, error, self.CHIP_RATE, n, -self.VERY_EARLY_LATE_CHIPS
                ),
                truth,
            )
            e = np.vdot(
                self.local_replica(
                    prn, e1c, error, self.CHIP_RATE, n, -self.EARLY_LATE_CHIPS
                ),
                truth,
            )
            l = np.vdot(
                self.local_replica(
                    prn, e1c, error, self.CHIP_RATE, n, +self.EARLY_LATE_CHIPS
                ),
                truth,
            )
            vl = np.vdot(
                self.local_replica(
                    prn, e1c, error, self.CHIP_RATE, n, +self.VERY_EARLY_LATE_CHIPS
                ),
                truth,
            )
            d.append(veml_discriminator(ve, e, l, vl))
        slope, intercept = np.polyfit(offsets, d, 1)
        if abs(slope) < 1e-6:
            raise RuntimeError("VEML discriminator calibration failed")
        return (float(slope), float(intercept))

    def run_continuous_tracking(
        self,
        mm,
        acq,
        secondary_sync,
        secondary,
        e1b,
        e1c,
        first_tracking_boundary,
        first_tracking_epoch_number,
    ):
        prn = acq["prn"]
        slope, intercept = self.calibrate_veml_slope(prn, e1c)
        print(
            f"\nVEML calibration: slope {slope:+.5f}/chip, intercept {intercept:+.5f}"
        )
        # Carrier states.
        frequency_reference_hz = float(secondary_sync["refined_doppler_hz"])
        pll_integrator_hz = 0.0
        # The absolute phase reference is arbitrary; the pilot PLL will settle it.
        carrier_phase_rad = 0.0
        # Code states.
        dll_integrator_chips_s = 0.0
        dll_rate_correction = 0.0
        boundary = float(first_tracking_boundary)
        previous_pilot = None
        previous_prompt_time = None
        pilot_history = deque(maxlen=max(250, round(1.0 / self.CODE_PERIOD_S)))
        integration_history = deque(maxlen=pilot_history.maxlen)
        current_lock = np.nan
        current_cn0 = np.nan
        # Results.
        out = {
            key: []
            for key in [
                "time",
                "raw_start_sample",
                "epoch_samples",
                "boundary_sample",
                "prompt",
                "pilot_prompt",
                "very_early",
                "early",
                "prompt_mag",
                "late",
                "very_late",
                "doppler_hz",
                "code_rate_chips_s",
                "dll_error_chips",
                "pll_error_rad",
                "fll_error_hz",
                "carrier_lock",
                "cn0_dbhz",
                "tracking_state",
                "secondary_sign",
            ]
        }
        start_tracking_sample = boundary
        epoch_number = int(first_tracking_epoch_number)
        print("\nCONTINUOUS PILOT TRACKING")
        print("------------------------------")
        print(f"Initial Doppler      {frequency_reference_hz:+.3f} Hz")
        print(f"First boundary       sample {boundary:.3f}")
        print("Integration length   variable, code-boundary driven")
        while True:
            track_time = (boundary - start_tracking_sample) / self.FS
            if track_time < FLL_PULLIN_SECONDS:
                state = 0
                pll_bw = PLL_WIDE_BW_HZ
                dll_bw = DLL_WIDE_BW_HZ
            elif track_time < NARROW_AFTER_SECONDS:
                state = 1
                pll_bw = PLL_WIDE_BW_HZ
                dll_bw = DLL_WIDE_BW_HZ
            else:
                state = 2
                pll_bw = PLL_NARROW_BW_HZ
                dll_bw = DLL_NARROW_BW_HZ
            pll_kp, pll_ki = pll_gains(pll_bw)
            dll_kp, dll_ki = dll_gains(dll_bw)
            # Current frequency command BEFORE seeing this epoch's discriminator.
            freq_pre = frequency_reference_hz + pll_integrator_hz
            # Carrier aiding predicts most of the code Doppler.
            carrier_aided_code_rate = self.CHIP_RATE * (
                1.0 + freq_pre / self.CARRIER_HZ
            )
            code_rate = carrier_aided_code_rate + dll_rate_correction
            (start_idx, stop_idx, start_chip, next_boundary) = self.epoch_geometry(
                boundary, code_rate
            )
            if start_idx < 0 or stop_idx > len(mm):
                break
            x = read_complex(mm, start_idx, stop_idx)
            n = len(x)
            if n < 10:
                break
            dt = n / self.FS
            sample_index = np.arange(n, dtype=np.float64)
            carrier = np.exp(
                -1j
                * (carrier_phase_rad + 2.0 * np.pi * freq_pre * sample_index / self.FS)
            )
            wiped = x * carrier
            # Pilot VEML correlators.
            r_ve = self.local_replica(
                prn, e1c, start_chip, code_rate, n, -self.VERY_EARLY_LATE_CHIPS
            )
            r_e = self.local_replica(
                prn, e1c, start_chip, code_rate, n, -self.EARLY_LATE_CHIPS
            )
            r_p = self.local_replica(prn, e1c, start_chip, code_rate, n, 0.0)
            r_l = self.local_replica(
                prn, e1c, start_chip, code_rate, n, +self.EARLY_LATE_CHIPS
            )
            r_vl = self.local_replica(
                prn, e1c, start_chip, code_rate, n, +self.VERY_EARLY_LATE_CHIPS
            )
            VE = np.vdot(r_ve, wiped)
            E = np.vdot(r_e, wiped)
            P = np.vdot(r_p, wiped)
            L = np.vdot(r_l, wiped)
            VL = np.vdot(r_vl, wiped)
            # Known pilot secondary-code sign for THIS TRUE primary-code epoch.
            secondary_sign = float(
                secondary[
                    (epoch_number + secondary_sync["secondary_shift"])
                    % self.SECONDARY_LENGTH
                ]
            )
            VE *= secondary_sign
            E *= secondary_sign
            P *= secondary_sign
            L *= secondary_sign
            VL *= secondary_sign
            # Parallel E1-B data Prompt. E1-B and E1-C use the same code timing /
            # carrier timing but different primary codes.
            r_data = self.local_replica(prn, e1b, start_chip, code_rate, n, 0.0)
            data_prompt = np.vdot(r_data, wiped)
            if self.signal.name != "e1":
                # E5 data and pilot are in quadrature. Global sign is resolved by CRC.
                data_prompt *= 1j
                data_secondary = e1b.secondary_code
                data_prompt *= data_secondary[
                    (epoch_number + secondary_sync["secondary_shift"])
                    % len(data_secondary)
                ]
            # Carrier discriminators
            phase_error = math.atan2(P.imag, P.real)
            prompt_time = 0.5 * (start_idx + stop_idx) / self.FS
            if previous_pilot is None or previous_prompt_time is None:
                fll_error = 0.0
            else:
                delta_t = prompt_time - previous_prompt_time
                fll_error = np.angle(P * np.conj(previous_pilot)) / (
                    2.0 * np.pi * delta_t
                )
            previous_pilot = P
            previous_prompt_time = prompt_time
            # Carrier loop
            if state == 0:
                # FLL changes only the persistent frequency reference.
                frequency_reference_hz += fll_alpha(FLL_BW_HZ, dt) * fll_error
                # Small direct phase steering during pull-in.
                phase_steer = 0.20 * phase_error
                pll_integrator_hz = 0.0
                freq_cmd = frequency_reference_hz
            else:
                pll_integrator_hz += pll_ki * phase_error * dt
                proportional_hz = pll_kp * phase_error
                freq_cmd = frequency_reference_hz + pll_integrator_hz + proportional_hz
                phase_steer = 0.0
            # DLL
            d_raw = veml_discriminator(VE, E, L, VL)
            # local-minus-true code phase error in chips.
            code_error_chips = (d_raw - intercept) / slope
            # If local code is ahead (positive error), slow the local code.
            dll_integrator_chips_s += -dll_ki * code_error_chips * dt
            dll_rate_correction = -dll_kp * code_error_chips + dll_integrator_chips_s
            # Update continuous-time NCO states
            carrier_phase_rad = math.remainder(
                carrier_phase_rad + 2.0 * np.pi * freq_cmd * dt + phase_steer,
                2.0 * np.pi,
            )
            # The NEXT boundary was predicted with the pre-discriminator code rate.
            # Correct it slightly using the measured local-minus-true phase error.
            #
            # If the local epoch boundary is late, the local code phase is behind
            # (negative error); this shifts the next boundary earlier.
            phase_boundary_correction_samples = (
                code_error_chips * self.FS / max(code_rate, 1.0)
            )
            # Apply only a modest fraction as phase steering; the DLL frequency
            # state handles the long-term correction.
            next_boundary += 0.20 * phase_boundary_correction_samples
            epoch_boundary = boundary
            boundary = next_boundary
            # Quality monitoring
            pilot_history.append(P)
            integration_history.append(dt)
            if len(pilot_history) == pilot_history.maxlen:
                current_lock = carrier_lock_metric(pilot_history)
                current_cn0 = estimate_cn0(pilot_history, integration_history)
            # Save
            out["time"].append(track_time)
            out["raw_start_sample"].append(start_idx)
            out["epoch_samples"].append(n)
            out["boundary_sample"].append(epoch_boundary)
            out["prompt"].append(data_prompt)
            out["pilot_prompt"].append(P)
            out["very_early"].append(abs(VE))
            out["early"].append(abs(E))
            out["prompt_mag"].append(abs(P))
            out["late"].append(abs(L))
            out["very_late"].append(abs(VL))
            out["doppler_hz"].append(freq_cmd)
            out["code_rate_chips_s"].append(code_rate)
            out["dll_error_chips"].append(code_error_chips)
            out["pll_error_rad"].append(phase_error)
            out["fll_error_hz"].append(fll_error)
            out["carrier_lock"].append(current_lock)
            out["cn0_dbhz"].append(current_cn0)
            out["tracking_state"].append(state)
            out["secondary_sign"].append(secondary_sign)
            epoch_number += 1
        for key in out:
            out[key] = np.asarray(out[key])
        out["veml_slope"] = slope
        out["veml_intercept"] = intercept
        return out


from .signals import SIGNALS

_DEFAULT_ENGINE = TrackingEngine(SIGNALS["e1"], FS)


def acquire_prn(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.acquire_prn(*args, **kwargs)


def calibrate_veml_slope(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.calibrate_veml_slope(*args, **kwargs)


def collect_aligned_prompts(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.collect_aligned_prompts(*args, **kwargs)


def epoch_geometry(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.epoch_geometry(*args, **kwargs)


def first_nominal_blocks(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.first_nominal_blocks(*args, **kwargs)


def local_replica(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.local_replica(*args, **kwargs)


def next_primary_boundary_from_phase(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.next_primary_boundary_from_phase(*args, **kwargs)


def nominal_epoch_replica(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.nominal_epoch_replica(*args, **kwargs)


def resolve_code_phase_at_sample_zero(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.resolve_code_phase_at_sample_zero(*args, **kwargs)


def run_continuous_tracking(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.run_continuous_tracking(*args, **kwargs)


def synchronize_secondary_code(*args, **kwargs):
    """Compatibility entry point for the original E1 receiver."""
    return _DEFAULT_ENGINE.synchronize_secondary_code(*args, **kwargs)


def main():
    args = parse_args()
    capture_dir = args.capture_dir.expanduser().resolve()
    raw_path, capture_metadata = recording(capture_dir)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.prn is not None and not (1 <= args.prn <= 50):
        raise ValueError("--prn must be 1..50")
    if args.secondary_sync_epochs < 25:
        raise ValueError("--secondary-sync-epochs must be >= 25")
    args.output.mkdir(parents=True)
    e1b, e1c, code_root = load_codes(args)
    secondary = np.asarray(e1c.secondary_code, dtype=np.float64)
    if len(secondary) != SECONDARY_LENGTH:
        raise RuntimeError("Unexpected E1-C secondary-code length")
    mm = open_sc16(raw_path)
    duration = len(mm) / FS
    print("=" * 80)
    print("GALILEO E1 CONTINUOUS CODE-BOUNDARY PILOT TRACKER")
    print("=" * 80)
    print(f"Input                {raw_path}")
    print(f"Duration             {duration:.3f} s")
    print(f"Sample rate          {FS / 1e6:.3f} MS/s")
    print(f"Nominal code epoch   {NOMINAL_SAMPLES_PER_CODE:.0f} samples")
    print("Actual epoch length  allowed to vary continuously")
    print(f"Code source          {code_root}")
    # 1. Acquisition
    from .acquisition import load_seed

    acquisition = load_seed(args.acquisition, raw_path, args.prn)
    acq_blocks = first_nominal_blocks(mm, 1)
    first_block = acq_blocks[0]
    phase0 = resolve_code_phase_at_sample_zero(first_block, acquisition, e1c)
    acquisition["code_phase_sample0_chips"] = phase0
    initial_code_rate = CHIP_RATE * (1.0 + acquisition["doppler_hz"] / E1_HZ)
    first_boundary = next_primary_boundary_from_phase(phase0, initial_code_rate)
    print("\nCONTINUOUS-TIME HANDOFF")
    print("-----------------------")
    print(f"Code phase at sample 0  {phase0:.6f} chips")
    print(f"First true boundary     {first_boundary:.3f} raw samples")
    print(f"Nominal acquisition fd  {acquisition['doppler_hz']:+.3f} Hz")
    # 2. Aligned E1-C epochs + secondary synchronization
    aligned = collect_aligned_prompts(
        mm=mm,
        prn=acquisition["prn"],
        e1c=e1c,
        first_boundary=first_boundary,
        initial_doppler_hz=acquisition["doppler_hz"],
        count=args.secondary_sync_epochs,
    )
    if len(aligned["prompts"]) < 25:
        raise RuntimeError("Recording is too short for secondary-code synchronization")
    secsync = synchronize_secondary_code(aligned, secondary, acquisition["doppler_hz"])
    print("\nE1-C SECONDARY-CODE SYNCHRONIZATION")
    print("---------------------------------")
    print(f"Secondary shift      {secsync['secondary_shift']} / 25")
    print(f"Residual Doppler     {secsync['residual_hz']:+.3f} Hz")
    print(f"Refined Doppler      {secsync['refined_doppler_hz']:+.3f} Hz")
    unique_lengths, length_counts = np.unique(aligned["lengths"], return_counts=True)
    print(
        "Aligned epoch lengths "
        + ", ".join(
            f"{int(n)} samples × {int(c)}"
            for n, c in zip(unique_lengths, length_counts)
        )
    )
    # Start tracking at the next exact predicted boundary after sync.
    first_tracking_boundary = float(aligned["end_boundary"])
    # The first secondary-sync prompt was epoch number 0 relative to our
    # chosen first boundary; tracking begins after N such epochs.
    first_tracking_epoch_number = int(len(aligned["prompts"]))
    # 3. Continuous tracking
    tracking = run_continuous_tracking(
        mm=mm,
        acq=acquisition,
        secondary_sync=secsync,
        secondary=secondary,
        e1b=e1b,
        e1c=e1c,
        first_tracking_boundary=first_tracking_boundary,
        first_tracking_epoch_number=first_tracking_epoch_number,
    )
    if not len(tracking["time"]):
        raise RuntimeError("No tracking output produced")
    # 4. Save outputs
    np.savez(
        args.output / "continuous_tracking.npz",
        **{
            key: value
            for key, value in tracking.items()
            if isinstance(value, np.ndarray)
        },
    )
    # Existing I/NAV decoder compatibility.
    np.savez(
        args.output / "tracking_validation.npz",
        time=tracking["time"],
        prompt=tracking["prompt"],
        pilot_prompt=tracking["pilot_prompt"],
        early=tracking["early"],
        prompt_mag=tracking["prompt_mag"],
        late=tracking["late"],
        pll_error=tracking["pll_error_rad"],
        dll_error=tracking["dll_error_chips"],
        doppler_hz=tracking["doppler_hz"],
        cn0_dbhz=tracking["cn0_dbhz"],
        carrier_lock=tracking["carrier_lock"],
        raw_start_sample=tracking["raw_start_sample"],
        boundary_sample=tracking["boundary_sample"],
        epoch_samples=tracking["epoch_samples"],
    )
    finite_lock = tracking["carrier_lock"][-250:]
    finite_lock = finite_lock[np.isfinite(finite_lock)]
    finite_cn0 = tracking["cn0_dbhz"][-250:]
    finite_cn0 = finite_cn0[np.isfinite(finite_cn0)]
    final_lock = float(np.median(finite_lock[-250:])) if len(finite_lock) else None
    final_cn0 = float(np.median(finite_cn0[-250:])) if len(finite_cn0) else None
    metadata = {
        "recording": fingerprint(raw_path),
        "capture_metadata": capture_metadata,
        "boundary_sample_convention": "start of each integration in recording sample coordinates",
        "architecture": {
            "tracking_component": ("Galileo E1-C pilot"),
            "telemetry_component": ("Galileo E1-B"),
            "continuous_code_boundaries": True,
            "variable_epoch_sample_count": True,
            "secondary_code_wipeoff": True,
            "carrier_loop": ("FLL pull-in + pilot PLL"),
            "code_loop": ("carrier-aided VEML DLL"),
        },
        "input": {"file": str(raw_path), "sample_rate_sps": FS, "duration_s": duration},
        "acquisition": {
            "prn": acquisition["prn"],
            "metric_db": acquisition["metric_db"],
            "doppler_hz": acquisition["doppler_hz"],
            "code_phase_sample0_chips": (phase0),
            "first_boundary_sample": (first_boundary),
        },
        "secondary_sync": secsync,
        "loop_settings": {
            "fll_bw_hz": (FLL_BW_HZ),
            "pll_wide_bw_hz": (PLL_WIDE_BW_HZ),
            "pll_narrow_bw_hz": (PLL_NARROW_BW_HZ),
            "dll_wide_bw_hz": (DLL_WIDE_BW_HZ),
            "dll_narrow_bw_hz": (DLL_NARROW_BW_HZ),
            "early_late_chips": (EARLY_LATE_CHIPS),
            "very_early_late_chips": (VERY_EARLY_LATE_CHIPS),
            "veml_slope": tracking["veml_slope"],
            "veml_intercept": tracking["veml_intercept"],
        },
        "final": {
            "doppler_hz": float(tracking["doppler_hz"][-1]),
            "carrier_lock": (final_lock),
            "cn0_dbhz": (final_cn0),
            "tracking_seconds": float(tracking["time"][-1]),
            "data_symbols": int(len(tracking["prompt"])),
            "epoch_sample_lengths": {
                str(int(v)): int(c)
                for v, c in zip(
                    *np.unique(tracking["epoch_samples"], return_counts=True)
                )
            },
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\nFINAL SUMMARY")
    print("-------------")
    print(f"PRN                    E{acquisition['prn']:02d}")
    print(f"Final Doppler          {tracking['doppler_hz'][-1]:+.3f} Hz")
    print(
        f"Carrier lock           "
        f"{final_lock if final_lock is not None else float('nan'):.3f}"
    )
    if final_cn0 is None:
        print("C/N₀                   not reported (lock threshold not met)")
    else:
        print(f"C/N₀                   {final_cn0:.2f} dB-Hz")
    unique_lengths, counts = np.unique(tracking["epoch_samples"], return_counts=True)
    print(
        "Tracking epoch lengths "
        + ", ".join(f"{int(n)}×{int(c)}" for n, c in zip(unique_lengths, counts))
    )
    print(f"E1-B Prompt symbols    {len(tracking['prompt'])}")
    print(
        "\nIf the pilot locks, tracking_validation.npz can be fed directly "
        "to 03_galileo_e1_decode.py."
    )
    make_plot(acquisition, secsync, tracking, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
