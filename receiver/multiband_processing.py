"""Signal-aware acquisition, tracking and telemetry, retaining original ADC coordinates."""

import json
import numpy as np
import matplotlib.pyplot as plt
from .common import fingerprint, write_json, heading, finish
from .signals import SIGNALS
from .multiband_capture import recording
from .tracking import TrackingEngine, open_sc16
from .positioning import serial


def identity(raw, rate):
    return {**fingerprint(raw), "sample_rate_sps": rate}


def acquire(args):
    signal = SIGNALS[args.band]
    raw, meta, rate = recording(args.capture, args.band)
    if (
        args.dwells < 1
        or args.top < 1
        or not args.prns
        or any(not 1 <= p <= 50 for p in args.prns)
    ):
        raise ValueError("Positive dwells/top and Galileo PRNs 1–50 are required.")
    if any(
        not np.isfinite(v) or v <= 0 for v in (args.doppler_max, args.doppler_step)
    ) or not np.isfinite(args.threshold_db):
        raise ValueError(
            "Finite positive Doppler search settings and finite threshold required."
        )
    engine = TrackingEngine(signal, rate)
    _, pilot = signal.codes()
    mm = open_sc16(raw)
    n = round(rate * signal.period)
    required = args.dwells * 2 * n
    if len(mm) < required:
        raise ValueError(f"Need at least {required / rate:g} s for acquisition.")
    iq = mm[:required].astype(np.float32)
    blocks = (iq[:, 0] + 1j * iq[:, 1]).reshape(args.dwells, 2 * n)
    blocks -= np.mean(blocks)
    rms = np.sqrt(np.mean(abs(blocks) ** 2))
    if rms <= 0:
        raise ValueError("Zero-power capture.")
    blocks /= rms
    dopplers = np.arange(
        -args.doppler_max, args.doppler_max + args.doppler_step * 0.5, args.doppler_step
    )
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    strongest = []
    for prn in sorted(set(args.prns)):
        # One code replica padded to two periods: each tested lag integrates an
        # aligned primary epoch, avoiding secondary sign transitions at the true lag.
        template = np.r_[engine.nominal_epoch_replica(prn, pilot), np.zeros(n)]
        code_fft = np.conj(np.fft.fft(template))
        power = np.empty((len(dopplers), n), dtype=np.float32)
        t = np.arange(2 * n) / rate
        for k, fd in enumerate(dopplers):
            z = blocks * np.exp(-2j * np.pi * fd * t)
            corr = np.fft.ifft(np.fft.fft(z, axis=1) * code_fft, axis=1)[:, :n]
            power[k] = np.sum(abs(corr) ** 2, axis=0)
        di, ci = np.unravel_index(np.argmax(power), power.shape)
        floor = float(np.median(power))
        metric = float(
            10 * np.log10(max(float(power[di, ci]), 1e-30) / max(floor, 1e-30))
        )
        row = dict(
            prn=prn,
            doppler_hz=float(dopplers[di]),
            code_index=int(ci),
            metric_db=metric,
            detected=metric >= args.threshold_db,
        )
        rows.append(row)
        strongest.append((row, power))
        strongest.sort(key=lambda item: item[0]["metric_db"], reverse=True)
        strongest = strongest[: args.top]
        print(
            f"{args.band.upper()} E{prn:02}: {metric:.2f} dB peak/floor; {dopplers[di]:+.0f} Hz",
            flush=True,
        )
    selected = []
    for rank, (row, power) in enumerate(strongest):
        if row["detected"]:
            # Secondary synchronization searches a narrow residual window; refine
            # coarse Doppler first so a 250 Hz grid cannot leave a 125 Hz residual.
            fine = np.arange(
                row["doppler_hz"] - args.doppler_step,
                row["doppler_hz"] + args.doppler_step + 12.5,
                25.0,
            )
            template = np.r_[
                engine.nominal_epoch_replica(row["prn"], pilot), np.zeros(n)
            ]
            code_fft = np.conj(np.fft.fft(template))
            best = None
            for fd in fine:
                corr = np.fft.ifft(
                    np.fft.fft(blocks * np.exp(-2j * np.pi * fd * t), axis=1)
                    * code_fft,
                    axis=1,
                )[:, :n]
                fine_power = np.sum(abs(corr) ** 2, axis=0)
                ci = int(np.argmax(fine_power))
                peak = float(fine_power[ci])
                if best is None or peak > best[0]:
                    best = (peak, float(fd), ci)
            row.update(
                doppler_hz=best[1], code_index=best[2], fine_doppler_step_hz=25.0
            )
            selected.append(row["prn"])
            write_json(args.output / f"E{row['prn']:02}_seed.json", row)
        if rank == 0 or row["detected"]:
            np.savez_compressed(
                args.output / f"E{row['prn']:02}_search.npz",
                power=power,
                dopplers=dopplers,
            )
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            order = sorted(rows, key=lambda r: r["metric_db"], reverse=True)
            ax[0].bar(
                [f"E{r['prn']:02}" for r in order], [r["metric_db"] for r in order]
            )
            ax[0].axhline(args.threshold_db, color="red", ls="--")
            ax[0].tick_params(axis="x", rotation=90)
            ax[0].set(
                title="Which codes stand above the search floor?",
                ylabel="Peak / median floor [dB]",
            )
            image = ax[1].imshow(
                10 * np.log10(np.maximum(power, 1e-20) / np.max(power)),
                origin="lower",
                aspect="auto",
                extent=[0, signal.code_length, dopplers[0], dopplers[-1]],
            )
            ax[1].set(
                xlabel="Primary-code lag [chips]",
                ylabel="Doppler [Hz]",
                title=f"E{row['prn']:02} delay–Doppler evidence",
            )
            fig.colorbar(image, ax=ax[1], label="Relative power [dB]")
            heading(
                fig,
                "01",
                f"{args.band.upper()} pilot acquisition",
                f"{signal.period * 1000:g} ms coherent × {args.dwells} noncoherent dwells",
                "The detection gate is heuristic, not a calibrated false-alarm probability. Tracking and CRC provide independent checks.",
            )
            finish(fig, args.output / f"01_acquisition_E{row['prn']:02}")
    write_json(
        args.output / "acquisition.json",
        dict(
            schema_version=2,
            band=args.band,
            recording=identity(raw, rate),
            capture_metadata=meta,
            selected_prns=selected,
            ranked_candidates=sorted(rows, key=lambda r: r["metric_db"], reverse=True),
            replica="sinBOC(1,1)" if args.band == "e1" else "E5 sideband BPSK(10)",
            coherent_ms=signal.period * 1000,
            noncoherent_dwells=args.dwells,
            threshold_db=args.threshold_db,
        ),
    )
    return 0 if selected else 3


def track(args):
    summary = json.loads((args.acquisition / "acquisition.json").read_text())
    band = summary["band"]
    signal = SIGNALS[band]
    raw, meta, rate = recording(args.capture, band)
    if summary["recording"] != identity(raw, rate):
        raise ValueError("Acquisition belongs to a different or modified recording.")
    selected = summary["selected_prns"]
    if not selected:
        raise ValueError("No above-gate acquisition candidates.")
    prn = args.prn if args.prn is not None else selected[0]
    if prn not in selected:
        raise ValueError("Requested PRN has no above-gate seed.")
    acq = json.loads((args.acquisition / f"E{prn:02}_seed.json").read_text())
    engine = TrackingEngine(signal, rate)
    data, pilot = signal.codes()
    secondary = signal.pilot_secondary(pilot, prn)
    mm = open_sc16(raw)
    first = engine.first_nominal_blocks(mm, 1)[0]
    phase = engine.resolve_code_phase_at_sample_zero(first, acq, pilot)
    code_rate = signal.chip_rate * (1 + acq["doppler_hz"] / signal.frequency_hz)
    boundary = engine.next_primary_boundary_from_phase(phase, code_rate)
    aligned = engine.collect_aligned_prompts(
        mm, prn, pilot, boundary, acq["doppler_hz"], 3 * signal.secondary_length
    )
    if len(aligned["prompts"]) < 2 * signal.secondary_length:
        raise ValueError(
            "Need at least two complete pilot secondary periods for synchronization."
        )
    sync = engine.synchronize_secondary_code(aligned, secondary, acq["doppler_hz"])
    result = engine.run_continuous_tracking(
        mm,
        acq,
        sync,
        secondary,
        data,
        pilot,
        aligned["end_boundary"],
        len(aligned["prompts"]),
    )
    if not len(result["time"]):
        raise ValueError("No samples remain after secondary synchronization.")
    result["primary_epoch"] = np.arange(len(result["time"])) + len(aligned["prompts"])
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "continuous_tracking.npz",
        **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
    )
    # Data symbols start at a data-secondary boundary. Preserve their exact
    # first primary epoch, never assign time using a nominal file-grid index.
    epochs = result["primary_epoch"] + sync["secondary_shift"]
    start = (
        int(np.flatnonzero(epochs % signal.symbol_epochs == 0)[0])
        if np.any(epochs % signal.symbol_epochs == 0)
        else len(epochs)
    )
    count = (len(epochs) - start) // signal.symbol_epochs
    indices = start + np.arange(count) * signal.symbol_epochs
    prompts = (
        result["prompt"][start : start + count * signal.symbol_epochs]
        .reshape(count, signal.symbol_epochs)
        .sum(axis=1)
    )
    np.savez_compressed(
        args.output / "tracking_validation.npz",
        prompt=prompts,
        time=result["time"][indices],
        raw_start_sample=result["raw_start_sample"][indices],
        boundary_sample=result["boundary_sample"][indices],
        primary_epoch=result["primary_epoch"][indices],
    )
    metadata = dict(
        schema_version=2,
        band=band,
        recording=identity(raw, rate),
        capture_metadata=meta,
        boundary_sample_convention="start of each integration in recording sample coordinates",
        input=dict(sample_rate_sps=rate, duration_s=len(mm) / rate),
        acquisition=acq,
        secondary_sync=sync,
        primary_period_s=signal.period,
        symbol_period_s=signal.period * signal.symbol_epochs,
        replica="sinBOC(1,1)" if band == "e1" else "BPSK sideband",
        status="tracked; navigation CRC and observation quality gates still required",
    )
    write_json(args.output / "metadata.json", serial(metadata))
    from .figures import tracking_figures

    tracking_figures(acq, sync, result, args.output, signal=signal, sample_rate=rate)
    return 0


def decode(args):
    if not np.isfinite(args.skip_seconds) or args.skip_seconds < 0:
        raise ValueError("Skip seconds must be finite and nonnegative.")
    from . import telemetry

    meta = json.loads((args.tracking / "metadata.json").read_text())
    signal = SIGNALS[meta["band"]]
    with np.load(args.tracking / "tracking_validation.npz", allow_pickle=False) as d:
        prompt = np.array(d["prompt"])
        boundaries = np.array(d["boundary_sample"])
        times = np.array(d["time"])
    keep = times >= args.skip_seconds
    prompt, boundaries, times = prompt[keep], boundaries[keep], times[keep]
    if np.any(~np.isfinite(prompt)) or np.any(np.diff(boundaries) <= 0):
        raise ValueError("Invalid tracked symbol sequence.")
    args.output.mkdir(parents=True, exist_ok=False)
    soft = prompt.real
    pages = []
    parts = []
    scores = []
    if signal.navigation == "I/NAV" and len(soft) >= 500:
        sync, _ = telemetry.sync_offset_search(soft)
        orientation, _ = telemetry.choose_viterbi_orientation(soft, sync)
        scores = sync["scores"].tolist()
        parts = orientation["parts"]
        for p in orientation["pages"]:
            part = parts[p["even_part_index"]]
            idx = part["symbol_start"]
            row = dict(
                crc_ok=p["crc_ok"],
                word_type=p["word_type"],
                word_bits_128="".join(map(str, p["word"])),
                time_start_s=float(times[idx]),
                boundary_sample=float(boundaries[idx]),
                symbol_start=idx,
            )
            if signal.name == "e1":
                row.update(
                    osnma_bits="".join(map(str, p["osnma"])),
                    sar_bits="".join(map(str, p["sar"])),
                )
            else:
                # These bit positions are reserved on E5b, not OSNMA/SAR.
                row["reserved_1_bits"] = "".join(
                    map(
                        str,
                        np.r_[
                            p["osnma"],
                            p["sar"],
                            parts[p["odd_part_index"]]["bits"][80:82],
                        ],
                    )
                )
            pages.append(row)
    elif signal.navigation == "F/NAV" and len(soft) >= 500:
        from .fnav import decode_stream

        pages, scores = decode_stream(soft, times, boundaries)
    output = dict(
        schema_version=2,
        band=signal.name,
        navigation=signal.navigation,
        satellite_prn=meta["acquisition"]["prn"],
        recording=meta["recording"],
        capture_metadata=meta["capture_metadata"],
        pages=pages,
        symbols_available=len(soft),
        symbol_period_s=meta["symbol_period_s"],
        crc_pass=sum(p["crc_ok"] for p in pages),
        reason=None if pages else "Insufficient complete navigation pages",
    )
    filename = (
        "decoded_fnav.json" if signal.navigation == "F/NAV" else "decoded_inav.json"
    )
    write_json(args.output / filename, serial(output))
    np.savez_compressed(
        args.output / "telemetry_diagnostics.npz",
        soft_symbols=soft,
        boundary_sample=boundaries,
        sync_scores=np.asarray(scores),
    )
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].plot(times, soft)
    ax[0, 0].set(
        title="Tracked data → soft symbols",
        xlabel="Tracking time [s]",
        ylabel="Correlation",
    )
    ax[0, 1].plot(scores)
    ax[0, 1].set(
        title="Repeated preamble alignment",
        xlabel="Symbol offset",
        ylabel="Normalized score",
    )
    ax[1, 0].scatter(
        [p["time_start_s"] for p in pages],
        [p["word_type"] for p in pages],
        c=["#087f8c" if p["crc_ok"] else "#bf4b45" for p in pages],
    )
    ax[1, 0].set(
        title="Navigation pages: teal = CRC pass",
        xlabel="Tracking time [s]",
        ylabel="Word/page type",
    )
    ax[1, 1].axis("off")
    ax[1, 1].text(
        0.02,
        0.8,
        f"{output['crc_pass']} / {len(pages)} pages pass CRC\n{len(soft)} soft symbols\n{signal.navigation}\n\nCRC detects bit errors; it is not authentication.",
        va="top",
    )
    heading(
        fig,
        "03",
        f"{signal.name.upper()} navigation decoding",
        f"E{output['satellite_prn']:02} · {signal.navigation}",
        "Only CRC-valid pages may enter orbit and time assembly. Missing data are retained as missing evidence.",
    )
    finish(fig, args.output / "03_telemetry")
    return 0 if output["crc_pass"] else 3
