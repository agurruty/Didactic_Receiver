"""Measured-data views: a question, a visible result, and explicit measurement limits."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from .common import FS, TEAL, GOLD, RED, MUTED, heading, finish


def raw_signal(raw, meta, output):
    mm = np.memmap(raw, dtype="<i2", mode="r").reshape(-1, 2)
    total, sq, clipped = np.zeros(2), np.zeros(2), 0
    times, levels, clips = [], [], []
    chunk = int(FS)
    for start in range(0, len(mm), chunk):
        iq = mm[start : start + chunk].astype(np.float64)
        total += iq.sum(axis=0)
        sq += (iq * iq).sum(axis=0)
        c = np.any((iq <= -32768) | (iq >= 32767), axis=1).sum()
        clipped += int(c)
        times.append(start / FS)
        levels.append(np.sqrt(np.mean(np.sum(iq * iq, axis=1))))
        clips.append(100 * c / len(iq))
    means = total / len(mm)
    rms = float(np.sqrt(sq.sum() / len(mm)))
    stats = dict(
        file=str(raw),
        samples=len(mm),
        duration_s=len(mm) / FS,
        rms_counts=rms,
        i_mean_counts=float(means[0]),
        q_mean_counts=float(means[1]),
        clip_fraction=clipped / len(mm),
        statistics_scope="entire recording",
    )
    n = min(len(mm), 262144)
    z = mm[:n, 0].astype(float) + 1j * mm[:n, 1].astype(float)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    t = np.arange(min(n, 1250)) / FS * 1e6
    ax[0, 0].plot(t, z.real[: len(t)], label="I", alpha=0.85)
    ax[0, 0].plot(t, z.imag[: len(t)], label="Q", alpha=0.7)
    ax[0, 0].set(
        title="01  The antenna becomes numbers\nFirst 100 µs (or available samples)",
        xlabel="Time from recording start [µs]",
        ylabel="SC16 counts",
    )
    ax[0, 0].legend(ncol=2)
    if n >= 8:
        f, psd = welch(z, fs=FS, nperseg=min(8192, n), return_onesided=False)
        ax[0, 1].plot(np.fft.fftshift(f) / 1e6, 10 * np.log10(np.maximum(np.fft.fftshift(psd), 1e-30)))
    ax[0, 1].set(
        title="02  Energy across the receiver bandwidth\nWelch PSD of the initial inspection window",
        xlabel="Offset from 1575.42 MHz [MHz]",
        ylabel="Power density [dB counts²/Hz]",
    )
    bins = np.linspace(-32768, 32768, 129)
    ax[1, 0].hist(z.real, bins=bins, histtype="step", label="I")
    ax[1, 0].hist(z.imag, bins=bins, histtype="step", label="Q")
    ax[1, 0].set(
        title="03  Is the digital range being used well?\nInitial window; dashed lines mark SC16 rails",
        xlabel="SC16 counts",
        ylabel="Samples per bin",
        yscale="log",
    )
    ax[1, 0].axvline(-32768, color=RED, ls="--")
    ax[1, 0].axvline(32767, color=RED, ls="--")
    ax[1, 0].legend()
    ax[1, 1].plot(times, levels, marker=".", label="Complex RMS")
    ax[1, 1].set(
        title="04  Did the recording stay stable?\nEvery sample contributes to one-second blocks",
        xlabel="Time from recording start [s]",
        ylabel="Complex RMS [counts]",
    )
    right = ax[1, 1].twinx()
    right.plot(times, clips, color=RED, marker=".", alpha=0.65)
    right.set_ylabel("Samples touching an SC16 rail [%]", color=RED)
    for a in ax.flat:
        a.grid(True)
    heading(
        fig,
        "00",
        "From radio waves to I/Q",
        f"{len(mm)/FS:.2f} s recorded  ·  {rms:,.0f} counts RMS  ·  {100*clipped/len(mm):.5f}% at digital rails",
        "Noise-like I/Q is expected. A spectrum alone cannot identify a satellite.\nSC16 counts are uncalibrated; digital rails do not diagnose all analog/ADC overload. PSD and histogram use the initial window only.",
    )
    finish(fig, output / "00_raw_signal")
    return stats


def acquisition_figure(rows, acq, args, output):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    shown = rows[:16]
    ax[0, 0].barh(
        [f"E{r['prn']:02d}" for r in shown],
        [r["metric_db"] for r in shown],
        color=[TEAL if r["detected"] else "#b9c5ce" for r in shown],
    )
    ax[0, 0].invert_yaxis()
    ax[0, 0].axvline(
        args.threshold_db, color=GOLD, ls="--", label=f"Heuristic gate: {args.threshold_db:g} dB"
    )
    ax[0, 0].set(
        title=f"01  Which codes stand out?\nTop {len(shown)} of {len(rows)} searched PRNs",
        xlabel="Coarse peak / guarded median map floor [dB]",
    )
    ax[0, 0].legend(fontsize=8)
    power = acq["fine_map"].astype(float)
    peak = max(power.max(), 1e-30)
    db = 10 * np.log10(np.maximum(power, 1e-30) / peak)
    ci = acq["code_index"]
    di = int(np.argmax(power) // power.shape[1])
    dx = 4092 / power.shape[1]
    fd = acq["fine_dopplers"]
    df = fd[1] - fd[0] if len(fd) > 1 else args.fine_step
    # Coarse map shows the complete searched domain; cuts below use fine bins.
    coarse = acq["coarse_map"].astype(float)
    cdi, cci = np.unravel_index(np.argmax(coarse), coarse.shape)
    cdop = acq["coarse_dopplers"]
    cdf = cdop[1] - cdop[0] if len(cdop) > 1 else args.doppler_step
    coarse_db = 10 * np.log10(np.maximum(coarse, 1e-30) / max(coarse.max(), 1e-30))
    group = 25
    trimmed = (coarse_db.shape[1] // group) * group
    display = coarse_db[:, :trimmed].reshape(len(cdop), -1, group).max(axis=2)
    im = ax[0, 1].imshow(
        display,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=-18,
        vmax=0,
        extent=[-0.5 * dx, (trimmed - 0.5) * dx, cdop[0] - 0.5 * cdf, cdop[-1] + 0.5 * cdf],
    )
    ax[0, 1].scatter(cci * dx, cdop[cdi], s=160, marker="+", color=GOLD, linewidth=2, clip_on=False)
    ax[0, 1].set(
        title=f"02  Full delay–Doppler search · E{acq['prn']:02d}\nGold cross: strongest coarse-grid cell",
        xlabel="Circular code delay [chips]",
        ylabel="Doppler [Hz]",
    )
    fig.colorbar(im, ax=ax[0, 1], label="Power relative to peak [dB]", shrink=0.8)
    offsets = np.arange(-int(3 / dx), int(3 / dx) + 1)
    ax[1, 0].plot(offsets * dx, db[di, (ci + offsets) % power.shape[1]])
    ax[1, 0].scatter(0, 0, color=GOLD, s=70, zorder=5)
    ax[1, 0].set(
        title="03  Zoom into code alignment\nBOC side peaks remain visible; false locks are possible",
        xlabel="Delay relative to selected peak [chips]",
        ylabel="Correlation relative to peak [dB]",
    )
    ax[1, 1].plot(fd, db[:, ci], marker=".", label="Fixed winning delay")
    ax[1, 1].scatter(fd[di], 0, color=GOLD, s=70, zorder=5)
    ax[1, 1].set(
        title="04  Remove the carrier frequency mismatch\nPeak is a grid estimate; tracking refines it",
        xlabel="Doppler [Hz]",
        ylabel="Correlation relative to peak [dB]",
    )
    for a in (ax[0, 0], ax[1, 0], ax[1, 1]):
        a.grid(True)
    detected = next(r["detected"] for r in rows if r["prn"] == acq["prn"])
    heading(
        fig,
        "01",
        "Find a satellite hidden in noise",
        f"E{acq['prn']:02d}  ·  {acq['doppler_hz']:+.0f} Hz  ·  {ci*dx:.3f} chips  ·  {'candidate above gate' if detected else 'BELOW GATE'}",
        f"{args.dwells} × 4 ms noncoherent E1-C search; sinBOC(1,1) replica approximates CBOC. Map columns show maxima of 25 samples; full selected maps are stored.\nPeak/floor is not C/N₀. Delay is modulo 4 ms (≈1,199 km), not a pseudorange. The detection gate is not a false-alarm calibration.",
    )
    finish(fig, output / f"01_acquisition_E{acq['prn']:02d}")


def tracking_figures(acq, sync, d, output):
    t = d["time"]
    prn = acq["prn"]
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    signals = [
        ("doppler_hz", "Carrier follows frequency motion", "Doppler [Hz]", 1),
        ("code_rate_chips_s", "Code clock follows the received chips", "Code rate − 1.023 Mcps [chips/s]", 1),
        ("pll_error_rad", "PLL corrects phase error", "Phase residual [degrees]", 180 / np.pi),
        ("dll_error_chips", "DLL corrects code misalignment", "VEML residual [chips]", 1),
        ("carrier_lock", "Does the pilot phase stay concentrated?", "Phase concentration [0–1]", 1),
        ("cn0_dbhz", "How much carrier rises above noise?", "Estimated C/N₀ [dB-Hz]", 1),
    ]
    for a, (key, title, label, scale) in zip(ax.flat, signals):
        y = d[key] * scale - (1_023_000 if key == "code_rate_chips_s" else 0)
        a.plot(t, y)
        a.set(title=title, xlabel="Time since tracking began [s]", ylabel=label)
        a.grid(True)
        a.set_xlim(0, max(float(t[-1]), 0.004))
        a.ticklabel_format(axis="y", style="plain", useOffset=False)
        for switch in (1, 5):
            if switch < t[-1]:
                a.axvline(switch, color=MUTED, ls=":", alpha=0.6)
    ax[2, 0].axhline(0.7, color=GOLD, ls="--")
    ax[2, 0].set_ylim(-0.05, 1.05)
    if not np.any(np.isfinite(d["cn0_dbhz"])):
        ax[2, 1].text(
            0.5,
            0.5,
            "C/N₀ unavailable\nphase-concentration gate not met",
            ha="center",
            transform=ax[2, 1].transAxes,
        )
    tail = d["carrier_lock"][-250:]
    tail = tail[np.isfinite(tail)]
    lock = f"{np.median(tail):.3f}" if len(tail) else "unavailable"
    heading(
        fig,
        "02",
        "Keep the local replica aligned",
        f"E{prn:02d}  ·  {len(t):,} tracked symbols  ·  final-window phase concentration {lock}",
        "Dotted lines: scheduled FLL → wide PLL at 1 s → narrow PLL at 5 s; these transitions do not prove lock.\nC/N₀ uses a finite coherent-pilot window and is withheld below 0.7 phase concentration. Doppler includes receiver oscillator error.",
    )
    finish(fig, output / "02_tracking_loops")
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    for a, key, title in [
        (ax[0, 0], "pilot_prompt", "Pilot after secondary-code wipeoff"),
        (ax[0, 1], "prompt", "Data channel keeps the symbol signs"),
    ]:
        z = d[key]
        scale = max(float(np.median(np.abs(z))), 1e-30)
        a.scatter(z.real / scale, z.imag / scale, c=t, s=4, cmap="viridis", alpha=0.5, rasterized=True)
        a.axhline(0, color=MUTED, lw=0.6)
        a.axvline(0, color=MUTED, lw=0.6)
        a.set(title=title, xlabel="I / median magnitude", ylabel="Q / median magnitude", aspect="equal")
    n = min(len(t), 500)
    ax[0, 2].plot(t[:n], d["prompt"].real[:n], label="I")
    ax[0, 2].plot(t[:n], d["prompt"].imag[:n], label="Q", alpha=0.65)
    ax[0, 2].set(
        title="The first two seconds of E1-B symbols",
        xlabel="Tracking time [s]",
        ylabel="Prompt correlation [counts]",
    )
    ax[0, 2].legend()
    norm = max(float(np.median(d["prompt_mag"])), 1e-30)
    for key, label in [
        ("very_early", "VE"),
        ("early", "E"),
        ("prompt_mag", "P"),
        ("late", "L"),
        ("very_late", "VL"),
    ]:
        ax[1, 0].plot(t, d[key] / norm, label=label, alpha=0.7, lw=0.8)
    ax[1, 0].set(
        title="Five correlators straddle the code peak",
        xlabel="Tracking time [s]",
        ylabel="Magnitude / median prompt",
    )
    ax[1, 0].legend(ncol=5, fontsize=8)
    ax[1, 1].plot(t, d["epoch_samples"] - 50000, lw=0.7)
    ax[1, 1].set(
        title="Code periods need not contain 50,000 samples",
        xlabel="Tracking time [s]",
        ylabel="Epoch length − 50,000 [samples]",
    )
    ax[1, 2].plot(t, d["fll_error_hz"])
    ax[1, 2].set(
        title="FLL observes residual frequency mismatch",
        xlabel="Tracking time [s]",
        ylabel="Frequency discriminator [Hz]",
    )
    for a in ax.flat:
        a.grid(True)
    heading(
        fig,
        "02",
        "From a spreading code to data symbols",
        f"E{prn:02d}  ·  secondary shift {sync['secondary_shift']}/25  ·  residual correction {sync['residual_hz']:+.2f} Hz",
        "Constellation color progresses from early (purple) to late (yellow); all samples and transients are shown.\nA concentrated pilot and two data clusters support lock; navigation CRC is the next independent check. Absolute sample indices are preserved.",
    )
    finish(fig, output / "02_tracking_symbols")


def telemetry_figure(prompt, soft, sync, result, prn, output):
    from .telemetry import deinterleave

    parts, pages = result["parts"], result["pages"]
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    z = prompt / max(float(np.median(np.abs(prompt))), 1e-30)
    ax[0, 0].scatter(z.real, z.imag, s=5, alpha=0.35, rasterized=True)
    ax[0, 0].set(
        title="01  One correlation becomes one coded symbol",
        xlabel="Prompt I / median magnitude",
        ylabel="Prompt Q / median magnitude",
        aspect="equal",
    )
    lim = max(1.3, float(np.max(np.abs(np.r_[z.real, z.imag]))) * 1.05)
    ax[0, 0].set(xlim=(-lim, lim), ylim=(-lim, lim))
    ax[0, 0].axhline(0, color=MUTED, lw=0.7)
    ax[0, 0].axvline(0, color=MUTED, lw=0.7)
    ax[0, 1].plot(np.arange(250), sync["scores"])
    ax[0, 1].axvline(
        sync["best_offset"], color=GOLD, ls="--", label=f"Selected offset: {sync['best_offset']}"
    )
    ax[0, 1].set(
        title="02  Find the repeating 0101100000 sync word",
        xlabel="Candidate offset [symbols modulo 250]",
        ylabel="Mean normalized sync correlation",
    )
    ax[0, 1].legend(fontsize=8)
    start = sync["best_offset"]
    stop = min(len(soft), start + 80)
    idx = np.arange(start, stop)
    ax[1, 0].step((idx - start) * 4, sync["polarity"] * soft[start:stop], where="mid", lw=0.8)
    ax[1, 0].axvspan(-2, 38, color=GOLD, alpha=0.16, label="10 unencoded sync symbols")
    ax[1, 0].axhline(0, color=MUTED, lw=0.7)
    ax[1, 0].set(
        title="03  The first page part begins with sync",
        xlabel="Time since candidate page-part boundary [ms]",
        ylabel="Normalized soft symbol (signed I)",
    )
    ax[1, 0].legend(fontsize=8)
    if parts:
        p0 = parts[0]
        inter = sync["polarity"] * soft[p0["symbol_start"] + 10 : p0["symbol_start"] + 250]
        field = np.vstack(
            [inter.reshape(8, 30), np.full((1, 30), np.nan), deinterleave(inter).reshape(8, 30)]
        )
        lim = max(float(np.nanmax(np.abs(field))), 1e-30)
        im = ax[1, 1].imshow(
            field, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest"
        )
        ax[1, 1].set(
            title="04  Undo the permutation before FEC",
            xlabel="Column (same row-major layout in both blocks)",
            yticks=[3.5, 12.5],
            yticklabels=["Received", "FEC order"],
        )
        fig.colorbar(im, ax=ax[1, 1], label="Soft symbol", shrink=0.8)
        x = np.array([p["time_start_s"] for p in parts])
        ax[2, 0].plot(x, [p["sync_corr"] for p in parts], marker=".", label="Sync")
        ax[2, 0].set(
            title="05  Check sync and the decoding fit",
            xlabel="Decoder-relative page-part time [s]",
            ylabel="Normalized sync correlation",
            ylim=(-1.05, 1.05),
        )
        right = ax[2, 0].twinx()
        right.plot(x, [p["path_metric"] / 240 for p in parts], color=GOLD, marker="x", ms=4)
        right.set_ylabel("FEC cost / coded symbol", color=GOLD)
    else:
        for a in (ax[1, 1], ax[2, 0]):
            a.text(0.5, 0.5, "No complete page part", ha="center", va="center", transform=a.transAxes)
            a.set_axis_off()
    if pages:
        for valid, label, color, marker in [(True, "CRC pass", TEAL, "o"), (False, "CRC fail", RED, "x")]:
            pp = [p for p in pages if p["crc_ok"] == valid]
            if pp:
                ax[2, 1].scatter(
                    [p["time_start_s"] for p in pp],
                    [p["word_type"] for p in pp],
                    color=color,
                    marker=marker,
                    label=label,
                    s=45,
                )
        ax[2, 1].legend(fontsize=8)
    else:
        ax[2, 1].text(
            0.5,
            0.5,
            "No complete even + odd nominal page",
            ha="center",
            va="center",
            transform=ax[2, 1].transAxes,
        )
    ax[2, 1].set(
        title="06  CRC is the independent integrity check",
        xlabel="Decoder-relative page time [s]",
        ylabel="I/NAV word type",
    )
    for a in (ax[0, 0], ax[0, 1], ax[1, 0], ax[2, 0], ax[2, 1]):
        a.grid(True)
    passes = sum(p["crc_ok"] for p in pages)
    heading(
        fig,
        "03",
        "From noisy symbols to intact words",
        f"E{prn:02d}  ·  {passes}/{len(pages)} complete pages pass CRC  ·  {len(parts)} page parts decoded",
        "A 4 ms symbol is still convolutionally encoded. The interleaver view precedes G2 inversion; Viterbi uses soft amplitudes.\nFEC cost is not a bit-error rate; zero tail bits are forced by the decoder. CRC checks integrity, not satellite authenticity (OSNMA not verified).",
    )
    finish(fig, output / "03_telemetry")
