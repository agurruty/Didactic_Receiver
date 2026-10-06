"""Missing evidence is a meaningful receiver result, not an invented orbit."""

import numpy as np
import matplotlib.pyplot as plt
from .common import TEAL, RED, heading, finish


def incomplete_navigation(raw, output, reason):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    messages = raw.get("messages", [])
    ax[0].scatter([m["time_start_s"] for m in messages], [m["word_type"] for m in messages], color=TEAL)
    ax[0].set(
        title="What was recovered with a passing CRC?",
        xlabel="Decoder-relative time [s]",
        ylabel="I/NAV word type",
    )
    ax[0].grid(True)
    coverage = raw.get("ephemeris_word_coverage", {})
    if coverage:
        keys = list(coverage)
        values = np.array([[int(w in coverage[k]) for w in (1, 2, 3, 4)] for k in keys])
        from matplotlib.colors import ListedColormap

        ax[1].imshow(values, aspect="auto", vmin=0, vmax=1, cmap=ListedColormap(["#e5eaf0", TEAL]))
        ax[1].set(
            xticks=range(4),
            xticklabels=["WT1", "WT2", "WT3", "WT4"],
            yticks=range(len(keys)),
            yticklabels=keys,
            title="Four words must share one issue of data",
            ylabel="IODnav",
        )
        for i in range(len(keys)):
            for j in range(4):
                ax[1].text(
                    j,
                    i,
                    "received" if values[i, j] else "missing",
                    ha="center",
                    va="center",
                    color="white" if values[i, j] else RED,
                )
    else:
        ax[1].text(
            0.5,
            0.5,
            "No CRC-valid ephemeris words\nNo orbit can be propagated",
            ha="center",
            va="center",
            transform=ax[1].transAxes,
        )
        ax[1].set_axis_off()
    heading(
        fig,
        "04",
        "What the navigation message tells us",
        f"E{raw['satellite_prn']:02d}  ·  {reason}",
        "No missing fields are replaced with nominal orbital values. More clean telemetry may recover the required words.\nEven a complete satellite orbit does not supply a receiver pseudorange or a receiver position.",
    )
    finish(fig, output / "04_navigation_readiness")


def navigation_dashboard(raw_pages, ephem, validation, anchor, state, orbit, output):
    from .common import GOLD, MUTED
    from .navigation import wrap_week_time, decode_sisa

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(221, projection="3d")
    xyz = np.array([[q[k] / 1000 for k in ("x_m", "y_m", "z_m")] for q in orbit])
    ax.plot(*xyz.T, color=TEAL, lw=2)
    point = np.array([state[k] / 1000 for k in ("x_m", "y_m", "z_m")])
    ax.scatter(*point, color=GOLD, s=70)
    u, v = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 24)
    earth = 6378.137
    ax.plot_surface(
        earth * np.outer(np.cos(u), np.sin(v)),
        earth * np.outer(np.sin(u), np.sin(v)),
        earth * np.outer(np.ones_like(u), np.cos(v)),
        color="#82b3c4",
        alpha=0.8,
        linewidth=0,
    )
    limit = max(earth, float(np.max(np.abs(xyz)))) * 1.05
    ax.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        zlim=(-limit, limit),
        xlabel="ECEF X [km]",
        ylabel="ECEF Y [km]",
        zlabel="ECEF Z [km]",
        title="01  The broadcast orbit places the satellite in space",
    )
    ax.set_box_aspect((1, 1, 1))
    ax.locator_params(nbins=3)
    ax = fig.add_subplot(222)
    times = [m["time_start_s"] for m in raw_pages]
    words = [m["word_type"] for m in raw_pages]
    ax.scatter(times, words, color=TEAL, s=35)
    ax.set(
        title="02  Which CRC-valid words made this possible?",
        xlabel="Decoder-relative time [s]",
        ylabel="I/NAV word type",
    )
    ax.grid(True)
    ax.text(
        0.02,
        0.96,
        f"WT1–4 share IODnav {ephem['IODnav']}\n{sum(c['pass'] for c in validation)}/{len(validation)} physical checks passed",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax = fig.add_subplot(223)
    dt = np.array([wrap_week_time(q["tow_s"] - state["tow_s"]) / 60 for q in orbit])
    poly = np.array([q["clock_polynomial_s"] * 1e9 for q in orbit])
    rel = np.array([q["relativistic_correction_s"] * 1e9 for q in orbit])
    # Put small effects on a common scale by explicitly removing the central polynomial bias.
    bias = state["clock_polynomial_s"] * 1e9
    ax.plot(dt, poly - bias, label=f"Polynomial − {bias:+.2f} ns")
    ax.plot(dt, rel, label="Relativity", color=GOLD)
    ax.axvline(0, color=MUTED, ls=":")
    ax.set(
        title="03  Clock drift and relativity change the timing",
        xlabel="Time from evaluation epoch [min]",
        ylabel="Clock terms [ns; polynomial bias removed]",
    )
    ax.legend(fontsize=9)
    ax.grid(True)
    ax = fig.add_subplot(224)
    ax.set_axis_off()
    sisa = decode_sisa(int(ephem["SISA_index"]))
    accuracy = f"{sisa['metres']:.2f} m" if sisa["kind"] == "value" else sisa["kind"]
    rows = [
        (
            "Satellite position",
            f"{state['x_m']/1000:,.1f}, {state['y_m']/1000:,.1f}, {state['z_m']/1000:,.1f} km ECEF",
        ),
        (
            "Orbit size / shape",
            f"a = {ephem['semi_major_axis_m']/1000:,.1f} km   ·   e = {ephem['eccentricity']:.6f}",
        ),
        ("Broadcast SISA", accuracy + " (not receiver position accuracy)"),
        ("Clock model", f"{state['clock_total_s']*1e6:+.4f} µs; E1 BGD not applied"),
        (
            "Evaluation time",
            f"TOW {state['tow_s']:.3f} s · {anchor.get('evaluation_epoch_source','unspecified')}",
        ),
        ("Pseudorange / position", "Not formed; common receive epochs and time scales still required"),
    ]
    for i, (label, value) in enumerate(rows):
        y = 0.95 - i * 0.155
        ax.text(0, y, label, weight="bold", fontsize=10, va="top")
        ax.text(0, y - 0.055, value, fontsize=9, va="top")
    ax.set_title("04  What we know, and what remains to measure")
    heading(
        fig,
        "04",
        "Navigation bits become a satellite orbit",
        f"E{ephem['SVID']:02d}  ·  coherent ephemeris recovered  ·  satellite clock and ECEF model available",
        "The arc is a broadcast-model prediction; the gold point is its evaluation epoch. Earth and orbit use equal spatial scales.\nFull fields, health/DVS, validation details and orbit samples remain in JSON/CSV. No receiver range or receiver position is inferred from this orbit.",
    )
    finish(fig, output / "04_navigation_orbit")
