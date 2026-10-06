"""Shared artifact contracts, CLI setup and plot presentation."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FS = 12_500_000.0
TEAL, GOLD, RED, INK, MUTED = "#087f8c", "#db9b32", "#bf4b45", "#183247", "#71818e"


def launch(main):
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--no-show", action="store_true")
    opts, remaining = p.parse_known_args()
    sys.argv[1:] = remaining
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/didactic-matplotlib")
    if opts.no_show or not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#f5f7fa",
            "axes.facecolor": "white",
            "axes.edgecolor": "#dce3e9",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.titlepad": 12,
            "grid.alpha": 0.25,
            "grid.color": "#aebbc7",
            "lines.linewidth": 1.3,
            "axes.prop_cycle": cycler(color=[TEAL, GOLD, "#6574ad", RED]),
            "legend.frameon": False,
            "savefig.dpi": 160,
            "image.cmap": "viridis",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


def recording(directory):
    directory = Path(directory).expanduser().resolve()
    meta_path = directory / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rate = meta.get("raw_sample_rate_sps", meta.get("sample_rate_sps"))
    if rate is None:
        raise ValueError("Recording metadata must declare sample_rate_sps or raw_sample_rate_sps.")
    if not math.isfinite(float(rate)) or abs(float(rate) - FS) > 0.001:
        raise ValueError(f"This receiver requires 12.5 MS/s; metadata declares {rate}.")
    if meta.get("valid") is False or meta.get("errors"):
        raise ValueError("Recording metadata reports stream errors; continuity is required.")
    files = [directory / name for name in ("e1.sc16", "galileo_e1.sc16") if (directory / name).exists()]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one e1.sc16 or galileo_e1.sc16 in {directory}.")
    raw = files[0]
    frequency = meta.get("actual_center_frequency_hz")
    for channel in meta.get("channels", []):
        if channel.get("file") == raw.name:
            frequency = channel.get("actual_center_frequency_hz", frequency)
    if frequency is not None and (
        not math.isfinite(float(frequency)) or abs(float(frequency) - 1_575_420_000.0) > 1
    ):
        raise ValueError("This receiver requires an E1 recording centered at 1575.42 MHz.")
    if raw.stat().st_size == 0 or raw.stat().st_size % 4:
        raise ValueError("SC16 must contain complete little-endian int16 I/Q pairs.")
    expected = meta.get("captured_samples", meta.get("samples_per_channel"))
    if expected is not None and int(expected) * 4 != raw.stat().st_size:
        raise ValueError("SC16 file length does not match recording metadata.")
    return raw, meta


def fingerprint(path):
    """Identity plus a bounded content digest, without rereading multi-GB captures."""
    path = Path(path).resolve()
    st = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        digest.update(f.read(1048576))
        f.seek(max(0, st.st_size - 1048576))
        digest.update(f.read(1048576))
    return {
        "path": str(path),
        "bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "edge_sha256": digest.hexdigest(),
        "sample_rate_sps": FS,
    }


def heading(fig, stage, title, result, note):
    fig.suptitle(f"{stage}   /   {title}\n{result}", fontsize=18, x=0.04, ha="left", y=0.985)
    fig.text(0.04, 0.025, note, fontsize=9, color=MUTED, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.84, bottom=0.14, hspace=0.65, wspace=0.35)


def finish(fig, path):
    import matplotlib.pyplot as plt

    path = Path(path)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    print(f'Figure: {path.with_suffix(".png")}')
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
