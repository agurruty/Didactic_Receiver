"""Single-band and paired E1/E5 solutions with explicit synchronization gates."""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from .common import write_json, heading, finish
from .observables import load_channel, observation, week_delta, WEEK
from .position_models import screened_solution, RejectedSolution, satellite_state
from .positioning import serial, csv_write
from .dual_frequency import combine
from .ionosphere import BroadcastIonosphere


def validate_session(channels):
    """Different recordings are never paired merely because their lengths match."""
    ids = {c["capture_metadata"].get("session_id") for c in channels}
    if None in ids or len(ids) != 1:
        raise ValueError("E1/E5 requires a common timed-capture session_id.")
    starts = {
        c["capture_metadata"].get("scheduled_start_device_time_s") for c in channels
    }
    if None in starts or len(starts) != 1:
        raise ValueError("E1/E5 channels do not share a scheduled device start.")
    for c in channels:
        meta = c["capture_metadata"]
        if (
            meta.get("valid") is not True
            or meta.get("synchronization") != "single timed multi-channel UHD streamer"
        ):
            raise ValueError(
                "E1/E5 requires valid synchronized multi-channel capture metadata."
            )
        if meta.get("session_start_offset_s") != 0:
            raise ValueError(
                "Nonzero/unknown recording start offsets are not supported for dual-frequency pairing."
            )
        if (
            abs(
                meta["first_sample_device_time_s"]
                - meta["scheduled_start_device_time_s"]
            )
            > 1e-6
        ):
            raise ValueError(
                "Actual first-sample device time disagrees with the common scheduled start."
            )


def fnav_clock_for_e1(e1, e5_channel):
    """Use the F/NAV E1/E5a clock for BOTH members of that combination (ICD table 71)."""
    from .navigation import wrap_week_time

    tx = e1["transmit_sv_s"] % WEEK
    candidates = [e for e in e5_channel["ephemerides"] if e["IODnav"] == e1["iodnav"]]
    if not candidates:
        raise ValueError("No matched F/NAV IODnav for E1/E5a clock correction.")
    ephem = min(candidates, key=lambda e: abs(wrap_week_time(tx - e["toe_s"])))
    sample = e1["receive_time_s"] * e5_channel["sample_rate_sps"]
    health = min(
        e5_channel["health_pages"], key=lambda p: abs(p["boundary_sample"] - sample)
    )
    bgd = health["decoded"]["BGD_E1_E5a_s"]
    sat, clock, state, gst = satellite_state(ephem, tx, bgd)
    return dict(
        e1,
        satellite_ecef_m=sat,
        satellite_clock_e1_s=clock,
        bgd_s=bgd,
        transmit_gst_tow_s=gst,
        clock_polynomial_s=state["clock_polynomial_s"],
        relativity_s=state["relativistic_correction_s"],
    )


def position(args):
    from .position_map import validate_map_arguments

    validate_map_arguments(args)
    if (
        not math.isfinite(args.min_cn0)
        or not math.isfinite(args.elevation_mask)
        or not 0 <= args.elevation_mask < 90
    ):
        raise ValueError("Finite C/N0 and elevation mask in [0,90) required.")
    if not math.isfinite(args.interval) or args.interval <= 0:
        raise ValueError("Interval must be finite and positive.")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (args.code_sigma_e1, args.code_sigma_e5, args.max_pdop)
    ):
        raise ValueError(
            "Code noise assumptions and PDOP gate must be finite and positive."
        )
    config = dict(
        min_lock=0.7,
        min_cn0_dbhz=args.min_cn0,
        max_ephemeris_age_s=14400.0,
        elevation_mask_deg=args.elevation_mask,
        max_pdop=args.max_pdop,
        code_sigma_m=args.code_sigma_e1,
        iono_model_sigma_m=10.0,
        higher_order_iono_sigma_m=0.1,
    )
    channels = []
    rejected = []
    for directory in sorted(set(args.run.glob("*/E*")) | set(args.run.glob("E*"))):
        if not (directory / "02_tracking/metadata.json").exists():
            continue
        try:
            peers = []
            band = directory.parent.name
            if band in ("e1", "e5b"):
                other = "e5b" if band == "e1" else "e1"
                peer = (
                    args.run / other / directory.name / "03_telemetry/decoded_inav.json"
                )
                if peer.exists():
                    candidate = json.loads(peer.read_text())
                    local_meta = json.loads(
                        (directory / "02_tracking/metadata.json").read_text()
                    )
                    try:
                        validate_session(
                            [
                                {
                                    "capture_metadata": local_meta.get(
                                        "capture_metadata", {}
                                    )
                                },
                                {
                                    "capture_metadata": candidate.get(
                                        "capture_metadata", {}
                                    )
                                },
                            ]
                        )
                    except ValueError:
                        pass  # An unrelated session never contributes navigation pages.
                    else:
                        peers.append(candidate)
            channels.append(load_channel(directory, config, peers))
        except (ValueError, KeyError, OSError) as exc:
            rejected.append(dict(channel=str(directory), reason=str(exc)))
    keys = [(c["band"], c["prn"]) for c in channels]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate PRN/band observations.")
    available = {c["band"] for c in channels}
    requested = args.mode
    if requested == "auto":
        requested = (
            "e1+e5b"
            if {"e1", "e5b"} <= available
            else (
                "e1+e5a"
                if {"e1", "e5a"} <= available
                else next((b for b in ("e5b", "e5a", "e1") if b in available), "e5b")
            )
        )
    bands = requested.split("+")
    dual = len(bands) == 2
    channels = [c for c in channels if c["band"] in bands]
    if dual and channels:
        validate_session(channels)
    for band in bands:
        identities = [c["recording"] for c in channels if c["band"] == band]
        if identities and any(r != identities[0] for r in identities):
            raise ValueError("Same-band channels belong to different recordings.")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = channels[0]["week"] if channels else 0
    epochs = []
    ranges = []
    combinations = []
    origin = None
    broadcast = None
    broadcast_error = None
    if channels:
        origins = [
            s["origin_s"]
            + week_delta(c["week"], reference) * WEEK
            - c["boundaries"][s["start"]] / c["sample_rate_sps"]
            + c["primary_period_s"] * s["start"]
            for c in channels
            for s in c["segments"]
        ]
        if np.ptp(origins) > 0.5:
            raise ValueError("Navigation transmit times disagree by more than 0.5 s.")
        origin = float(np.median(origins) + 0.080)
        if not dual and args.ionosphere == "ntcm-g":
            try:
                broadcast = BroadcastIonosphere(channels, reference, args.gst_era)
            except ValueError as exc:
                broadcast_error = str(exc)
        start = max(
            0.0,
            min(
                c["boundaries"][s["start"]] / c["sample_rate_sps"]
                for c in channels
                for s in c["segments"]
            ),
        )
        stop = max(
            c["boundaries"][s["stop"] - 1] / c["sample_rate_sps"]
            for c in channels
            for s in c["segments"]
        )
        start = math.ceil(start / args.interval) * args.interval
        count = max(0, math.floor((stop - start) / args.interval) + 1)
        if count > 1_000_000:
            raise ValueError("Too many solution epochs; increase --interval.")
        initial = None
        for index in range(count):
            time = start + index * args.interval
            by_band = {b: {} for b in bands}
            failures = []
            for c in channels:
                try:
                    o = observation(
                        c, time * c["sample_rate_sps"], reference, origin, config
                    )
                    o["code_sigma_m"] = (
                        args.code_sigma_e1 if c["band"] == "e1" else args.code_sigma_e5
                    )
                    by_band[c["band"]][c["prn"]] = o
                except (ValueError, KeyError) as exc:
                    failures.append(dict(prn=c["prn"], band=c["band"], reason=str(exc)))
            obs = []
            if dual:
                for prn in sorted(set(by_band["e1"]) & set(by_band[bands[1]])):
                    try:
                        first = by_band["e1"][prn]
                        if bands[1] == "e5a":
                            other = next(
                                c
                                for c in channels
                                if c["band"] == "e5a" and c["prn"] == prn
                            )
                            first = fnav_clock_for_e1(first, other)
                        o, diagnostic = combine(
                            first,
                            by_band[bands[1]][prn],
                            args.code_sigma_e1,
                            args.code_sigma_e5,
                        )
                        obs.append(o)
                        combinations.append(diagnostic)
                    except ValueError as exc:
                        failures.append(dict(prn=prn, reason=str(exc)))
            else:
                obs = list(by_band[bands[0]].values())
            row = dict(
                receive_time_s=time,
                input_rejections=failures,
                available_prns=[o["prn"] for o in obs],
            )
            try:
                if broadcast_error:
                    raise ValueError(broadcast_error)
                if broadcast is not None:
                    context = broadcast.at(time, 1.0)
                    for o in obs:
                        o["broadcast_iono"] = context
                row.update(screened_solution(obs, config, initial))
                if row["accepted"]:
                    initial = row["state"]
            except RejectedSolution as exc:
                row.update(exc.result)
            except (ValueError, np.linalg.LinAlgError) as exc:
                row.update(accepted=False, reason=str(exc))
            details = {d["prn"]: d for d in row.get("observations", [])}
            for o in obs:
                ranges.append(
                    {
                        **o,
                        **details.get(o["prn"], {}),
                        "used_in_accepted_solution": bool(
                            row["accepted"] and o["prn"] in details
                        ),
                    }
                )
            epochs.append(row)
    mode = (
        "E1/E5 first-order ionosphere-free code combination"
        if dual
        else (
            "broadcast NTCM-G scaled to signal frequency"
            if args.ionosphere == "ntcm-g"
            else "uncorrected ionosphere"
        )
    )
    summary = dict(
        schema_version=2,
        mode=requested,
        status="position estimates available"
        if any(e["accepted"] for e in epochs)
        else "insufficient consistent evidence",
        accepted_epochs=sum(e["accepted"] for e in epochs),
        attempted_epochs=len(epochs),
        epochs=epochs,
        channel_rejections=rejected,
        navigation_fusion="CRC-valid same-session E1/E5b orbit and clock words may complete a common IODnav; time anchors stay local",
        configuration=config,
        corrections=dict(ionosphere=mode),
        clock=dict(
            reference_gst_week=reference,
            receiver_clock_at_sample_zero_s=origin,
            device_timestamp_interpreted=False,
        ),
        limitations=[
            "Hardware capture and real-signal E5 accuracy require field validation.",
            "IF combinations amplify code noise; assumed code sigmas are configurable, not measured accuracy.",
            "Unknown common RF delays enter the solved clock; direction-dependent inter-frequency antenna/multipath delays can bias XYZ.",
            "Formal covariance assumes independent errors. No carrier-phase ambiguity fixing, PPP, RTK, OSNMA validation, or joint AltBOC tracking is claimed.",
        ],
    )
    write_json(args.output / "position_solution.json", serial(summary))
    write_json(args.output / "pseudoranges.json", serial(ranges))
    write_json(args.output / "dual_frequency.json", serial(combinations))
    csv_write(
        args.output / "position.csv",
        serial(epochs),
        [
            "receive_time_s",
            "accepted",
            "reason",
            "latitude_deg",
            "longitude_deg",
            "ellipsoidal_height_m",
            "pdop",
            "residual_rms_m",
        ],
    )
    csv_write(
        args.output / "dual_frequency.csv",
        serial(combinations),
        [
            "receive_time_s",
            "prn",
            "pair",
            "geometry_free_code_m",
            "apparent_e1_iono_m",
            "ionosphere_free_range_m",
            "combined_code_sigma_m",
        ],
    )
    from .position_figures import position_figures

    position_figures(summary, ranges, args.output)
    if dual:
        combination_figure(combinations, args.output)
    from .position_map import render_map

    render_map(summary, args.output, args)
    return 0 if summary["accepted_epochs"] else 3


def combination_figure(rows, output):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for prn in sorted({r["prn"] for r in rows}):
        rr = [r for r in rows if r["prn"] == prn]
        t = [r["receive_time_s"] for r in rr]
        ax[0, 0].plot(t, [r["geometry_free_code_m"] for r in rr], label=f"E{prn:02}")
        ax[0, 1].plot(t, [r["apparent_e1_iono_m"] for r in rr])
    ax[0, 0].set(
        title="What differs between simultaneous bands?",
        xlabel="Capture time [s]",
        ylabel="P(E5) − P(E1), clock corrected [m]",
    )
    ax[0, 1].set(
        title="Apparent E1 ionospheric delay",
        xlabel="Capture time [s]",
        ylabel="Code-derived delay including biases [m]",
    )
    if rows:
        r = rows[0]
        ax[1, 0].bar(["E1 weight", "E5 weight"], [r["alpha"], r["beta"]])
        text = f"Equal-noise amplification: {r['equal_noise_amplification']:.2f}×\nAssumed combined code sigma: {r['combined_code_sigma_m']:.2f} m"
        ax[0, 0].legend(ncol=4)
    else:
        text = "No simultaneous, healthy, time-anchored E1/E5 pairs."
    ax[1, 0].set(title="Cancel 1/f² delay with signed weights", ylabel="Coefficient")
    ax[1, 1].axis("off")
    ax[1, 1].text(
        0.02,
        0.9,
        text
        + "\n\nFirst-order ionosphere cancels.\nCode noise and multipath do not.\nReceiver inter-frequency delay contaminates\nabsolute TEC inferred from code alone.",
        va="top",
    )
    heading(
        fig,
        "05",
        "What does a second frequency buy us?",
        "Same satellite · same receive epoch · explicit clock convention",
        "These are code observations, not carrier-phase precision. The plot reports apparent ionosphere, not calibrated electron content.",
    )
    finish(fig, output / "05_dual_frequency")
