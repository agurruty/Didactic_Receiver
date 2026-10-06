# Galileo E1 + E5 · From samples to a position on Earth

The `feature/galileo-e5` branch adds signal-aware **E5a, E5b and synchronized
E1+E5 capture/processing** through `python galileo.py`. Start with the
[E1/E5 guide](docs/E5_RECEIVER.md) for hardware routing, storage requirements,
commands, official ESA/EUSPA references and the software-validation boundary.
E5 is experimental pending real-hardware and real-signal validation; no
centimetre-level or joint AltBOC performance is claimed.

```bash
python galileo.py capture --bands e1+e5b --duration 120 \
  --output recordings/e1-e5b-001 --plan
python galileo.py --help
```

The original E1 walkthrough below remains valid for 12.5 MS/s E1 recordings.
New multi-band recordings use per-band subdirectories and the new CLI.

Each numbered script answers one receiver question, saves its numerical evidence, and produces a figure. The live-tested correlation, pilot tracking, secondary-code synchronization, FEC and orbit algorithms live in `receiver/`; the entry scripts stay small. Use `--help` for the parameters and `--no-show` to save figures without opening windows. Figures are exported as PNG and PDF.

| Milestone | Question | Evidence saved |
| --- | --- | --- |
| `00_galileo_e1_capture.py` | What did the antenna and digitizer deliver? | SC16 + hardware metadata, full-record RMS/clipping statistics, waveform/PSD/histogram/stability figure |
| `01_galileo_e1_acquire.py` | Which satellite codes stand out, at what delay and Doppler? | All searched PRN scores, selected candidates' complete coarse/fine maps and tracking seeds, ranking and search figures |
| `02_galileo_e1_track.py` | Can the local carrier and code keep following one signal? | Every integration's correlators, loop residuals, code/carrier rates, phase concentration, C/N₀ estimate and recording sample indices; two figures |
| `03_galileo_e1_decode.py` | Can the symbols become intact navigation words? | Sync search, soft symbols, all attempted nominal pages including CRC failures, raw navigation/CRC/OSNMA/SAR bits and timing; telemetry figure |
| `04_galileo_e1_message.py` | What do those words say about the satellite? | CRC-valid field parsing, coherent ephemeris, orbit/clock CSV and status JSON; orbit dashboard or an explicit missing-evidence figure |
| `05_galileo_e1_position.py` | Can simultaneous code measurements locate the antenna? | Page-to-code timing, raw/corrected ranges, XYZ + clock, geometry, residuals, conditional covariance, ionosphere diagnostics and a street-map finale |

## Setup from a fresh clone

Use Linux with Bash and Python 3.11 or newer (the smoke checks use Python 3.13).

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/Didactic_Receiver.git
cd Didactic_Receiver
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run_receiver.sh --help
```

Replace `YOUR_USERNAME` with the repository owner. If already cloned, run
`git submodule update --init --recursive`. On Debian/Ubuntu, install
`python3-venv` if creating the environment fails because venv/ensurepip is missing.
`GNSS-DSP-tools/` is a pinned, unmodified upstream submodule; its license is
in `GNSS-DSP-tools/LICENSE`. Optional Numba can accelerate upstream routines.
For a new hardware capture, install UHD and its Python bindings separately and
ensure `python -c 'import uhd'` works in the interpreter used for capture.

No recordings, metadata, generated results, or map caches are distributed.
Supply your own matching SC16 capture and `metadata.json`, or use stage 00 to
capture them. `.gitignore` allows only the listed source/configuration paths;
add new source paths explicitly when extending the project. Keep all captures
under `recordings/` and all generated artifacts under `runs/`.

## Run an existing recording

Activate the environment created above. UHD is needed only for a new capture.

```bash
./run_receiver.sh recordings --output runs/first-look --top 1 --skip-position
# Process a larger shortlist from the SAME capture and attempt positioning:
./run_receiver.sh recordings --output runs/position-session --top 8
```

`PYTHON=/path/to/python ./run_receiver.sh ...` chooses the interpreter. The runner processes channels sequentially, with common original recording coordinates; stage 05 then constructs simultaneous measurements from their fractional code boundaries and decoded time anchors. The default shortlist is eight above-gate candidates. All usable channels enter positioning; acquisition rank alone does not choose the final geometry. It saves `index.html`, `run.json`, `channel_summary.json`, per-stage logs, plots and numerical artifacts. Open `index.html` locally. No web server is required. Processing works offline; use `--map-source none` to disable online street-map tiles.

Create `recordings/` and place your own `e1.sc16` and its matching `metadata.json` there before running. Do not pair an unrelated recording with existing metadata. A recording directory must contain exactly one `e1.sc16` or legacy `galileo_e1.sc16`, plus metadata declaring `sample_rate_sps` or `raw_sample_rate_sps`. The receiver currently requires **12.5 MS/s, E1 centered at 1575.42 MHz, little-endian signed int16 I/Q pairs**. Known stream errors and inconsistent sample counts are rejected.

Every output directory must be new. Existing captures/results are never overwritten. Exit codes: **0** stage completed, **3** insufficient signal/navigation evidence, **1** an input or execution error. A failed channel does not prevent the runner from attempting other selected channels. Stage logs explain the failure; output directories may contain partial results.

## Capture once

```bash
python3 00_galileo_e1_capture.py --output recordings/session-01 \
  --duration 30 --gain 90 --subdev A:0 --antenna RX1
```

`--args` sets the USRP address and transport configuration. `--subdev` selects one receive chain on either TwinRX daughterboard (`A:0`, `A:1`, `B:0`, `B:1`); this E1 teaching workflow records **one connected RF input**, not all four chains. Keep the GPSDO antenna/feed separate from the GNSS RF input. The capture waits for GPS/reference lock, schedules the start in device time, streams directly to disk, and records the applied tuning/gain. At 12.5 MS/s SC16 uses **50 MB/s**. Check disk throughput and free space before a long recording. Receive errors leave metadata marked invalid.

Duration controls both available tracking time and the chance of recovering a complete ephemeris; 30 s is a starting point, not a guarantee. Gain controls digital level and can introduce overload: the raw figure shows SC16 rails but cannot diagnose all analog overload. A noise-like waveform is normal for spread-spectrum GNSS.

Inspect an old capture without recapturing:

```bash
python3 00_galileo_e1_capture.py --input recordings --output runs/raw-inspection
```

The former `01_galileo_e1_diagnostic.py` is not part of this repository. Its combined capture/E1-B search has been replaced by stages 00 and 01. The new search uses **E1-C**, matching the tracker, so it can hand off the exact pilot acquisition instead of searching E1-B and then independently reacquiring E1-C.

## Walk through stages 00–05

Run these blocks **in order, in the same Bash terminal**, from the repository root. They use one capture and one run directory throughout. Close each figure window to continue, or append `--no-show` to a Python stage command to save its figures without opening them.

Choose the capture that actually contains your SC16 file and matching metadata. Use `capture=recordings/session-01` if you just made the capture above. Choose a fresh `run` directory for each walkthrough; the commands do not overwrite earlier results.

```bash
capture=recordings
run=runs/walkthrough-01
```

### 00 · Inspect the recorded signal

This reads the existing capture; it does not start another hardware recording. To make a new recording first, use [Capture once](#capture-once) above and point `capture` to that directory.

```bash
python3 00_galileo_e1_capture.py --input "$capture" --output "$run/00_signal"
```

Look at the waveform, spectrum and clipping statistics before continuing.

### 01 · Find candidate satellites

```bash
python3 01_galileo_e1_acquire.py "$capture" --output "$run/01_acquisition" --top 8
```

The ranking and search maps show which codes stand above the acquisition gate. Eight is a processing shortlist, not a requirement to use exactly eight satellites. Positioning needs at least four usable channels with suitable geometry, so start with more than four when the recording allows it.

Read the actual selected PRNs from acquisition, then choose the strongest for the first pass through stages 02–04:

```bash
mapfile -t prns < <(python3 -c 'import json, sys; print(*json.load(open(sys.argv[1]))["selected_prns"], sep="\n")' "$run/01_acquisition/acquisition.json")
printf 'Selected PRNs: %s\n' "${prns[*]}"
prn=${prns[0]}
channel=$(printf '%s/E%02d' "$run" "$prn")
```

Continue only if acquisition succeeded and the list is nonempty. You can set `prn` to another number in that list before setting `channel`. Acquisition saves above-threshold seeds, and tracking checks their recording identity. The identity includes path, size, modification time and hashes of the first/last MiB; it is not a full-file integrity hash. Move or modify the recording and you must reacquire.

### 02 · Track one satellite

```bash
python3 02_galileo_e1_track.py "$capture" --acquisition "$run/01_acquisition" \
  --prn "$prn" --output "$channel/02_tracking"
```

Inspect the carrier/code loops, constellation and C/N₀. Tracking saves the fractional code boundaries needed later for ranging.

### 03 · Decode its navigation pages

```bash
python3 03_galileo_e1_decode.py "$channel/02_tracking" \
  --output "$channel/03_telemetry"
```

Look for CRC-valid pages and time anchors. Tracking a signal does not guarantee enough navigation data was recovered.

### 04 · Interpret the broadcast message

```bash
python3 04_galileo_e1_message.py "$channel/03_telemetry/decoded_inav.json" \
  --output "$channel/04_navigation"
```

The dashboard shows the decoded ephemeris, predicted satellite orbit and clock, or explains what evidence is missing. This is the satellite's position; stage 05 estimates the antenna's position.

### Process the remaining candidate satellites

Stage 05 combines several satellites from the **same capture**. This loop repeats stages 02–04 for the remaining selected PRNs, leaving the channel just inspected in place. It saves their plots without opening windows and continues to another candidate if a stage fails; read the printed reason for each failure.

```bash
for other in "${prns[@]}"; do
  if [ "$other" = "$prn" ]; then continue; fi
  other_channel=$(printf '%s/E%02d' "$run" "$other")
  python3 02_galileo_e1_track.py "$capture" --acquisition "$run/01_acquisition" \
    --prn "$other" --output "$other_channel/02_tracking" --no-show || continue
  python3 03_galileo_e1_decode.py "$other_channel/02_tracking" \
    --output "$other_channel/03_telemetry" --no-show || continue
  python3 04_galileo_e1_message.py "$other_channel/03_telemetry/decoded_inav.json" \
    --output "$other_channel/04_navigation" --no-show || continue
done
```

### 05 · Form pseudoranges and estimate the antenna position

Pass the **run directory**, which contains all the `E##` folders:

```bash
python3 05_galileo_e1_position.py "$run" --output "$run/05_position" --interval 1
```

This attempts one solution per second over the available tracking coverage. No calibration file is required to attempt a solution: the receiver estimates a clock offset. Stage 05 now applies **broadcast NTCM-G ionosphere correction by default**, using decoded WT5/WT6 and the capture calendar metadata. Missing model inputs are reported rather than silently replaced with zero correction. Use `--ionosphere none` only for an explicitly uncorrected comparison. Do not fill calibration or VTEC arguments with guessed values. The [pseudorange and position details](#pseudorange-and-position) below explain those optional inputs.

Open `05_pseudoranges.png`, `05_position_geometry.png`, `05_ionosphere.png` and **`05_position_map.png`** in `$run/05_position/` (PDFs are saved too). The geometry and map figures are produced only when a position is accepted. The map shows streets, the final estimate, the accepted epoch estimates, a conditional 95% horizontal ellipse and a wider-area inset. If fewer than four channels satisfy timing, navigation and geometry requirements, the first dashboard explains why a fix is unavailable. Numerical results, rejected measurements and uncertainty are retained in JSON/CSV.

Your folders now follow the layout stage 05 expects:

```text
runs/walkthrough-01/
├── 00_signal/
├── 01_acquisition/
├── E##/                    # one folder per selected PRN
│   ├── 02_tracking/
│   ├── 03_telemetry/
│   └── 04_navigation/
└── 05_position/
```

To rerun only positioning, give it a new output directory. For example, restrict the receive epochs to seconds 10–120 of the original recording (choose an interval your capture actually covers):

```bash
python3 05_galileo_e1_position.py "$run" --output "$run/05_position-review" \
  --start 10 --stop 120 --interval 1 --no-show
```

The walkthrough saves each stage's plots in its output folder. For automatic processing plus an `index.html` gallery, use `run_receiver.sh` as shown in [Run an existing recording](#run-an-existing-recording), with a separate fresh output directory.

The most useful controls are:

- Acquisition: `--prns 11 26` narrows the search; `--dwells 20` sums twenty 4 ms coherent powers; `--doppler-max 7500`, `--doppler-step 250` and `--fine-step 25` control frequency coverage and cost. `--threshold-db 7` is a heuristic peak-to-map-floor gate, not a calibrated false-alarm probability. All searched PRNs remain in JSON even if only the top 16 fit the ranking plot.
- Tracking: `--secondary-sync-epochs 75` uses 300 ms to synchronize the 25-symbol pilot secondary code. The existing FLL/PLL/DLL settings are grouped near the top of `receiver/tracking.py`. The 1 s and 5 s transitions are scheduled bandwidth changes, not independent lock declarations.
- Telemetry: `--skip-seconds 0.4` discards initial tracking settling. Increase it if the early loop is unstable, while retaining enough complete pages to decode. Every 4 ms E1-B primary-code epoch is one **coded symbol**, not one navigation data bit.
- Navigation: `--span-minutes 90` plots broadcast-model predictions ±90 min around the evaluation epoch. `--tow` overrides that epoch in Galileo seconds of week; `--gst-week` supplies week context. Without a WT5/WT6 time anchor, the plot explicitly falls back to ephemeris reference time `toe`, not capture time.
- Positioning: `--interval 1` controls solution spacing; `--start` and `--stop` select receive times in seconds from recording start. `--output` must be new. `--ionosphere ntcm-g` is the default; `--ionosphere none` disables the correction for comparison. Optional calibration and external ionosphere inputs are described below.

## Read the figures honestly

**Acquisition:** noncoherent power is compared with the median map floor after excluding the peak neighborhood. This score is **not C/N₀**. Fine-grid delay is circular, modulo 4 ms; multiplying it by light speed does not produce a complete pseudorange. The map display max-pools columns to preserve a narrow peak at screen resolution; raw selected maps are saved unpooled. Code/Doppler cuts use the original bins, and the code cut shows BOC side peaks. The sinBOC(1,1) local replica is the original narrowband approximation to Galileo CBOC, not a full-band matched replica. False code-peak locks remain possible.

**Tracking:** raw residuals and transients remain visible. Different units have separate axes. Pilot/data constellations include the entire tracked interval; their color encodes time. C/N₀ is a finite-window coherent-pilot estimate, suppressed below 0.7 phase concentration, not a calibrated instrument reading. Oscillator mismatch contributes to apparent Doppler. The final summary uses the actual last 250 epochs, so an earlier high C/N₀ cannot disguise a later missing estimate. A separate data-symbol view exposes the variable integration lengths and all five VEML correlators.

**Telemetry:** page sync, interleaving, G2 inversion, soft Viterbi decoding and CRC each have a visible role. Before/after interleaver images use the same row-major display convention so the permutation is actually visible. The decoder forces a zero end state: zero tail bits are therefore not independent evidence of successful reception. CRC supplies an independent integrity check, but **does not authenticate the satellite**; OSNMA bits are retained without authentication. Alert pages are not decoded by this nominal-page path. Missing page pairings are visible in the page-part records. Raw sample indices and fractional start boundaries are carried to each decoded page, separately from decoder-relative time.

**Navigation:** only CRC-passing words are assembled, and WT1–4 must share an IODnav. Field scales and signed angles follow the [Galileo OS SIS ICD v2.2](https://www.gsc-europa.eu/sites/default/files/sites/all/files/Galileo_OS_SIS_ICD_v2.2.pdf), tables 42–47 and 67–70. Plausibility checks gate orbit propagation; they are not full navigation-solution integrity monitoring. The ECEF orbit is predicted from broadcast data, not measured by this receiver. The clock plot contains polynomial plus relativity; E1 BGD is parsed but not applied to this plotted clock model. Health, DVS and SISA are preserved, not silently treated as proof of ranging readiness. RedCED, RS-FEC2 and almanac coverage are reported; missing full ephemerides are never filled with nominal values. A missing/invalid ephemeris produces a readiness figure and exit 3.

## Pseudorange and position

The walkthrough above reaches this milestone directly; the runner also attempts it after processing the selected channels. For an existing runner output named `runs/position-session`, you can repeat only stage 05 with:

```bash
python3 05_galileo_e1_position.py runs/position-session \
  --output runs/position-refined --interval 1 --no-show
```

The input directory contains `E##/02_tracking/{metadata.json,continuous_tracking.npz}` and `E##/03_telemetry/decoded_inav.json`. At least four distinct satellites from the same recording need valid ephemerides, health, SISA and overlapping, time-anchored tracking. A missing fix produces an explanatory figure and exit 3. Old artifacts without fractional **start** boundaries and recording identities must be regenerated.

### Put an existing solution on the map

The geographic finale runs automatically with stage 05. For results you already computed, use **map-only mode**: its input is the folder containing `position_solution.json`, and it never reruns tracking, decoding or the position solver.

```bash
python3 05_galileo_e1_position.py runs/position --map-only \
  --output runs/position-map --no-show
```

For the walkthrough's directory layout, the equivalent is:

```bash
python3 05_galileo_e1_position.py "$run/05_position" --map-only \
  --output "$run/05_map-review" --map-width 600 --no-show
```

`--map-width` is the minimum detail-view width in ground metres (default 1200). The viewport expands if needed to include all accepted estimates and the final ellipse. The inset gives wider geographic context. Coordinates use Web Mercator; the ellipse is constructed in local east/north metres and transformed into the map projection, and the scale bar accounts for local map distortion.

The figure distinguishes the **last accepted estimate**, the cluster of epoch estimates, and the **conditional 95% horizontal uncertainty ellipse**. Its displayed dimensions are **semi-axes**, not full widths. Neither a tight cluster nor a small ellipse establishes accuracy against the true antenna location. With four satellites there is no residual redundancy; residual ionosphere-model errors and other biases can move the entire cluster. Actual error is shown as unknown unless you supply an independent reference.

If you have an independently established antenna latitude/longitude, add `--map-reference LAT LON --map-reference-source "reference provenance"`. These are degrees, in latitude-then-longitude order. The map adds a reference marker and reports horizontal offset from it; reference-coordinate uncertainty is not included in the GNSS ellipse. A point guessed from a map is not surveyed truth. The reference changes only the presentation, never the solver.

OpenStreetMap tiles require internet access on first use. Only the two displayed viewports are requested, with visible attribution and a persistent seven-day cache in `runs/.map-cache/osm`; `--map-cache` selects another cache directory. See the [OSM tile policy](https://operations.osmfoundation.org/policies/tiles/). No API key is required. Network failures leave a clearly marked incomplete basemap and preserve the numerical solution. `--map-source none` disables tile requests. These map options also work with `run_receiver.sh`.

In addition to PNG/PDF, the output contains `position_map.json` (projection, ellipse, reference and tile provenance), `position_map.geojson` (estimates and uncertainty for a GIS) and `open_in_openstreetmap.html` (a link to pan/zoom around the final coordinate in a browser; the external map shows the point, not the uncertainty ellipse). Choose a new output directory for each rendering.

### How time becomes range

CRC-valid WT5 supplies GST week/TOW; WT6 adds TOW anchors with week resolved from nearby WT5. TOW refers to the beginning of the transmitted page, including its synchronization symbols. Each anchor is matched to a saved fractional code boundary. Counted 4 ms primary-code periods resolve the integer-period ambiguity within an uninterrupted tracking segment; interpolation supplies only the fractional period at a common receive sample. Conflicting anchors, lock/C/N₀ failures and discontinuities break this connection. The implementation never extrapolates across an unanchored gap. It cannot prove the absence of every undetected false code lock.

All channels are evaluated at identical original recording sample numbers. The default receiver clock has a navigation-aligned, arbitrary origin: median extrapolated transmit epoch plus 80 ms. **80 ms is a coordinate convention, not an assumed propagation time or an ambiguity solution.** Changing it adds the same constant to every pseudorange and the fitted receiver clock; XYZ is invariant. The fitted clock then yields a navigation-derived receive GST estimate. The sample interval uses the declared 12.5 MS/s; a new clock offset is solved at every receive epoch rather than assuming the oscillator remains perfectly synchronized.

`first_sample_device_time_s` stays a device timestamp. There is no assumed GPSDO-to-GST, Unix-epoch or leap-second conversion. Position timestamps retain broadcast GST week modulo 4096. For NTCM-G, the full week era is resolved separately from the capture `created_utc` calendar context, and model UTC is computed from decoded WT6 at the solved receive GST. The host date selects only the era, never the precise receive time; its week must agree within one week. If that metadata is unavailable or incorrect, `--gst-era 0` explicitly selects the first 4096-week era (starting in 1999); only use an era established for your capture. The chosen source is preserved in `clock.ionosphere_calendar`. Without common hardware-delay calibration, inferred receive time refers to antenna arrival, with common path delay absorbed into the clock. Differential delays can bias position.

### Corrections and what remains uncertain

The observation equation is `Praw + c·δtSV(E1) − hardware − tropo − iono − Shapiro = geometric_range + receiver_clock_metres`. E1 satellite clock uses the I/NAV polynomial, eccentric-orbit relativity and **minus BGD(E1,E5b)**. Satellite coordinates are evaluated at corrected transmit GST and rotated into reception ECEF using geometric propagation time, avoiding use of the clock-biased raw pseudorange for Earth rotation. Troposphere uses standard-atmosphere Saastamoinen zenith terms and Black–Eisner mapping; local weather is not measured.

**The default is NTCM-G v1.0**, the Galileo broadcast single-frequency ionosphere model described by the [European GNSS Service Centre](https://www.gsc-europa.eu/sites/default/files/NTCM-G_Ionospheric_Model_Description_-_v1.0.pdf). No new RF recording or external TEC download is needed when your decoded pages already contain valid WT5 coefficients and WT6 time parameters.

The model combines `ai0/ai1/ai2`, UTC day/time and receiver-to-satellite geometry. It predicts vertical electron content at each signal's 450 km ionospheric pierce point, maps it to slant electron content, and converts it into E1 code delay. This delay is recomputed during position iteration and **subtracted** from the raw-range model. The coefficients are not treated as VTEC, and the implementation is NTCM-G, not NeQuick-G or dual-frequency cancellation.

Only CRC-valid pages from sources with usable E1 health/data-validity flags are eligible. At each receive epoch, all satellites use one shared coefficient set from the latest decoded WT5 at or before that epoch. Before the first decoded page, offline processing uses the earliest available page, explicitly marked `offline_backfill` with a negative page age. WT5/WT6 ages are limited to four hours; the UTC polynomial reference is limited to fourteen days. All-zero coefficients, missing calendar context or missing UTC data reject the modelled solution rather than inventing a correction. The five reserved disturbance flags are preserved, not interpreted as an accuracy guarantee. UTC conversion explicitly rejects the ±6-hour interval around an actual broadcast leap-second change; it does not silently approximate the leap second.

**Correction and uncertainty are separate.** `--iono-model-sigma 10` retains a 10 m assumed residual zenith standard uncertainty after modelling, mapped to each path. This is a configurable processing assumption, not an NTCM-G accuracy claim or a bound on disturbed conditions. Applying a model does not justify shrinking the ellipse without evidence; spatially correlated model errors remain outside the independent-error covariance assumption.

For your existing directory containing `runs/E04`, `runs/E21`, etc., rerun only positioning into a fresh directory:

```bash
python3 05_galileo_e1_position.py runs --output runs/position-ntcm-review \
  --ionosphere ntcm-g --no-show
```

The input is the parent of the `E##` folders, **not** the old `runs/position` result folder. The previously computed result is left intact. To reproduce the earlier uncorrected mode:

```bash
python3 05_galileo_e1_position.py runs --output runs/position-uncorrected-review \
  --ionosphere none --no-show
```

The NTCM-G run also computes an uncorrected comparison using the **same selected satellites and ionosphere sigmas**, so its `05_ionosphere` figure can show the delay per satellite, vertical versus slant TEC and the correction-induced ENU shift. This comparison does not use known antenna coordinates or claim an accuracy improvement. Its states, shifts and residual-check status are retained in each epoch's `ionosphere_comparison`; UTC, coefficients, page provenance, pierce points and delays remain in the numerical artifacts.

External alternatives remain available. Choose one instead of `--ionosphere`:

- `--ionosphere-csv path.csv --ionosphere-source "model/data provenance"`: columns `prn,receive_time_s,delay_m,sigma_m`, with nonnegative E1 slant code delay in metres and positive uncertainty. Time is seconds from recording start. Interpolation requires bracketing entries within 120 s; missing satellites and extrapolation are rejected. Corrections must match the capture and its satellite lines of sight.
- `--vtec VALUE --vtec-sigma SIGMA --ionosphere-source "measurement/model provenance"`: externally determined VTEC and uncertainty in TECU. A 450 km thin-shell approximation treats this VTEC as constant over the run and region. Use it only where that approximation is justified; it is not a global or broadcast ionosphere model.

The numerical model follows the [Galileo OS SIS ICD](https://www.gsc-europa.eu/sites/default/files/sites/all/files/Galileo_OS_SIS_ICD_v2.2.pdf), [ESA satellite-coordinate guidance](https://gssc.esa.int/navipedia/index.php/Satellite_Coordinates_Computation) and [tropospheric-delay guidance](https://gssc.esa.int/navipedia/index.php/Tropospheric_Delay). External atmospheric inputs retain their source in the output.

### Optional measured calibration

`--calibration calibration.json` accepts independently established time/path calibration. Omit unavailable fields; never replace them with guessed values. The schema is:

```text
source: description of the measurement and reference plane
first_sample_gst: {week: GST week modulo 4096, tow_s: seconds of week, uncertainty_s: standard uncertainty}
common_delay_ns: measured positive antenna/cable/receiver delay to the sample reference plane
common_delay_sigma_ns: its standard uncertainty
differential_delay_ns: {"PRN": additional per-satellite delay in ns}
differential_delay_sigma_ns: {"PRN": its standard uncertainty in ns}
recording: optional exact recording identity object from tracking metadata
```

Use valid JSON with numeric values and remove absent fields. A calibrated GST origin still leaves a residual clock unknown. Positive calibrated hardware delay is subtracted from raw range; unknown common delay remains inseparable from the clock. Common calibration uncertainty is reported separately from formal XYZ/clock covariance. Differential uncertainties enter measurement weights. Per-satellite delay calibration must have a physical basis; it is not a way to force a desired position.

### Read the final dashboards

`05_ionosphere` explains the broadcast-model correction and its effect on the estimate. `05_pseudoranges` connects common epochs, raw ranges, additive corrections, receiver clock and PDOP. `05_position_geometry` shows the estimated antenna location, satellite sky geometry, ENU scatter, a conditional 95% horizontal ellipse and residuals. Height is ellipsoidal. Scatter is relative to the first accepted estimate, not surveyed truth; the globe is an illustrative sphere, not a map or geoid model. `05_position_map` provides the street-level context and wider-area inset, using the same saved final estimate and covariance.

Weighted least squares estimates XYZ and clock from **all** usable satellites. Elevation and PDOP gates protect against unsuitable geometry. Residual consistency uses assumed independent range errors; a unique redundant leave-one-out solution may exclude one measurement when at least six are available. Ambiguous exclusions reject the epoch. Exactly four satellites provide no residual redundancy, even if their residuals are essentially zero. CRC is not authentication, and these checks are not certified RAIM. Correlated multipath, BOC side-peak locks, atmosphere-model errors and differential hardware biases can exceed the displayed formal uncertainty.

`position_solution.json` preserves clock conventions, calibration, model settings, per-epoch covariance, timing anchors, rejections and residuals. `pseudoranges.json` / `.csv` retain raw observations, individual corrections and use/exclusion flags; `position.csv` exposes the solution timeline. Rejected computed fits and excluded-measurement residuals remain available. Each output directory must be new. The full runner retains a nonzero status if any requested channel or milestone fails, even when other channels yield a position.

## Verification

```bash
python3 -m unittest discover -s tests -v
```

The ionosphere tests include twelve published NTCM-G Annex D cases spanning high, medium and low solar activity, polar and equatorial locations, and different local times. They match the published STEC rounding precision. Additional checks cover calendar-era provenance, GST–UTC polynomial and week rollover, explicit leap-transition rejection, stale/missing data, coherent coefficient selection, offline backfill, correction sign and receiver-clock-origin invariance. A synthetic range-model test recovers known XYZ/clock after applying the correction.

The existing four-satellite recording was reprocessed with NTCM-G: 178/178 epochs were accepted, with modelled E1 delays of approximately 3.65–5.95 m. The final estimate shifted about 0.85 m horizontally and 3.87 m downward relative to the uncorrected fit. These are changes in estimated position, not measured accuracy improvements; no surveyed reference was supplied, and four satellites still provide no residual redundancy.
