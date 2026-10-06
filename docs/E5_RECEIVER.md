# Galileo E1 + E5: a measured, inspectable receiver

This feature lives in `feature/galileo-e5` in the existing repository. There is
one signal-aware tracking engine and one orbit/position model. The six original
E1 scripts and `run_receiver.sh` remain available for existing recordings.

The new entry point is `python galileo.py`. Every processing stage preserves
numerical artifacts and generates PNG/PDF teaching figures. `run` creates a
local HTML gallery. Recordings and generated results remain ignored by Git.

## Supported processing and validation boundary

| Mode | RF centre | Processing |
| --- | ---: | --- |
| E1 | 1575.420 MHz | Existing sinBOC(1,1) E1-C pilot tracking and E1-B I/NAV |
| E5a | 1176.450 MHz | E5a-Q pilot, E5a-I secondary wipeoff, 20 ms F/NAV symbols |
| E5b | 1207.140 MHz | E5b-Q pilot, E5b-I secondary wipeoff, 4 ms I/NAV symbols |
| E1 + E5b | Two simultaneous RF channels | Same-PRN, same-epoch ionosphere-free code positioning; shared I/NAV ephemeris assembly |
| E1 + E5a | Two simultaneous RF channels | Same-PRN, same-epoch ionosphere-free code positioning using the matched F/NAV clock |
| Full-band E5 | 1191.795 MHz, 100 MS/s | RF capture and extraction of E5a/E5b for independent sideband processing |

**E5 support is an experimental implementation, not a field-validated accuracy
claim.** Deterministic RF tests exercise acquisition and tracking against known
Doppler and fractional delay. Independently encoded navigation vectors test
FEC/CRC and timing. Mock UHD tests exercise synchronized writing and timestamp
gaps. Existing recorded E1 artifacts are used for regression. Actual X300 E5
capture, over-the-air E5 navigation, and dual-frequency position accuracy still
need validation on your hardware against independent truth.

Full-band capture does **not** imply joint AltBOC correlation/tracking. Neither
PPP, RTK, carrier-phase ambiguity resolution nor OSNMA verification is implemented.
E1 retains its original simplified sinBOC replica and associated side-peak risk.
The current E5a path processes the conventional I/Q components; it does not exploit
the E5a quasi-pilot extension introduced in ICD 2.2.

## Recommended first experiment: E1 + E5b

E5b shares the I/NAV format with E1. This makes it the clearest first comparison
and lets complementary CRC-valid orbit/clock words complete the same IODnav set.
It still requires an independently decoded time anchor and healthy tracking on
each participating signal. E5a adds F/NAV and is also available.

Use an environment that contains both the dependencies in `requirements.txt`
and UHD's Python bindings for hardware capture. Offline processing does not
require UHD. Check `python -c 'import uhd'` in the capture environment.

First inspect the plan; this does not access hardware or create the output:

```bash
python galileo.py capture --bands e1+e5b --duration 120 \
  --subdev 'A:0 A:1' --antennas RX1 RX1 \
  --output recordings/e1-e5b-001 --plan
```

Then record, using the antenna routing that matches the physical setup:

```bash
python galileo.py capture --bands e1+e5b --duration 120 \
  --subdev 'A:0 A:1' --antennas RX1 RX1 \
  --output recordings/e1-e5b-001

python galileo.py run recordings/e1-e5b-001 --bands e1+e5b \
  --output runs/e1-e5b-001 --top 8 --map-source none --no-show
```

Open `runs/e1-e5b-001/index.html`. Positioning defaults to E1+E5b when both bands
have usable channels. At least four **paired satellites**, not four total signal
channels, are required. Each run/output directory must be new.

With one antenna connected to TwinRX RX1, `RX1 RX1` routes that input to both
channels. If the antenna feed is externally distributed to both inputs, use
`RX1 RX2`. TwinRX's internal shared-input routing changes gain behavior and
path delay: the two RF channels share some front-end components. Read back
settings and stream continuity are recorded, but RF delay/antenna calibration
is not inferred. Select appropriate gain for the actual feed and inspect
clipping; the default 90 dB comes from the original E1 setup, not an E5 calibration.
Keep the GPSDO reference feed separate from the navigation RF input.

| Capture | Rate per channel | SC16 payload | Storage for 120 s |
| --- | ---: | ---: | ---: |
| E1 only, default | 12.5 MS/s | 50 MB/s | 6 GB |
| E5a or E5b only | 25 MS/s | 100 MB/s | 12 GB |
| E1 + E5a or E5b | 25 MS/s | 200 MB/s | 24 GB |
| Full-band E5 | 100 MS/s | 400 MB/s | 48 GB |
| E1 + full-band E5 | 100 MS/s | 800 MB/s | 96 GB |

Units are decimal; processing artifacts need additional space. 10 GbE alone
is not proof of sustained disk/host throughput. The capture refuses insufficient
free space, unexpected applied RF/rate settings, stream errors, absent receive
timestamps, and discontinuous sample timing. Interrupted captures are marked
invalid. Run a short capture and inspect it before committing to a long session.

## E5-only and E5a

```bash
python galileo.py capture --bands e5b --duration 120 \
  --subdev A:0 --antennas RX1 --output recordings/e5b-001
python galileo.py run recordings/e5b-001 --bands e5b \
  --output runs/e5b-001 --map-source none --no-show

python galileo.py capture --bands e1+e5a --duration 120 \
  --subdev 'A:0 A:1' --antennas RX1 RX1 --output recordings/e1-e5a-001
python galileo.py run recordings/e1-e5a-001 --bands e1+e5a \
  --output runs/e1-e5a-001 --map-source none --no-show
```

For E5a alone, use `--bands e5a` in both commands. F/NAV pages last 10 seconds;
short captures may not contain the complete clock/ephemeris set. A 120-second
capture gives more opportunity but does not guarantee successful navigation.

## Walk through the evidence

```bash
capture=recordings/e1-e5b-001
python galileo.py inspect "$capture" --band e5b --output runs/e5b-inspection --no-show
python galileo.py acquire "$capture" --band e5b --output runs/e5b-acquisition --top 8 --no-show
```

Read `selected_prns` in `runs/e5b-acquisition/acquisition.json`, then choose an
actually detected PRN. The example below uses PRN 11 only as a placeholder:

```bash
prn=11
python galileo.py track "$capture" --acquisition runs/e5b-acquisition \
  --prn "$prn" --output runs/e5b-tracking --no-show
python galileo.py decode runs/e5b-tracking --output runs/e5b-telemetry --no-show
python galileo.py message runs/e5b-telemetry/decoded_inav.json \
  --output runs/e5b-navigation --no-show
```

E5a produces `decoded_fnav.json`. For positioning use the directory structure
created by `run`: `<run>/<band>/E##/{02_tracking,03_telemetry,04_navigation}`.

Compare modes from the same processed session without retracking:

```bash
python galileo.py position runs/e1-e5b-001 --mode e1 \
  --output runs/compare-e1 --map-source none --no-show
python galileo.py position runs/e1-e5b-001 --mode e5b \
  --output runs/compare-e5b --map-source none --no-show
python galileo.py position runs/e1-e5b-001 --mode e1+e5b \
  --output runs/compare-dual --map-source none --no-show
```

Single-band solutions default to broadcast NTCM-G, scaled by inverse frequency
squared. Missing coefficients/calendar information yield an explicit rejection;
`--ionosphere none` is an explicitly uncorrected comparison. These mode runs can
use different satellite sets, so differences in their positions are not proof
of accuracy improvement. Compare accepted epochs/PRNs, geometry and residuals,
and use an independently surveyed antenna location when evaluating accuracy.

## Why the combination is physically meaningful

For matched frequencies, the code combination is

```
P_IF = alpha * P_E1 + beta * P_E5
alpha = f_E1² / (f_E1² - f_E5²)
beta  = -f_E5² / (f_E1² - f_E5²)
```

It cancels the first-order dispersive ionosphere. It does not remove thermal
noise, multipath, higher-order ionosphere, antenna effects or incorrect timing.
Noise variance is propagated as `alpha² sigma_E1² + beta² sigma_E5²`, assuming
independent code errors; common satellite orbit/clock error is not multiplied
as though it were two independent errors. The configurable 10 m E1 / 3 m E5
zenith code sigmas are explicit assumptions, not measured accuracies.

The implementation uses fractional primary-code boundaries and decoded page
TOW, with 4 ms E1 and 1 ms E5 epochs. It never interprets an absolute USRP device
timestamp as GST. Device timing establishes stream simultaneity only. Separately
recorded E1/E5 sessions are rejected for combination.

I/NAV supplies the E1/E5b clock, F/NAV the E1/E5a clock. Single-band clock
corrections apply the appropriate BGD factor. For an E1/E5a pair, E1 is explicitly
recomputed using the matched F/NAV clock before combination. Weighted matched
BGD terms cancel; they are not subtracted again. Distinct IODnav sets and
inconsistent satellite states are rejected.

`05_dual_frequency.png` and `dual_frequency.json/csv` show the geometry-free
code difference, apparent E1 ionosphere, signed weights and noise amplification.
The apparent ionosphere includes receiver inter-frequency delay and multipath;
it is **not calibrated TEC**. Common delay is absorbed by the estimated receiver
clock, while direction-dependent and differential biases can affect position.

## Full-band E5 capture and sideband extraction

```bash
python galileo.py capture --bands e5 --duration 120 \
  --output recordings/e5-wide-001 --plan
# Remove --plan after verifying the storage/transport and RF setup.
python galileo.py capture --bands e5 --duration 120 \
  --output recordings/e5-wide-001
python galileo.py split-e5 recordings/e5-wide-001 --output recordings/e5-split-001
python galileo.py run recordings/e5-split-001 --bands e5a+e5b \
  --mode e5b --output runs/e5-wide-001 --map-source none --no-show
```

Both sidebands are tracked/decoded, with E5b selected for the final single-band
solution. The E5a/E5b pair alone has poor frequency separation for ionosphere
cancellation and is not offered as an IF solution. To combine with E1, the
preferred first experiment remains direct E1+E5b or E1+E5a capture.

The channelizer translates ±15.345 MHz, applies a 257-tap symmetric FIR and
outputs 25 MS/s. Filter delay is compensated, original time zero is preserved,
and chunk overlaps prevent discontinuities. The first/last 32 output samples
are edge transients; initial acquisition/tracking needs usable signal away from
those edges. Output amplitude is scaled by 0.5 to leave quantization headroom.
Source fingerprints and transformation settings are retained in metadata.

The published TwinRX instantaneous bandwidth does not establish the actual
antenna/feed/preselector response over the entire E5 band. Verify both sidebands
and their attenuation on the real setup before using full-band captures for
precision comparisons, particularly near the front-end filter-band boundary.

## Engineering references

- [EUSPA/GSC Galileo OS documents in force](https://www.gsc-europa.eu/electronic-library/programme-reference-documents/galileo-in-force/open-service): authoritative version index.
- [Galileo OS SIS ICD 2.2, November 2025](https://www.gsc-europa.eu/sites/default/files/sites/all/files/Galileo_OS_SIS_ICD_v2.2.pdf): §§2–3 signals/codes; §4.2 F/NAV; §4.3 I/NAV; §5.1.2 page timing; §§5.1.3–5.1.5 clock/BGD conventions. This implementation targets that version, not an unspecified future ICD.
- [ESA Galileo signal plan](https://gssc.esa.int/navipedia/index.php/GALILEO_Signal_Plan): RF centres, reference bandwidths and AltBOC/sideband distinction.
- [ESA ionosphere-free combination](https://gssc.esa.int/navipedia/index.php/Ionosphere-free_Combination_for_Dual_Frequency_Receivers): frequency-squared combination and clock convention.
- [EUSPA/GSC NTCM-G model](https://www.gsc-europa.eu/sites/default/files/NTCM-G_Ionospheric_Model_Description_-_v1.0.pdf): single-frequency model inherited from the E1 receiver.
- [Ettus TwinRX manual](https://files.ettus.com/manual/page_twinrx.html): 200 MHz clock, bandwidth/sample-rate limits, independent tuning and antenna/gain routing.

## Reproduce software validation

```bash
python -m unittest discover -s tests -v
MPLBACKEND=Agg PYTHONPATH=. python tests/rf_smoke.py
```

The optional RF smoke check generates temporary synthetic SC16 recordings in
`/tmp`, including non-grid Doppler and fractional delay. These tests do not
transmit RF and do not access the USRP. No test recordings are committed.
