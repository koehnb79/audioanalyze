# Guitar Rhythm Analyzer

A local Python CLI for exploratory rhythm analysis of guitar audio and audio extracted from video.

## Version 0.8.0 — listening-led event review

Version 0.8.0 adds optional visual event maps to the v0.7 multi-detector report. The maps show a compact 20 ms peak-amplitude envelope for the full mix, HPSS harmonic component, and HPSS percussive component, with spectral-flux and Librosa onset detections overlaid.

Run the full collection and create maps only for the manually annotated reference recordings:

```bash
python rhythm_event_validation.py ./recordings --recursive --plots --output ./guitar_rhythm_v080
```

Use `--plots-all` instead of `--plots` to create maps for every recording; this can produce hundreds of image files and substantially increase runtime. The output includes the v0.7 summary and event CSVs plus an `event_maps/` folder.

The event maps are a human-review aid, not ground-truth annotations. They show detected event timing against the signal envelope, but do not establish which visible transient is a guitar attack, drum hit, or metronome click. Listen to the recording while inspecting the map. Compare the nine known reference examples first and add confidently unaccompanied controls before attempting source classification.

Version 0.8.0 does not change the detection thresholds or automatically classify recordings. Keep prior v0.7 outputs as a separate baseline.

## Version 0.7.0 — multi-detector attack diagnostics

Version 0.7.0 replaces the single-detector event report with a diagnostic comparison of two onset methods:

- Positive spectral-flux peaks, using a recording-relative robust threshold.
- Librosa's onset-strength/onset-detection pipeline.

Both methods are applied to the full mix and the HPSS harmonic and percussive components. The report compares detector agreement, event density, inter-onset intervals, and harmonic/percussive timing relationships.

A second event-level CSV, `rhythm_attack_events.csv`, contains per-event descriptors: spectral centroid, spectral flatness, spectral rolloff, local RMS peak, pre-event RMS, peak-to-pre-event ratio, and detector strength. These features can help identify why a transient detector responds differently to attacks, but they do not identify drums, metronomes, or guitar.

The summary CSV is `rhythm_attack_summary.csv`; `analysis_errors.csv` records failures. The tool retains listening annotations and always leaves `algorithm_verdict` as `not_classified`.

### Run v0.7.0

```bash
python rhythm_event_validation.py ./recordings --recursive --output ./guitar_rhythm_v070
```

Review the nine listening-reference recordings first. In particular, compare the two onset detectors on the same recording. If their event lists disagree substantially, timing statistics from either detector should be treated cautiously. HPSS components are not isolated instruments, spectral descriptors are not source labels, and events during low harmonic energy are not proof of an external rhythm source.

The event CSV can be large because it contains one row per detected event. Keep the v0.6.0 report as a separate baseline; v0.7.0 does not overwrite it or alter the primary longitudinal analyzer.

## Version 0.6.0 — event-level rhythm diagnostics

Version 0.6.0 adds a separate event-level analysis tool, `rhythm_event_validation.py`. It detects transient/onset events independently in the HPSS harmonic and percussive components, then reports:

- Event counts and event timestamps for each component.
- Inter-onset interval median, median absolute deviation, coefficient of variation, and implied median pulse.
- The proportion of percussive-component events occurring near a harmonic-component event (within 60 ms), versus events with no nearby harmonic event.
- The proportion of percussive events occurring during the lower quartile of harmonic RMS energy.
- Overlapping 20-second windows with event counts, percussive event density, and event coincidence.

These are signal relationships, not instrument recognition. HPSS can place guitar strums in the percussive component, and guitar attacks can be detected independently from harmonic onsets. An event outside the 60 ms coincidence window is not thereby proven to be a drum or metronome. The lower-quartile energy measure is a coarse activity proxy, not a reliable guitar-silence detector.

### Run v0.6.0

```bash
python rhythm_event_validation.py ./recordings --recursive --output ./guitar_rhythm_v060
```

It writes `rhythm_event_validation.csv` and `analysis_errors.csv`. The event report is separate from the v0.3/v0.4 longitudinal analyzer and v0.5 periodicity validation. It does not modify or filter the main dataset.

Review the manually annotated examples first. In particular, compare the distributions of independent percussive events and their timing across the known drum, metronome, and likely-metronome recordings. If the measures do not separate these examples, do not use them as a source classifier.

## Version 0.5.1 — richer rhythm diagnostics

Version 0.5.1 improves the validation report without introducing an automatic source classifier.

- Reports up to six distinct autocorrelation tempo candidates for the full mix, harmonic component, and percussive component, with relative strengths.
- Adds an octave-family view that folds half/double-time candidates into a common 60–120 BPM comparison band. This helps expose metrical ambiguity; it does not decide which pulse a listener intended.
- Stores overlapping 20-second window measurements for all three signal components, including each window's candidate octave family and periodicity strength.
- Adds octave-folded window-to-window BPM variability and a descriptive measure of how often percussive periodicity strength exceeds harmonic periodicity strength.
- Keeps manual listening labels separate and leaves `algorithm_verdict` as `not_classified`.

The added measures are review aids, not evidence that a percussive signal is drums or a metronome. HPSS separates signal characteristics, not instruments. A guitar's own repeated strumming can produce strong periodicity; drums can be syncopated or masked; and the strongest autocorrelation peak may represent half-time, double-time, or a subdivision. Window CV is only meaningful when there are at least three valid windows.

### Run v0.5.1

```bash
python rhythm_reference_validation.py ./recordings --recursive --output ./guitar_rhythm_v051_validation
```

Compare the new output against the manually annotated examples first. Preserve the previous v0.5.0 CSV as a baseline; do not replace it or use the new measurements to automatically filter the main longitudinal dataset.

## Version 0.5.0 — rhythm-reference validation

Version 0.5.0 adds a separate exploratory validation tool. This is intentionally not another automatic metronome/drum classifier.

- Adds `rhythm_reference_validation.py`, which measures periodicity in the full mix, harmonic component, and percussive component.
- Reports the strongest autocorrelation tempo candidate, normalized peak strength, runner-up strength, and overlapping-window percussive tempo candidates.
- Carries manual listening annotations separately from algorithmic measurements.
- Does not label unannotated recordings as self-paced and does not automatically exclude recordings from longitudinal analysis.
- Includes `reference_annotations.csv`, an initial user-listening reference set. These labels are listening annotations, not verified production metadata.

The initial reference set contains seven user-identified drum recordings, one user-identified metronome recording, and one recording the user considers likely metronome-influenced. The September 22 recording is deliberately labeled `likely_metronome`, not confirmed.

### Run the validation tool

Install the project dependencies as described below, then run the tool against the original recordings:

```bash
python rhythm_reference_validation.py ./recordings --recursive --output ./guitar_rhythm_v050_validation
```

It writes `rhythm_reference_validation.csv` and `analysis_errors.csv`. The report is a diagnostic aid for comparing known listening examples. It is not a source-identification model: autocorrelation can find periodicity in guitar playing itself, and metronome/drum patterns may be masked, syncopated, or expressed at half/double tempo.

### Validation rules

- Keep user listening labels and computed features in separate fields.
- Do not use a lack of detected periodicity as proof that a recording is self-paced.
- Do not automatically remove or exclude any recording based on these features.
- Review the annotated examples first, compare full/harmonic/percussive evidence, and only then decide whether a future classifier is justified.

## Version 0.4.0

- Adds conservative rhythm-reference screening using harmonic/percussive source separation (HPSS), percussive transient timing, interval regularity, and independence from harmonic attacks.
- The v0.4 heuristic missed all eight current 2020 listening examples; its `self_paced_eligible` field must not be treated as ground truth. It is retained for reproducibility, not as a trusted filter.
- Keeps all recordings in the master CSV/JSON/SQLite outputs and additionally exports screening subsets. These subsets are heuristic and should not be used to select a clean self-paced cohort without manual review.

## Version 0.3.1

- Date resolution: embedded media creation timestamp via FFprobe, then date in filename (YYYY_MM_DD or YYYYMMDD), then filesystem modification time. Each row records date, year, and provenance so fallback dates can be reviewed.
- Self-paced pulse estimation is now the default. Percussive audio is no longer assumed to be an independent drum/metronome reference.
- Removed overfit onset-to-grid residual calculations. Grid-error fields remain null and carry a status explaining that no independent reference phase is available.
- Added overlapping 30-second tempo windows every 15 seconds, with local BPM and within-recording tempo-window CV when at least three windows are available.
- Retains phrase statistics, onset timestamps, diagnostic plots, CSV/JSON/SQLite output, and FFmpeg decoding fallback.

## Install

Python 3.10–3.12 recommended. FFmpeg is required for video and fallback decoding; FFprobe reads embedded creation timestamps.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the main analyzer

```bash
python guitar_rhythm_analyzer.py ./recordings --recursive --plots --output ./guitar_rhythm_v040
```

Self-paced is the default. `--reference-bpm` can record a known tempo but cannot provide independent beat phase. The legacy `--mode reference` option is retained for compatibility; it does not calculate grid accuracy.

## Date fields

- `embedded_creation_time`: creation timestamp exposed by FFprobe.
- `filename_date`: date parsed from the filename.
- `file_modified_fallback`: filesystem modification time, which may be a copy/edit date rather than the performance date.

Review fallback dates before longitudinal comparisons. Embedded tags may also be missing or inaccurate; provenance is always retained.

## Outputs

The main analyzer writes CSV/JSON measurements, date provenance, tempo windows, phrase summaries, raw onset timestamps, and rhythm-reference screening. SQLite retains versioned result rows. Optional plots are saved under `onset_plots/`; decode failures are listed in `analysis_errors.json`.

The v0.5 validation tool writes a separate validation CSV and error CSV. It does not alter the v0.4 outputs.

## Interpretation

Rhythm-reference detection is not proof of an external backing track. HPSS can classify guitar attacks as percussive; autocorrelation can find periodicity in the guitar itself. Onsets and phrase boundaries are heuristic. Within-phrase interval CV is affected by subdivision, syncopation, swing, arpeggiation, expressive timing, and recording conditions. Tempo estimates can jump between half-time and double-time.

No absolute synchronization accuracy is reported. That would require a separate reference track or independently supplied beat timestamps including phase; BPM alone is insufficient.

This is exploratory audio analysis, not a validated motor assessment or clinical/diagnostic instrument. Compare similar passages and inspect diagnostic plots rather than interpreting a single metric as evidence of decline or improvement.
