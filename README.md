# Guitar Rhythm Analyzer

A local Python CLI for exploratory rhythm analysis of guitar audio and audio extracted from video.

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
