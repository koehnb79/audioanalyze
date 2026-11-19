# Guitar Rhythm Analyzer

A local Python CLI for exploratory rhythm analysis of guitar audio and audio extracted from video.

## Version 0.9.0-dev (internal rhythm analysis)

- Date resolution: embedded media creation timestamp via FFprobe, then date in filename (YYYY_MM_DD or YYYYMMDD), then filesystem modification time. Each row records date, year, and provenance so fallback dates can be reviewed.
- Self-paced pulse estimation is now the default. Percussive audio is no longer assumed to be an independent drum/metronome reference.
- Removed overfit onset-to-grid residual calculations. Grid-error fields remain null and carry a status explaining that no independent reference phase is available.
- Added overlapping 30-second tempo windows every 15 seconds, with local BPM and within-recording tempo-window CV when at least three windows are available.
- Retains phrase statistics, onset timestamps, diagnostic plots, CSV/JSON/SQLite output, and FFmpeg decoding fallback.


## v0.9.0-dev additions

This development branch adds exploratory internal-pulse measurements to the existing analyzer:

- Beat-interval median, coefficient of variation, and median absolute deviation in milliseconds.
- Descriptive beat-interval drift slope (milliseconds per minute).
- Beat-tracker time coverage.
- Optional human recording-context labels (solo, accompaniment, or uncertain) from recording_annotations.csv, matched by exact filename.
- Existing phrase/onset, tempo-window, date provenance, plot, CSV/JSON/SQLite, and FFmpeg support remain.

The beat tracker estimates beats from the same recording being measured. These new metrics are **not independent ground-truth timing errors**, and they do not establish millisecond accuracy against a true measure grid. They are intended for cautious within-recording and across-recording exploration. A beat tracker may follow drums/backing tracks when present, which is why context labels must be reviewed before comparing solo performances.

The optional recording_annotations.csv template is in the repository. Fill in exact filenames and use:
- solo: you are confident there is no automated accompaniment.
- accompaniment: drums, metronome, backing track, or other external time source is present.
- uncertain: unknown or mixed.

Unannotated recordings default to uncertain; the analyzer does not guess whether accompaniment is present.

## Install

Python 3.10–3.12 recommended. FFmpeg is required for video and fallback decoding; FFprobe reads embedded creation timestamps.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python guitar_rhythm_analyzer.py ./recordings --recursive --plots --output ./guitar_rhythm_v090
```

Self-paced is the default. `--reference-bpm` can record a known tempo but cannot provide independent beat phase. The legacy `--mode reference` option is retained for compatibility; it does not calculate grid accuracy.

## Date fields

- `embedded_creation_time`: creation timestamp exposed by FFprobe.
- `filename_date`: date parsed from the filename.
- `file_modified_fallback`: filesystem modification time, which may be a copy/edit date rather than the performance date.

Review fallback dates before longitudinal comparisons. Embedded tags may also be missing or inaccurate; provenance is always retained.

## Outputs

CSV and JSON measurements include recording date provenance, tempo windows, phrase summaries, and raw onset timestamps. SQLite retains versioned result rows. Optional plots are saved under `onset_plots/`; decode failures are listed in `analysis_errors.json`.

## Interpretation

Onsets and phrase boundaries are heuristic. Within-phrase interval CV is affected by subdivision, syncopation, swing, arpeggiation, expressive timing, and recording conditions. Tempo windows can jump between half-time and double-time; tempo-window CV describes variation in estimated local tempo, not necessarily timing error.

No absolute synchronization accuracy is reported in 0.3.1. That would require a separate reference track or independently supplied beat timestamps including phase; BPM alone is insufficient.

This is exploratory audio analysis, not a validated motor assessment or clinical/diagnostic instrument. Compare similar passages and inspect diagnostic plots rather than interpreting a single metric as evidence of decline or improvement.


## v0.9.1-dev pulse diagnostics

This branch adds onset-envelope autocorrelation candidates alongside the Librosa beat
tracker. The candidates expose alternative periodicities and possible half-time or
double-time interpretations; they are not confirmed beat labels or ground truth.

New fields:
- `pulse_autocorrelation_candidates_json`: ranked periodicity candidates with BPM,
  period, and normalized autocorrelation.
- `pulse_autocorrelation_top_bpm`: strongest autocorrelation candidate.
- `tracker_vs_autocorrelation_octave_adjusted_difference_percent`: difference between
  the beat track and strongest autocorrelation candidate after allowing half/double-time
  equivalence.
- `beat_tracker_time_span_percent`: elapsed recording span between first and last
  detected beats, expressed as a percentage of recording duration.

The old `beat_tracker_coverage_percent` field remains for compatibility, but it is
only a time-span measure, not a measure of tracker accuracy. Pulse intervals are filtered
to the supported 0.25–2.0 second range before calculating the pulse-interval CV.

The autocorrelation and beat tracker both analyze the same audio, so agreement between
them is not independent validation. Listening, manual beat annotation, and a separate
pulse-reference test are still needed before making strong claims about rhythmic
improvement.
