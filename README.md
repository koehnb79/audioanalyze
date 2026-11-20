# Guitar Rhythm Analyzer

## Version 0.9.4.1-dev (missed-click-tolerant reference fitting)

The experimental reference detector now fits pulse period and phase while allowing observed gaps to be integer multiples of the underlying period. This addresses missed transient detections without treating every long gap as an irregular click interval.

- Searches a bounded 40–240 BPM period range and refines candidate period/phase using robust inlier fitting.
- Reports click-fit inlier count, median and 90th-percentile phase residual, within-20-ms percentage, and estimated missing grid slots.
- Retains raw observed-gap CV as a diagnostic only; it is no longer used by itself to reject a candidate.
- Uses conservative fit-quality gates before calculating guitar-to-reference timing.
- Adds synthetic coverage for missing clicks and outlier sequences.

**Important:** a periodic guitar pattern can still mimic a click train. A good mathematical phase fit does not establish the acoustic source or prove that the candidate is a metronome. Candidate trains still require waveform/audio review; mono-source separation is not guaranteed.

## Version 0.9.4.0-dev (experimental independent metronome reference)

The v0.9.4 development branch adds an experimental click-train detector for recordings explicitly tagged `[Metronome]` in the filename.

- Builds short-window spectral signatures around detected transients and searches for a repeated signature.
- Checks whether the candidate transient train is sufficiently regular before treating it as a possible click reference.
- Excludes candidate click events from the guitar-event list before calculating guitar-to-click timing. In a mono mix, a guitar attack coincident with a click may also be excluded; reliable source separation is not guaranteed.
- Reports signed beat-relative timing residuals, median absolute error, 90th-percentile error, and percentages within 10, 20, 30, and 50 ms.
- Reports sixteenth-grid residuals separately. This is a nearest-grid diagnostic, not proof that a particular note was intended on that subdivision.
- Leaves non-`[Metronome]` recordings untouched and records why independent reference analysis was not attempted.
- Adds synthetic tests for perfect alignment, early/late offsets, subdivision-vs-beat distinction, irregular clicks, and insufficient reference events.

**Important validation limit:** a repeated guitar attack pattern can resemble a metronome click train. A candidate click train must be reviewed against the audio and diagnostic plot. These measurements should not be treated as verified synchronization accuracy until click detection and guitar/click separation are confirmed for the recording. A known BPM alone still does not provide independent beat phase.

Run tests with:

```bash
python -m unittest discover -s tests -v
```

## Version 0.9.3.1-dev (local-grid validation corrections)

The v0.9.3 development branch builds on the v0.9.2 grid-analysis branch. It retains the v0.9.2 fields for comparison and adds a separate local-grid event report:

- Fits beat grids independently in overlapping 12-second windows advanced every 6 seconds; it no longer relies only on a single recording-wide fitted grid.
- Estimates the local beat period while accounting for intervals that are integer multiples of the underlying period, preserving evidence of missing beats.
- Assigns each detected onset an explicit beat index, subdivision label, predicted grid timestamp, and signed residual in milliseconds.
- Supports beat, eighth-note, triplet, sixteenth, sextuplet, and thirty-second-note candidate grids. An event is assigned to the coarsest subdivision whose residual is within the default 18 ms tolerance; otherwise it is marked `off-grid`. The best finer candidate and all candidate residuals remain available for inspection.
- Preserves the unsubdivided beat-grid residual separately from the selected-subdivision residual, so subdivision fitting cannot conceal deviation from the beat pulse.
- Missing-beat gaps and half/double-time ambiguity are separate flags. Missing beat timestamps alone no longer force the octave-ambiguity flag.
- Reports event coverage and unassigned-event counts.
- Retains each event's raw residual and flags missing-beat proximity, local tempo changes, and half/double-time ambiguity.
- Adds deterministic synthetic tests for perfect grids, controlled timing offsets, missing beats, tempo changes, half/double-time ambiguity, and event/subdivision fields.

The local grid is still anchored to beat timestamps estimated from the same recording. These residuals are mathematical consistency measures, not independent ground-truth timing errors. Beat tracker errors and onset detector errors remain possible; ambiguous windows are explicitly surfaced rather than silently treated as certain.

Run synthetic tests with:

```bash
python -m unittest discover -s tests -v

# Optional: change the subdivision tolerance in milliseconds when calling
# analyze_local_grid(..., subdivision_tolerance_ms=18.0)
```


A local Python CLI for exploratory rhythm analysis of guitar audio and audio extracted from video.

## Version 0.9.2-dev (local pulse and grid diagnostics)

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


## v0.9.2-dev local pulse and mathematical-grid diagnostics

- Each overlapping 30-second tempo window now includes ranked onset-envelope autocorrelation periodicity candidates, exposing local half-time/double-time alternatives.
- Beat timestamps are fitted to a constant-period grid by least squares. The analyzer reports beat-grid residuals (median absolute, median signed, 90th percentile absolute, within 50 ms) and retains the residual series.
- Detected guitar onsets are also compared with quarter-beat subdivisions of that fitted grid, with the same summary statistics and raw residual series.
- Grid fitting is performed per recording across the detected beat track; local candidate profiles are retained per window to help identify tempo-interpretation changes.

**Important interpretation limit:** the beat grid is fitted to the beat track detected from the same audio. It is therefore a mathematical consistency measure of the algorithm's beat sequence, not an independent external clock and not proof of absolute timing accuracy. Onset-to-quarter-grid results depend on onset detection and the chosen beat/subdivision interpretation; missed notes, extra transients, and syncopated attacks can affect them. Do not interpret these fields as clinical measurements. The purpose is to consistently quantify deviation from an idealized grid under a fixed algorithm, then compare like with like over time.

New v0.9.2 fields include `fitted_grid_bpm`, `beat_grid_*`, `onset_quarter_grid_*`, `grid_fit_status`, and per-window `autocorrelation_candidates`.
