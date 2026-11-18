# Guitar Rhythm Analyzer

A local Python command-line tool for exploratory rhythm analysis of guitar recordings. It accepts audio files and extracts audio tracks from video files; video frames are not analyzed.

## Version 0.3.0

Version 0.3 adds exploratory pulse-grid analysis. It compares detected guitar/audio onsets with a beat grid estimated from the recording, and can optionally use a supplied reference BPM. Automatic percussive-component detection is only a candidate: guitar attacks can also appear in the percussive signal, so it must not be assumed to be a true drum/metronome track.

This version is oriented toward short playing snippets. It adds heuristic phrase segmentation, within-phrase timing summaries, data-quality flags, optional onset diagnostic plots, an FFmpeg decoding fallback, and versioned SQLite result storage.

## Features

- Batch processing of audio and video recordings.
- Estimated onset timestamps retained in JSON and CSV.
- Phrase segmentation based on unusually long gaps between detected onsets.
- Within-phrase interval variability, median phrase variability, and interval spread.
- Legacy all-onset variability retained under an explicitly labelled field for comparison.
- Estimated pulse tempo and beat-grid timing residuals (median, 90th percentile, signed offset, and share within 50 ms).\n- Candidate pulse estimates from the full mix and percussive component; source and confidence labels are retained.\n- Selectable `auto`, `self-paced`, and `reference` analysis modes, plus optional known reference BPM.
- Optional waveform plots with detected onsets and within-phrase interval plots.
- CSV, JSON, and SQLite exports; SQLite stores versioned JSON payloads to allow future schema changes.

## Supported formats

Audio: WAV, MP3, FLAC, M4A, AAC, OGG, AIFF.

Video: MP4, MOV, MKV, AVI, WebM, MPEG, MPG, M4V.

Actual support depends on installed decoders. FFmpeg is required for video and is also used as a fallback for audio files that the normal decoder cannot read.

## Requirements

- Python 3.10–3.12 recommended.
- FFmpeg installed and available as `ffmpeg` for video and fallback decoding.
- Python packages in `requirements.txt`.

### Install

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run

Analyze a folder (default minimum duration is now 3 seconds):

```bash
python guitar_rhythm_analyzer.py ./recordings
```

Include nested folders and generate diagnostic plots:

```bash
python guitar_rhythm_analyzer.py ./recordings --recursive --plots --mode auto
```

Choose a mode:\n\n```bash\n# Independent playing: infer a pulse from the recording itself\npython guitar_rhythm_analyzer.py ./recordings --mode self-paced\n\n# Recordings with drums/metronome: experimental percussive reference tracking\npython guitar_rhythm_analyzer.py ./recordings --mode reference\n\n# If you know the metronome BPM, supply it explicitly\npython guitar_rhythm_analyzer.py ./recordings --mode reference --reference-bpm 100\n```\n\nChoose an output directory:

```bash
python guitar_rhythm_analyzer.py ./recordings --recursive --plots --output ./rhythm_results
```

Analyze a single video:

```bash
python guitar_rhythm_analyzer.py "./recordings/guitar_video.mp4" --plots
```

Change the minimum duration or the onset-count threshold used for the quality flag:

```bash
python guitar_rhythm_analyzer.py ./recordings --min-duration 2 --min-onsets 5
```

## Output

- `rhythm_measurements.csv` — measurements for spreadsheet analysis.
- `rhythm_measurements.json` — structured results, phrase details, and detected onset timestamps.
- `rhythm_history.sqlite` — append-only `analysis_runs` table with versioned JSON results.
- `analysis_errors.json` — files that could not be analyzed.
- `onset_plots/` — created when `--plots` is supplied; waveform/onset and phrase-interval plots.

File modification time is saved as metadata, not treated as the recording date. Use the recording date encoded in a filename or a separate metadata workflow for historical comparisons.

## Understanding the measurements

**Onsets** are estimated moments when new musical events begin. They are not guaranteed to correspond one-to-one with notes. Pick noise, chord attacks, resonance, and background instruments can affect detection.

**Phrases** are estimated by splitting the onset sequence at gaps longer than an adaptive threshold. This is a practical heuristic, not a reliable understanding of musical phrasing. Inspect the plots and adjust interpretation accordingly.

**Within-phrase interval CV** describes the relative spread of detected intervals that fall inside the estimated phrases. It is less affected by long pauses than the legacy all-onset CV, but it is still affected by changes in subdivision, syncopation, swing, arpeggiation, and deliberate expressive timing. It is not a pure measure of rhythmic precision.

**Quality flag** is a simple data-quantity warning based on detected onset and interval counts. It is not a statistical confidence score and does not verify that the detected events are correct.

**Tempo estimates** infer a pulse from onset strength. They may select half-time or double-time and are not equivalent to note-level timing accuracy.

## Version 0.3 pulse-grid measurements\n\n`pulse_bpm` is the selected pulse estimate. `pulse_source` and `pulse_confidence` describe where it came from and how cautiously it should be interpreted. `grid_subdivision` is the tested number of equal subdivisions per beat. The grid residuals are the signed distance from each detected onset to its nearest inferred grid position; summary fields report median absolute error, signed median, 90th percentile absolute error, and percentage within 50 ms. These are alignment-to-an-inferred-grid measures, not absolute ground-truth accuracy.\n\nIn `auto` mode, a beat estimate from the percussive component is recorded as a candidate when available, but automatic detection cannot reliably prove that drums or a metronome are present. In `reference` mode, the percussive component is used experimentally. A supplied `--reference-bpm` overrides tempo selection but the grid phase is still inferred from detected onsets. For strongest external-reference measurements, a future iteration should support manually supplied reference beat timestamps or a separate reference audio track.\n\nFor self-paced playing, the inferred grid is derived from the same performance being measured. This can make residuals look artificially small; use them alongside tempo drift, beat-interval variability, onset plots, and comparable repeated passages rather than as a standalone score. The current pulse tracker does not yet separate guitar attacks from drum attacks.\n\n## Longitudinal use and limitations

This is an exploratory audio-analysis tool, not a validated motor assessment or clinical/diagnostic instrument. Do not interpret any single metric as evidence of muscular decline or improvement. Microphone placement, instrument, strings, effects, recording quality, playing style, and musical material can all change the measurements.

For the strongest comparisons, group recordings by playing style (for example, repeated riff, strumming, fingerpicking, scale/exercise, or free improvisation) and compare similar passages where possible. Retain the original media and inspect diagnostic plots before drawing conclusions. Version 0.2 is a step toward better short-snippet analysis; manual onset correction and pattern-aware timing analysis remain future work.
