# Guitar Rhythm Analyzer

A local Python command-line tool for exploratory rhythm analysis of guitar recordings.
It accepts ordinary audio files and extracts the audio track from video files before
analyzing it. Video frames are not analyzed.

## Features

- Batch processing of audio and video recordings.
- Onset (note/strum attack) detection and timestamp retention.
- Inter-onset interval mean, median, standard deviation, median absolute deviation,
  and coefficient of variation (CV).
- Estimated pulse tempo and tempo variability.
- Timing variability in the first, middle, and final thirds of each recording.
- Export to CSV, JSON, and SQLite for historical comparisons.

## Supported formats

Audio: WAV, MP3, FLAC, M4A, AAC, OGG, AIFF.

Video: MP4, MOV, MKV, AVI, WebM, MPEG, MPG, M4V.

Actual format support depends on the installed audio decoders and FFmpeg build.

## Requirements

- Python 3.10–3.12 recommended.
- FFmpeg installed and available as the `ffmpeg` command for video files.
- Python packages listed in `requirements.txt`.

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

Analyze files in a folder:

```bash
python guitar_rhythm_analyzer.py ./recordings
```

Include nested folders:

```bash
python guitar_rhythm_analyzer.py ./recordings --recursive
```

Choose an output location:

```bash
python guitar_rhythm_analyzer.py ./recordings --recursive --output ./rhythm_results
```

Analyze one video:

```bash
python guitar_rhythm_analyzer.py "./recordings/guitar_video.mp4"
```

Set the minimum media duration (default 8 seconds):

```bash
python guitar_rhythm_analyzer.py ./recordings --recursive --min-duration 5
```

## Output

The output folder contains:

- `rhythm_measurements.csv` — convenient for Excel and spreadsheet analysis.
- `rhythm_measurements.json` — structured results, including all detected onset times.
- `rhythm_history.sqlite` — SQLite database; each run is appended as a new snapshot.
- `analysis_errors.json` — created when files are skipped or fail to decode.

The file modification timestamp is saved as metadata, not assumed to be the recording date.
For reliable historical analysis, provide or maintain recording dates and playing-style
labels in a future metadata workflow.

## Understanding the measurements

**Onsets** are estimated moments at which new musical events begin. The detected events
are not guaranteed to correspond one-to-one with played notes: chord attacks, pick noise,
string resonance, and background instruments can affect detection.

**Inter-onset interval (IOI)** is the elapsed time between successive detected attacks.
Its standard deviation measures spread in seconds; its CV expresses spread relative to
the average interval and can help compare different tempos.

**Tempo estimates** infer a pulse from the onset-strength pattern. Automatic tempo
estimation can choose half-time or double-time, and a local tempo estimate is not the
same thing as note-level timing accuracy.

## Limitations and responsible interpretation

This is version 0.1.0, an exploratory prototype. It is not a validated motor assessment
or a clinical/diagnostic instrument. The measurements can be affected by musical style,
rests, syncopation, swing, rubato, deliberate expressive timing, distortion, compression,
noise, microphone placement, and changes in instrument or playing technique.

A simple IOI CV must not be interpreted as a pure measure of rhythmic ability. A future
version should add onset visualization and manual correction, playing-style grouping,
pattern-aware beat tracking, and confidence/quality indicators. For longitudinal work,
compare similar passages and playing styles where possible, and retain the original media.
