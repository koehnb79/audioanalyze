#!/usr/bin/env python3
"""
Guitar Rhythm Analyzer
Batch-analyze audio files and the audio tracks of video files.

Install:
    python -m pip install -r requirements.txt

Video support requires FFmpeg installed and available on PATH.

Examples:
    python guitar_rhythm_analyzer.py ./recordings
    python guitar_rhythm_analyzer.py ./recordings --recursive
    python guitar_rhythm_analyzer.py ./recordings --recursive --output ./results

This is an exploratory audio-analysis tool, not a validated clinical assessment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg", ".m4v"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
ANALYSIS_VERSION = "0.1.0"


def extract_video_audio(video_path: Path, wav_path: Path) -> None:
    """Extract mono 22.05 kHz PCM audio from a video using FFmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and make sure 'ffmpeg' is on PATH "
            "to analyze video files."
        )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-vn", "-ac", "1", "-ar", "22050",
        "-c:a", "pcm_s16le", str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "FFmpeg could not extract audio.")


def load_recording(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file or extract and load the audio track of a video."""
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        with tempfile.TemporaryDirectory(prefix="guitar_rhythm_") as folder:
            wav_path = Path(folder) / "extracted_audio.wav"
            extract_video_audio(path, wav_path)
            audio, sample_rate = librosa.load(wav_path, sr=22050, mono=True)
    else:
        audio, sample_rate = librosa.load(path, sr=22050, mono=True)
    if audio.size == 0:
        raise RuntimeError("No audio samples could be decoded.")
    return audio, sample_rate


def finite_or_none(value: object) -> Optional[float]:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def coefficient_of_variation(values: np.ndarray) -> Optional[float]:
    if values.size < 2:
        return None
    mean = float(np.mean(values))
    if mean <= 0:
        return None
    return finite_or_none(float(np.std(values, ddof=1) / mean * 100))


def analyze_recording(path: Path, min_duration: float = 8.0) -> dict:
    audio, sample_rate = load_recording(path)
    duration = float(librosa.get_duration(y=audio, sr=sample_rate))
    if duration < min_duration:
        raise RuntimeError(
            f"Recording is {duration:.1f}s; minimum is {min_duration:.1f}s."
        )

    hop_length = 256
    onset_envelope = librosa.onset.onset_strength(
        y=audio, sr=sample_rate, hop_length=hop_length, aggregate=np.median
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=False,
        units="frames",
        wait=2,
    )
    onset_times = np.asarray(
        librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length),
        dtype=float,
    )

    # Suppress extremely close detections that are often duplicate transient peaks.
    if onset_times.size:
        onset_times = onset_times[np.r_[True, np.diff(onset_times) >= 0.045]]

    intervals = np.diff(onset_times)
    intervals = intervals[(intervals >= 0.045) & (intervals <= 3.0)]

    mean_interval = finite_or_none(np.mean(intervals)) if intervals.size else None
    median_interval = finite_or_none(np.median(intervals)) if intervals.size else None
    interval_sd = (
        finite_or_none(np.std(intervals, ddof=1)) if intervals.size > 1 else None
    )
    interval_mad = (
        finite_or_none(np.median(np.abs(intervals - np.median(intervals))))
        if intervals.size else None
    )

    tempo_track = librosa.feature.tempo(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=None,
    )
    tempo_track = np.asarray(tempo_track, dtype=float).reshape(-1)
    tempo_track = tempo_track[np.isfinite(tempo_track) & (tempo_track > 0)]
    tempo_median = finite_or_none(np.median(tempo_track)) if tempo_track.size else None
    tempo_sd = (
        finite_or_none(np.std(tempo_track, ddof=1)) if tempo_track.size > 1 else None
    )

    # Interval CV in each third can reveal within-recording changes.
    third_cvs = []
    for section in range(3):
        start = duration * section / 3
        end = duration * (section + 1) / 3
        local_onsets = onset_times[(onset_times >= start) & (onset_times < end)]
        local_intervals = np.diff(local_onsets)
        local_intervals = local_intervals[
            (local_intervals >= 0.045) & (local_intervals <= 3.0)
        ]
        third_cvs.append(coefficient_of_variation(local_intervals))

    file_stat = path.stat()
    return {
        "filename": path.name,
        "path": str(path.resolve()),
        "media_type": "video" if path.suffix.lower() in VIDEO_EXTENSIONS else "audio",
        "file_modified_local": datetime.fromtimestamp(file_stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 3),
        "sample_rate_hz": sample_rate,
        "detected_onsets": int(onset_times.size),
        "usable_intervals": int(intervals.size),
        "mean_inter_onset_seconds": mean_interval,
        "median_inter_onset_seconds": median_interval,
        "inter_onset_sd_ms": finite_or_none(interval_sd * 1000) if interval_sd is not None else None,
        "inter_onset_cv_percent": coefficient_of_variation(intervals),
        "inter_onset_mad_ms": finite_or_none(interval_mad * 1000) if interval_mad is not None else None,
        "estimated_tempo_median_bpm": tempo_median,
        "estimated_tempo_sd_bpm": tempo_sd,
        "first_third_interval_cv_percent": third_cvs[0],
        "middle_third_interval_cv_percent": third_cvs[1],
        "last_third_interval_cv_percent": third_cvs[2],
        "onset_times_seconds": json.dumps([round(float(t), 4) for t in onset_times]),
        "analysis_version": ANALYSIS_VERSION,
        "interpretation_note": "Automatic onset estimates; review before interpreting.",
    }


def save_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "rhythm_measurements.csv"
    json_file = output_dir / "rhythm_measurements.json"
    database_file = output_dir / "rhythm_history.sqlite"

    if not rows:
        print("No recordings were successfully analyzed; no result files written.")
        return

    fieldnames = list(rows[0].keys())
    with csv_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Each run is appended as a snapshot, so later algorithm versions can coexist.
    with sqlite3.connect(database_file) as connection:
        columns = ", ".join(f'"{name}" TEXT' for name in fieldnames)
        connection.execute(f"CREATE TABLE IF NOT EXISTS recordings ({columns})")
        names = ", ".join(f'"{name}"' for name in fieldnames)
        placeholders = ", ".join("?" for _ in fieldnames)
        connection.executemany(
            f"INSERT INTO recordings ({names}) VALUES ({placeholders})",
            [
                [str(row[name]) if row[name] is not None else None for name in fieldnames]
                for row in rows
            ],
        )

    print(f"Results saved in: {output_dir.resolve()}")
    print(f"  {csv_file.name}")
    print(f"  {json_file.name}")
    print(f"  {database_file.name}")


def discover_files(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in SUPPORTED_EXTENSIONS else []
    iterator = source.rglob("*") if recursive else source.iterdir()
    return sorted(
        item for item in iterator
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze guitar rhythm features from audio and video recordings."
    )
    parser.add_argument("input", type=Path, help="Media file or folder containing recordings.")
    parser.add_argument("--recursive", action="store_true", help="Include subfolders.")
    parser.add_argument("--output", type=Path, default=Path("guitar_rhythm_results"))
    parser.add_argument(
        "--min-duration", type=float, default=8.0,
        help="Skip media shorter than this many seconds (default: 8).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input path does not exist: {args.input}", file=sys.stderr)
        return 2

    files = discover_files(args.input, args.recursive)
    if not files:
        print("No supported audio/video files found.")
        return 1

    print(f"Found {len(files)} media file(s).")
    successful: list[dict] = []
    failures: list[dict] = []

    for index, media_file in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {media_file}")
        try:
            measurement = analyze_recording(media_file, args.min_duration)
            successful.append(measurement)
            cv = measurement["inter_onset_cv_percent"]
            tempo = measurement["estimated_tempo_median_bpm"]
            cv_text = f"{cv:.2f}%" if cv is not None else "unavailable"
            tempo_text = f"{tempo:.1f} BPM" if tempo is not None else "unavailable"
            print(
                f"  {measurement['detected_onsets']} onsets; "
                f"inter-onset CV {cv_text}; estimated pulse {tempo_text}"
            )
        except Exception as error:
            failures.append({"file": str(media_file), "error": str(error)})
            print(f"  SKIPPED: {error}", file=sys.stderr)

    save_results(successful, args.output)
    if failures:
        args.output.mkdir(parents=True, exist_ok=True)
        error_file = args.output / "analysis_errors.json"
        error_file.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"{len(failures)} file(s) skipped; details: {error_file}")

    print(
        "\nCaution: automatic onset detection can miss quiet notes or count noise as attacks. "
        "Compare similar playing styles and inspect detections before drawing conclusions."
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
