#!/usr/bin/env python3
"""Event-level guitar/percussive timing diagnostics (exploratory; no source classifier)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np

VERSION = "0.6.0"
ANNOTATION_FILE = Path(__file__).with_name("reference_annotations.csv")
SR = 22050
HOP = 256
MATCH_TOLERANCE_SECONDS = 0.060
MIN_EVENT_SPACING_SECONDS = 0.080


def load_annotations() -> dict:
    result = {}
    if ANNOTATION_FILE.exists():
        with ANNOTATION_FILE.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                result[row["filename_stem"]] = (row["manual_label"], row["label_source"])
    return result


ANNOTATIONS = load_annotations()


def clean_events(times: np.ndarray) -> np.ndarray:
    """Suppress duplicate detections closer than the minimum event spacing."""
    times = np.sort(np.asarray(times, dtype=float))
    kept = []
    for t in times:
        if not kept or t - kept[-1] >= MIN_EVENT_SPACING_SECONDS:
            kept.append(float(t))
    return np.asarray(kept, dtype=float)


def detect_events(signal: np.ndarray) -> np.ndarray:
    onset_frames = librosa.onset.onset_detect(
        y=signal, sr=SR, hop_length=HOP, backtrack=False,
        units="frames", normalize=True
    )
    return clean_events(librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP))


def nearest_matches(source: np.ndarray, target: np.ndarray) -> tuple[int, np.ndarray]:
    """Count source events with a target event within tolerance; one-to-one not required."""
    if not len(source) or not len(target):
        return 0, np.array([], dtype=float)
    distances = np.min(np.abs(source[:, None] - target[None, :]), axis=1)
    matched = distances <= MATCH_TOLERANCE_SECONDS
    return int(np.sum(matched)), distances


def interval_metrics(times: np.ndarray, prefix: str) -> dict:
    intervals = np.diff(times)
    plausible = intervals[(intervals >= 0.10) & (intervals <= 2.0)]
    if len(plausible):
        med = float(np.median(plausible))
        mad = float(np.median(np.abs(plausible - med)))
        cv = float(np.std(plausible, ddof=1) / np.mean(plausible) * 100) if len(plausible) >= 2 else None
        bpm = 60.0 / med
    else:
        med = mad = cv = bpm = None
    return {
        f"{prefix}_event_count": int(len(times)),
        f"{prefix}_interval_count": int(len(plausible)),
        f"{prefix}_median_interval_seconds": round(med, 4) if med is not None else None,
        f"{prefix}_interval_mad_seconds": round(mad, 4) if mad is not None else None,
        f"{prefix}_interval_cv_percent": round(cv, 3) if cv is not None else None,
        f"{prefix}_implied_median_bpm": round(bpm, 2) if bpm is not None else None,
        f"{prefix}_event_times_seconds": json.dumps(np.round(times, 3).tolist()),
    }


def analyze(path: Path) -> dict:
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < SR * 5:
        raise ValueError("Recording shorter than five seconds")
    harmonic, percussive = librosa.effects.hpss(y)
    h_events = detect_events(harmonic)
    p_events = detect_events(percussive)
    matched_count, distances = nearest_matches(p_events, h_events)
    independent_count = len(p_events) - matched_count
    duration = len(y) / SR

    row = {
        "filename": path.name, "stem": path.stem,
        "duration_seconds": round(duration, 3), "analysis_version": VERSION,
        **interval_metrics(h_events, "harmonic"),
        **interval_metrics(p_events, "percussive"),
        "percussive_events_near_harmonic_count": matched_count,
        "percussive_events_independent_count": independent_count,
        "percussive_events_near_harmonic_fraction": (
            round(matched_count / len(p_events), 4) if len(p_events) else None
        ),
        "percussive_events_independent_fraction": (
            round(independent_count / len(p_events), 4) if len(p_events) else None
        ),
        "median_nearest_harmonic_distance_seconds": (
            round(float(np.median(distances)), 4) if len(distances) else None
        ),
    }

    # Relative harmonic energy is a coarse activity proxy. Report, don't label:
    # percussive attacks during low harmonic energy may be backing rhythm, but
    # HPSS leakage and ringing guitar notes can produce the same pattern.
    rms = librosa.feature.rms(y=harmonic, frame_length=2048, hop_length=HOP)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
    if len(rms):
        threshold = float(np.quantile(rms, 0.25))
        low_activity = np.interp(p_events, rms_times, rms) <= threshold if len(p_events) else np.array([])
        row["harmonic_rms_lower_quartile"] = round(threshold, 7)
        row["percussive_events_during_low_harmonic_energy_count"] = int(np.sum(low_activity))
        row["percussive_events_during_low_harmonic_energy_fraction"] = (
            round(float(np.mean(low_activity)), 4) if len(low_activity) else None
        )
    else:
        row["harmonic_rms_lower_quartile"] = None
        row["percussive_events_during_low_harmonic_energy_count"] = 0
        row["percussive_events_during_low_harmonic_energy_fraction"] = None

    # Window-level event density and event overlap reveal whether relationships
    # persist across a track; short windows are descriptive, not classification.
    window, step = 20.0, 10.0
    windows = []
    start = 0.0
    while start + window <= duration:
        h = h_events[(h_events >= start) & (h_events < start + window)]
        p = p_events[(p_events >= start) & (p_events < start + window)]
        matched, _ = nearest_matches(p, h)
        windows.append({
            "start_seconds": round(start, 2),
            "harmonic_events": int(len(h)),
            "percussive_events": int(len(p)),
            "percussive_independent_fraction": round((len(p) - matched) / len(p), 4) if len(p) else None,
            "percussive_events_per_second": round(len(p) / window, 4),
        })
        start += step
    row["window_count"] = len(windows)
    row["window_event_metrics_json"] = json.dumps(windows, separators=(",", ":"))

    label = ANNOTATIONS.get(path.stem)
    row["listening_label"] = label[0] if label else "unannotated"
    row["listening_label_source"] = label[1] if label else "none"
    row["algorithm_verdict"] = "not_classified"
    row["interpretation"] = (
        "Event timing and harmonic/percussive coincidence only. HPSS does not "
        "identify instruments; independent percussive events may still be guitar "
        "attacks. Do not infer backing-track presence or self-paced status."
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("guitar_rhythm_v060"))
    args = parser.parse_args()
    extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif"}
    files = ([args.input] if args.input.is_file() else
             sorted(p for p in (args.input.rglob("*") if args.recursive else args.input.iterdir())
                    if p.is_file() and p.suffix.lower() in extensions))
    rows, errors = [], []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        try:
            rows.append(analyze(path))
        except Exception as exc:
            errors.append({"filename": path.name, "error": str(exc)})
            print(f"  ERROR: {exc}")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output / "rhythm_event_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "analysis_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "error"])
        writer.writeheader()
        writer.writerows(errors)
    print(f"Analyzed {len(rows)} recording(s); {len(errors)} error(s).")
    print(f"Event report: {args.output / 'rhythm_event_validation.csv'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
