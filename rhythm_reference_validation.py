#!/usr/bin/env python3
"""Exploratory rhythm-reference validation for manually annotated guitar recordings.

This tool deliberately reports signal features rather than deciding whether a
recording contains a metronome or drums. User listening annotations are kept
separate from computed measurements.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np

VERSION = "0.5.0"
ANNOTATION_FILE = Path(__file__).with_name("reference_annotations.csv")


def load_annotations() -> dict:
    labels = {}
    if ANNOTATION_FILE.exists():
        with ANNOTATION_FILE.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[row["filename_stem"]] = (
                    row["manual_label"], row["label_source"]
                )
    return labels


ANNOTATIONS = load_annotations()


def safe_cv(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if values.size < 3 or np.mean(values) <= 0:
        return None
    return float(np.std(values, ddof=1) / np.mean(values) * 100)


def periodicity_features(envelope: np.ndarray, sr: int, hop: int) -> dict:
    """Autocorrelation peaks over plausible beat and subdivision periods."""
    x = np.asarray(envelope, dtype=float)
    if x.size < 8 or not np.any(x):
        return {"peak_bpm": None, "peak_strength": None, "runner_up_strength": None,
                "peak_ratio": None, "periodicity_cv_percent": None}
    x = x - np.mean(x)
    corr = librosa.autocorrelate(x, max_size=min(len(x) - 1, int(2.0 * sr / hop)))
    lo = max(1, int(round(60.0 / 180.0 * sr / hop)))
    hi = min(len(corr) - 1, int(round(60.0 / 45.0 * sr / hop)))
    if hi <= lo:
        return {"peak_bpm": None, "peak_strength": None, "runner_up_strength": None,
                "peak_ratio": None, "periodicity_cv_percent": None}
    segment = corr[lo:hi + 1]
    # Local maxima, not every neighboring lag, are candidates.
    peaks = np.where((segment[1:-1] > segment[:-2]) &
                     (segment[1:-1] >= segment[2:]))[0] + lo + 1
    if not len(peaks):
        peaks = np.array([int(np.argmax(segment)) + lo])
    ranked = peaks[np.argsort(corr[peaks])[::-1]]
    best = int(ranked[0])
    best_value = float(corr[best])
    second = float(corr[ranked[1]]) if len(ranked) > 1 else None
    ratio = best_value / second if second is not None and second > 0 else None
    bpm = 60.0 * sr / (best * hop)
    return {"peak_bpm": round(bpm, 3),
            "peak_strength": round(best_value / max(float(corr[0]), 1e-12), 5),
            "runner_up_strength": round(second / max(float(corr[0]), 1e-12), 5) if second is not None else None,
            "peak_ratio": round(ratio, 4) if ratio is not None else None,
            "periodicity_cv_percent": None}


def analyze(path: Path) -> dict:
    y, sr = librosa.load(path, sr=22050, mono=True)
    if len(y) < sr * 5:
        raise ValueError("Recording shorter than five seconds")
    hop = 256
    harmonic, percussive = librosa.effects.hpss(y)
    signals = {"full": y, "harmonic": harmonic, "percussive": percussive}
    row = {"filename": path.name, "stem": path.stem,
           "duration_seconds": round(len(y) / sr, 3),
           "analysis_version": VERSION}
    for name, signal in signals.items():
        env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop,
                                           aggregate=np.median)
        features = periodicity_features(env, sr, hop)
        for key, value in features.items():
            row[f"{name}_{key}"] = value
    # Compare tempo-periodicity evidence in consecutive overlapping windows.
    window = int(20 * sr)
    step = int(10 * sr)
    window_bpms = []
    for start in range(0, max(0, len(y) - window + 1), step):
        seg = percussive[start:start + window]
        env = librosa.onset.onset_strength(y=seg, sr=sr, hop_length=hop,
                                           aggregate=np.median)
        f = periodicity_features(env, sr, hop)
        if f["peak_bpm"] is not None:
            window_bpms.append(f["peak_bpm"])
    row["percussive_window_count"] = len(window_bpms)
    row["percussive_window_bpm_values"] = ";".join(str(x) for x in window_bpms)
    row["percussive_window_bpm_cv_percent"] = (
        round(safe_cv(np.asarray(window_bpms)), 3)
        if len(window_bpms) >= 3 and safe_cv(np.asarray(window_bpms)) is not None else None
    )
    label = ANNOTATIONS.get(path.stem)
    row["listening_label"] = label[0] if label else "unannotated"
    row["listening_label_source"] = label[1] if label else "none"
    row["algorithm_verdict"] = "not_classified"
    row["interpretation"] = (
        "Exploratory periodicity features only; do not infer source identity or "
        "self-paced status from these measurements."
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Media file or folder containing audio.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("guitar_rhythm_v050_validation"))
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
    with (args.output / "rhythm_reference_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "analysis_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "error"])
        writer.writeheader()
        writer.writerows(errors)
    print(f"Analyzed {len(rows)} recording(s); {len(errors)} error(s).")
    print(f"Validation report: {args.output / 'rhythm_reference_validation.csv'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
