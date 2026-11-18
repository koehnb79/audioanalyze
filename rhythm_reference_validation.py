#!/usr/bin/env python3
"""Exploratory rhythm-reference validation for manually annotated guitar recordings.

Reports signal features only. It does not identify a sound source or decide that
a recording is self-paced. Manual listening labels remain separate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np

VERSION = "0.5.1"
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
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 3 or np.mean(values) <= 0:
        return None
    return float(np.std(values, ddof=1) / np.mean(values) * 100)


def fold_octave(bpm: float, low: float = 60.0, high: float = 120.0) -> float:
    """Fold half/double-time equivalents into a common comparison band."""
    while bpm < low:
        bpm *= 2.0
    while bpm > high:
        bpm /= 2.0
    return bpm


def periodicity_features(envelope: np.ndarray, sr: int, hop: int) -> dict:
    """Return several autocorrelation candidates; BPM is not source identity."""
    empty = {"peak_bpm": None, "peak_strength": None,
             "runner_up_strength": None, "peak_ratio": None,
             "periodicity_cv_percent": None, "candidate_bpms": "[]",
             "candidate_strengths": "[]", "octave_family_bpms": "[]"}
    x = np.asarray(envelope, dtype=float)
    if x.size < 8 or not np.any(x):
        return empty
    x = x - np.mean(x)
    corr = librosa.autocorrelate(
        x, max_size=min(len(x) - 1, int(2.0 * sr / hop))
    )
    lo = max(1, int(round(60.0 / 180.0 * sr / hop)))
    hi = min(len(corr) - 1, int(round(60.0 / 45.0 * sr / hop)))
    if hi <= lo:
        return empty
    segment = corr[lo:hi + 1]
    peaks = np.where(
        (segment[1:-1] > segment[:-2]) &
        (segment[1:-1] >= segment[2:])
    )[0] + lo + 1
    if not len(peaks):
        peaks = np.array([int(np.argmax(segment)) + lo])
    ranked = peaks[np.argsort(corr[peaks])[::-1]]
    # Keep distinct lag peaks. Neighboring peaks are not counted as separate candidates.
    selected = []
    min_separation = max(1, int(round(0.035 * sr / hop)))
    for peak in ranked:
        if all(abs(int(peak) - prior) >= min_separation for prior in selected):
            selected.append(int(peak))
        if len(selected) == 6:
            break
    strengths = [float(corr[p] / max(float(corr[0]), 1e-12)) for p in selected]
    bpms = [60.0 * sr / (p * hop) for p in selected]
    best_value = strengths[0]
    second = strengths[1] if len(strengths) > 1 else None
    ratio = best_value / second if second is not None and second > 0 else None
    family = sorted({round(fold_octave(b), 3) for b in bpms})
    return {
        "peak_bpm": round(bpms[0], 3),
        "peak_strength": round(best_value, 5),
        "runner_up_strength": round(second, 5) if second is not None else None,
        "peak_ratio": round(ratio, 4) if ratio is not None else None,
        "periodicity_cv_percent": None,
        "candidate_bpms": json.dumps([round(b, 3) for b in bpms]),
        "candidate_strengths": json.dumps([round(s, 5) for s in strengths]),
        "octave_family_bpms": json.dumps(family),
    }


def onset_envelope(signal: np.ndarray, sr: int, hop: int) -> np.ndarray:
    return librosa.onset.onset_strength(
        y=signal, sr=sr, hop_length=hop, aggregate=np.median
    )


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
        features = periodicity_features(onset_envelope(signal, sr, hop), sr, hop)
        for key, value in features.items():
            row[f"{name}_{key}"] = value

    # Overlapping 20-second windows (10-second hop). Keep all component estimates
    # and compare octave-folded candidates rather than raw half/double-time BPM.
    window, step = int(20 * sr), int(10 * sr)
    window_rows = []
    for start in range(0, max(0, len(y) - window + 1), step):
        item = {"start_seconds": round(start / sr, 2)}
        for name, signal in signals.items():
            seg = signal[start:start + window]
            f = periodicity_features(onset_envelope(seg, sr, hop), sr, hop)
            item[f"{name}_bpm"] = f["peak_bpm"]
            item[f"{name}_strength"] = f["peak_strength"]
            item[f"{name}_octave_family_bpms"] = json.loads(f["octave_family_bpms"])
        window_rows.append(item)

    row["window_count"] = len(window_rows)
    row["window_features_json"] = json.dumps(window_rows, separators=(",", ":"))
    for name in signals:
        vals = [w[f"{name}_bpm"] for w in window_rows
                if w[f"{name}_bpm"] is not None]
        folded = [fold_octave(v) for v in vals]
        row[f"{name}_window_bpm_cv_percent"] = (
            round(safe_cv(np.asarray(folded)), 3)
            if safe_cv(np.asarray(folded)) is not None else None
        )
        row[f"{name}_window_bpm_values"] = ";".join(str(round(v, 3)) for v in vals)

    # How often does percussive periodicity persist in windows where harmonic
    # onset activity is relatively low? This is a clue for review, not proof of drums.
    if window_rows:
        p_strength = [w["percussive_strength"] or 0 for w in window_rows]
        h_strength = [w["harmonic_strength"] or 0 for w in window_rows]
        row["percussive_stronger_window_fraction"] = round(
            float(np.mean(np.asarray(p_strength) > np.asarray(h_strength))), 3
        )
        row["percussive_window_strength_median"] = round(float(np.median(p_strength)), 5)
    else:
        row["percussive_stronger_window_fraction"] = None
        row["percussive_window_strength_median"] = None

    label = ANNOTATIONS.get(path.stem)
    row["listening_label"] = label[0] if label else "unannotated"
    row["listening_label_source"] = label[1] if label else "none"
    row["algorithm_verdict"] = "not_classified"
    row["interpretation"] = (
        "Exploratory periodicity features only. Candidate tempos include possible "
        "half/double-time interpretations; HPSS does not identify drums or a "
        "metronome. Review listening labels and audio; do not infer self-paced status."
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Media file or folder containing audio.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("guitar_rhythm_v051_validation"))
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
