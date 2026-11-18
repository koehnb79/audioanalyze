#!/usr/bin/env python3
"""Multi-method attack diagnostics for guitar recordings; not a source classifier."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks

VERSION = "0.7.0"
ANNOTATION_FILE = Path(__file__).with_name("reference_annotations.csv")
SR = 22050
HOP = 256
FRAME = 2048
MIN_GAP = 0.075
MATCH_TOLERANCE = 0.060


def load_annotations() -> dict:
    labels = {}
    if ANNOTATION_FILE.exists():
        with ANNOTATION_FILE.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[row["filename_stem"]] = (row["manual_label"], row["label_source"])
    return labels


ANNOTATIONS = load_annotations()


def suppress_close(times: np.ndarray, strengths: np.ndarray, min_gap: float = MIN_GAP):
    if not len(times):
        return np.array([], dtype=int)
    order = np.argsort(strengths)[::-1]
    chosen = []
    for idx in order:
        if all(abs(float(times[idx]) - float(times[j])) >= min_gap for j in chosen):
            chosen.append(int(idx))
    return np.asarray(sorted(chosen), dtype=int)


def frame_times(n: int) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n), sr=SR, hop_length=HOP)


def detect_flux_events(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive spectral flux peaks with robust, recording-relative threshold."""
    S = np.abs(librosa.stft(signal, n_fft=FRAME, hop_length=HOP))
    flux = np.maximum(0.0, np.diff(S, axis=1)).sum(axis=0)
    if not len(flux) or np.max(flux) <= 0:
        return np.array([]), np.array([]), flux
    # Smooth only a little; retain sharp attacks while reducing isolated bin noise.
    smooth = np.convolve(flux, np.ones(3) / 3, mode="same") if len(flux) >= 3 else flux
    med = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - med)))
    threshold = med + max(3.0 * 1.4826 * mad, float(np.quantile(smooth, 0.75) - med))
    peaks, props = find_peaks(smooth, height=threshold, prominence=max(mad, 1e-12),
                              distance=max(1, int(MIN_GAP * SR / HOP)))
    # Align diff-derived frames to the centre of the corresponding STFT frame.
    times = librosa.frames_to_time(peaks + 1, sr=SR, hop_length=HOP)
    return times, smooth[peaks], smooth


def detect_librosa_events(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    env = librosa.onset.onset_strength(y=signal, sr=SR, hop_length=HOP)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP,
                                        backtrack=False, units="frames")
    frames = np.asarray(frames, dtype=int)
    times = librosa.frames_to_time(frames, sr=SR, hop_length=HOP)
    strengths = env[frames] if len(frames) else np.array([])
    return times, strengths


def event_descriptors(y: np.ndarray, times: np.ndarray) -> list[dict]:
    """Describe local transient spectrum and short-term energy; not instrument labels."""
    S = np.abs(librosa.stft(y, n_fft=FRAME, hop_length=HOP))
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0]
    rms = librosa.feature.rms(S=S)[0]
    tframes = frame_times(S.shape[1])
    out = []
    for t in times:
        ix = int(np.argmin(np.abs(tframes - t)))
        before, after = max(0, ix - 2), min(len(rms), ix + 4)
        base = float(np.median(rms[max(0, ix - 8):max(1, ix - 2)])) if ix >= 2 else 0.0
        peak = float(np.max(rms[before:after])) if after > before else 0.0
        out.append({
            "time_seconds": round(float(t), 4),
            "spectral_centroid_hz": round(float(cent[ix]), 1),
            "spectral_flatness": round(float(flat[ix]), 5),
            "spectral_rolloff_hz": round(float(roll[ix]), 1),
            "rms_peak": round(peak, 7),
            "rms_pre_event": round(base, 7),
            "rms_peak_to_pre": round(peak / max(base, 1e-8), 3),
        })
    return out


def nearest_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a):
        return np.array([])
    if not len(b):
        return np.full(len(a), np.nan)
    return np.min(np.abs(a[:, None] - b[None, :]), axis=1)


def summarize_times(times: np.ndarray, prefix: str, duration: float) -> dict:
    intervals = np.diff(times)
    plausible = intervals[(intervals >= 0.08) & (intervals <= 2.5)]
    med = float(np.median(plausible)) if len(plausible) else None
    cv = (float(np.std(plausible, ddof=1) / np.mean(plausible) * 100)
          if len(plausible) >= 3 and np.mean(plausible) > 0 else None)
    return {
        f"{prefix}_count": int(len(times)),
        f"{prefix}_median_interval_s": round(med, 4) if med is not None else None,
        f"{prefix}_interval_cv_pct": round(cv, 3) if cv is not None else None,
        f"{prefix}_events_per_minute": round(len(times) * 60 / max(duration, 1e-9), 2),
    }


def analyze(path: Path) -> tuple[dict, list[dict]]:
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < SR * 5:
        raise ValueError("Recording shorter than five seconds")
    harmonic, percussive = librosa.effects.hpss(y)
    signals = {"full": y, "harmonic": harmonic, "percussive": percussive}
    events = {}
    detail_rows = []
    for component, signal in signals.items():
        flux_t, flux_strength, flux_env = detect_flux_events(signal)
        librosa_t, librosa_strength = detect_librosa_events(signal)
        events[f"{component}_flux"] = flux_t
        events[f"{component}_librosa"] = librosa_t
        for detector, times, strengths in (
            ("spectral_flux", flux_t, flux_strength),
            ("librosa_onset", librosa_t, librosa_strength),
        ):
            descriptors = event_descriptors(signal, times)
            for i, desc in enumerate(descriptors):
                detail_rows.append({
                    "filename": path.name, "stem": path.stem,
                    "component": component, "detector": detector,
                    "event_index": i, **desc,
                    "detector_strength": round(float(strengths[i]), 7) if i < len(strengths) else None,
                })

    duration = len(y) / SR
    row = {"filename": path.name, "stem": path.stem,
           "duration_seconds": round(duration, 3), "analysis_version": VERSION}
    for key, times in events.items():
        row.update(summarize_times(times, key, duration))

    # Detector agreement tests whether the onset list is method-dependent.
    for component in signals:
        a, b = events[f"{component}_flux"], events[f"{component}_librosa"]
        distances = nearest_distance(a, b)
        row[f"{component}_flux_events_matching_librosa_fraction"] = (
            round(float(np.mean(distances <= MATCH_TOLERANCE)), 4) if len(distances) else None
        )

    # Compare harmonic/percussive event timing independently for each detector.
    for detector in ("flux", "librosa"):
        h, p = events[f"harmonic_{detector}"], events[f"percussive_{detector}"]
        d = nearest_distance(p, h)
        row[f"{detector}_percussive_independent_fraction"] = (
            round(float(np.mean(d > MATCH_TOLERANCE)), 4) if len(d) else None
        )
        row[f"{detector}_percussive_median_harmonic_distance_s"] = (
            round(float(np.nanmedian(d)), 4) if len(d) and np.any(np.isfinite(d)) else None
        )

    # Quiet-harmonic context uses a local RMS threshold; it is not a silence label.
    h_rms = librosa.feature.rms(y=harmonic, frame_length=FRAME, hop_length=HOP)[0]
    rms_t = frame_times(len(h_rms))
    threshold = float(np.quantile(h_rms, 0.20))
    for detector in ("flux", "librosa"):
        p = events[f"percussive_{detector}"]
        vals = np.interp(p, rms_t, h_rms) if len(p) else np.array([])
        row[f"{detector}_percussive_events_in_low_harmonic_rms_fraction"] = (
            round(float(np.mean(vals <= threshold)), 4) if len(vals) else None
        )
    row["harmonic_rms_20th_percentile"] = round(threshold, 7)

    # Windowed agreement and attack counts reveal whether the detector behaviour
    # changes during a recording. Windows are descriptive only.
    windows = []
    start, win, step = 0.0, 20.0, 10.0
    while start + win <= duration:
        w = {"start_seconds": round(start, 2)}
        for key, times in events.items():
            w[key + "_count"] = int(np.sum((times >= start) & (times < start + win)))
        windows.append(w)
        start += step
    row["window_count"] = len(windows)
    row["window_counts_json"] = json.dumps(windows, separators=(",", ":"))

    label = ANNOTATIONS.get(path.stem)
    row["listening_label"] = label[0] if label else "unannotated"
    row["listening_label_source"] = label[1] if label else "none"
    row["algorithm_verdict"] = "not_classified"
    row["interpretation"] = (
        "Multi-detector onset and transient descriptors. Spectral features are "
        "not instrument identity; HPSS components are not isolated sources. "
        "Do not infer backing-track presence or self-paced status."
    )
    return row, detail_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("guitar_rhythm_v070"))
    args = parser.parse_args()
    extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif"}
    files = ([args.input] if args.input.is_file() else
             sorted(p for p in (args.input.rglob("*") if args.recursive else args.input.iterdir())
                    if p.is_file() and p.suffix.lower() in extensions))
    summaries, details, errors = [], [], []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        try:
            summary, event_rows = analyze(path)
            summaries.append(summary)
            details.extend(event_rows)
        except Exception as exc:
            errors.append({"filename": path.name, "error": str(exc)})
            print(f"  ERROR: {exc}")
    args.output.mkdir(parents=True, exist_ok=True)
    def write_csv(name: str, rows: list[dict]):
        if not rows:
            return
        fields = list(dict.fromkeys(k for item in rows for k in item))
        with (args.output / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write_csv("rhythm_attack_summary.csv", summaries)
    write_csv("rhythm_attack_events.csv", details)
    write_csv("analysis_errors.csv", errors)
    print(f"Analyzed {len(summaries)} recording(s); {len(details)} event rows; {len(errors)} error(s).")
    print(f"Output folder: {args.output}")
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
