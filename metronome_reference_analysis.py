"""Experimental independent metronome-click reference analysis.

A repeated transient can be proposed as a click train only when its spectral
signature repeats and its intervals are sufficiently regular. This is a
candidate detector, not a guaranteed source separator: mono mixtures can make
guitar attacks and metronome clicks inseparable.
"""
from __future__ import annotations

import json
import math
import numpy as np


def _transient_signatures(y: np.ndarray, sr: int, onset_times: np.ndarray,
                          frame_seconds: float = 0.045) -> np.ndarray:
    """Build normalized magnitude signatures immediately after each onset."""
    n_fft = 2048
    length = max(32, int(round(frame_seconds * sr)))
    signatures = []
    for onset in onset_times:
        start = max(0, int(round(float(onset) * sr)))
        segment = np.asarray(y[start:start + length], dtype=float)
        if segment.size < length:
            segment = np.pad(segment, (0, length - segment.size))
        window = np.hanning(length)
        spectrum = np.abs(np.fft.rfft(segment * window, n=n_fft))
        # Ignore DC and normalize so similarity emphasizes spectral shape.
        spectrum = spectrum[1:]
        norm = float(np.linalg.norm(spectrum))
        signatures.append(spectrum / norm if norm > 1e-12 else spectrum)
    return np.asarray(signatures, dtype=float)


def _largest_similarity_cluster(signatures: np.ndarray,
                                threshold: float = 0.94) -> np.ndarray:
    """Greedy medoid-like cluster of repeated transient spectral signatures."""
    count = len(signatures)
    if count < 8:
        return np.array([], dtype=int)
    similarity = signatures @ signatures.T
    best = np.array([], dtype=int)
    # Candidate medoids are actual events; this avoids a fragile global k-means.
    for i in range(count):
        members = np.flatnonzero(similarity[i] >= threshold)
        if len(members) > len(best):
            best = members
    return best


def analyze_reference_timing(click_times: np.ndarray,
                             guitar_onsets: np.ndarray) -> dict:
    """Compare attacks with a supplied independent click train."""
    clicks = np.sort(np.asarray(click_times, dtype=float))
    onsets = np.sort(np.asarray(guitar_onsets, dtype=float))
    empty = {
        "reference_timing_status": "insufficient_clicks",
        "reference_click_count": int(clicks.size),
        "reference_guitar_event_count": int(onsets.size),
        "reference_guitar_event_times_seconds_json": json.dumps(np.round(onsets, 4).tolist()),
        "reference_bpm": None,
        "reference_interval_cv_percent": None,
        "reference_guitar_beat_median_abs_error_ms": None,
        "reference_guitar_beat_median_signed_error_ms": None,
        "reference_guitar_beat_p90_abs_error_ms": None,
        "reference_guitar_beat_within_10ms_percent": None,
        "reference_guitar_beat_within_20ms_percent": None,
        "reference_guitar_beat_within_30ms_percent": None,
        "reference_guitar_beat_within_50ms_percent": None,
        "reference_guitar_beat_residuals_ms_json": "[]",
        "reference_guitar_sixteenth_median_abs_error_ms": None,
        "reference_guitar_sixteenth_residuals_ms_json": "[]",
    }
    if clicks.size < 8 or onsets.size == 0:
        return empty
    intervals = np.diff(clicks)
    valid = intervals[(intervals >= 0.25) & (intervals <= 2.0)]
    if valid.size < 6:
        return empty
    period = float(np.median(valid))
    if period <= 0:
        return empty
    interval_cv = float(np.std(valid, ddof=1) / np.mean(valid) * 100.0)
    # Robust phase/period fit with click indices. Integer gaps preserve missed-click evidence.
    indices = np.rint((clicks - clicks[0]) / period)
    if len(np.unique(indices)) < 6:
        return empty
    slope, phase = np.polyfit(indices, clicks, 1)
    if not math.isfinite(slope) or slope <= 0:
        return empty
    period = float(slope)
    click_residuals = clicks - (phase + indices * period)
    if interval_cv > 8.0 or float(np.median(np.abs(click_residuals))) > 0.012:
        result = dict(empty)
        result.update({
            "reference_timing_status": "click_train_irregular",
            "reference_bpm": round(60.0 / period, 3),
            "reference_interval_cv_percent": round(interval_cv, 3),
        })
        return result

    # Nearest beat is the strict beat-alignment measure. Sixteenth-grid results
    # are reported separately and must not be mistaken for intended note targets.
    beat_index = np.rint((onsets - phase) / period)
    beat_residuals = (onsets - (phase + beat_index * period)) * 1000.0
    sixteenth = period / 4.0
    sixteenth_index = np.rint((onsets - phase) / sixteenth)
    sixteenth_residuals = (onsets - (phase + sixteenth_index * sixteenth)) * 1000.0
    absolute = np.abs(beat_residuals)
    result = dict(empty)
    result.update({
        "reference_timing_status": "analyzed",
        "reference_bpm": round(60.0 / period, 3),
        "reference_interval_cv_percent": round(interval_cv, 3),
        "reference_guitar_beat_median_abs_error_ms": round(float(np.median(absolute)), 3),
        "reference_guitar_beat_median_signed_error_ms": round(float(np.median(beat_residuals)), 3),
        "reference_guitar_beat_p90_abs_error_ms": round(float(np.percentile(absolute, 90)), 3),
        "reference_guitar_beat_within_10ms_percent": round(float(np.mean(absolute <= 10) * 100), 2),
        "reference_guitar_beat_within_20ms_percent": round(float(np.mean(absolute <= 20) * 100), 2),
        "reference_guitar_beat_within_30ms_percent": round(float(np.mean(absolute <= 30) * 100), 2),
        "reference_guitar_beat_within_50ms_percent": round(float(np.mean(absolute <= 50) * 100), 2),
        "reference_guitar_beat_residuals_ms_json": json.dumps(np.round(beat_residuals, 3).tolist()),
        "reference_guitar_sixteenth_median_abs_error_ms": round(float(np.median(np.abs(sixteenth_residuals))), 3),
        "reference_guitar_sixteenth_residuals_ms_json": __import__("json").dumps(np.round(sixteenth_residuals, 3).tolist()),
    })
    return result


def detect_metronome_reference(y: np.ndarray, sr: int,
                               onset_times: np.ndarray) -> dict:
    """Propose a repeated click train from onset spectral similarity."""
    onsets = np.sort(np.asarray(onset_times, dtype=float))
    empty = {
        "metronome_detection_status": "insufficient_onsets",
        "metronome_candidate_count": 0,
        "metronome_candidate_times_seconds_json": "[]",
        "metronome_detection_note": "No independent click train was established.",
    }
    if onsets.size < 8:
        return empty
    signatures = _transient_signatures(np.asarray(y, dtype=float), sr, onsets)
    cluster = _largest_similarity_cluster(signatures)
    if cluster.size < max(8, int(math.ceil(onsets.size * 0.12))):
        return empty
    candidate_times = onsets[cluster]
    # Remove detected click events from the guitar-event list. If a guitar note
    # coincides with a click in a mono mix, it may also be removed; source separation
    # cannot be guaranteed from a single mixed channel.
    keep = np.ones(onsets.size, dtype=bool)
    keep[cluster] = False
    guitar_candidates = onsets[keep]
    timing = analyze_reference_timing(candidate_times, guitar_candidates)
    if timing["reference_timing_status"] != "analyzed":
        result = dict(empty)
        result.update({
            "metronome_detection_status": "candidate_train_not_verified",
            "metronome_candidate_count": int(cluster.size),
            "metronome_candidate_times_seconds_json": __import__("json").dumps(np.round(candidate_times, 4).tolist()),
            "metronome_detection_note": "A repeated transient signature was found, but its timing did not pass click-train checks.",
        })
        return result
    return {
        "metronome_detection_status": "candidate_click_train",
        "metronome_candidate_count": int(cluster.size),
        "metronome_candidate_times_seconds_json": __import__("json").dumps(np.round(candidate_times, 4).tolist()),
        "metronome_detection_note": "Candidate click train inferred from repeated spectral signatures. Review audio/diagnostic plot; mono-source separation is not guaranteed.",
        **timing,
    }
