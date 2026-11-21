"""Experimental metronome-reference analysis.

Repeated transient signatures propose possible click events. A robust phase/period
fit allows integer-multiple gaps (missed detections), but timing regularity alone
cannot prove that a repeated guitar pattern is a metronome.
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
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(length), n=n_fft))[1:]
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
    for i in range(count):
        members = np.flatnonzero(similarity[i] >= threshold)
        if len(members) > len(best):
            best = members
    return best


def _wrapped_residuals(times: np.ndarray, period: float,
                       phase: float) -> np.ndarray:
    return (times - phase + period / 2.0) % period - period / 2.0


def _fit_period_phase_with_missed_events(clicks: np.ndarray) -> dict | None:
    """Robustly fit a periodic grid while allowing gaps of 1..N pulse periods.

    Search is intentionally bounded to 40-240 BPM (0.25-1.5 seconds). A candidate
    must have at least 8 events and 6 distinct pulse indices. We score fits by
    inlier count, residual size, and the number of apparently missing grid slots.
    The last criterion avoids preferring an artificial half-period when the data
    actually supports a slower pulse. It cannot resolve all tempo/subdivision
    ambiguities; those remain an interpretation limit.
    """
    times = np.sort(np.asarray(clicks, dtype=float))
    if times.size < 8 or not np.all(np.isfinite(times)):
        return None

    best = None
    # A 1 ms grid is adequate for this exploratory detector; refine by regression.
    for trial_period in np.arange(0.250, 1.5001, 0.001):
        angles = 2.0 * np.pi * times / trial_period
        phase = (math.atan2(float(np.mean(np.sin(angles))),
                            float(np.mean(np.cos(angles)))) *
                 trial_period / (2.0 * np.pi)) % trial_period
        residuals = _wrapped_residuals(times, trial_period, phase)
        inliers = np.abs(residuals) <= 0.030
        if int(np.sum(inliers)) < 8:
            continue

        fit_times = times[inliers]
        indices = np.rint((fit_times - phase) / trial_period).astype(int)
        if np.unique(indices).size < 6:
            continue
        slope, intercept = np.polyfit(indices, fit_times, 1)
        if not (0.250 <= slope <= 1.500) or not math.isfinite(slope):
            continue

        # Reassign and refit once using the estimated line.
        all_indices = np.rint((times - intercept) / slope).astype(int)
        all_residuals = times - (intercept + all_indices * slope)
        inliers = np.abs(all_residuals) <= 0.030
        if int(np.sum(inliers)) < 8:
            continue
        indices = all_indices[inliers]
        fit_times = times[inliers]
        if np.unique(indices).size < 6:
            continue
        slope, intercept = np.polyfit(indices, fit_times, 1)
        if not (0.250 <= slope <= 1.500) or not math.isfinite(slope):
            continue
        all_indices = np.rint((times - intercept) / slope).astype(int)
        all_residuals = times - (intercept + all_indices * slope)
        inlier_residuals = all_residuals[np.abs(all_residuals) <= 0.030]
        if inlier_residuals.size < 8:
            continue
        unique_indices = np.unique(all_indices[np.abs(all_residuals) <= 0.030])
        missing_slots = max(0, int(unique_indices[-1] - unique_indices[0] + 1
                                   - unique_indices.size))
        median_abs = float(np.median(np.abs(inlier_residuals)))
        p90_abs = float(np.percentile(np.abs(inlier_residuals), 90))
        within20 = float(np.mean(np.abs(inlier_residuals) <= 0.020))
        score = (int(inlier_residuals.size), -median_abs, -p90_abs,
                 -missing_slots, float(slope))
        if best is None or score > best["score"]:
            best = {
                "score": score, "period": float(slope),
                "phase": float(intercept), "indices": all_indices,
                "residuals": all_residuals, "inlier_mask": np.abs(all_residuals) <= 0.030,
                "median_abs_ms": median_abs * 1000.0,
                "p90_abs_ms": p90_abs * 1000.0,
                "within_20ms_percent": within20 * 100.0,
                "missing_slots": missing_slots,
            }
    return best


def analyze_reference_timing(click_times: np.ndarray,
                             guitar_onsets: np.ndarray) -> dict:
    """Compare attacks with a supplied or proposed independent click train."""
    clicks = np.sort(np.asarray(click_times, dtype=float))
    onsets = np.sort(np.asarray(guitar_onsets, dtype=float))
    empty = {
        "reference_timing_status": "insufficient_clicks",
        "reference_click_count": int(clicks.size),
        "reference_guitar_event_count": int(onsets.size),
        "reference_guitar_event_times_seconds_json": json.dumps(np.round(onsets, 4).tolist()),
        "reference_bpm": None,
        "reference_interval_cv_percent": None,
        "reference_click_fit_inlier_count": 0,
        "reference_click_fit_median_abs_residual_ms": None,
        "reference_click_fit_p90_abs_residual_ms": None,
        "reference_click_fit_within_20ms_percent": None,
        "reference_click_fit_missing_grid_slots": None,
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
    fit = _fit_period_phase_with_missed_events(clicks)
    if fit is None:
        return empty

    gaps = np.diff(clicks)
    gap_cv = (float(np.std(gaps, ddof=1) / np.mean(gaps) * 100.0)
              if gaps.size > 1 and np.mean(gaps) > 0 else None)
    inlier_count = int(np.sum(fit["inlier_mask"]))
    fit_fraction = inlier_count / clicks.size
    # Conservative gate: candidate must be phase-consistent, not merely periodic-ish.
    verified = (fit_fraction >= 0.75 and
                fit["median_abs_ms"] <= 12.0 and
                fit["p90_abs_ms"] <= 25.0 and
                fit["within_20ms_percent"] >= 75.0)
    result = dict(empty)
    result.update({
        "reference_timing_status": "analyzed" if verified else "click_train_irregular",
        "reference_bpm": round(60.0 / fit["period"], 3),
        # This is the raw observed-gap CV; missed detections can inflate it.
        "reference_interval_cv_percent": round(gap_cv, 3) if gap_cv is not None else None,
        "reference_click_fit_inlier_count": inlier_count,
        "reference_click_fit_median_abs_residual_ms": round(fit["median_abs_ms"], 3),
        "reference_click_fit_p90_abs_residual_ms": round(fit["p90_abs_ms"], 3),
        "reference_click_fit_within_20ms_percent": round(fit["within_20ms_percent"], 2),
        "reference_click_fit_missing_grid_slots": int(fit["missing_slots"]),
    })
    if not verified:
        return result

    period, phase = fit["period"], fit["phase"]
    beat_index = np.rint((onsets - phase) / period)
    beat_residuals = (onsets - (phase + beat_index * period)) * 1000.0
    sixteenth = period / 4.0
    sixteenth_index = np.rint((onsets - phase) / sixteenth)
    sixteenth_residuals = (onsets - (phase + sixteenth_index * sixteenth)) * 1000.0
    absolute = np.abs(beat_residuals)
    result.update({
        "reference_guitar_beat_median_abs_error_ms": round(float(np.median(absolute)), 3),
        "reference_guitar_beat_median_signed_error_ms": round(float(np.median(beat_residuals)), 3),
        "reference_guitar_beat_p90_abs_error_ms": round(float(np.percentile(absolute, 90)), 3),
        "reference_guitar_beat_within_10ms_percent": round(float(np.mean(absolute <= 10) * 100), 2),
        "reference_guitar_beat_within_20ms_percent": round(float(np.mean(absolute <= 20) * 100), 2),
        "reference_guitar_beat_within_30ms_percent": round(float(np.mean(absolute <= 30) * 100), 2),
        "reference_guitar_beat_within_50ms_percent": round(float(np.mean(absolute <= 50) * 100), 2),
        "reference_guitar_beat_residuals_ms_json": json.dumps(np.round(beat_residuals, 3).tolist()),
        "reference_guitar_sixteenth_median_abs_error_ms": round(float(np.median(np.abs(sixteenth_residuals))), 3),
        "reference_guitar_sixteenth_residuals_ms_json": json.dumps(np.round(sixteenth_residuals, 3).tolist()),
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
    keep = np.ones(onsets.size, dtype=bool)
    keep[cluster] = False
    guitar_candidates = onsets[keep]
    timing = analyze_reference_timing(candidate_times, guitar_candidates)
    if timing["reference_timing_status"] != "analyzed":
        result = dict(empty)
        result.update({
            "metronome_detection_status": "candidate_train_not_verified",
            "metronome_candidate_count": int(cluster.size),
            "metronome_candidate_times_seconds_json": json.dumps(np.round(candidate_times, 4).tolist()),
            "metronome_detection_note": (
                "A repeated transient signature was found, but its phase fit did not "
                "meet conservative validation thresholds. Inspect the waveform/audio."
            ),
            **{k: v for k, v in timing.items() if k.startswith("reference_click_fit_")
               or k in ("reference_bpm", "reference_interval_cv_percent",
                        "reference_timing_status")},
        })
        return result
    return {
        "metronome_detection_status": "candidate_click_train",
        "metronome_candidate_count": int(cluster.size),
        "metronome_candidate_times_seconds_json": json.dumps(np.round(candidate_times, 4).tolist()),
        "metronome_detection_note": (
            "Candidate click train inferred from repeated spectral signatures and "
            "a regular phase fit. This is not acoustic source verification; inspect "
            "audio because repeated guitar attacks can produce the same pattern."
        ),
        **timing,
    }
