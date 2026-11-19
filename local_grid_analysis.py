"""Local beat-grid analysis for Guitar Rhythm Analyzer v0.9.3.

The grid is anchored to beat-tracker timestamps, never to the guitar onsets being
scored. Local windows allow tempo changes. Every onset retains its signed residual
and explicit beat/subdivision identity. This is a mathematical consistency measure
against an estimated pulse, not independent ground truth.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

SUBDIVISIONS = {
    1: "beat",
    2: "eighth",
    3: "triplet",
    4: "sixteenth",
    6: "sextuplet",
    8: "thirty-second",
}


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _period_candidates(intervals: np.ndarray) -> list[tuple[float, float, float]]:
    """Return (period, fit_cost, missed_beat_fraction) candidates."""
    candidates = set()
    for interval in intervals:
        for multiple in range(1, 5):
            period = float(interval / multiple)
            if 0.25 <= period <= 2.0:
                candidates.add(round(period, 6))
    scored = []
    for period in candidates:
        multiples = np.maximum(1, np.rint(intervals / period)).astype(int)
        valid = (multiples >= 1) & (multiples <= 4)
        errors = np.abs(intervals[valid] / period - multiples[valid])
        if not errors.size:
            continue
        missed = float(np.mean(multiples[valid] > 1))
        # A faster sub-pulse can explain every interval, but is penalized for
        # inventing beats. This does not erase missing-beat evidence.
        cost = float(np.median(errors) + 0.10 * np.mean(errors)
                     + 0.10 * missed)
        scored.append((period, cost, missed))
    return sorted(scored, key=lambda item: item[1])


def _fit_local_grid(beats: np.ndarray) -> dict | None:
    if beats.size < 3:
        return None
    beats = np.sort(beats)
    intervals = np.diff(beats)
    intervals = intervals[(intervals >= 0.25) & (intervals <= 8.0)]
    if intervals.size < 2:
        return None
    candidates = _period_candidates(intervals)
    if not candidates:
        return None
    period, cost, missed_fraction = candidates[0]

    # Circular mean gives a phase even when one or more beats are absent.
    angles = 2.0 * np.pi * np.mod(beats, period) / period
    vector = np.mean(np.exp(1j * angles))
    phase = float((np.angle(vector) % (2.0 * np.pi)) * period / (2.0 * np.pi))
    beat_numbers = np.rint((beats - phase) / period).astype(int)
    beat_residuals = (beats - (phase + beat_numbers * period)) * 1000.0
    median_abs = float(np.median(np.abs(beat_residuals)))

    missing = []
    for left, right in zip(beat_numbers[:-1], beat_numbers[1:]):
        if right - left > 1:
            missing.append(int(right - left - 1))

    # Missing tracker beats and octave ambiguity are distinct observations.
    # Only a near-equivalent competing period can establish octave ambiguity.
    competing = [
        p for p, c, _ in candidates[1:]
        if ((1.8 <= p / period <= 2.2) or
            (1.8 <= period / p <= 2.2)) and c <= cost + 0.08
    ]
    octave_ambiguous = bool(competing)
    return {
        "period_seconds": period,
        "bpm": 60.0 / period,
        "phase_seconds": phase,
        "fit_cost": cost,
        "beat_fit_median_abs_error_ms": median_abs,
        "missing_beat_count": int(sum(missing)),
        "half_double_time_ambiguous": bool(octave_ambiguous),
    }


def _assign_event(onset: float, grid: dict, tolerance_ms: float = 18.0) -> dict:
    """Assign the coarsest supported subdivision within a fixed timing tolerance.

    The finest grid is retained as a diagnostic candidate, but is not automatically
    selected merely because it produces a smaller residual.
    """
    period = grid["period_seconds"]
    phase = grid["phase_seconds"]
    candidates = []
    for count, label in SUBDIVISIONS.items():
        step = period / count
        index = int(np.rint((onset - phase) / step))
        predicted = phase + index * step
        residual = (onset - predicted) * 1000.0
        candidates.append({
            "count": count, "label": label, "predicted": predicted,
            "residual": residual, "index": index,
        })

    # Select the least complex grid that explains this event to within the
    # declared tolerance. If none does, retain the best candidate but call it
    # off-grid rather than silently promoting it to a very fine subdivision.
    selected = next((c for c in candidates
                     if abs(c["residual"]) <= tolerance_ms), None)
    best = min(candidates, key=lambda c: abs(c["residual"]))
    chosen = selected or best
    count = chosen["count"] if selected else 0
    label = chosen["label"] if selected else "off-grid"
    predicted = chosen["predicted"]
    residual = chosen["residual"]
    beat_index = int(math.floor((predicted - phase) / period + 1e-8))
    subdivision_index = (int(round(
        (predicted - (phase + beat_index * period)) /
        (period / count))) if count > 1 else 0)

    return {
        "onset_seconds": round(float(onset), 6),
        "predicted_grid_seconds": round(float(predicted), 6),
        "signed_residual_ms": round(float(residual), 3),
        "absolute_residual_ms": round(abs(float(residual)), 3),
        "beat_index": beat_index,
        "subdivision": label,
        "subdivisions_per_beat": count,
        "subdivision_index": subdivision_index,
        "subdivision_tolerance_ms": float(tolerance_ms),
        "subdivision_assignment_confident": bool(selected),
        "best_candidate_subdivision": best["label"],
        "best_candidate_residual_ms": round(float(best["residual"]), 3),
        "candidate_residuals_ms": {
            c["label"]: round(float(c["residual"]), 3) for c in candidates
        },
        "beat_confidence": "estimated",
        "missing_beat_flag": False,
        "half_double_time_ambiguity_flag": grid["half_double_time_ambiguous"],
    }

def analyze_local_grid(
    onset_times: Iterable[float],
    beat_times: Iterable[float],
    window_seconds: float = 12.0,
    hop_seconds: float = 6.0,
    subdivision_tolerance_ms: float = 18.0,
) -> dict:
    """Analyze events against overlapping, locally fitted pulse grids."""
    onsets = np.sort(_finite_array(onset_times))
    beats = np.sort(_finite_array(beat_times))
    empty = {"local_grid_status": "insufficient_data", "local_grid_windows": [],
             "local_grid_events": [], "local_grid_summary": {}}
    if onsets.size == 0 or beats.size < 3:
        return empty
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    start = float(min(onsets[0], beats[0]))
    end = float(max(onsets[-1], beats[-1]))
    starts = list(np.arange(start, max(start + 1e-9, end - window_seconds) + 1e-9,
                            hop_seconds))
    if not starts or starts[-1] + window_seconds < end:
        starts.append(max(start, end - window_seconds))
    windows = []
    event_choices: dict[int, tuple[float, dict]] = {}

    for window_start in sorted(set(round(float(x), 6) for x in starts)):
        window_end = window_start + window_seconds
        local_beats = beats[(beats >= window_start) & (beats <= window_end)]
        fit = _fit_local_grid(local_beats)
        if fit is None:
            continue
        local_events_idx = np.where((onsets >= window_start) &
                                    (onsets <= window_end))[0]
        window = {
            "start_seconds": round(window_start, 3),
            "end_seconds": round(window_end, 3),
            "tempo_bpm": round(fit["bpm"], 3),
            "beat_period_seconds": round(fit["period_seconds"], 6),
            "phase_seconds": round(fit["phase_seconds"], 6),
            "beat_fit_median_abs_error_ms": round(fit["beat_fit_median_abs_error_ms"], 3),
            "missing_beat_count": fit["missing_beat_count"],
            "half_double_time_ambiguous": fit["half_double_time_ambiguous"],
            "event_count": int(len(local_events_idx)),
        }
        windows.append(window)
        quality = fit["beat_fit_median_abs_error_ms"] + 100.0 * fit["fit_cost"]
        beat_numbers = np.rint((local_beats - fit["phase_seconds"]) /
                               fit["period_seconds"]).astype(int)
        beat_gaps = set()
        for left, right in zip(beat_numbers[:-1], beat_numbers[1:]):
            if right - left > 1:
                beat_gaps.update(range(int(left + 1), int(right)))
        window["missing_beat_indices"] = sorted(beat_gaps)
        for event_idx in local_events_idx:
            assigned = _assign_event(float(onsets[event_idx]), fit,
                                     tolerance_ms=subdivision_tolerance_ms)
            assigned["missing_beat_flag"] = assigned["beat_index"] in beat_gaps
            assigned["missing_beats_before_nearby"] = any(
                0 < assigned["beat_index"] - missing <= 1 for missing in beat_gaps
            )
            previous = event_choices.get(int(event_idx))
            if previous is None or quality < previous[0]:
                event_choices[int(event_idx)] = (quality, assigned)

    if not windows:
        return empty

    events = [event_choices[i][1] for i in sorted(event_choices)]
    # Detect local tempo shifts between neighboring overlapping estimates.
    for i, window in enumerate(windows):
        neighbor_tempos = []
        if i:
            neighbor_tempos.append(windows[i - 1]["tempo_bpm"])
        if i + 1 < len(windows):
            neighbor_tempos.append(windows[i + 1]["tempo_bpm"])
        window["tempo_change_flag"] = bool(
            neighbor_tempos and any(
                max(window["tempo_bpm"], bpm) / min(window["tempo_bpm"], bpm) >= 1.20
                for bpm in neighbor_tempos if bpm > 0
            )
        )
    transition_times = [
        windows[i]["start_seconds"] for i in range(1, len(windows))
        if windows[i]["tempo_change_flag"]
    ]
    for event in events:
        event["tempo_change_nearby_flag"] = any(
            abs(event["onset_seconds"] - t) <= window_seconds / 2
            for t in transition_times
        )

    residuals = np.asarray([e["signed_residual_ms"] for e in events], dtype=float)
    abs_residuals = np.abs(residuals)
    summary = {
        "event_count": int(len(events)),
        "median_absolute_residual_ms": round(float(np.median(abs_residuals)), 3),
        "median_signed_residual_ms": round(float(np.median(residuals)), 3),
        "p90_absolute_residual_ms": round(float(np.percentile(abs_residuals, 90)), 3),
        "within_50ms_percent": round(float(np.mean(abs_residuals <= 50.0) * 100), 2),
        "raw_signed_residuals_ms": np.round(residuals, 3).tolist(),
        "subdivision_counts": {
            **{label: sum(e["subdivision"] == label for e in events)
               for label in SUBDIVISIONS.values()},
            "off-grid": sum(e["subdivision"] == "off-grid" for e in events),
        },
        "event_coverage_percent": round(
            float(len(events) / onsets.size * 100.0), 2
        ),
        "unassigned_event_count": int(onsets.size - len(events)),
        "subdivision_tolerance_ms": 18.0,
        "missing_beat_count_estimate": int(len({
            beat_index for window in windows
            for beat_index in window.get("missing_beat_indices", [])
        })),
        "tempo_change_window_count": int(sum(w["tempo_change_flag"] for w in windows)),
        "ambiguous_window_count": int(sum(w["half_double_time_ambiguous"] for w in windows)),
    }
    return {
        "local_grid_status": "analyzed",
        "local_grid_window_seconds": float(window_seconds),
        "local_grid_hop_seconds": float(hop_seconds),
        "subdivision_tolerance_ms": float(subdivision_tolerance_ms),
        "local_grid_windows": windows,
        "local_grid_events": events,
        "local_grid_summary": summary,
        "local_grid_caveat": (
            "Local grids are anchored to beat timestamps estimated from this same audio. "
            "Residuals are mathematical consistency measures, not independent ground-truth "
            "timing errors. Beat/subdivision identity and ambiguity flags must be reviewed."
        ),
    }
