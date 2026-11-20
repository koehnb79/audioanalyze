"""Missed-beat-tolerant global beat-grid diagnostics.

This is an additional diagnostic. It does not replace the legacy ordinal-index
fit or the overlapping local-grid analysis. Beat timestamps are still estimates
from the recording, not an independent timing reference.
"""
from __future__ import annotations

import json
import math
from typing import Iterable

import numpy as np


def _fit_candidate(beats: np.ndarray, period: float) -> dict | None:
    """Fit a candidate pulse period while allowing integer-multiple beat gaps."""
    angles = 2.0 * np.pi * np.mod(beats, period) / period
    vector = np.mean(np.exp(1j * angles))
    if abs(vector) < 1e-8:
        return None
    phase = float((np.angle(vector) % (2.0 * np.pi)) * period / (2.0 * np.pi))

    indices = np.rint((beats - phase) / period).astype(int)
    if np.unique(indices).size < 3:
        return None
    for _ in range(2):
        slope, intercept = np.polyfit(indices.astype(float), beats, 1)
        if not math.isfinite(slope) or slope <= 0:
            return None
        period = float(slope)
        phase = float(intercept)
        indices = np.rint((beats - phase) / period).astype(int)

    if np.unique(indices).size < 3:
        return None
    slope, intercept = np.polyfit(indices.astype(float), beats, 1)
    if not math.isfinite(slope) or slope <= 0:
        return None
    period = float(slope)
    phase = float(intercept)
    residuals = beats - (phase + indices * period)
    gaps = np.diff(indices)
    if np.any(gaps < 1) or np.any(gaps > 4):
        return None

    normalized = np.abs(residuals) / period
    missed_fraction = float(np.mean(gaps > 1))
    fractional_interval_error = np.abs(np.diff(beats) / period - gaps)
    cost = float(np.median(normalized) +
                 0.10 * np.mean(normalized) +
                 0.10 * np.mean(fractional_interval_error) +
                 0.10 * missed_fraction)
    return {
        "period": period,
        "phase": phase,
        "indices": indices,
        "residuals": residuals,
        "gaps": gaps,
        "missed_fraction": missed_fraction,
        "cost": cost,
    }


def missed_beat_tolerant_grid(beat_times: Iterable[float]) -> dict:
    """Estimate a global beat grid while permitting up to three missed beats.

    The candidate score penalizes unnecessary inserted beats. If a half/double
    period remains similarly plausible, the result is explicitly marked ambiguous.
    """
    beats = np.asarray(list(beat_times), dtype=float).reshape(-1)
    beats = np.sort(beats[np.isfinite(beats)])
    result = {
        "missed_beat_grid_status": "insufficient_detected_beats",
        "missed_beat_grid_bpm": None,
        "missed_beat_grid_period_seconds": None,
        "missed_beat_grid_phase_seconds": None,
        "missed_beat_grid_median_abs_error_ms": None,
        "missed_beat_grid_p90_abs_error_ms": None,
        "missed_beat_grid_missing_slots_estimate": None,
        "missed_beat_grid_octave_ambiguous": None,
        "missed_beat_grid_candidate_cost": None,
        "missed_beat_grid_residuals_ms_json": "[]",
    }
    if beats.size < 4:
        return result
    intervals = np.diff(beats)
    intervals = intervals[(intervals >= 0.25) & (intervals <= 8.0)]
    if intervals.size < 3:
        return result

    candidate_periods = {
        round(float(interval / multiple), 6)
        for interval in intervals
        for multiple in range(1, 5)
        if 0.25 <= interval / multiple <= 2.0
    }
    candidates = []
    for candidate_period in candidate_periods:
        fit = _fit_candidate(beats, candidate_period)
        if fit is not None:
            candidates.append(fit)
    if not candidates:
        result["missed_beat_grid_status"] = "no_valid_candidate"
        return result

    candidates.sort(key=lambda item: item["cost"])
    best = candidates[0]
    octave_alternatives = [
        candidate for candidate in candidates[1:]
        if (1.8 <= candidate["period"] / best["period"] <= 2.2 or
            1.8 <= best["period"] / candidate["period"] <= 2.2)
        and candidate["cost"] <= best["cost"] + 0.12
    ]
    ambiguous = bool(octave_alternatives)
    missing_slots = int(np.sum(np.maximum(best["gaps"] - 1, 0)))
    residual_ms = best["residuals"] * 1000.0
    result.update({
        "missed_beat_grid_status": (
            "octave_ambiguous" if ambiguous else "fitted_with_integer_gaps"
        ),
        "missed_beat_grid_bpm": round(60.0 / best["period"], 3),
        "missed_beat_grid_period_seconds": round(best["period"], 6),
        "missed_beat_grid_phase_seconds": round(best["phase"], 6),
        "missed_beat_grid_median_abs_error_ms": round(
            float(np.median(np.abs(residual_ms))), 3
        ),
        "missed_beat_grid_p90_abs_error_ms": round(
            float(np.percentile(np.abs(residual_ms), 90)), 3
        ),
        "missed_beat_grid_missing_slots_estimate": missing_slots,
        "missed_beat_grid_octave_ambiguous": ambiguous,
        "missed_beat_grid_candidate_cost": round(best["cost"], 6),
        "missed_beat_grid_residuals_ms_json": json.dumps(
            np.round(residual_ms, 3).tolist()
        ),
    })
    return result
