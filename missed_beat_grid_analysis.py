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
    fractional_interval_error = np.abs(np.diff(beats) / period - gaps)
    fit_error = float(np.median(normalized) +
                      0.10 * np.mean(normalized) +
                      0.10 * np.mean(fractional_interval_error))
    missing_slots = int(np.sum(np.maximum(gaps - 1, 0)))
    total_grid_steps = int(np.sum(gaps))
    missing_slot_fraction = (
        float(missing_slots / total_grid_steps) if total_grid_steps else 0.0
    )
    # Complexity is proportional to the actual number of inserted grid slots,
    # not merely the number of intervals that contain at least one gap.
    cost = float(fit_error + 0.25 * missing_slot_fraction)
    return {
        "period": period,
        "phase": phase,
        "indices": indices,
        "residuals": residuals,
        "gaps": gaps,
        "missing_slots": missing_slots,
        "missing_slot_fraction": missing_slot_fraction,
        "fit_error": fit_error,
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
        "missed_beat_grid_pulse_alias_ambiguous": None,
        "missed_beat_grid_candidate_cost": None,
        "missed_beat_grid_residuals_ms_json": "[]",
        "missed_beat_grid_interpretation_note": (
            "The selected period is the least-cost explanation of detected beat timestamps. "
            "A 2x or 3x faster pulse with systematic missed detections may fit these timestamps "
            "equally well. The least-complex grid is reported, but timestamps alone cannot establish "
            "the intended musical beat."
        ),
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
    # Compare timing fit independently of the complexity penalty. A cleaner,
    # slower grid can win the score while a 2x/3x faster pulse fits the same
    # timestamps just as well with systematic missed detections.
    pulse_alias_alternatives = [
        (candidate, factor)
        for candidate in candidates[1:]
        for factor in (2.0, 3.0)
        if (
            abs(candidate["period"] * factor / best["period"] - 1.0) <= 0.06
            or abs(best["period"] * factor / candidate["period"] - 1.0) <= 0.06
        )
        and candidate["fit_error"] <= best["fit_error"] + 0.01
    ]
    ambiguous = bool(pulse_alias_alternatives)
    octave_ambiguous = any(factor == 2.0 for _, factor in pulse_alias_alternatives)
    missing_slots = best["missing_slots"]
    residual_ms = best["residuals"] * 1000.0
    result.update({
        "missed_beat_grid_status": (
            "pulse_alias_ambiguous" if ambiguous else "fitted_with_integer_gaps"
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
        "missed_beat_grid_octave_ambiguous": octave_ambiguous,
        "missed_beat_grid_pulse_alias_ambiguous": ambiguous,
        "missed_beat_grid_candidate_cost": round(best["cost"], 6),
        "missed_beat_grid_residuals_ms_json": json.dumps(
            np.round(residual_ms, 3).tolist()
        ),
    })
    return result
