"""Non-destructive comparison of tempo/pulse estimates.

This module deliberately reports competing estimates rather than selecting a
"correct" musical beat. Half/double-time equivalence is an interpretation
relationship, not evidence that either estimate is the intended meter.
"""
from __future__ import annotations

import json
import math
from typing import Iterable


def build_pulse_hypotheses(
    tracker_bpm: float | None,
    phrase_bpm: float | None,
    autocorrelation_candidates: Iterable[dict] | None = None,
    min_bpm: float = 30.0,
    max_bpm: float = 240.0,
    octave_tolerance_percent: float = 3.5,
) -> dict:
    """Return source estimates and octave-equivalence groups without choosing.

    The input estimates are preserved verbatim (after numeric validation).
    Candidates are grouped when their tempos are within the configured
    tolerance of a 1:2 or 2:1 relationship. No group is declared the musical
    pulse; the audio or a human beat annotation is needed for that.
    """
    estimates = []

    def add(source: str, value) -> None:
        try:
            bpm = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(bpm) or not min_bpm <= bpm <= max_bpm:
            return
        estimates.append({"source": source, "bpm": round(bpm, 3)})

    add("beat_tracker", tracker_bpm)
    add("phrase_estimator_median", phrase_bpm)
    for index, item in enumerate(autocorrelation_candidates or []):
        if isinstance(item, dict):
            add(f"onset_autocorrelation_{index + 1}", item.get("bpm"))

    relationships = []
    for i, left in enumerate(estimates):
        for right in estimates[i + 1:]:
            ratio = left["bpm"] / right["bpm"]
            octave_ratio = min(abs(ratio / 2.0 - 1.0),
                               abs(ratio - 2.0),
                               abs(ratio - 0.5) / 0.5)
            if octave_ratio * 100.0 <= octave_tolerance_percent:
                relationships.append({
                    "source_a": left["source"],
                    "bpm_a": left["bpm"],
                    "source_b": right["source"],
                    "bpm_b": right["bpm"],
                    "relationship": "approximately_half_double_time",
                    "ratio_deviation_percent": round(octave_ratio * 100.0, 3),
                })

    has_octave_conflict = bool(relationships)
    status = "octave_interpretation_conflict" if has_octave_conflict else (
        "multiple_estimates" if len(estimates) > 1 else
        "single_estimate" if estimates else "insufficient_estimates"
    )
    return {
        "pulse_hypothesis_status": status,
        "pulse_hypothesis_estimates_json": json.dumps(
            estimates, separators=(",", ":")
        ),
        "pulse_hypothesis_octave_relationships_json": __import__("json").dumps(
            relationships, separators=(",", ":")
        ),
        "pulse_hypothesis_selected_bpm": None,
        "pulse_hypothesis_note": (
            "Competing tempo estimates are reported without selecting a musical "
            "beat. Octave-related estimates may represent half-time/double-time "
            "interpretations. Listening or independently annotated beats are "
            "required to identify the intended musical pulse."
        ),
    }
