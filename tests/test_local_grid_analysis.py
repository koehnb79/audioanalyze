"""Synthetic tests for the v0.9.3 local-grid engine."""
import unittest

import numpy as np

from local_grid_analysis import analyze_local_grid


def beat_grid(period=0.5, count=40, start=0.0):
    return start + np.arange(count, dtype=float) * period


class LocalGridTests(unittest.TestCase):
    def test_perfect_grid_has_near_zero_residuals(self):
        beats = beat_grid()
        onsets = beats + 0.0
        result = analyze_local_grid(onsets, beats, window_seconds=12, hop_seconds=6)
        self.assertEqual(result["local_grid_status"], "analyzed")
        self.assertLess(result["local_grid_summary"]["median_absolute_residual_ms"], 1.0)

    def test_controlled_timing_errors_preserve_signed_residuals(self):
        beats = beat_grid()
        offsets = np.resize(np.array([0.020, -0.030, 0.045, -0.010]), beats.size)
        onsets = beats + offsets
        result = analyze_local_grid(onsets, beats, window_seconds=12, hop_seconds=6)
        residuals = result["local_grid_summary"]["raw_signed_residuals_ms"]
        self.assertEqual(len(residuals), len(onsets))
        self.assertGreater(max(residuals), 15.0)
        self.assertLess(min(residuals), -25.0)

    def test_missing_beats_are_detected_without_stretching_local_period(self):
        complete = beat_grid(count=48)
        detected = np.delete(complete, [8, 9, 25])
        onsets = complete
        result = analyze_local_grid(onsets, detected, window_seconds=12, hop_seconds=6)
        self.assertGreater(result["local_grid_summary"]["missing_beat_count_estimate"], 0)
        self.assertTrue(any(e["missing_beat_flag"] for e in result["local_grid_events"]))

    def test_tempo_change_is_localized(self):
        first = np.arange(24) * 0.5
        second = first[-1] + 0.5 + np.arange(24) * 0.38
        beats = np.concatenate([first, second])
        onsets = beats.copy()
        result = analyze_local_grid(onsets, beats, window_seconds=8, hop_seconds=4)
        tempos = [w["tempo_bpm"] for w in result["local_grid_windows"]]
        self.assertGreater(max(tempos) - min(tempos), 10.0)
        self.assertGreater(result["local_grid_summary"]["tempo_change_window_count"], 0)

    def test_half_double_time_ambiguity_flag_is_reported_for_skipped_beats(self):
        complete = beat_grid(count=40)
        detected = np.delete(complete, np.arange(1, 40, 3))
        result = analyze_local_grid(complete, detected, window_seconds=12, hop_seconds=6)
        self.assertGreater(result["local_grid_summary"]["ambiguous_window_count"], 0)
        self.assertTrue(all("half_double_time_ambiguous" in w for w in result["local_grid_windows"]))

    def test_subdivision_identity_and_raw_event_fields_are_present(self):
        beats = beat_grid(count=30)
        onsets = np.sort(np.concatenate([beats, beats[:-1] + 0.25]))
        result = analyze_local_grid(onsets, beats, window_seconds=12, hop_seconds=6)
        events = result["local_grid_events"]
        self.assertTrue(events)
        self.assertTrue(all("subdivision" in e and "beat_index" in e for e in events))
        self.assertTrue(all("signed_residual_ms" in e for e in events))


if __name__ == "__main__":
    unittest.main()
