"""Synthetic tests for independent reference timing calculations."""
import unittest
import numpy as np

from metronome_reference_analysis import analyze_reference_timing


class MetronomeReferenceTests(unittest.TestCase):
    def test_perfect_beat_aligned_attacks(self):
        clicks = np.arange(40, dtype=float) * 0.5
        guitar = clicks + 0.008
        result = analyze_reference_timing(clicks, guitar)
        self.assertEqual(result["reference_timing_status"], "analyzed")
        self.assertAlmostEqual(result["reference_guitar_beat_median_signed_error_ms"], 8.0, places=2)
        self.assertAlmostEqual(result["reference_guitar_beat_median_abs_error_ms"], 8.0, places=2)

    def test_signed_offsets_distinguish_early_and_late(self):
        clicks = np.arange(40, dtype=float) * 0.5
        guitar = clicks - 0.025
        result = analyze_reference_timing(clicks, guitar)
        self.assertLess(result["reference_guitar_beat_median_signed_error_ms"], -24.0)

    def test_sixteenth_grid_is_separate_from_beat_alignment(self):
        clicks = np.arange(40, dtype=float) * 0.5
        guitar = clicks[:-1] + 0.125
        result = analyze_reference_timing(clicks, guitar)
        self.assertGreater(result["reference_guitar_beat_median_abs_error_ms"], 100)
        self.assertLess(result["reference_guitar_sixteenth_median_abs_error_ms"], 1)

    def test_missing_click_detections_allow_integer_multiple_gaps(self):
        full = np.arange(60, dtype=float) * 0.3
        clicks = np.delete(full, [5, 13, 14, 29, 42, 43, 51])
        guitar = np.arange(60, dtype=float) * 0.3 + 0.009
        result = analyze_reference_timing(clicks, guitar)
        self.assertEqual(result["reference_timing_status"], "analyzed")
        self.assertAlmostEqual(result["reference_bpm"], 200.0, delta=0.5)
        self.assertGreater(result["reference_click_fit_missing_grid_slots"], 0)
        self.assertAlmostEqual(result["reference_guitar_beat_median_signed_error_ms"], 9.0, delta=1.0)

    def test_outlier_clicks_do_not_silently_pass_validation(self):
        clicks = np.arange(40, dtype=float) * 0.5
        clicks[10] += 0.09
        result = analyze_reference_timing(clicks, np.arange(40) * 0.5)
        self.assertNotEqual(result["reference_timing_status"], "analyzed")

    def test_irregular_clicks_are_rejected(self):
        clicks = np.arange(40, dtype=float) * 0.5
        clicks[20:] += 0.08
        result = analyze_reference_timing(clicks, clicks.copy())
        self.assertEqual(result["reference_timing_status"], "click_train_irregular")

    def test_too_few_clicks_do_not_produce_accuracy(self):
        result = analyze_reference_timing(np.arange(5) * 0.5, np.arange(5) * 0.5)
        self.assertEqual(result["reference_timing_status"], "insufficient_clicks")
        self.assertIsNone(result["reference_guitar_beat_median_abs_error_ms"])


if __name__ == "__main__":
    unittest.main()
