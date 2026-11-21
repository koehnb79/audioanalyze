import unittest

import numpy as np

from missed_beat_grid_analysis import missed_beat_tolerant_grid


class MissedBeatGridTests(unittest.TestCase):
    def test_regular_grid_preserves_expected_period(self):
        beats = 0.12 + np.arange(12) * 0.5
        result = missed_beat_tolerant_grid(beats)
        # A 240 BPM pulse with every other beat missed fits the same timestamps.
        self.assertEqual(result["missed_beat_grid_status"], "octave_ambiguous")
        self.assertTrue(result["missed_beat_grid_octave_ambiguous"])
        self.assertAlmostEqual(result["missed_beat_grid_bpm"], 120.0, places=2)
        self.assertEqual(result["missed_beat_grid_missing_slots_estimate"], 0)
        self.assertLess(result["missed_beat_grid_median_abs_error_ms"], 0.01)

    def test_skipped_detection_does_not_stretch_period(self):
        beats = np.array([0.12, 0.62, 1.62, 2.12, 2.62, 3.12, 3.62])
        result = missed_beat_tolerant_grid(beats)
        self.assertAlmostEqual(result["missed_beat_grid_bpm"], 120.0, places=2)
        self.assertEqual(result["missed_beat_grid_missing_slots_estimate"], 1)
        self.assertLess(result["missed_beat_grid_median_abs_error_ms"], 0.01)

    def test_jittered_beats_retain_low_residuals(self):
        beats = np.array([0.10, 0.602, 1.098, 1.601, 2.099, 2.600, 3.102])
        result = missed_beat_tolerant_grid(beats)
        self.assertAlmostEqual(result["missed_beat_grid_bpm"], 120.0, delta=1.0)
        self.assertLess(result["missed_beat_grid_median_abs_error_ms"], 3.0)

    def test_uniformly_spaced_detections_cannot_prove_intended_pulse(self):
        # The least-cost explanation is 60 BPM, but timestamps cannot exclude
        # a 120 BPM pulse whose alternate beats were never detected.
        beats = 0.1 + np.arange(8) * 1.0
        result = missed_beat_tolerant_grid(beats)
        self.assertAlmostEqual(result["missed_beat_grid_bpm"], 60.0, places=2)
        self.assertIn("systematic missed detections",
                      result["missed_beat_grid_interpretation_note"])

    def test_two_x_faster_alias_is_reported(self):
        beats = 0.12 + np.arange(10) * 0.5
        result = missed_beat_tolerant_grid(beats)
        self.assertAlmostEqual(result["missed_beat_grid_bpm"], 120.0, places=2)
        self.assertTrue(result["missed_beat_grid_octave_ambiguous"])
        self.assertEqual(result["missed_beat_grid_status"], "octave_ambiguous")

    def test_three_x_faster_alias_is_reported(self):
        beats = 0.12 + np.arange(10) * 0.75
        result = missed_beat_tolerant_grid(beats)
        self.assertAlmostEqual(result["missed_beat_grid_bpm"], 80.0, places=2)
        self.assertTrue(result["missed_beat_grid_octave_ambiguous"])
        self.assertEqual(result["missed_beat_grid_status"], "octave_ambiguous")

    def test_too_few_beats_do_not_produce_a_fit(self):
        result = missed_beat_tolerant_grid([0.1, 0.6, 1.1])
        self.assertEqual(result["missed_beat_grid_status"], "insufficient_detected_beats")
        self.assertIsNone(result["missed_beat_grid_bpm"])


if __name__ == "__main__":
    unittest.main()
