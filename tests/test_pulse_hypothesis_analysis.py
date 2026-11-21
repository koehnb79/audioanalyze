"""Tests for non-destructive pulse hypothesis reporting."""
import json
import unittest

from pulse_hypothesis_analysis import build_pulse_hypotheses


class PulseHypothesisTests(unittest.TestCase):
    def test_reports_octave_conflict_without_selecting_a_tempo(self):
        result = build_pulse_hypotheses(
            tracker_bpm=184.0,
            phrase_bpm=92.0,
            autocorrelation_candidates=[{"bpm": 91.5}],
        )
        self.assertEqual(result["pulse_hypothesis_status"],
                         "octave_interpretation_conflict")
        estimates = json.loads(result["pulse_hypothesis_estimates_json"])
        self.assertEqual([item["bpm"] for item in estimates],
                         [184.0, 92.0, 91.5])
        self.assertIsNone(result["pulse_hypothesis_selected_bpm"])
        relationships = json.loads(
            result["pulse_hypothesis_octave_relationships_json"])
        self.assertTrue(relationships)

    def test_does_not_flag_close_non_octave_estimates(self):
        result = build_pulse_hypotheses(
            tracker_bpm=96.0,
            phrase_bpm=98.0,
            autocorrelation_candidates=[{"bpm": 97.0}],
        )
        self.assertEqual(result["pulse_hypothesis_status"], "multiple_estimates")
        self.assertEqual(json.loads(
            result["pulse_hypothesis_octave_relationships_json"]), [])

    def test_rejects_invalid_and_out_of_range_values(self):
        result = build_pulse_hypotheses(
            tracker_bpm=float("nan"),
            phrase_bpm=500,
            autocorrelation_candidates=[{"bpm": None}],
        )
        self.assertEqual(result["pulse_hypothesis_status"],
                         "insufficient_estimates")
        self.assertEqual(json.loads(
            result["pulse_hypothesis_estimates_json"]), [])


if __name__ == "__main__":
    unittest.main()
