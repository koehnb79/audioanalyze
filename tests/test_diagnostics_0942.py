import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for name, func in (
    ("local_grid_analysis", "analyze_local_grid"),
    ("metronome_reference_analysis", "detect_metronome_reference"),
):
    module = types.ModuleType(name)
    setattr(module, func, lambda *args, **kwargs: {})
    sys.modules[name] = module

spec = importlib.util.spec_from_file_location("rhythm_diagnostics", ROOT / "guitar_rhythm_analyzer.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


class DiagnosticTests(unittest.TestCase):
    def test_two_beat_intervals_do_not_round_none(self):
        with patch.object(analyzer.librosa.onset, "onset_strength", return_value=np.ones(100)), \
             patch.object(analyzer.librosa.beat, "beat_track", return_value=(120.0, np.array([0.1, 0.6, 1.1]))), \
             patch.object(analyzer, "pulse_autocorrelation_candidates", return_value=[]), \
             patch.object(analyzer, "fitted_grid_metrics", return_value={}):
            result = analyzer.pulse_analysis(np.ones(1000), 22050, np.array([0.1, 0.6, 1.1]))
        self.assertIsNone(result["pulse_interval_cv_percent"])
        self.assertEqual(result["beat_interval_count"], 2)

    def test_three_beat_intervals_have_numeric_cv(self):
        with patch.object(analyzer.librosa.onset, "onset_strength", return_value=np.ones(100)), \
             patch.object(analyzer.librosa.beat, "beat_track", return_value=(120.0, np.array([0.1, 0.6, 1.1, 1.6]))), \
             patch.object(analyzer, "pulse_autocorrelation_candidates", return_value=[]), \
             patch.object(analyzer, "fitted_grid_metrics", return_value={}):
            result = analyzer.pulse_analysis(np.ones(1000), 22050, np.array([0.1, 0.6, 1.1, 1.6]))
        self.assertEqual(result["pulse_interval_cv_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
