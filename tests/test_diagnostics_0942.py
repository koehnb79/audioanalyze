import importlib.util
import sys
import tempfile
import csv
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The analyzer diagnostics tests need lightweight stand-ins while importing the
# main script, but must not leave those stand-ins in sys.modules: the real
# local-grid and metronome test modules are imported later by unittest discovery.
_saved_modules = {
    name: sys.modules.get(name)
    for name in ("local_grid_analysis", "metronome_reference_analysis")
}
try:
    for name, func in (
        ("local_grid_analysis", "analyze_local_grid"),
        ("metronome_reference_analysis", "detect_metronome_reference"),
    ):
        module = types.ModuleType(name)
        setattr(module, func, lambda *args, **kwargs: {})
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location(
        "rhythm_diagnostics", ROOT / "guitar_rhythm_analyzer.py"
    )
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
finally:
    for name, previous in _saved_modules.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class DiagnosticTests(unittest.TestCase):
    def test_two_beat_intervals_do_not_round_none(self):
        with patch.object(analyzer.librosa.onset, "onset_strength", return_value=np.ones(100)), \
             patch.object(analyzer.librosa.beat, "beat_track", return_value=(120.0, np.array([0.1, 0.6, 1.1]))), \
             patch.object(analyzer, "pulse_autocorrelation_candidates", return_value=[]), \
             patch.object(analyzer, "fitted_grid_metrics", return_value={}):
            result = analyzer.pulse_analysis(np.ones(1000), 22050, np.array([0.1, 0.6, 1.1]))
        self.assertIsNone(result["pulse_interval_cv_percent"])
        self.assertEqual(result["beat_interval_count"], 2)

    def test_csv_writer_includes_fields_added_by_later_rows(self):
        rows = [
            {"filename": "ordinary.mp3", "analysis_version": "0.9.4.2-dev"},
            {"filename": "metronome.mp3", "analysis_version": "0.9.4.2-dev",
             "reference_bpm": 120.0, "reference_timing_status": "analyzed"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            analyzer.save_results(rows, Path(folder))
            with (Path(folder) / "rhythm_measurements.csv").open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertIn("reference_bpm", reader.fieldnames)
                saved = list(reader)
        self.assertEqual(saved[0]["filename"], "ordinary.mp3")
        self.assertEqual(saved[1]["reference_bpm"], "120.0")

    def test_three_beat_intervals_have_numeric_cv(self):
        with patch.object(analyzer.librosa.onset, "onset_strength", return_value=np.ones(100)), \
             patch.object(analyzer.librosa.beat, "beat_track", return_value=(120.0, np.array([0.1, 0.6, 1.1, 1.6]))), \
             patch.object(analyzer, "pulse_autocorrelation_candidates", return_value=[]), \
             patch.object(analyzer, "fitted_grid_metrics", return_value={}):
            result = analyzer.pulse_analysis(np.ones(1000), 22050, np.array([0.1, 0.6, 1.1, 1.6]))
        self.assertEqual(result["pulse_interval_cv_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
