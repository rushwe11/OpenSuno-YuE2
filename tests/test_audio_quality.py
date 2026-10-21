import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import audio_quality


class AudioQualityTest(unittest.TestCase):
    def tone(self, frequency, seconds=1, sample_rate=48000):
        t = np.arange(seconds * sample_rate) / sample_rate
        mono = np.sin(2 * np.pi * frequency * t).astype(np.float32) * .2
        return np.column_stack([mono, mono])

    def test_normal_musical_audio_passes(self):
        audio = self.tone(440) + self.tone(110) * .5
        report = audio_quality.analyze(audio)
        self.assertTrue(report['passed'], report)
        self.assertGreater(report['energy_above_250_percent'], 2)

    def test_sub_only_failure_is_rejected(self):
        report = audio_quality.analyze(self.tone(43))
        self.assertFalse(report['passed'])
        self.assertIn('degenerate sub-bass-only output', report['reasons'])

    def test_silence_is_rejected(self):
        report = audio_quality.analyze(np.zeros((48000, 2), dtype=np.float32))
        self.assertFalse(report['passed'])
        self.assertIn('near-silent output', report['reasons'])

    def test_rejected_candidate_and_metrics_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            written = []
            with self.assertRaisesRegex(RuntimeError, 'quality gate'):
                audio_quality.check_and_save(path, self.tone(43), lambda target, audio: (written.append(target), target.touch()))
            self.assertEqual(written, [path / 'audio-rejected.wav'])
            self.assertTrue((path / 'quality.json').is_file())


if __name__ == '__main__':
    unittest.main()
