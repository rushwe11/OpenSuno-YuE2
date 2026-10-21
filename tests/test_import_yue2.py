import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from import_yue2 import import_all
import song_continuation


class ImportYuE2Test(unittest.TestCase):
    def test_completed_song_becomes_extendable_opensuno_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'YuE2'
            source = root / 'runs/studio/job/song'
            source.mkdir(parents=True)
            audio = np.zeros((48000, 2), dtype=np.float32)
            sf.write(source / 'audio.flac', audio, 48000)
            request = {'id': 'night-song', 'style': 'R&B, piano', 'lyrics': '[Verse]\nStay',
                       'cot': 'off', 'seed': 42, 'cfg_scale': 1.01}
            (source / 'request.json').write_text(json.dumps(request))
            (source / 'result.json').write_text(json.dumps({
                'status': 'complete', 'identity': 'abc', 'audio_seconds': 1,
                'timing': {'e2e_seconds': 12}, 'weights': {'mot': {}, 'vae': {}}
            }))
            (source / 'config.json').write_text(json.dumps({'generation': {
                'abc': {'temperature': .7, 'top_p': .9, 'top_k': 30, 'repetition_penalty': 1.005,
                        'penalty_window': 100, 'min_tokens': 32, 'max_tokens': 4096},
                'semantic': {'temperature': 1, 'top_p': .95, 'top_k': 100, 'repetition_penalty': 1.2,
                             'penalty_window': 50, 'min_tokens': 25, 'max_tokens': 25},
                'ode_steps': 2,
            }}))
            np.save(source / 'semantic.npy', np.arange(25, dtype=np.int32))
            np.save(source / 'prefix.npy', np.arange(8, dtype=np.int32))
            library = Path(directory) / 'library'
            report = import_all(root, library)
            self.assertEqual(len(report['imported']), 1)
            destination = library / report['imported'][0]['id']
            self.assertTrue((destination / 'audio.wav').is_file())
            self.assertEqual(json.loads((destination / 'tokens.json').read_text()), list(range(25)))
            self.assertEqual(json.loads((destination / 'render-prefix.json').read_text()), list(range(8)))
            settings = json.loads((destination / 'settings.json').read_text())
            self.assertEqual((settings['model'], settings['title'], settings['seed']),
                             ('mps-pytorch', 'Night Song', 42))
            saved = song_continuation.song_files(destination, 1)
            self.assertEqual(saved['codec'], list(range(25)))
            self.assertEqual(saved['seconds'], 1)
            again = import_all(root, library)
            self.assertEqual(len(again['existing']), 1)


if __name__ == '__main__':
    unittest.main()
