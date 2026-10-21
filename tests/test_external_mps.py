import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))

from runtime_platform import external_yue2_root, external_yue2_python
from model_registry import is_external_mps, is_upstream_torch


class ExternalMpsTest(unittest.TestCase):
    def make_project(self, root):
        for path in ('.venv/bin/python', 'models/YuE2-3B/model.safetensors',
                     'models/YuE2-Vae/model.safetensors'):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    def test_environment_override_discovers_existing_project(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'YuE2'
            project.mkdir()
            self.make_project(project)
            checkout = Path(directory) / 'fork'
            checkout.mkdir()
            with patch.dict(os.environ, {'OPENSUNO_YUE2_ROOT': str(project)}):
                self.assertEqual(external_yue2_root(checkout), project.resolve())
                self.assertEqual(external_yue2_python(checkout), project.resolve() / '.venv/bin/python')

    def test_sibling_checkout_is_discovered_without_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            self.make_project(project)
            checkout = project / 'OpenSuno-YuE2'
            checkout.mkdir()
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(external_yue2_root(checkout), project.resolve())

    def test_incomplete_project_is_not_offered(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / 'fork'
            checkout.mkdir()
            with patch.dict(os.environ, {'OPENSUNO_YUE2_ROOT': str(Path(directory) / 'missing')}):
                self.assertIsNone(external_yue2_root(checkout))

    def test_registry_classification(self):
        self.assertTrue(is_external_mps('mps-pytorch'))
        self.assertTrue(is_upstream_torch('mps-pytorch'))
        self.assertTrue(is_upstream_torch('cuda-bf16'))
        self.assertFalse(is_upstream_torch('bf16'))

    def test_nonfinite_logits_are_sanitized_in_fp32(self):
        try:
            import torch
            from mps_sampling import safe_mps_logits
        except ImportError:
            self.skipTest('torch is installed in the external YuE2 environment')
        logits = torch.tensor([[float('nan'), float('inf'), -float('inf'), 2]], dtype=torch.bfloat16)
        repaired = safe_mps_logits(logits)
        self.assertEqual(repaired.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(repaired).all()))
        self.assertEqual(repaired.tolist(), [[-10000., 10000., -10000., 2.]])


if __name__ == '__main__':
    unittest.main()
