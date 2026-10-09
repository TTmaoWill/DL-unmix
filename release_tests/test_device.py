"""Device regression tests; CUDA execution requires an allocated visible GPU."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dlunmix import DLUnmix, FitConfig
from dlunmix.api import _resolve_device, _train_epoch
from dlunmix.cli import main
from dlunmix.synthetic import make_synthetic


class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference, cls.target = make_synthetic()
        donors = cls.reference[0].index.tolist()
        cls.splits = dict(train_donors=donors[:12], validation_donors=donors[12:])
        cls.config = FitConfig(candidate_epochs=(1, 2))
        cls.cpu = DLUnmix(cls.config).fit(*cls.reference, **cls.splits)

    def test_explicit_cpu_matches_default_and_portable_roundtrip(self):
        explicit = DLUnmix(self.config).fit(*self.reference, **self.splits, device="cpu")
        self.assertEqual(explicit.device, torch.device("cpu"))
        for key, value in self.cpu.model_.state_dict().items():
            torch.testing.assert_close(value, explicit.model_.state_dict()[key], rtol=0, atol=0)
        expected = self.cpu.predict(*self.target[:2])
        np.testing.assert_array_equal(expected, explicit.predict(*self.target[:2], device="cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            explicit.save(path)
            weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
            self.assertTrue(all(x.device.type == "cpu" for x in weights.values()))
            for restored in (DLUnmix.load(path), DLUnmix.load(path, device="cpu")):
                np.testing.assert_array_equal(expected, restored.predict(*self.target[:2]))

    def test_invalid_and_unavailable_devices_fail_without_mutation(self):
        before = self.cpu.predict(*self.target[:2])
        for device in ("mps", "cpu:0", "cuda:-1", "not-a-device", None):
            with self.subTest(device=device), self.assertRaises(ValueError):
                DLUnmix(device=device)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA requested but unavailable"):
                self.cpu.fit(*self.reference, **self.splits, device="cuda")
            with self.assertRaisesRegex(ValueError, "CUDA requested but unavailable"):
                self.cpu.predict(*self.target[:2], device="cuda")
            with self.assertRaisesRegex(ValueError, "CUDA requested but unavailable"):
                DLUnmix.load("unused", device="cuda")
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()) as err:
                out = Path(tmp) / "demo"
                with self.assertRaises(SystemExit) as raised:
                    main(["demo", "--out", str(out), "--device", "cuda"])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("CUDA requested but unavailable", err.getvalue())
                self.assertFalse(out.exists())
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(ValueError, "index 1 is unavailable"):
                _resolve_device("cuda:1")
        self.assertEqual(self.cpu.device, torch.device("cpu"))
        np.testing.assert_array_equal(before, self.cpu.predict(*self.target[:2]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA execution skipped: no visible CUDA device")
    def test_cuda_inference_and_cpu_artifact_portability(self):
        expected = self.cpu.predict(*self.target[:2])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cpu"
            self.cpu.save(path)
            cuda = DLUnmix.load(path, device="cuda:0")
            self.assertTrue(all(t.device.type == "cuda" for t in list(cuda.model_.parameters()) + list(cuda.model_.buffers())))
            actual = cuda.predict(*self.target[:2])
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)
            cuda.save(Path(tmp) / "gpu")
            restored = DLUnmix.load(Path(tmp) / "gpu")
            np.testing.assert_array_equal(expected, restored.predict(*self.target[:2]))
            np.testing.assert_array_equal(expected, cuda.predict(*self.target[:2], device="cpu"))
            self.assertEqual(cuda.device, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA execution skipped: no visible CUDA device")
    def test_tiny_cuda_fit_tensor_placement_and_save_load(self):
        def checked_epoch(model, loader, optimizer, left, right):
            self.assertTrue(all(t.device.type == "cuda" for t in loader.dataset.tensors))
            self.assertEqual(left.device.type, "cuda")
            self.assertEqual(right.device.type, "cuda")
            return _train_epoch(model, loader, optimizer, left, right)
        with patch("dlunmix.api._train_epoch", side_effect=checked_epoch):
            fitted = DLUnmix(FitConfig(candidate_epochs=(1,))).fit(*self.reference, **self.splits, device="cuda")
        pred = fitted.predict(*self.target[:2])
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            fitted.save(path)
            cpu = DLUnmix.load(path)
            np.testing.assert_allclose(pred, cpu.predict(*self.target[:2]), rtol=1e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
